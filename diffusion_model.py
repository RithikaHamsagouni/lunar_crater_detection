"""
diffusion_model.py — Corrected conditional diffusion model
Fixes:
- DDIM NaN bug
- UpBlock channel mismatch
- Cosine noise schedule
- Safer timestep embeddings
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


# ─────────────────────────────────────────────
# Sinusoidal time embedding
# ─────────────────────────────────────────────
class SinusoidalPE(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = max(1, self.dim // 2)

        if half == 1:
            freqs = torch.ones(1, device=t.device)
        else:
            freqs = torch.exp(
                -math.log(10000) *
                torch.arange(half, device=t.device) /
                (half - 1)
            )

        args = t[:, None].float() * freqs[None]

        emb = torch.cat([args.sin(), args.cos()], dim=-1)

        if emb.shape[1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[1]))

        return emb


# ─────────────────────────────────────────────
# Residual block
# ─────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, dropout=0.0):
        super().__init__()

        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.time_proj = nn.Linear(time_dim, out_ch)

        self.norm2 = nn.GroupNorm(8, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.skip = (
            nn.Conv2d(in_ch, out_ch, 1)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))

        t_proj = self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = h + t_proj

        h = self.conv2(
            self.dropout(F.silu(self.norm2(h)))
        )

        return h + self.skip(x)


# ─────────────────────────────────────────────
# Attention
# ─────────────────────────────────────────────
class AttentionBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()

        self.norm = nn.GroupNorm(8, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape

        h = self.norm(x)

        q, k, v = self.qkv(h).chunk(3, dim=1)

        q = q.reshape(B, C, H * W)
        k = k.reshape(B, C, H * W)
        v = v.reshape(B, C, H * W)

        scale = C ** -0.5

        attn = torch.einsum(
            "bci,bcj->bij",
            q * scale,
            k
        ).softmax(dim=-1)

        out = torch.einsum(
            "bij,bcj->bci",
            attn,
            v
        )

        out = out.reshape(B, C, H, W)

        return x + self.proj(out)


# ─────────────────────────────────────────────
# Down block
# ─────────────────────────────────────────────
class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, dropout, use_attn=False):
        super().__init__()

        self.res1 = ResBlock(in_ch, out_ch, time_dim, dropout)
        self.res2 = ResBlock(out_ch, out_ch, time_dim, dropout)

        self.attn = AttentionBlock(out_ch) if use_attn else nn.Identity()

        self.down = nn.Conv2d(
            out_ch,
            out_ch,
            kernel_size=3,
            stride=2,
            padding=1
        )

    def forward(self, x, t):
        x = self.res1(x, t)
        x = self.res2(x, t)
        x = self.attn(x)

        skip = x

        x = self.down(x)

        return x, skip


# ─────────────────────────────────────────────
# Up block (FIXED)
# ─────────────────────────────────────────────
class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, time_dim, dropout, use_attn=False):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_ch,
            out_ch,
            kernel_size=2,
            stride=2
        )

        self.res1 = ResBlock(
            out_ch + skip_ch,
            out_ch,
            time_dim,
            dropout
        )

        self.res2 = ResBlock(
            out_ch,
            out_ch,
            time_dim,
            dropout
        )

        self.attn = AttentionBlock(out_ch) if use_attn else nn.Identity()

    def forward(self, x, skip, t):
        x = self.up(x)

        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        x = torch.cat([x, skip], dim=1)

        x = self.res1(x, t)
        x = self.res2(x, t)

        return self.attn(x)


# ─────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────
class CraterDiffusionUNet(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()

        d = cfg.MODEL_DIM
        dr = cfg.DROPOUT_RATE
        time_dim = d * 4

        self.time_emb = nn.Sequential(
            SinusoidalPE(d),
            nn.Linear(d, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Conditioning encoder
        self.cond_in = nn.Conv2d(1, d, 3, padding=1)
        self.cond_d1 = nn.Conv2d(d, d * 2, 3, stride=2, padding=1)
        self.cond_d2 = nn.Conv2d(d * 2, d * 4, 3, stride=2, padding=1)
        self.cond_d3 = nn.Conv2d(d * 4, d * 8, 3, stride=2, padding=1)

        # Mask input
        self.mask_in = nn.Conv2d(1, d, 3, padding=1)

        self.fuse = nn.Conv2d(d * 2, d, 1)

        # Encoder
        self.down1 = DownBlock(d, d * 2, time_dim, dr)
        self.down2 = DownBlock(d * 2, d * 4, time_dim, dr)
        self.down3 = DownBlock(d * 4, d * 8, time_dim, dr, use_attn=True)

        # Bottleneck
        self.mid1 = ResBlock(d * 16, d * 8, time_dim, dr)
        self.mid_attn = AttentionBlock(d * 8)
        self.mid2 = ResBlock(d * 8, d * 8, time_dim, dr)

        # Decoder
        self.up3 = UpBlock(d * 8, d * 8 + d * 4, d * 4, time_dim, dr, use_attn=True)
        self.up2 = UpBlock(d * 4, d * 4 + d * 2, d * 2, time_dim, dr)
        self.up1 = UpBlock(d * 2, d * 2 + d, d, time_dim, dr)

        self.out_norm = nn.GroupNorm(8, d)
        self.out_conv = nn.Conv2d(d, 1, 1)

    def forward(self, noisy_mask, image, t):
        t_emb = self.time_emb(t)

        c0 = F.silu(self.cond_in(image))
        c1 = F.silu(self.cond_d1(c0))
        c2 = F.silu(self.cond_d2(c1))
        c3 = F.silu(self.cond_d3(c2))

        m = F.silu(self.mask_in(noisy_mask))

        x = self.fuse(torch.cat([m, c0], dim=1))

        x, s1 = self.down1(x, t_emb)
        x, s2 = self.down2(x, t_emb)
        x, s3 = self.down3(x, t_emb)

        x = torch.cat([x, c3], dim=1)

        x = self.mid1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid2(x, t_emb)

        x = self.up3(x, torch.cat([s3, c2], dim=1), t_emb)
        x = self.up2(x, torch.cat([s2, c1], dim=1), t_emb)
        x = self.up1(x, torch.cat([s1, c0], dim=1), t_emb)

        return self.out_conv(
            F.silu(self.out_norm(x))
        )


# ─────────────────────────────────────────────
# Noise schedule
# ─────────────────────────────────────────────
class NoiseSchedule:
    def __init__(self, cfg: Config):
        T = cfg.TIMESTEPS

        if cfg.USE_COSINE_SCHEDULE:
            steps = torch.arange(T + 1, dtype=torch.float32)

            f = torch.cos(
                ((steps / T) + 0.008) / 1.008 * math.pi / 2
            ) ** 2

            alpha_hat = f / f[0]

            betas = (
                1 - alpha_hat[1:] / alpha_hat[:-1]
            ).clamp(0, 0.999)

        else:
            betas = torch.linspace(
                cfg.BETA_START,
                cfg.BETA_END,
                T
            )

        alphas = 1.0 - betas
        alpha_hat = torch.cumprod(alphas, dim=0)

        self.T = T
        self.betas = betas
        self.alphas = alphas
        self.alpha_hat = alpha_hat

        self.sqrt_alpha_hat = torch.sqrt(alpha_hat)
        self.sqrt_one_minus_alpha_hat = torch.sqrt(1 - alpha_hat)

    def to(self, device):
        for attr in vars(self):
            val = getattr(self, attr)
            if torch.is_tensor(val):
                setattr(self, attr, val.to(device))
        return self
    def q_sample(self, x0, t, noise=None):
        """
        Forward diffusion process: x0 → xt
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_ah = self.sqrt_alpha_hat[t][:, None, None, None]
        sqrt_om = self.sqrt_one_minus_alpha_hat[t][:, None, None, None]

        return sqrt_ah * x0 + sqrt_om * noise, noise

    @torch.no_grad()
    def ddim_sample(
        self,
        model,
        image: torch.Tensor,
        steps: int = 50,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        DDIM reverse process (faster than DDPM).
        Returns the denoised mask [B, 1, H, W] in range [-1, 1].
        """
        B, _, H, W = image.shape
        device = image.device
        x = torch.randn(B, 1, H, W, device=device)

        step_seq = torch.linspace(self.T - 1, 0, steps, dtype=torch.long)
        for i, t_val in enumerate(step_seq):
            t = torch.full((B,), t_val, device=device, dtype=torch.long)
            eps = model(x, image, t)

            ah_t  = self.alpha_hat[t_val]
            ah_t1 = self.alpha_hat[step_seq[i + 1]] if i + 1 < len(step_seq) else torch.tensor(1.0)

            x0_pred = (x - (1 - ah_t).sqrt() * eps) / ah_t.sqrt()
            x0_pred = x0_pred.clamp(-1, 1)

            if i + 1 < len(step_seq):
                sigma = eta * ((1 - ah_t1) / (1 - ah_t) * (1 - ah_t / ah_t1)).sqrt()
                noise = torch.randn_like(x) if eta > 0 else 0
                x = ah_t1.sqrt() * x0_pred + (1 - ah_t1 - sigma**2).sqrt() * eps + sigma * noise
            else:
                x = x0_pred

        return x

# ─────────────────────────────────────────────
# Build model + noise schedule
# ─────────────────────────────────────────────
def build_model(cfg):
    """
    Creates crater diffusion model and corresponding noise schedule.
    """
    model = CraterDiffusionUNet(cfg).to(cfg.DEVICE)
    schedule = NoiseSchedule(cfg).to(cfg.DEVICE)
    return model, schedule