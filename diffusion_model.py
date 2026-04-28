"""
diffusion_model.py — Conditional DDPM/DDIM diffusion model for
lunar crater segmentation. The NAC image is the conditioning signal;
the model learns to denoise a noisy segmentation mask back to clean binary.

Architecture:
  Encoder (NAC image) → time embedding + skip connections → Diffusion U-Net
  → denoised mask at each timestep

References:
  - Ho et al. 2020  (DDPM)
  - Song et al. 2021 (DDIM)
  - Rombach et al. 2022 (conditioning strategy)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config


# ─────────────────────────────────────────────────────────────
# Sinusoidal time embedding
# ─────────────────────────────────────────────────────────────

class SinusoidalPE(nn.Module):
    """Maps scalar timestep t → embedding vector of size `dim`."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t[:, None].float() * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)   # [B, dim]


# ─────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """Residual block conditioned on time embedding."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.drop  = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip  = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(self.drop(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Self-attention on spatial features (efficient version)."""

    def __init__(self, ch: int):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.qkv  = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        q, k, v = self.qkv(h).chunk(3, dim=1)
        scale = (C ** -0.5)
        attn = torch.einsum(
            "bci,bcj->bij",
            q.reshape(B, C, -1) * scale,
            k.reshape(B, C, -1)
        ).softmax(-1)
        out = torch.einsum("bij,bcj->bci", attn, v.reshape(B, C, -1))
        return x + self.proj(out.reshape(B, C, H, W))


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, dropout, use_attn=False):
        super().__init__()
        self.res1 = ResBlock(in_ch, out_ch, time_dim, dropout)
        self.res2 = ResBlock(out_ch, out_ch, time_dim, dropout)
        self.attn = AttentionBlock(out_ch) if use_attn else nn.Identity()
        self.down = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x, t):
        x = self.res1(x, t)
        x = self.res2(x, t)
        x = self.attn(x)
        return self.down(x), x   # (downsampled, skip)


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, time_dim, dropout, use_attn=False):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)
        self.res1 = ResBlock(in_ch + skip_ch, out_ch, time_dim, dropout)
        self.res2 = ResBlock(out_ch, out_ch, time_dim, dropout)
        self.attn = AttentionBlock(out_ch) if use_attn else nn.Identity()

    def forward(self, x, skip, t):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res1(x, t)
        x = self.res2(x, t)
        return self.attn(x)


# ─────────────────────────────────────────────────────────────
# Conditional Diffusion U-Net
# ─────────────────────────────────────────────────────────────

class CraterDiffusionUNet(nn.Module):
    """
    DDPM U-Net conditioned on the NAC image.

    Input:
      noisy_mask  [B, 1, H, W]   — noisy segmentation at timestep t
      image       [B, 1, H, W]   — conditioning NAC image
      t           [B]             — integer timestep indices

    Output:
      predicted noise  [B, 1, H, W]
    """

    def __init__(self, cfg: Config):
        super().__init__()
        d = cfg.MODEL_DIM
        T = cfg.TIMESTEPS
        dr = cfg.DROPOUT_RATE
        time_dim = d * 4

        # Time MLP
        self.time_emb = nn.Sequential(
            SinusoidalPE(d),
            nn.Linear(d, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Conditioning encoder (shares weights with decoder via skip)
        self.cond_in = nn.Conv2d(1, d, 3, padding=1)
        self.cond_d1 = nn.Conv2d(d,   d*2, 3, stride=2, padding=1)
        self.cond_d2 = nn.Conv2d(d*2, d*4, 3, stride=2, padding=1)
        self.cond_d3 = nn.Conv2d(d*4, d*8, 3, stride=2, padding=1)

        # Noisy mask input
        self.mask_in = nn.Conv2d(1, d, 3, padding=1)

        # Fuse mask + conditioning at entry
        self.fuse = nn.Conv2d(d * 2, d, 1)

        # Encoder
        self.down1 = DownBlock(d,   d*2, time_dim, dr, use_attn=False)
        self.down2 = DownBlock(d*2, d*4, time_dim, dr, use_attn=False)
        self.down3 = DownBlock(d*4, d*8, time_dim, dr, use_attn=True)

        # Bottleneck
        self.mid1 = ResBlock(d*8 + d*8, d*8, time_dim, dr)  # +cond
        self.mid_attn = AttentionBlock(d*8)
        self.mid2 = ResBlock(d*8, d*8, time_dim, dr)

        # Decoder
        self.up3 = UpBlock(d*8, d*8 + d*4, d*4, time_dim, dr, use_attn=True)
        self.up2 = UpBlock(d*4, d*4 + d*2, d*2, time_dim, dr, use_attn=False)
        self.up1 = UpBlock(d*2, d*2 + d,   d,   time_dim, dr, use_attn=False)

        # Output
        self.out_norm = nn.GroupNorm(8, d)
        self.out_conv = nn.Conv2d(d, 1, 1)

    def forward(
        self,
        noisy_mask: torch.Tensor,
        image: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:

        t_emb = self.time_emb(t)                      # [B, time_dim]

        # Conditioning pyramid
        c0 = F.silu(self.cond_in(image))
        c1 = F.silu(self.cond_d1(c0))
        c2 = F.silu(self.cond_d2(c1))
        c3 = F.silu(self.cond_d3(c2))

        # Fuse noisy mask + conditioning at full resolution
        m = F.silu(self.mask_in(noisy_mask))
        x = self.fuse(torch.cat([m, c0], dim=1))

        # Encode
        x, s1 = self.down1(x, t_emb)
        x, s2 = self.down2(x, t_emb)
        x, s3 = self.down3(x, t_emb)

        # Bottleneck (inject c3 conditioning)
        x = torch.cat([x, c3], dim=1)
        x = self.mid1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid2(x, t_emb)

        # Decode (inject conditioning into skip connections)
        x = self.up3(x, torch.cat([s3, c2], dim=1), t_emb)
        x = self.up2(x, torch.cat([s2, c1], dim=1), t_emb)
        x = self.up1(x, torch.cat([s1, c0], dim=1), t_emb)

        return self.out_conv(F.silu(self.out_norm(x)))


# ─────────────────────────────────────────────────────────────
# Noise schedule (linear beta)
# ─────────────────────────────────────────────────────────────

class NoiseSchedule:
    """Pre-computes DDPM coefficients for forward/reverse diffusion."""

    def __init__(self, cfg: Config):
        T = cfg.TIMESTEPS
        betas = torch.linspace(cfg.BETA_START, cfg.BETA_END, T)
        alphas = 1.0 - betas
        alpha_hat = torch.cumprod(alphas, dim=0)

        self.T = T
        self.betas     = betas
        self.alphas    = alphas
        self.alpha_hat = alpha_hat
        self.sqrt_alpha_hat       = alpha_hat.sqrt()
        self.sqrt_one_minus_alpha = (1 - alpha_hat).sqrt()

    def to(self, device):
        for attr in ["betas", "alphas", "alpha_hat",
                     "sqrt_alpha_hat", "sqrt_one_minus_alpha"]:
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward process: add noise to x0 at timestep t."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ah = self.sqrt_alpha_hat[t][:, None, None, None]
        sqrt_om = self.sqrt_one_minus_alpha[t][:, None, None, None]
        return sqrt_ah * x0 + sqrt_om * noise, noise

    @torch.no_grad()
    def ddim_sample(
        self,
        model: CraterDiffusionUNet,
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

            sigma = eta * ((1 - ah_t1) / (1 - ah_t) * (1 - ah_t / ah_t1)).sqrt()
            noise  = torch.randn_like(x) if eta > 0 else 0
            x = ah_t1.sqrt() * x0_pred + (1 - ah_t1 - sigma**2).sqrt() * eps + sigma * noise

        return x


def build_model(cfg: Config) -> tuple[CraterDiffusionUNet, NoiseSchedule]:
    model    = CraterDiffusionUNet(cfg).to(cfg.DEVICE)
    schedule = NoiseSchedule(cfg).to(cfg.DEVICE)
    return model, schedule