"""
probability_head.py — Converts diffusion model raw outputs into
calibrated per-pixel crater probabilities with uncertainty estimates.

Method: Monte Carlo dropout (MC-Dropout) — run the model N times with
dropout active at inference, then aggregate statistics across runs.

Outputs per pixel:
  p_mean      — mean P(crater) across MC samples
  p_std       — epistemic uncertainty (model doesn't know)
  p_entropy   — predictive entropy (combined uncertainty)
  p_aleatoric — data uncertainty (irreducible noise)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from config import Config


@dataclass
class ProbabilisticOutput:
    """All probability maps for a single image patch."""
    p_mean:      torch.Tensor   # [B, 1, H, W]  mean crater probability
    p_std:       torch.Tensor   # [B, 1, H, W]  epistemic std
    p_entropy:   torch.Tensor   # [B, 1, H, W]  predictive entropy
    p_aleatoric: torch.Tensor   # [B, 1, H, W]  aleatoric uncertainty
    binary_mask: torch.Tensor   # [B, 1, H, W]  thresholded at 0.5
    confidence:  torch.Tensor   # [B, 1, H, W]  1 - entropy (normalised)


class MCDropoutWrapper(nn.Module):
    """
    Wraps a model and keeps dropout active during inference for
    Monte Carlo uncertainty estimation.
    """

    def __init__(self, model: nn.Module, dropout_rate: float = 0.15):
        super().__init__()
        self.model = model
        self._patch_dropout(dropout_rate)

    def _patch_dropout(self, rate: float):
        """Replace all existing Dropout layers with the given rate."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.p = rate

    def _enable_dropout(self):
        """Force all Dropout layers to train mode (active during inference)."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def forward(self, *args, **kwargs):
        self._enable_dropout()
        return self.model(*args, **kwargs)


class ProbabilityHead(nn.Module):
    """
    Runs MC-Dropout inference and returns calibrated probability maps.

    Usage:
        prob_head = ProbabilityHead(model, cfg)
        out = prob_head(noisy_mask, image, t)
        print(out.p_mean)      # P(crater) per pixel
        print(out.p_std)       # uncertainty
    """

    def __init__(self, diffusion_model: nn.Module, cfg: Config):
        super().__init__()
        self.mc_model   = MCDropoutWrapper(diffusion_model, cfg.DROPOUT_RATE)
        self.n_samples  = cfg.MC_SAMPLES
        self.device     = cfg.DEVICE

        # Temperature scaling parameter (learned during calibration)
        self.temperature = nn.Parameter(torch.ones(1))

    @torch.no_grad()
    def forward(
        self,
        raw_mask: torch.Tensor,    # [B, 1, H, W] — diffusion denoised output
        image: torch.Tensor,       # [B, 1, H, W] — conditioning NAC image
        t: torch.Tensor,           # [B]           — timestep
        threshold: float = 0.5,
    ) -> ProbabilisticOutput:
        """
        Run N MC-Dropout forward passes and compute probability statistics.
        """
        samples = []

        for _ in range(self.n_samples):
            logits = self.mc_model(raw_mask, image, t)
            # Temperature-scaled sigmoid → probability
            prob = torch.sigmoid(logits / self.temperature)
            samples.append(prob)

        # Stack: [N, B, 1, H, W]
        samples = torch.stack(samples, dim=0)

        # ── Aggregate statistics ─────────────────────────────────────────
        p_mean = samples.mean(dim=0)                        # E[p]
        p_std  = samples.std(dim=0)                         # Var[p]^0.5

        # Predictive entropy: H[p] = -p log p - (1-p) log(1-p)
        eps = 1e-6
        p_entropy = -(
            p_mean * (p_mean + eps).log() +
            (1 - p_mean) * (1 - p_mean + eps).log()
        )

        # Aleatoric uncertainty: E[H(p)] across MC samples
        sample_entropy = -(
            samples * (samples + eps).log() +
            (1 - samples) * (1 - samples + eps).log()
        )
        p_aleatoric = sample_entropy.mean(dim=0)

        # Confidence: 1 − normalised entropy (0=uncertain, 1=certain)
        max_entropy = torch.log(torch.tensor(2.0))
        confidence  = 1.0 - (p_entropy / max_entropy).clamp(0, 1)

        # Binary mask at threshold
        binary_mask = (p_mean >= threshold).float()

        return ProbabilisticOutput(
            p_mean      = p_mean,
            p_std       = p_std,
            p_entropy   = p_entropy,
            p_aleatoric = p_aleatoric,
            binary_mask = binary_mask,
            confidence  = confidence,
        )

    def calibrate(self, val_logits: torch.Tensor, val_labels: torch.Tensor):
        """
        Optimise temperature scaling on a validation set.
        Minimises NLL post-hoc without retraining.
        """
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval_nll():
            optimizer.zero_grad()
            scaled = torch.sigmoid(val_logits / self.temperature)
            loss = F.binary_cross_entropy(scaled, val_labels)
            loss.backward()
            return loss

        optimizer.step(eval_nll)
        print(f"  [calibration] temperature = {self.temperature.item():.4f}")


def summarise_probabilities(out: ProbabilisticOutput) -> dict:
    """
    Compute scalar summary statistics from a ProbabilisticOutput.
    Useful for logging or the crater catalog.
    """
    b = out.p_mean.shape[0]
    results = []

    for i in range(b):
        pm = out.p_mean[i, 0]
        detected = out.binary_mask[i, 0].bool()

        results.append({
            "max_p_crater":    pm.max().item(),
            "mean_p_crater":   pm[detected].mean().item() if detected.any() else 0.0,
            "pixel_coverage":  detected.float().mean().item(),
            "mean_confidence": out.confidence[i, 0].mean().item(),
            "mean_uncertainty":out.p_std[i, 0].mean().item(),
            "epistemic_unc":   out.p_std[i, 0].mean().item(),
            "aleatoric_unc":   out.p_aleatoric[i, 0].mean().item(),
        })

    return results


def extract_crater_instances(
    binary_mask: torch.Tensor,
    p_mean: torch.Tensor,
    min_pixels: int = 20,
) -> list[dict]:
    """
    Simple connected-component extraction from binary mask.
    Returns list of crater instance dicts with centroid, radius, p_crater.

    For production use, replace with scipy.ndimage.label or
    skimage.measure.label for full connected component labelling.
    """
    from scipy import ndimage

    mask_np = binary_mask[0, 0].cpu().numpy().astype(np.uint8)
    labeled, n_components = ndimage.label(mask_np)

    craters = []
    for label_id in range(1, n_components + 1):
        region = labeled == label_id
        if region.sum() < min_pixels:
            continue

        ys, xs = np.where(region)
        cy, cx = ys.mean(), xs.mean()
        radius  = max((ys.max() - ys.min()), (xs.max() - xs.min())) / 2
        p_vals  = p_mean[0, 0].cpu().numpy()[region]

        craters.append({
            "centroid_y": float(cy),
            "centroid_x": float(cx),
            "radius_px":  float(radius),
            "p_crater":   float(p_vals.mean()),
            "p_max":      float(p_vals.max()),
            "area_px":    int(region.sum()),
        })

    return craters