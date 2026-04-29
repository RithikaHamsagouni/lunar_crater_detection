# probability_head.py
"""
Probability head for lunar crater diffusion model
Fixed for:
- Stable MC Dropout
- Proper CPU/GPU compatibility
- Safer entropy calculations
- Correct crater extraction
- Better confidence estimates
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from dataclasses import dataclass
from scipy import ndimage

from config import Config


# ─────────────────────────────────────────────
# Output container
# ─────────────────────────────────────────────
@dataclass
class ProbabilisticOutput:
    p_mean: torch.Tensor
    p_std: torch.Tensor
    p_entropy: torch.Tensor
    p_aleatoric: torch.Tensor
    binary_mask: torch.Tensor
    confidence: torch.Tensor


# ─────────────────────────────────────────────
# MC Dropout Wrapper
# ─────────────────────────────────────────────
class MCDropoutWrapper(nn.Module):
    """
    Keeps dropout active during inference
    """

    def __init__(self, model, dropout_rate=0.15):
        super().__init__()

        self.model = model

        self._patch_dropout(dropout_rate)

    def _patch_dropout(self, rate):
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.p = rate

    def _enable_dropout(self):
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def forward(self, *args, **kwargs):
        self._enable_dropout()

        return self.model(*args, **kwargs)


# ─────────────────────────────────────────────
# Probability Head
# ─────────────────────────────────────────────
class ProbabilityHead(nn.Module):
    """
    Converts diffusion logits → crater probabilities
    """

    def __init__(self, diffusion_model, cfg: Config):
        super().__init__()

        self.mc_model = MCDropoutWrapper(
            diffusion_model,
            cfg.DROPOUT_RATE
        )

        self.n_samples = cfg.MC_SAMPLES
        self.device = cfg.DEVICE

        # Temperature scaling
        self.temperature = nn.Parameter(
            torch.ones(1)
        )

    @torch.no_grad()
    def forward(
        self,
        raw_mask,
        image,
        t,
        threshold=0.5
    ):
        """
        MC-Dropout inference
        """
        samples = []

        for _ in range(self.n_samples):
            logits = self.mc_model(
                raw_mask,
                image,
                t
            )

            probs = torch.sigmoid(
                logits / self.temperature.clamp(min=1e-3)
            )

            samples.append(probs)

        # [N, B, 1, H, W]
        samples = torch.stack(
            samples,
            dim=0
        )

        # Mean probability
        p_mean = samples.mean(dim=0)

        # Epistemic uncertainty
        p_std = samples.std(dim=0)

        eps = 1e-6

        # Predictive entropy
        p_entropy = -(
            p_mean * torch.log(p_mean + eps)
            + (1 - p_mean) * torch.log(1 - p_mean + eps)
        )

        # Aleatoric uncertainty
        sample_entropy = -(
            samples * torch.log(samples + eps)
            + (1 - samples) * torch.log(1 - samples + eps)
        )

        p_aleatoric = sample_entropy.mean(dim=0)

        # Confidence
        max_entropy = torch.log(
            torch.tensor(
                2.0,
                device=p_mean.device
            )
        )

        confidence = 1.0 - (
            p_entropy / max_entropy
        ).clamp(0, 1)

        # Binary mask
        binary_mask = (
            p_mean >= threshold
        ).float()

        return ProbabilisticOutput(
            p_mean=p_mean,
            p_std=p_std,
            p_entropy=p_entropy,
            p_aleatoric=p_aleatoric,
            binary_mask=binary_mask,
            confidence=confidence,
        )

    # ─────────────────────────────────────────
    # Calibration
    # ─────────────────────────────────────────
    def calibrate(
        self,
        val_logits,
        val_labels
    ):
        optimizer = torch.optim.LBFGS(
            [self.temperature],
            lr=0.01,
            max_iter=50
        )

        def closure():
            optimizer.zero_grad()

            scaled = torch.sigmoid(
                val_logits / self.temperature.clamp(min=1e-3)
            )

            loss = F.binary_cross_entropy(
                scaled,
                val_labels
            )

            loss.backward()

            return loss

        optimizer.step(closure)

        print(
            f"[Calibration] temperature={self.temperature.item():.4f}"
        )


# ─────────────────────────────────────────────
# Summaries
# ─────────────────────────────────────────────
def summarise_probabilities(out):
    """
    Converts probability maps → scalar stats
    """
    batch_size = out.p_mean.shape[0]

    summaries = []

    for i in range(batch_size):
        pm = out.p_mean[i, 0]

        detected = out.binary_mask[i, 0].bool()

        summaries.append({
            "max_p_crater": pm.max().item(),

            "mean_p_crater":
                pm[detected].mean().item()
                if detected.any()
                else 0.0,

            "pixel_coverage":
                detected.float().mean().item(),

            "mean_confidence":
                out.confidence[i, 0].mean().item(),

            "epistemic_uncertainty":
                out.p_std[i, 0].mean().item(),

            "aleatoric_uncertainty":
                out.p_aleatoric[i, 0].mean().item(),
        })

    return summaries


# ─────────────────────────────────────────────
# Crater Instance Extraction
# ─────────────────────────────────────────────
def extract_crater_instances(
    binary_mask,
    p_mean,
    min_pixels=20
):
    """
    Connected component crater extraction
    Returns:
    [
      {
        centroid_x,
        centroid_y,
        radius_px,
        p_crater,
        p_max,
        area_px
      }
    ]
    """

    mask_np = (
        binary_mask[0, 0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.uint8)
    )

    labeled, n_components = ndimage.label(mask_np)

    craters = []

    prob_map = (
        p_mean[0, 0]
        .detach()
        .cpu()
        .numpy()
    )

    for label_id in range(1, n_components + 1):
        region = labeled == label_id

        area = region.sum()

        if area < min_pixels:
            continue

        ys, xs = np.where(region)

        if len(xs) == 0 or len(ys) == 0:
            continue

        centroid_x = xs.mean()
        centroid_y = ys.mean()

        width = xs.max() - xs.min()
        height = ys.max() - ys.min()

        radius = max(width, height) / 2.0

        p_vals = prob_map[region]

        craters.append({
            "centroid_x": float(centroid_x),
            "centroid_y": float(centroid_y),

            "radius_px": float(
                max(3.0, radius)
            ),

            "p_crater": float(
                p_vals.mean()
            ),

            "p_max": float(
                p_vals.max()
            ),

            "area_px": int(area),
        })

    # Sort by confidence
    craters = sorted(
        craters,
        key=lambda x: x["p_crater"],
        reverse=True
    )

    return craters