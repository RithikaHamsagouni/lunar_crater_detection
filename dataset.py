# probability_head.py

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

    def __init__(self, model, dropout_rate=0.10):
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

    def __init__(
        self,
        diffusion_model,
        cfg: Config
    ):
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
        threshold=0.20
    ):

        samples = []

        for _ in range(self.n_samples):

            logits = self.mc_model(
                raw_mask,
                image,
                t
            )

            logits = logits / self.temperature.clamp(min=1e-3)

            probs = torch.sigmoid(logits)

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

        # Smooth probabilities
        p_mean = torch.clamp(
            p_mean,
            0.0,
            1.0
        )

        # Lower threshold for sparse crater detection
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


# ─────────────────────────────────────────────
# Summary statistics
# ─────────────────────────────────────────────
def summarise_probabilities(out):

    batch_size = out.p_mean.shape[0]

    summaries = []

    for i in range(batch_size):

        pm = out.p_mean[i, 0]

        detected = out.binary_mask[i, 0].bool()

        summaries.append({

            "max_p_crater":
                pm.max().item(),

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
# Crater extraction
# ─────────────────────────────────────────────
def extract_crater_instances(
    binary_mask,
    p_mean,
    min_pixels=8
):
    """
    Connected component crater extraction
    """

    mask_np = (
        binary_mask[0, 0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.uint8)
    )

    # Morphological cleanup
    mask_np = ndimage.binary_opening(
        mask_np,
        structure=np.ones((3, 3))
    )

    mask_np = ndimage.binary_closing(
        mask_np,
        structure=np.ones((3, 3))
    )

    labeled, n_components = ndimage.label(
        mask_np
    )

    prob_map = (
        p_mean[0, 0]
        .detach()
        .cpu()
        .numpy()
    )

    craters = []

    for label_id in range(1, n_components + 1):

        region = labeled == label_id

        area = region.sum()

        if area < min_pixels:
            continue

        ys, xs = np.where(region)

        if len(xs) == 0:
            continue

        centroid_x = xs.mean()
        centroid_y = ys.mean()

        width = xs.max() - xs.min()
        height = ys.max() - ys.min()

        radius = max(width, height) / 2.0

        p_vals = prob_map[region]

        mean_prob = p_vals.mean()
        max_prob = p_vals.max()

        # Reject weak detections
        if mean_prob < 0.15:
            continue

        craters.append({

            "centroid_x":
                float(centroid_x),

            "centroid_y":
                float(centroid_y),

            "radius_px":
                float(max(3.0, radius)),

            "p_crater":
                float(mean_prob),

            "p_max":
                float(max_prob),

            "area_px":
                int(area),
        })

    # Sort by confidence
    craters = sorted(
        craters,
        key=lambda x: x["p_crater"],
        reverse=True
    )

    return craters
