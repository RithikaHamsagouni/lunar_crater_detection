# crater_classifier.py
"""
Crater classification pipeline
Fixed for:
- Stable morphology calculations
- Better degradation scoring
- Safe overlap detection
- Proper crater type assignment
- Cleaner CNN override support
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy import ndimage

from config import Config


# ─────────────────────────────────────────────
# Null features
# ─────────────────────────────────────────────
def _null_features():
    return {
        "circularity": 0.0,
        "rim_sharpness": 0.0,
        "rim_completeness": 0.0,
        "depth_proxy": 0.0,
        "texture_roughness": 0.0,
        "area_px": 0,
        "radius_px": 0.0,
        "diameter_m": 0.0,
    }


# ─────────────────────────────────────────────
# Morphology Features
# ─────────────────────────────────────────────
def compute_morphology_features(
    image_patch,
    mask,
    resolution_m=1.0
):
    ys, xs = np.where(mask)

    if len(xs) < 4 or len(ys) < 4:
        return _null_features()

    area = float(mask.sum())

    if area <= 0:
        return _null_features()

    # Radius
    radius = max(
        xs.max() - xs.min(),
        ys.max() - ys.min()
    ) / 2.0

    # Rim mask
    dilated = ndimage.binary_dilation(mask)

    rim_mask = dilated & (~mask)

    perimeter = max(
        rim_mask.sum(),
        1
    )

    # Circularity
    circularity = (
        (4 * np.pi * area)
        / (perimeter ** 2 + 1e-6)
    )

    circularity = float(
        np.clip(circularity, 0, 1)
    )

    # Gradient
    gy, gx = np.gradient(
        image_patch.astype(np.float32)
    )

    grad_mag = np.sqrt(
        gx**2 + gy**2
    )

    # Rim sharpness
    rim_sharpness = (
        grad_mag[rim_mask].mean()
        if rim_mask.any()
        else 0.0
    )

    rim_sharpness = float(
        np.clip(rim_sharpness, 0, 1)
    )

    # Rim completeness
    if rim_mask.any():
        threshold = grad_mag[rim_mask].mean() * 0.5

        rim_completeness = (
            grad_mag[rim_mask] > threshold
        ).mean()
    else:
        rim_completeness = 0.0

    rim_completeness = float(
        np.clip(rim_completeness, 0, 1)
    )

    # Depth proxy
    eroded = ndimage.binary_erosion(
        mask,
        iterations=max(1, int(radius * 0.25))
    )

    if eroded.any() and rim_mask.any():
        floor_intensity = image_patch[
            eroded
        ].mean()

        rim_intensity = image_patch[
            rim_mask
        ].mean()

        depth_proxy = max(
            0.0,
            rim_intensity - floor_intensity
        )
    else:
        depth_proxy = 0.0

    depth_proxy = float(
        np.clip(depth_proxy, 0, 1)
    )

    # Texture
    texture_roughness = float(
        image_patch[mask].std()
    )

    diameter_m = (
        2 * radius * resolution_m
    )

    return {
        "circularity": circularity,
        "rim_sharpness": rim_sharpness,
        "rim_completeness": rim_completeness,
        "depth_proxy": depth_proxy,
        "texture_roughness": texture_roughness,
        "area_px": int(area),
        "radius_px": float(radius),
        "diameter_m": float(diameter_m),
    }


# ─────────────────────────────────────────────
# Degradation Score
# ─────────────────────────────────────────────
def compute_degradation_score(features):
    """
    0 → fresh
    1 → heavily degraded
    """

    sharpness_score = 1 - features["rim_sharpness"]
    depth_score = 1 - features["depth_proxy"]
    completeness_score = 1 - features["rim_completeness"]
    circularity_score = 1 - features["circularity"]

    score = (
        0.35 * sharpness_score
        + 0.30 * depth_score
        + 0.20 * completeness_score
        + 0.15 * circularity_score
    )

    return float(
        np.clip(score, 0, 1)
    )


# ─────────────────────────────────────────────
# Age Label
# ─────────────────────────────────────────────
def age_label(degradation_score):
    if degradation_score < 0.30:
        return "young"

    elif degradation_score < 0.65:
        return "intermediate"

    return "old"


# ─────────────────────────────────────────────
# IoU Matrix
# ─────────────────────────────────────────────
def compute_pairwise_iou(masks):
    n = len(masks)

    iou_matrix = np.zeros(
        (n, n),
        dtype=np.float32
    )

    for i in range(n):
        for j in range(i + 1, n):
            intersection = (
                masks[i] & masks[j]
            ).sum()

            union = (
                masks[i] | masks[j]
            ).sum()

            iou = (
                intersection
                / (union + 1e-6)
            )

            iou_matrix[i, j] = iou
            iou_matrix[j, i] = iou

    return iou_matrix


# ─────────────────────────────────────────────
# Overlap detection
# ─────────────────────────────────────────────
def detect_overlapping(
    iou_matrix,
    threshold=0.15
):
    return (
        iou_matrix > threshold
    ).any(axis=1)


# ─────────────────────────────────────────────
# CNN Classifier
# ─────────────────────────────────────────────
class CraterTypeClassifier(nn.Module):
    CLASSES = [
        "fresh",
        "degraded",
        "overlapping",
        "uncertain"
    ]

    def __init__(self, dropout=0.3):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(4),
        )

        self.head = nn.Sequential(
            nn.Flatten(),

            nn.Linear(
                128 * 4 * 4,
                256
            ),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(
                256,
                4
            )
        )

    def forward(self, x):
        return self.head(
            self.backbone(x)
        )

    def predict_proba(self, x):
        with torch.no_grad():
            logits = self.forward(x)

            probs = F.softmax(
                logits,
                dim=-1
            )

        return {
            cls: probs[:, i].cpu().numpy()
            for i, cls in enumerate(self.CLASSES)
        }


# ─────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────
class CraterPipeline:
    def __init__(
        self,
        cfg: Config,
        cnn_model=None
    ):
        self.cfg = cfg
        self.cnn = cnn_model

        self.overlap_thresh = cfg.OVERLAP_IOU
        self.fresh_thresh = cfg.FRESH_THRESH
        self.degraded_thresh = cfg.DEGRADED_THRESH

    def classify(
        self,
        image_patch,
        instance_masks,
        p_craters,
        resolution_m=1.0
    ):
        if not instance_masks:
            return []

        results = []

        # Step 1: Morphology
        for i, (mask, p) in enumerate(
            zip(instance_masks, p_craters)
        ):
            features = compute_morphology_features(
                image_patch,
                mask,
                resolution_m
            )

            degradation = compute_degradation_score(
                features
            )

            results.append({
                "instance_id": i,
                "p_crater": float(p),
                "features": features,
                "degradation_score": degradation,
                "age_estimate": age_label(
                    degradation
                ),
                "crater_type": None,
                "overlap_iou": 0.0,
            })

        # Step 2: Overlap
        if len(instance_masks) > 1:
            iou_matrix = compute_pairwise_iou(
                instance_masks
            )

            overlaps = detect_overlapping(
                iou_matrix,
                self.overlap_thresh
            )

            max_ious = iou_matrix.max(axis=1)

        else:
            overlaps = np.array([False])
            max_ious = np.array([0.0])

        # Step 3: Assign classes
        for i, result in enumerate(results):
            p = result["p_crater"]

            deg = result[
                "degradation_score"
            ]

            result["overlap_iou"] = float(
                max_ious[i]
            )

            if overlaps[i]:
                crater_type = "overlapping"

            elif p < self.degraded_thresh:
                crater_type = "uncertain"

            elif deg < 0.25:
                crater_type = "fresh"

            elif deg < 0.60:
                crater_type = "degraded"

            else:
                crater_type = "heavily_degraded"

            result["crater_type"] = crater_type

        # Optional CNN refinement
        if self.cnn is not None:
            self._apply_cnn(
                image_patch,
                instance_masks,
                results
            )

        return results

    # ─────────────────────────────────────────
    # CNN override
    # ─────────────────────────────────────────
    def _apply_cnn(
        self,
        image,
        masks,
        results
    ):
        for i, (mask, result) in enumerate(
            zip(masks, results)
        ):
            ys, xs = np.where(mask)

            if len(xs) < 4:
                continue

            y1 = max(0, ys.min() - 4)
            y2 = min(
                image.shape[0],
                ys.max() + 4
            )

            x1 = max(0, xs.min() - 4)
            x2 = min(
                image.shape[1],
                xs.max() + 4
            )

            roi_img = torch.tensor(
                image[y1:y2, x1:x2]
            ).float()

            roi_mask = torch.tensor(
                mask[y1:y2, x1:x2]
            ).float()

            roi_img = F.interpolate(
                roi_img.unsqueeze(0).unsqueeze(0),
                size=(64, 64)
            )

            roi_mask = F.interpolate(
                roi_mask.unsqueeze(0).unsqueeze(0),
                size=(64, 64)
            )

            roi = torch.cat(
                [roi_img, roi_mask],
                dim=1
            )

            probs = self.cnn.predict_proba(
                roi
            )

            best_class = max(
                probs,
                key=lambda k: probs[k].mean()
            )

            result["cnn_type"] = best_class

            result["cnn_probs"] = {
                k: float(v.mean())
                for k, v in probs.items()
            }

            if result["crater_type"] in [
                "uncertain",
                "overlapping"
            ]:
                result["crater_type"] = best_class