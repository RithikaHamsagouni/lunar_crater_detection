"""
crater_classifier.py — Classifies detected craters into:
  1. Fresh       — sharp rim, bowl-shaped, high P(crater)
  2. Degraded    — eroded/infilled, time-worn, low rim contrast
  3. Overlapping — multiple craters sharing boundary (superposition)

Each crater instance gets:
  - crater_type   : "fresh" | "degraded" | "overlapping" | "uncertain"
  - degradation_score : 0.0 (fresh) → 1.0 (heavily eroded)
  - age_estimate  : relative age label ("young" | "intermediate" | "old")
  - overlap_iou   : IoU with nearest neighbour crater
  - p_crater      : detection confidence from probability head
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from config import Config


# ─────────────────────────────────────────────────────────────
# Morphological feature extractor
# ─────────────────────────────────────────────────────────────

def compute_morphology_features(
    image_patch: np.ndarray,       # [H, W] — raw NAC intensity (normalised)
    mask: np.ndarray,              # [H, W] — binary crater mask for this instance
    resolution_m: float = 1.0,    # metres per pixel
) -> dict:
    """
    Extract shape + texture features from a single crater instance.

    Features used for degradation scoring:
      rim_sharpness     — gradient magnitude along the rim
      rim_completeness  — fraction of rim that is detectable
      depth_proxy       — central brightness depression (bowl vs flat)
      circularity       — 4π·area / perimeter² (1.0 = perfect circle)
      texture_roughness — std-dev of pixel values inside the crater
    """

    ys, xs = np.where(mask)
    if len(ys) < 4:
        return _null_features()

    cy, cx = ys.mean(), xs.mean()
    radius  = max((ys.max() - ys.min()), (xs.max() - xs.min())) / 2.0
    area    = mask.sum()

    # Perimeter via morphological dilation − mask
    dilated   = ndimage.binary_dilation(mask)
    rim_mask  = dilated & ~mask
    perimeter = rim_mask.sum() + 1e-6

    # Circularity
    circularity = (4 * np.pi * area) / (perimeter ** 2)

    # Rim sharpness — gradient magnitude along rim pixels
    gy, gx = np.gradient(image_patch)
    grad_mag = np.sqrt(gy**2 + gx**2)
    rim_sharpness = float(grad_mag[rim_mask].mean()) if rim_mask.any() else 0.0

    # Rim completeness — fraction of the expected ring that has strong gradient
    rim_thresh    = grad_mag[rim_mask].mean() * 0.5 if rim_mask.any() else 0.0
    rim_completeness = float((grad_mag[rim_mask] > rim_thresh).mean()) if rim_mask.any() else 0.0

    # Depth proxy — difference between rim intensity and crater floor intensity
    floor_mask = mask.copy()
    eroded     = ndimage.binary_erosion(mask, iterations=max(1, int(radius * 0.3)))
    if eroded.any():
        floor_intensity = image_patch[eroded].mean()
        rim_intensity   = image_patch[rim_mask].mean() if rim_mask.any() else floor_intensity
        depth_proxy     = float(rim_intensity - floor_intensity)
    else:
        depth_proxy = 0.0

    # Texture roughness
    texture_roughness = float(image_patch[mask].std())

    # Diameter in metres
    diameter_m = 2 * radius * resolution_m

    return {
        "circularity":        min(circularity, 1.0),
        "rim_sharpness":      min(rim_sharpness, 1.0),
        "rim_completeness":   rim_completeness,
        "depth_proxy":        max(0.0, depth_proxy),
        "texture_roughness":  texture_roughness,
        "area_px":            area,
        "radius_px":          radius,
        "diameter_m":         diameter_m,
    }


def _null_features() -> dict:
    return {k: 0.0 for k in [
        "circularity", "rim_sharpness", "rim_completeness",
        "depth_proxy", "texture_roughness", "area_px", "radius_px", "diameter_m"
    ]}


# ─────────────────────────────────────────────────────────────
# Degradation scorer
# ─────────────────────────────────────────────────────────────

def compute_degradation_score(features: dict) -> float:
    """
    Composite score: 0.0 = fresh, 1.0 = heavily degraded.

    Weights derived from lunar morphology literature:
    - rim_sharpness and depth_proxy are the strongest indicators
    - circularity degrades as ejecta slumps and secondary craters fill the rim
    """
    w_sharpness     = 0.35
    w_depth         = 0.30
    w_completeness  = 0.20
    w_circularity   = 0.15

    # Invert: high sharpness → low degradation
    s = (
        (1 - features["rim_sharpness"])    * w_sharpness +
        (1 - features["depth_proxy"])      * w_depth +
        (1 - features["rim_completeness"]) * w_completeness +
        (1 - features["circularity"])      * w_circularity
    )
    return float(np.clip(s, 0, 1))


def age_label(degradation_score: float) -> str:
    if degradation_score < 0.30:
        return "young"         # Copernican / Eratosthenian
    elif degradation_score < 0.65:
        return "intermediate"  # Imbrian
    else:
        return "old"           # Nectarian / Pre-Nectarian


# ─────────────────────────────────────────────────────────────
# Overlap detector
# ─────────────────────────────────────────────────────────────

def compute_pairwise_iou(masks: list[np.ndarray]) -> np.ndarray:
    """Compute IoU matrix between all crater mask pairs."""
    n = len(masks)
    iou_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            intersection = (masks[i] & masks[j]).sum()
            union        = (masks[i] | masks[j]).sum()
            iou = intersection / (union + 1e-6)
            iou_matrix[i, j] = iou
            iou_matrix[j, i] = iou
    return iou_matrix


def detect_overlapping(iou_matrix: np.ndarray, threshold: float = 0.15) -> np.ndarray:
    """Returns boolean array — True if crater i overlaps any other crater."""
    return (iou_matrix > threshold).any(axis=1)


# ─────────────────────────────────────────────────────────────
# Deep classifier (optional CNN for morphology features)
# ─────────────────────────────────────────────────────────────

class CraterTypeClassifier(nn.Module):
    """
    Lightweight CNN that classifies individual crater ROI patches.
    Input: [B, 2, 64, 64] — (image_patch, mask_patch) stacked
    Output: [B, 4] logits for [fresh, degraded, overlapping, uncertain]
    """

    CLASSES = ["fresh", "degraded", "overlapping", "uncertain"]

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    def predict_proba(self, x: torch.Tensor) -> dict:
        with torch.no_grad():
            logits = self.forward(x)
            probs  = F.softmax(logits, dim=-1)
        return {cls: probs[:, i].cpu().numpy()
                for i, cls in enumerate(self.CLASSES)}


# ─────────────────────────────────────────────────────────────
# Main classifier pipeline
# ─────────────────────────────────────────────────────────────

class CraterPipeline:
    """
    End-to-end crater classification.
    Combines morphology-based degradation scoring with IoU-based
    overlap detection and the deep CNN classifier (if available).
    """

    def __init__(self, cfg: Config, cnn_model: CraterTypeClassifier = None):
        self.cfg       = cfg
        self.cnn       = cnn_model
        self.overlap_thresh = cfg.OVERLAP_IOU
        self.fresh_thresh   = cfg.FRESH_THRESH
        self.degraded_thresh= cfg.DEGRADED_THRESH

    def classify(
        self,
        image_patch: np.ndarray,      # [H, W] normalised NAC image
        instance_masks: list[np.ndarray],  # one binary mask per crater
        p_craters: list[float],       # P(crater) from probability head
        resolution_m: float = 1.0,
    ) -> list[dict]:
        """
        Returns a list of classification dicts — one per crater instance.
        """

        if not instance_masks:
            return []

        # Step 1: Morphology features + degradation score
        results = []
        all_features = []
        for i, (mask, p) in enumerate(zip(instance_masks, p_craters)):
            feats = compute_morphology_features(image_patch, mask, resolution_m)
            deg   = compute_degradation_score(feats)
            all_features.append(feats)
            results.append({
                "instance_id":      i,
                "p_crater":         p,
                "degradation_score":deg,
                "age_estimate":     age_label(deg),
                "features":         feats,
                "crater_type":      None,   # filled below
                "overlap_iou":      0.0,
            })

        # Step 2: Pairwise overlap detection
        if len(instance_masks) > 1:
            iou_mat   = compute_pairwise_iou(instance_masks)
            overlaps  = detect_overlapping(iou_mat, self.overlap_thresh)
            max_ious  = iou_mat.max(axis=1)
        else:
            overlaps = np.array([False])
            max_ious = np.array([0.0])

        # Step 3: Assign crater types
        for i, r in enumerate(results):
            deg = r["degradation_score"]
            p   = r["p_crater"]
            r["overlap_iou"] = float(max_ious[i])

            if overlaps[i]:
                r["crater_type"] = "overlapping"
            elif p < self.degraded_thresh:
                r["crater_type"] = "uncertain"
            elif deg < (1 - self.fresh_thresh):
                r["crater_type"] = "fresh"
            elif deg < (1 - self.degraded_thresh):
                r["crater_type"] = "degraded"
            else:
                r["crater_type"] = "heavily_degraded"

        # Step 4 (optional): CNN override
        if self.cnn is not None:
            self._apply_cnn(image_patch, instance_masks, results)

        return results

    def _apply_cnn(self, image, masks, results):
        """Refine classification using the deep CNN for each crater ROI."""
        import torch
        from torchvision.transforms.functional import resize

        for i, (mask, r) in enumerate(zip(masks, results)):
            ys, xs = np.where(mask)
            if len(ys) < 4:
                continue

            y1, y2 = max(0, ys.min()-4), min(image.shape[0], ys.max()+4)
            x1, x2 = max(0, xs.min()-4), min(image.shape[1], xs.max()+4)
            roi_img  = torch.tensor(image[y1:y2, x1:x2]).float().unsqueeze(0)
            roi_mask = torch.tensor(mask [y1:y2, x1:x2]).float().unsqueeze(0)

            roi = torch.stack([
                F.interpolate(roi_img.unsqueeze(0), (64,64)).squeeze(),
                F.interpolate(roi_mask.unsqueeze(0),(64,64)).squeeze(),
            ], dim=0).unsqueeze(0)

            probs = self.cnn.predict_proba(roi)
            best  = max(probs, key=lambda k: probs[k].mean())
            r["cnn_type"]  = best
            r["cnn_probs"] = {k: float(v.mean()) for k, v in probs.items()}
            # CNN overrides morphology for overlapping and uncertain cases
            if r["crater_type"] in ("uncertain", "overlapping"):
                r["crater_type"] = best


def format_catalog_entry(result: dict, lat: float, lon: float, resolution_m: float) -> dict:
    """Convert classifier output to a GeoJSON-ready catalog entry."""
    feats = result["features"]
    return {
        "type":             "Feature",
        "geometry": {
            "type":        "Point",
            "coordinates": [lon, lat],
        },
        "properties": {
            "crater_type":       result["crater_type"],
            "p_crater":          round(result["p_crater"], 4),
            "degradation_score": round(result["degradation_score"], 4),
            "age_estimate":      result["age_estimate"],
            "overlap_iou":       round(result["overlap_iou"], 4),
            "diameter_m":        round(feats["diameter_m"], 1),
            "rim_sharpness":     round(feats["rim_sharpness"], 4),
            "rim_completeness":  round(feats["rim_completeness"], 4),
            "depth_proxy":       round(feats["depth_proxy"], 4),
            "circularity":       round(feats["circularity"], 4),
            "resolution_m_px":   resolution_m,
        }
    }