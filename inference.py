# inference.py
"""
Inference for lunar crater detection
Fixed for:
- Proper checkpoint loading
- Safe CPU/GPU support
- Correct crater extraction
- Green crater circle + red center dot
- JSON + image saving
"""

import os
import json
import argparse
import numpy as np
import torch

from PIL import Image, ImageDraw
from scipy import ndimage

from config import Config
from diffusion_model import build_model
from probability_head import ProbabilityHead, extract_crater_instances
from crater_classifier import CraterPipeline


# ─────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────
def load_model(checkpoint_path, cfg):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    model, schedule = build_model(cfg)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=cfg.DEVICE
    )

    model.load_state_dict(checkpoint["model_state"])

    model.eval()

    print(
        f"Loaded checkpoint: epoch={checkpoint.get('epoch', '?')}, "
        f"val_dice={checkpoint.get('val_dice', 0):.4f}"
    )

    return model, schedule


# ─────────────────────────────────────────────
# Normalize image
# ─────────────────────────────────────────────
def norm_uint8(arr):
    arr = arr.astype(np.float32)

    arr = arr - arr.min()

    if arr.max() > 0:
        arr = (arr / arr.max()) * 255.0

    return arr.astype(np.uint8)


# ─────────────────────────────────────────────
# Core inference
# ─────────────────────────────────────────────
@torch.no_grad()
def run_inference_on_image(
    image_np,
    model,
    schedule,
    prob_head,
    pipeline,
    cfg
):
    image_tensor = torch.tensor(
        image_np,
        dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0).to(cfg.DEVICE)

    # DDIM reverse diffusion
    raw_mask = schedule.ddim_sample(
        model,
        image_tensor,
        steps=cfg.DDIM_STEPS
    )

    t = torch.zeros(
        1,
        dtype=torch.long,
        device=cfg.DEVICE
    )

    prob_out = prob_head(
        raw_mask,
        image_tensor,
        t
    )

    if not prob_out.binary_mask.any():
        return [], image_np

    # Extract crater instances
    instances = extract_crater_instances(
        prob_out.binary_mask,
        prob_out.p_mean,
        min_pixels=20
    )

    labeled_mask, _ = ndimage.label(
        prob_out.binary_mask[0, 0].cpu().numpy().astype(np.uint8)
    )

    instance_masks = []
    p_craters = []

    for inst in instances:
        cx = int(inst["centroid_x"])
        cy = int(inst["centroid_y"])

        label_id = labeled_mask[cy, cx]

        if label_id > 0:
            instance_masks.append(labeled_mask == label_id)
            p_craters.append(inst["p_crater"])

    classifications = pipeline.classify(
        image_np,
        instance_masks,
        p_craters,
        resolution_m=1.0
    )

    return list(zip(classifications, instances)), image_np


# ─────────────────────────────────────────────
# Save visual output
# ─────────────────────────────────────────────
def save_output_image(
    image_np,
    results,
    image_path,
    output_dir="outputs"
):
    os.makedirs(output_dir, exist_ok=True)

    image_rgb = Image.fromarray(
        norm_uint8(image_np)
    ).convert("RGB")

    draw = ImageDraw.Draw(image_rgb)

    color_map = {
        "fresh": (0, 255, 0),
        "degraded": (255, 255, 0),
        "heavily_degraded": (255, 165, 0),
        "overlapping": (255, 0, 0),
        "uncertain": (180, 180, 180),
    }

    catalog = []

    for i, (classification, instance) in enumerate(results):
        cx = float(instance["centroid_x"])
        cy = float(instance["centroid_y"])
        r = max(5, float(instance["radius_px"]))

        crater_type = classification.get(
            "crater_type",
            "uncertain"
        )

        color = color_map.get(
            crater_type,
            (255, 255, 255)
        )

        # Green circle
        draw.ellipse(
            [cx - r, cy - r, cx + r, cy + r],
            outline=color,
            width=2
        )

        # Red center dot
        draw.ellipse(
            [cx - 3, cy - 3, cx + 3, cy + 3],
            fill=(255, 0, 0)
        )

        # Label
        draw.text(
            (cx - r, max(0, cy - r - 12)),
            f"#{i+1}",
            fill=color
        )

        catalog.append({
            "crater_id": i + 1,
            "centroid_x": instance["centroid_x"],
            "centroid_y": instance["centroid_y"],
            "radius_px": instance["radius_px"],
            "p_crater": instance["p_crater"],
            "crater_type": crater_type,
            "age_estimate": classification.get("age_estimate"),
            "degradation_score": classification.get("degradation_score"),
        })

    basename = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    out_img = os.path.join(
        output_dir,
        f"{basename}_detected.jpg"
    )

    out_json = os.path.join(
        output_dir,
        f"{basename}_catalog.json"
    )

    image_rgb.save(out_img)

    with open(out_json, "w") as f:
        json.dump(catalog, f, indent=2)

    print(f"[*] Saved image: {out_img}")
    print(f"[*] Saved JSON : {out_json}")

    return out_img


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        required=True,
        help="Path to input image"
    )

    parser.add_argument(
        "--checkpoint",
        default="checkpoints/best_model.pt"
    )

    parser.add_argument(
        "--output-dir",
        default="outputs"
    )

    args = parser.parse_args()

    cfg = Config()

    print(f"Using device: {cfg.DEVICE}")

    model, schedule = load_model(
        args.checkpoint,
        cfg
    )

    prob_head = ProbabilityHead(
        model,
        cfg
    ).to(cfg.DEVICE)

    pipeline = CraterPipeline(
        cfg,
        cnn_model=None
    )

    image_np = np.array(
        Image.open(args.image).convert("L")
    ).astype(np.float32)

    print(f"Image shape: {image_np.shape}")

    results, _ = run_inference_on_image(
        image_np,
        model,
        schedule,
        prob_head,
        pipeline,
        cfg
    )

    print(f"Detected {len(results)} craters!")

    if results:
        save_output_image(
            image_np,
            results,
            args.image,
            args.output_dir
        )
    else:
        print("[!] No craters detected.")

        os.makedirs(args.output_dir, exist_ok=True)

        basename = os.path.splitext(
            os.path.basename(args.image)
        )[0]

        out_path = os.path.join(
            args.output_dir,
            f"{basename}_no_detections.jpg"
        )

        Image.fromarray(
            norm_uint8(image_np)
        ).save(out_path)

        print(f"[*] Saved plain image: {out_path}")


if __name__ == "__main__":
    main()