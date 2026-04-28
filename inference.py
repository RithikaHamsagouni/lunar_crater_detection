import os, json, argparse
import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy import ndimage

from config import Config
from diffusion_model import build_model
from probability_head import ProbabilityHead, extract_crater_instances
from crater_classifier import CraterPipeline, format_catalog_entry


# ─────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────
def load_model(checkpoint_path, cfg):
    model, schedule = build_model(cfg)
    ckpt = torch.load(checkpoint_path, map_location=cfg.DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint: epoch={ckpt['epoch']}, val_dice={ckpt['val_dice']:.4f}")
    return model, schedule


# ─────────────────────────────────────────────
# Core inference function
# ─────────────────────────────────────────────
@torch.no_grad()
def run_inference_on_image(image_np, model, schedule, prob_head, pipeline, cfg):
    image = torch.tensor(image_np).unsqueeze(0).unsqueeze(0).float().to(cfg.DEVICE)

    raw_mask = schedule.ddim_sample(model, image, steps=cfg.DDIM_STEPS)

    t = torch.zeros(1, dtype=torch.long, device=cfg.DEVICE)
    out = prob_head(raw_mask, image, t)

    if not out.binary_mask.any():
        return [], image_np

    instances = extract_crater_instances(out.binary_mask, out.p_mean, min_pixels=20)

    labeled, _ = ndimage.label(out.binary_mask[0, 0].cpu().numpy().astype(np.uint8))

    instance_masks, p_craters = [], []
    for inst in instances:
        lid = labeled[int(inst["centroid_y"]), int(inst["centroid_x"])]
        if lid > 0:
            instance_masks.append(labeled == lid)
            p_craters.append(inst["p_crater"])

    classifications = pipeline.classify(image_np, instance_masks, p_craters, resolution_m=1.0)

    return list(zip(classifications, instances)), image_np


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def norm_uint8(arr):
    a = arr - arr.min()
    if a.max() > 0:
        a = (a / a.max() * 255)
    return a.astype(np.uint8)


# ─────────────────────────────────────────────
# Visualize & save output image
# ─────────────────────────────────────────────
def save_output_image(image_np, results, image_path, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)

    pil_img = Image.fromarray(norm_uint8(image_np)).convert("RGB")
    draw = ImageDraw.Draw(pil_img)

    color_map = {
        "fresh":            (0, 255, 0),
        "degraded":         (0, 255, 255),
        "heavily_degraded": (0, 165, 255),
        "overlapping":      (0, 0, 255),
        "uncertain":        (255, 80, 80),
    }

    for i, (classification, instance) in enumerate(results):
        cx     = float(instance["centroid_x"])
        cy     = float(instance["centroid_y"])
        radius = float(instance["radius_px"])        # key defined in probability_head.py
        p      = float(instance["p_crater"])
        crater_type = classification.get("crater_type", "uncertain")
        color  = color_map.get(crater_type, (255, 255, 255))

        # Bounding ellipse
        x0, y0 = cx - radius, cy - radius
        x1, y1 = cx + radius, cy + radius
        draw.ellipse([x0, y0, x1, y1], outline=color, width=2)
        # Centre dot
        draw.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill=color)
        # Label
        draw.text((x0, max(0, y0 - 14)),
                  f"#{i+1} {crater_type} p={p:.2f}", fill=color)

    basename = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(output_dir, f"{basename}_detected.jpg")
    pil_img.save(out_path)
    print(f"[✓] Output image  → {os.path.abspath(out_path)}")

    # JSON catalog
    catalog = []
    for classification, instance in results:
        catalog.append({
            "centroid_x":        instance["centroid_x"],
            "centroid_y":        instance["centroid_y"],
            "radius_px":         instance["radius_px"],
            "area_px":           instance["area_px"],
            "p_crater":          instance["p_crater"],
            "p_max":             instance["p_max"],
            "crater_type":       classification.get("crater_type"),
            "degradation_score": classification.get("degradation_score"),
            "age_estimate":      classification.get("age_estimate"),
            "overlap_iou":       classification.get("overlap_iou"),
        })

    json_path = os.path.join(output_dir, f"{basename}_catalog.json")
    with open(json_path, "w") as f:
        json.dump(catalog, f, indent=2)
    print(f"[✓] JSON catalog  → {os.path.abspath(json_path)}")

    return out_path


# ─────────────────────────────────────────────
# CLI MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",      required=True, help="Path to input image")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    output_dir = args.output_dir   # argparse converts --output-dir → output_dir
    os.makedirs(output_dir, exist_ok=True)

    cfg = Config()
    print(f"Device: {cfg.DEVICE}")

    model, schedule = load_model(args.checkpoint, cfg)
    prob_head = ProbabilityHead(model, cfg).to(cfg.DEVICE)
    pipeline  = CraterPipeline(cfg, cnn_model=None)

    image_np = np.array(Image.open(args.image).convert("L")).astype(np.float32)
    print(f"Image loaded: {image_np.shape}  min={image_np.min():.1f}  max={image_np.max():.1f}")

    results, _ = run_inference_on_image(
        image_np, model, schedule, prob_head, pipeline, cfg
    )

    print(f"Detected {len(results)} craters!")

    if results:
        save_output_image(image_np, results, args.image, output_dir=output_dir)
    else:
        print("[!] No craters detected — saving plain image for verification.")
        basename = os.path.splitext(os.path.basename(args.image))[0]
        out_path = os.path.join(output_dir, f"{basename}_no_detections.jpg")
        Image.fromarray(norm_uint8(image_np)).save(out_path)
        print(f"[✓] Plain image → {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()