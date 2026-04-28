import os, json, argparse
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
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
# Visualize & save output image
# ─────────────────────────────────────────────
def save_output_image(image_np, results, image_path, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)

    # Normalize to 0–255 and convert to RGB for colored annotations
    norm = image_np - image_np.min()
    if norm.max() > 0:
        norm = (norm / norm.max() * 255).astype(np.uint8)
    else:
        norm = norm.astype(np.uint8)

    pil_img = Image.fromarray(norm).convert("RGB")
    draw = ImageDraw.Draw(pil_img)

    for i, (classification, instance) in enumerate(results):
        cx = instance["centroid_x"]
        cy = instance["centroid_y"]
        radius = instance.get("radius_px", 10)   # use radius if available, else default
        p = instance.get("p_crater", 0.0)

        # Draw circle around crater
        x0, y0 = cx - radius, cy - radius
        x1, y1 = cx + radius, cy + radius
        draw.ellipse([x0, y0, x1, y1], outline=(0, 255, 0), width=2)

        # Label with index and confidence
        label = f"#{i+1} {p:.2f}"
        draw.text((x0, y0 - 14), label, fill=(0, 255, 0))

    # Build output path
    basename = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(output_dir, f"{basename}_detected.jpg")
    pil_img.save(out_path)
    print(f"[✓] Output image saved → {out_path}")

    # Also save JSON catalog
    catalog = []
    for classification, instance in results:
        entry = {
            "centroid_x": instance["centroid_x"],
            "centroid_y": instance["centroid_y"],
            "radius_px":  instance.get("radius_px"),
            "p_crater":   instance.get("p_crater"),
            "classification": str(classification),
        }
        catalog.append(entry)

    json_path = os.path.join(output_dir, f"{basename}_catalog.json")
    with open(json_path, "w") as f:
        json.dump(catalog, f, indent=2)
    print(f"[✓] Catalog saved       → {json_path}")


# ─────────────────────────────────────────────
# CLI MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",      required=True, help="Path to input image")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--output-dir", default="outputs", help="Directory for results")
    args = parser.parse_args()

    cfg = Config()

    model, schedule = load_model(args.checkpoint, cfg)
    prob_head = ProbabilityHead(model, cfg).to(cfg.DEVICE)
    pipeline  = CraterPipeline(cfg, cnn_model=None)

    image_np = np.array(Image.open(args.image).convert("L")).astype(np.float32)

    results, _ = run_inference_on_image(
        image_np, model, schedule, prob_head, pipeline, cfg
    )

    print(f"Detected {len(results)} craters!")

    if results:
        save_output_image(image_np, results, args.image, output_dir=args.output_dir)
    else:
        print("[!] No craters detected — no output image generated.")
        # Save a plain copy anyway so you can confirm the image was read correctly
        basename = os.path.splitext(os.path.basename(args.image))[0]
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, f"{basename}_no_detections.jpg")
        Image.fromarray(image_np.astype(np.uint8)).save(out_path)
        print(f"[✓] Plain image saved  → {out_path}")


if __name__ == "__main__":
    main()