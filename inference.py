import torch
import cv2
import numpy as np
import argparse
import os
import segmentation_models_pytorch as smp

parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, required=True)
parser.add_argument('--checkpoint', type=str, required=True)
parser.add_argument('--threshold', type=float, default=0.3,
                    help='Detection threshold (lower = more craters detected, default=0.3)')
parser.add_argument('--min_area', type=int, default=50,
                    help='Minimum contour area in pixels (filters noise, default=50)')
parser.add_argument('--encoder', type=str, default='resnet34',
                    help='Encoder backbone used during training (default: resnet34)')
args = parser.parse_args()

# ── Load & preprocess image ──────────────────────────────────────────────────
orig_bgr = cv2.imread(args.image)
if orig_bgr is None:
    raise FileNotFoundError(f"Could not read image: {args.image}")

gray = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2GRAY)

# CLAHE gives better local contrast than plain equalizeHist
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(gray)

# Mild Gaussian blur to reduce sensor noise before model input
blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

tensor = torch.tensor(blurred / 255.0).unsqueeze(0).unsqueeze(0).float()

# ── Load model ───────────────────────────────────────────────────────────────
# Build the architecture (must match what train.py used)
model = smp.Unet(
    encoder_name=args.encoder,
    encoder_weights=None,   # no pretrained weights needed — we load from checkpoint
    in_channels=1,          # grayscale input
    classes=1,              # binary crater / no-crater output
)

# Load checkpoint — handles both full model saves and state_dict saves
ckpt = torch.load(args.checkpoint, map_location='cpu')

if isinstance(ckpt, dict):
    # Training checkpoint with extra keys (epoch, optimizer, loss, etc.)
    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    elif 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        # Assume the dict itself IS the state_dict
        state_dict = ckpt
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        # Strip 'module.' prefix added by DataParallel training
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
else:
    # Full model object saved with torch.save(model, ...)
    model = ckpt

model.eval()

with torch.no_grad():
    prob_map = model(tensor)[0][0].numpy()   # shape: (H, W), values 0–1

# ── Post-process probability map ─────────────────────────────────────────────
# 1. Smooth the probability map to merge nearby high-confidence regions
prob_smooth = cv2.GaussianBlur(prob_map, (5, 5), 0)

# 2. Threshold at user-defined level (default 0.3 instead of 0.7)
binary = (prob_smooth > args.threshold).astype(np.uint8) * 255

# 3. Morphological closing to fill small holes inside crater rims
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
binary_closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# 4. Remove tiny speckles (noise)
binary_clean = cv2.morphologyEx(binary_closed, cv2.MORPH_OPEN, kernel)

# ── Find contours & draw detections ──────────────────────────────────────────
contours, _ = cv2.findContours(binary_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

output_img = orig_bgr.copy()
detected_count = 0

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < args.min_area:
        continue  # skip noise

    # Fit a minimum enclosing circle for each detected region
    (cx, cy), radius = cv2.minEnclosingCircle(cnt)
    cx, cy, radius = int(cx), int(cy), int(radius)

    # Circularity check: real craters are roughly circular
    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        continue
    circularity = 4 * np.pi * area / (perimeter ** 2)

    if circularity < 0.2:   # skip very non-circular blobs
        continue

    # Color-code by confidence: green=high, yellow=medium, red=low
    cx_region = np.clip(cy, 0, prob_map.shape[0]-1)
    cy_region = np.clip(cx, 0, prob_map.shape[1]-1)
    confidence = float(prob_map[cx_region, cy_region])

    if confidence > 0.6:
        color = (0, 255, 0)      # green  – high confidence
    elif confidence > 0.4:
        color = (0, 255, 255)    # yellow – medium confidence
    else:
        color = (0, 100, 255)    # orange – low confidence

    cv2.circle(output_img, (cx, cy), radius, color, 2)
    cv2.circle(output_img, (cx, cy), 2, color, -1)   # center dot

    detected_count += 1

# ── Save outputs ──────────────────────────────────────────────────────────────
os.makedirs("outputs", exist_ok=True)

base = os.path.splitext(os.path.basename(args.image))[0]

# Detected image with circles
out_detected = f"outputs/{base}_detected.jpg"
cv2.imwrite(out_detected, output_img)

# Probability heatmap (viridis colormap for readability)
prob_u8 = (prob_map * 255).astype(np.uint8)
heatmap = cv2.applyColorMap(prob_u8, cv2.COLORMAP_VIRIDIS)
out_heatmap = f"outputs/{base}_heatmap.jpg"
cv2.imwrite(out_heatmap, heatmap)

# Binary mask
out_mask = f"outputs/{base}_mask.jpg"
cv2.imwrite(out_mask, binary_clean)

# Side-by-side comparison
h, w = orig_bgr.shape[:2]
heatmap_resized = cv2.resize(heatmap, (w, h))
comparison = np.hstack([orig_bgr, heatmap_resized, output_img])
out_compare = f"outputs/{base}_comparison.jpg"
cv2.imwrite(out_compare, comparison)

print(f"✅ Detected {detected_count} craters (threshold={args.threshold}, min_area={args.min_area})")
print(f"   - Detection image : {out_detected}")
print(f"   - Probability map : {out_heatmap}")
print(f"   - Binary mask     : {out_mask}")
print(f"   - Comparison      : {out_compare}")
print()
print("Tip: If still missing craters, lower --threshold (e.g. 0.2)")
print("Tip: If too many false positives, raise --threshold (e.g. 0.4) or --min_area")