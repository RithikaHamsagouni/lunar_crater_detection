"""
dataset.py — Safe crater dataset loader
Fixes:
- ZipFile multiprocessing bug
- Real augmentation
- Patch extraction
- Normalization consistency
"""

import os
import io
import zipfile
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageEnhance
import torchvision.transforms as T

from config import Config


# ─────────────────────────────────────────────
# Patch extraction
# ─────────────────────────────────────────────
def extract_patches(image, mask, patch_size, stride):
    img_patches, mask_patches = [], []

    H, W = image.shape

    if H < patch_size or W < patch_size:
        image = np.array(
            Image.fromarray(image).resize((patch_size, patch_size))
        )
        mask = np.array(
            Image.fromarray(mask).resize((patch_size, patch_size))
        )
        return [image], [mask]

    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            img_patches.append(image[y:y+patch_size, x:x+patch_size])
            mask_patches.append(mask[y:y+patch_size, x:x+patch_size])

    if len(img_patches) == 0:
        img_patches.append(image[:patch_size, :patch_size])
        mask_patches.append(mask[:patch_size, :patch_size])

    return img_patches, mask_patches


# ─────────────────────────────────────────────
# YOLO boxes → crater mask
# ─────────────────────────────────────────────
def yolo_boxes_to_mask(boxes, H, W):
    mask = np.zeros((H, W), dtype=np.uint8)

    for cx, cy, bw, bh in boxes:
        px = int(cx * W)
        py = int(cy * H)
        rx = max(1, int(bw * W / 2))
        ry = max(1, int(bh * H / 2))

        yy, xx = np.ogrid[:H, :W]
        ellipse = ((xx - px) / rx) ** 2 + ((yy - py) / ry) ** 2 <= 1
        mask[ellipse] = 1

    return mask


# ─────────────────────────────────────────────
# Lightweight augmentations
# ─────────────────────────────────────────────
def augment_pair(image, mask):
    if random.random() < 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

    if random.random() < 0.5:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

    if random.random() < 0.3:
        angle = random.choice([90, 180, 270])
        image = image.rotate(angle)
        mask = mask.rotate(angle)

    if random.random() < 0.3:
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(random.uniform(0.8, 1.2))

    return image, mask


# ─────────────────────────────────────────────
class CraterDataset(Dataset):
    def __init__(self, zip_path, split="train", patch_size=128, stride=64, augment=True):
        assert split in ("train", "valid", "test")

        self.zip_path = zip_path
        self.split = split
        self.patch_size = patch_size
        self.stride = stride
        self.augment = augment and split == "train"

        self.img_prefix = f"craters/{split}/images/"
        self.label_prefix = f"craters/{split}/labels/"

        self.samples = []
        self.zf = None

        with zipfile.ZipFile(zip_path, "r") as zf:
            all_names = set(zf.namelist())

            for name in sorted(all_names):
                if name.startswith(self.img_prefix) and name.endswith(".jpg"):
                    stem = os.path.basename(name).replace(".jpg", "")
                    label_name = self.label_prefix + stem + ".txt"

                    if label_name in all_names:
                        self.samples.append((name, label_name))

        if not self.samples:
            raise ValueError(f"No samples found for split={split}")

        self.to_tensor = T.ToTensor()

        print(f"[{split}] {len(self.samples)} samples")

    # Lazy ZipFile open (worker-safe)
    def _get_zip(self):
        if self.zf is None:
            self.zf = zipfile.ZipFile(self.zip_path, "r")
        return self.zf

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        zf = self._get_zip()

        img_name, label_name = self.samples[idx]

        img_bytes = zf.read(img_name)
        label_text = zf.read(label_name).decode()

        image = Image.open(io.BytesIO(img_bytes)).convert("L")
        W, H = image.size

        boxes = []

        for line in label_text.strip().splitlines():
            parts = line.split()
            if len(parts) == 5:
                _, cx, cy, bw, bh = map(float, parts)
                boxes.append([cx, cy, bw, bh])

        boxes = np.array(boxes) if boxes else np.zeros((0, 4))

        image_np = np.array(image)
        mask_np = yolo_boxes_to_mask(boxes, H, W)

        img_patches, mask_patches = extract_patches(
            image_np,
            mask_np,
            self.patch_size,
            self.stride
        )

        # Random patch
        patch_idx = random.randint(0, len(img_patches) - 1)

        image_patch = Image.fromarray(img_patches[patch_idx])
        mask_patch = Image.fromarray(mask_patches[patch_idx] * 255)

        if self.augment:
            image_patch, mask_patch = augment_pair(image_patch, mask_patch)

        img_t = self.to_tensor(image_patch).float()

        # Normalize exactly like training
        mean = img_t.mean()
        std = img_t.std()

        img_t = (img_t - mean) / (std + 1e-6)

        mask_t = (self.to_tensor(mask_patch) > 0.5).float()

        return {
            "image": img_t,
            "mask": mask_t,
            "identifier": os.path.basename(img_name),
            "n_craters": len(boxes),
        }


# ─────────────────────────────────────────────
def get_dataloaders(cfg: Config, zip_path):
    train_ds = CraterDataset(
        zip_path,
        split="train",
        patch_size=cfg.PATCH_SIZE,
        stride=cfg.STRIDE,
        augment=True,
    )

    val_ds = CraterDataset(
        zip_path,
        split="valid",
        patch_size=cfg.PATCH_SIZE,
        stride=cfg.STRIDE,
        augment=False,
    )

    test_ds = CraterDataset(
        zip_path,
        split="test",
        patch_size=cfg.PATCH_SIZE,
        stride=cfg.STRIDE,
        augment=False,
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )

    test_dl = DataLoader(
        test_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )

    return train_dl, val_dl, test_dl