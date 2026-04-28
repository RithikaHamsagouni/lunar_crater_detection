import os, io, zipfile
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from config import Config


# ─────────────────────────────────────────────
# NEW: Patch extraction (FIXED)
# ─────────────────────────────────────────────
def extract_patches(image: np.ndarray, patch_size: int, stride: int):
    patches, coords = [], []
    H, W = image.shape

    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            patch = image[y:y+patch_size, x:x+patch_size]
            patches.append(patch)
            coords.append((y, x))

    return patches, coords


# ─────────────────────────────────────────────
def yolo_boxes_to_mask(boxes: np.ndarray, H: int, W: int):
    mask = np.zeros((H, W), dtype=np.uint8)

    for cx, cy, bw, bh in boxes:
        px = int(cx * W)
        py = int(cy * H)
        rx = max(1, int(bw * W / 2))
        ry = max(1, int(bh * H / 2))

        yy, xx = np.ogrid[:H, :W]
        ellipse = ((xx - px)/rx)**2 + ((yy - py)/ry)**2 <= 1
        mask[ellipse] = 1

    return mask


# ─────────────────────────────────────────────
class CraterDataset(Dataset):
    def __init__(self, zip_path, split="train", patch_size=256, augment=True):
        assert split in ("train", "valid", "test")

        self.zip_path = zip_path
        self.split = split
        self.patch_size = patch_size
        self.augment = augment and split == "train"

        self.img_prefix = f"craters/{split}/images/"
        self.label_prefix = f"craters/{split}/labels/"

        # 🔥 FIX: Open zip ONCE
        self.zf = zipfile.ZipFile(zip_path, "r")
        all_names = set(self.zf.namelist())

        self.samples = []
        for name in sorted(all_names):
            if name.startswith(self.img_prefix) and name.endswith(".jpg"):
                stem = os.path.basename(name).replace(".jpg", "")
                label_name = self.label_prefix + stem + ".txt"
                if label_name in all_names:
                    self.samples.append((name, label_name))

        if len(self.samples) == 0:
            raise ValueError(f"No samples found in {split}. Check zip structure.")

        print(f"[{split}] {len(self.samples)} samples")

        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, label_name = self.samples[idx]

        img_bytes = self.zf.read(img_name)
        label_bytes = self.zf.read(label_name).decode()

        image = Image.open(io.BytesIO(img_bytes)).convert("L")
        W, H = image.size

        # Parse YOLO labels
        boxes = []
        for line in label_bytes.strip().splitlines():
            parts = line.split()
            if len(parts) == 5:
                _, cx, cy, bw, bh = map(float, parts)
                boxes.append([cx, cy, bw, bh])

        boxes = np.array(boxes) if boxes else np.zeros((0, 4))

        mask_np = yolo_boxes_to_mask(boxes, H, W)
        mask = Image.fromarray(mask_np * 255)

        # Resize
        image = image.resize((self.patch_size, self.patch_size))
        mask = mask.resize((self.patch_size, self.patch_size))

        img_t = self.to_tensor(image)

        # 🔥 FIX: safe normalization
        std = img_t.std()
        img_t = (img_t - img_t.mean()) / (std + 1e-6)

        mask_t = (self.to_tensor(mask) > 0.5).float()

        return {
            "image": img_t,
            "mask": mask_t,
            "identifier": os.path.basename(img_name),
            "n_craters": len(boxes),
        }


# ─────────────────────────────────────────────
def get_dataloaders(cfg: Config, zip_path: str):
    train_ds = CraterDataset(zip_path, "train", cfg.PATCH_SIZE, True)
    val_ds = CraterDataset(zip_path, "valid", cfg.PATCH_SIZE, False)
    test_ds = CraterDataset(zip_path, "test", cfg.PATCH_SIZE, False)

    train_dl = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE)
    test_dl = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE)

    return train_dl, val_dl, test_dl