"""
train.py — Training loop for the conditional crater diffusion model.

Fixes applied vs previous version:
  - Removed torch.cuda.amp (deprecated API, crashes on CPU)
  - Removed tensorboard dependency (optional, won't crash if missing)
  - Added try/except around every batch so errors are visible, not silent
  - Reduced default patch size to 128 for CPU training speed
  - Added --fast flag for quick smoke-test (2 epochs, 2 batches each)
  - checkpoint dir is created before saving (was missing)
  - GradScaler only used when CUDA is available

Run:
    python train.py                   # full training
    python train.py --fast            # smoke test: 2 epochs
    python train.py --epochs 5        # custom epoch count
"""

import os, sys, random, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from config import Config
from dataset import get_dataloaders
from diffusion_model import build_model

# ── TensorBoard is optional ────────────────────────────────────────────────
try:
    from torch.utils.tensorboard import SummaryWriter
    _TB = True
except ImportError:
    _TB = False
    print("[info] tensorboard not installed — logging to console only")
    print("       Install with:  pip install tensorboard")


# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
def diffusion_loss(model, schedule, image, mask, device):
    """
    DDPM noise-prediction loss + auxiliary segmentation BCE.
    Returns (total_loss, log_dict).
    """
    B = image.size(0)
    t = torch.randint(1, schedule.T, (B,), device=device)

    noise = torch.randn_like(mask)
    x_t, _ = schedule.q_sample(mask, t, noise)

    eps_pred = model(x_t, image, t)

    loss_mse = F.mse_loss(eps_pred, noise)

    sqrt_ah = schedule.sqrt_alpha_hat[t][:, None, None, None]
    sqrt_om = schedule.sqrt_one_minus_alpha[t][:, None, None, None]
    x0_pred = (x_t - sqrt_om * eps_pred) / (sqrt_ah + 1e-6)
    loss_bce = F.binary_cross_entropy_with_logits(x0_pred, mask.clamp(0, 1))

    loss = loss_mse + 0.5 * loss_bce
    return loss, {"mse": loss_mse.item(), "bce": loss_bce.item()}


# ─────────────────────────────────────────────────────────────────────────────
def evaluate(model, schedule, val_dl, device):
    """Validation Dice + BCE at a fixed mid-range timestep (t=200)."""
    model.eval()
    total_dice, total_bce, count = 0.0, 0.0, 0

    with torch.no_grad():
        for batch in val_dl:
            try:
                image = batch["image"].to(device)
                mask  = batch["mask"].to(device)
                t = torch.full((image.size(0),), schedule.T // 2, device=device, dtype=torch.long)


               
                noise  = torch.randn_like(mask)
                x_t, _ = schedule.q_sample(mask, t, noise)
                eps_pred = model(x_t, image, t)

                sqrt_ah = schedule.sqrt_alpha_hat[t][:, None, None, None]
                sqrt_om = schedule.sqrt_one_minus_alpha[t][:, None, None, None]
                x0_pred = (x_t - sqrt_om * eps_pred) / (sqrt_ah + 1e-6)

                pred_mask    = (torch.sigmoid(x0_pred) > 0.5).float()
                intersection = (pred_mask * mask).sum(dim=[2, 3])
                dice = (2 * intersection + 1) / (
                    pred_mask.sum(dim=[2, 3]) + mask.sum(dim=[2, 3]) + 1
                )
                total_dice += dice.mean().item()
                total_bce  += F.binary_cross_entropy_with_logits(x0_pred, mask).item()
                count += 1

            except Exception as e:
                print(f"  [val batch error] {e}")
                continue

    model.train()
    if count == 0:
        return {"val_dice": 0.0, "val_bce": 0.0}
    return {"val_dice": total_dice / count, "val_bce": total_bce / count}


# ─────────────────────────────────────────────────────────────────────────────
def train(args):
    cfg = Config()

    # ── Override config with CLI args ────────────────────────────────────
    if args.epochs:
        cfg.EPOCHS = args.epochs
    if args.batch_size:
        cfg.BATCH_SIZE = args.batch_size
    if args.patch_size:
        cfg.PATCH_SIZE = args.patch_size
    if args.fast:
        cfg.EPOCHS     = 2
        cfg.PATCH_SIZE = 128
        cfg.BATCH_SIZE = 4
        cfg.TIMESTEPS  = 100    # much faster noise schedule for testing
        print("[fast mode] epochs=2, patch=128, batch=4, timesteps=100")

    set_seed(cfg.SEED)
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(cfg.OUTPUT_DIR,     exist_ok=True)

    device = cfg.DEVICE
    use_cuda = (device == "cuda")
    print(f"Device     : {device}")
    print(f"Patch size : {cfg.PATCH_SIZE}")
    print(f"Batch size : {cfg.BATCH_SIZE}")
    print(f"Timesteps  : {cfg.TIMESTEPS}")
    print(f"Epochs     : {cfg.EPOCHS}")

    # ── Data ──────────────────────────────────────────────────────────────
    ZIP_PATH = args.zip
    if not os.path.exists(ZIP_PATH):
        print(f"ERROR: zip not found at '{ZIP_PATH}'")
        print("  Make sure archive__3_.zip is in the same folder as train.py")
        sys.exit(1)

    train_dl, val_dl, _ = get_dataloaders(cfg, zip_path=ZIP_PATH)
    print(f"Train batches: {len(train_dl)} | Val batches: {len(val_dl)}")

    # ── Fast mode: limit to 2 batches per epoch for smoke test ───────────
    max_batches = 2 if args.fast else None

    # ── Model ─────────────────────────────────────────────────────────────
    model, schedule = build_model(cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters : {n_params:,}")

    # ── Optimiser ─────────────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS, eta_min=cfg.LR * 0.01)

    # AMP scaler — only when CUDA available (no-op on CPU)
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    # Optional TensorBoard writer
    writer = SummaryWriter(cfg.LOG_DIR) if _TB else None

    best_dice   = 0.0
    global_step = 0
    log_path    = os.path.join(cfg.OUTPUT_DIR, "train_log.csv")

    # Write CSV header
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,val_dice,val_bce\n")

    print("\n── Starting training ────────────────────────────────────────")

    for epoch in range(1, cfg.EPOCHS + 1):
        model.train()
        epoch_loss   = 0.0
        batches_done = 0

        pbar = tqdm(train_dl, desc=f"Epoch {epoch}/{cfg.EPOCHS}", leave=True)

        for i, batch in enumerate(pbar):
            if max_batches and i >= max_batches:
                break

            try:
                image = batch["image"].to(device)
                mask  = batch["mask"].to(device)

                optimizer.zero_grad()

                # Forward (AMP autocast only on CUDA)
                with torch.amp.autocast("cuda", enabled=use_cuda):
                    loss, log = diffusion_loss(model, schedule, image, mask, device)

                # Backward
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()

                epoch_loss   += loss.item()
                batches_done += 1
                global_step  += 1

                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    mse=f"{log['mse']:.4f}",
                    bce=f"{log['bce']:.4f}",
                )

                if writer:
                    writer.add_scalar("train/loss", loss.item(), global_step)
                    writer.add_scalar("train/mse",  log["mse"],  global_step)
                    writer.add_scalar("train/bce",  log["bce"],  global_step)

            except Exception as e:
                print(f"\n  [ERROR in batch {i}]: {e}")
                import traceback; traceback.print_exc()
                continue

        scheduler.step()

        if batches_done == 0:
            print(f"  Epoch {epoch}: no batches completed — check errors above")
            continue

        avg_loss = epoch_loss / batches_done

        # ── Validate ──────────────────────────────────────────────────────
        val_metrics = evaluate(model, schedule, val_dl, device)
        vd  = val_metrics["val_dice"]
        vb  = val_metrics["val_bce"]

        print(
            f"  Epoch {epoch:3d} | "
            f"train_loss={avg_loss:.4f} | "
            f"val_dice={vd:.4f} | "
            f"val_bce={vb:.4f}"
        )

        if writer:
            writer.add_scalar("val/dice", vd, epoch)
            writer.add_scalar("val/bce",  vb, epoch)

        # Append to CSV log
        with open(log_path, "a") as f:
            f.write(f"{epoch},{avg_loss:.6f},{vd:.6f},{vb:.6f}\n")

        # ── Save checkpoint (always save last, save best separately) ──────
        last_path = os.path.join(cfg.CHECKPOINT_DIR, "last_model.pt")
        torch.save({
            "epoch":       epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "val_dice":    vd,
            "cfg_patch":   cfg.PATCH_SIZE,
            "cfg_ts":      cfg.TIMESTEPS,
        }, last_path)

        if vd > best_dice:
            best_dice = vd
            best_path = os.path.join(cfg.CHECKPOINT_DIR, "best_model.pt")
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "val_dice":    best_dice,
                "cfg_patch":   cfg.PATCH_SIZE,
                "cfg_ts":      cfg.TIMESTEPS,
            }, best_path)
            print(f"  ✓ Best model saved → {best_path}  (dice={best_dice:.4f})")

    if writer:
        writer.close()

    print(f"\nTraining complete. Log → {log_path}")
    print(f"Best val_dice = {best_dice:.4f}")
    print(f"Checkpoints  → {cfg.CHECKPOINT_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train crater diffusion model")
    parser.add_argument("--zip",        default="archive__3_.zip",
                        help="Path to crater dataset zip")
    parser.add_argument("--epochs",     type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--patch_size", type=int, default=None,
                        help="Override patch size (128 recommended for CPU)")
    parser.add_argument("--fast",       action="store_true",
                        help="Smoke test: 2 epochs, 2 batches, small patches")
    args = parser.parse_args()

    train(args)