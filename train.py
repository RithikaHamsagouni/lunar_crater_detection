# train.py
"""
Training loop for lunar crater diffusion model
Fixed for:
- CPU / GPU compatibility
- Kaggle / Colab support
- Stable checkpoint saving
- Better logging
- Fast smoke-test mode
"""

import os
import sys
import random
import argparse
import traceback

import torch
import torch.nn.functional as F
import numpy as np

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from config import Config
from dataset import get_dataloaders
from diffusion_model import build_model


# ─────────────────────────────────────────────
# Optional TensorBoard
# ─────────────────────────────────────────────
try:
    from torch.utils.tensorboard import SummaryWriter
    TB_AVAILABLE = True
except ImportError:
    TB_AVAILABLE = False
    print("[INFO] TensorBoard not installed. Logging to console only.")


# ─────────────────────────────────────────────
# Seed everything
# ─────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────
# Diffusion loss
# ─────────────────────────────────────────────
def diffusion_loss(model, schedule, image, mask, device):
    """
    DDPM + BCE hybrid loss
    """
    B = image.size(0)

    t = torch.randint(
        low=1,
        high=schedule.T,
        size=(B,),
        device=device
    )

    noise = torch.randn_like(mask)

    x_t, _ = schedule.q_sample(mask, t, noise)

    pred_noise = model(x_t, image, t)

    # Main DDPM loss
    mse_loss = F.mse_loss(pred_noise, noise)

    # Reconstruct x0
    sqrt_ah = schedule.sqrt_alpha_hat[t][:, None, None, None]
    sqrt_om = schedule.sqrt_one_minus_alpha_hat[t][:, None, None, None]

    x0_pred = (x_t - sqrt_om * pred_noise) / (sqrt_ah + 1e-6)

    # Auxiliary BCE
    bce_loss = F.binary_cross_entropy_with_logits(
        x0_pred,
        mask.clamp(0, 1)
    )

    total_loss = mse_loss + 0.5 * bce_loss

    return total_loss, {
        "mse": mse_loss.item(),
        "bce": bce_loss.item()
    }


# ─────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, schedule, val_loader, device):
    model.eval()

    total_dice = 0.0
    total_bce = 0.0
    batches = 0

    for batch in val_loader:
        try:
            image = batch["image"].to(device)
            mask = batch["mask"].to(device)

            B = image.size(0)

            t = torch.full(
                (B,),
                schedule.T // 2,
                device=device,
                dtype=torch.long
            )

            noise = torch.randn_like(mask)

            x_t, _ = schedule.q_sample(mask, t, noise)

            pred_noise = model(x_t, image, t)

            sqrt_ah = schedule.sqrt_alpha_hat[t][:, None, None, None]
            sqrt_om = schedule.sqrt_one_minus_alpha_hat[t][:, None, None, None]

            x0_pred = (x_t - sqrt_om * pred_noise) / (sqrt_ah + 1e-6)

            pred_mask = (torch.sigmoid(x0_pred) > 0.5).float()

            intersection = (pred_mask * mask).sum(dim=[2, 3])

            dice = (
                (2 * intersection + 1)
                / (pred_mask.sum(dim=[2, 3]) + mask.sum(dim=[2, 3]) + 1)
            )

            total_dice += dice.mean().item()

            total_bce += F.binary_cross_entropy_with_logits(
                x0_pred,
                mask
            ).item()

            batches += 1

        except Exception as e:
            print(f"[VAL ERROR] {e}")
            continue

    model.train()

    if batches == 0:
        return {
            "val_dice": 0.0,
            "val_bce": 999.0
        }

    return {
        "val_dice": total_dice / batches,
        "val_bce": total_bce / batches
    }


# ─────────────────────────────────────────────
# Main training
# ─────────────────────────────────────────────
def train(args):
    cfg = Config()

    # CLI overrides
    if args.epochs:
        cfg.EPOCHS = args.epochs

    if args.batch_size:
        cfg.BATCH_SIZE = args.batch_size

    if args.patch_size:
        cfg.PATCH_SIZE = args.patch_size

    if args.fast:
        cfg.EPOCHS = 2
        cfg.BATCH_SIZE = 4
        cfg.PATCH_SIZE = 128
        cfg.TIMESTEPS = 100
        print("[FAST MODE ENABLED]")

    set_seed(cfg.SEED)

    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    os.makedirs(cfg.LOG_DIR, exist_ok=True)

    device = cfg.DEVICE
    use_cuda = device == "cuda"

    print(f"Device       : {device}")
    print(f"Epochs       : {cfg.EPOCHS}")
    print(f"Batch size   : {cfg.BATCH_SIZE}")
    print(f"Patch size   : {cfg.PATCH_SIZE}")
    print(f"Timesteps    : {cfg.TIMESTEPS}")

    # Dataset zip
    zip_path = args.zip

    if not os.path.exists(zip_path):
        print(f"ERROR: Dataset zip not found: {zip_path}")
        sys.exit(1)

    # Loaders
    train_loader, val_loader, _ = get_dataloaders(
        cfg,
        zip_path=zip_path
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches  : {len(val_loader)}")

    # Model
    model, schedule = build_model(cfg)

    total_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"Trainable params: {total_params:,}")

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.LR,
        weight_decay=cfg.WEIGHT_DECAY
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg.EPOCHS,
        eta_min=cfg.LR * 0.01
    )

    # Mixed precision
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_cuda
    )

    # TensorBoard
    writer = SummaryWriter(cfg.LOG_DIR) if TB_AVAILABLE else None

    best_dice = 0.0
    global_step = 0

    log_csv = os.path.join(
        cfg.OUTPUT_DIR,
        "train_log.csv"
    )

    with open(log_csv, "w") as f:
        f.write("epoch,train_loss,val_dice,val_bce\n")

    print("\n---- TRAINING STARTED ----")

    max_batches = 2 if args.fast else None

    for epoch in range(1, cfg.EPOCHS + 1):
        model.train()

        epoch_loss = 0.0
        batches_done = 0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{cfg.EPOCHS}"
        )

        for batch_idx, batch in enumerate(progress):
            if max_batches and batch_idx >= max_batches:
                break

            try:
                image = batch["image"].to(device)
                mask = batch["mask"].to(device)

                optimizer.zero_grad()

                with torch.amp.autocast(
                    "cuda",
                    enabled=use_cuda
                ):
                    loss, logs = diffusion_loss(
                        model,
                        schedule,
                        image,
                        mask,
                        device
                    )

                scaler.scale(loss).backward()

                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    cfg.GRAD_CLIP
                )

                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()
                batches_done += 1
                global_step += 1

                progress.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "mse": f"{logs['mse']:.4f}",
                    "bce": f"{logs['bce']:.4f}",
                })

                if writer:
                    writer.add_scalar(
                        "train/loss",
                        loss.item(),
                        global_step
                    )

            except Exception as e:
                print(f"\n[TRAIN ERROR] Batch {batch_idx}: {e}")
                traceback.print_exc()
                continue

        scheduler.step()

        if batches_done == 0:
            print("No batches completed this epoch.")
            continue

        avg_train_loss = epoch_loss / batches_done

        # Validation
        val_metrics = evaluate(
            model,
            schedule,
            val_loader,
            device
        )

        val_dice = val_metrics["val_dice"]
        val_bce = val_metrics["val_bce"]

        print(
            f"Epoch {epoch}: "
            f"train_loss={avg_train_loss:.4f}, "
            f"val_dice={val_dice:.4f}, "
            f"val_bce={val_bce:.4f}"
        )

        # CSV log
        with open(log_csv, "a") as f:
            f.write(
                f"{epoch},{avg_train_loss:.6f},{val_dice:.6f},{val_bce:.6f}\n"
            )

        # Save last
        last_path = os.path.join(
            cfg.CHECKPOINT_DIR,
            "last_model.pt"
        )

        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "val_dice": val_dice,
            "cfg_patch": cfg.PATCH_SIZE,
            "cfg_timesteps": cfg.TIMESTEPS,
        }, last_path)

        # Save best
        if val_dice > best_dice:
            best_dice = val_dice

            best_path = os.path.join(
                cfg.CHECKPOINT_DIR,
                "best_model.pt"
            )

            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "val_dice": best_dice,
                "cfg_patch": cfg.PATCH_SIZE,
                "cfg_timesteps": cfg.TIMESTEPS,
            }, best_path)

            print(
                f"[*] Best model saved: {best_path} "
                f"(dice={best_dice:.4f})"
            )

    if writer:
        writer.close()

    print("\n---- TRAINING COMPLETE ----")
    print(f"Best Dice Score: {best_dice:.4f}")
    print(f"Logs saved to: {log_csv}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--zip",
        default="archive__3_.zip",
        help="Path to crater dataset zip"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=None
    )

    parser.add_argument(
        "--patch_size",
        type=int,
        default=None
    )

    parser.add_argument(
        "--fast",
        action="store_true"
    )

    args = parser.parse_args()

    train(args)