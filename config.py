"""
config.py — Central configuration for lunar crater diffusion system
Optimized for:
- VS Code (Windows laptop)
- Colab
- Kaggle
- CPU + GPU
"""

import os
import torch


class Config:
    # ─────────────────────────────────────────────
    # Base paths (absolute, cross-platform safe)
    # ─────────────────────────────────────────────
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    DATA_CSV  = os.path.join(BASE_DIR, "ser_portal_luna_lroc_pds_nac_edrcdr_260406.csv")
    IMAGE_DIR = os.path.join(BASE_DIR, "images")

    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
    OUTPUT_DIR     = os.path.join(BASE_DIR, "outputs")
    LOG_DIR        = os.path.join(BASE_DIR, "runs")

    # Auto-create required directories
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # ─────────────────────────────────────────────
    # Data
    # ─────────────────────────────────────────────
    PATCH_SIZE = 128          # CPU safe
    STRIDE     = 64
    VAL_SPLIT  = 0.15

    RESOLUTION_THRESH = 1.5
    QUALITY_THRESH    = 0

    # Training normalization stats
    # Since lunar grayscale is normalized dynamically,
    # these are fallback values for inference consistency.
    TRAIN_MEAN = 0.5
    TRAIN_STD  = 0.5

    # ─────────────────────────────────────────────
    # Model
    # ─────────────────────────────────────────────
    IN_CHANNELS  = 1
    OUT_CHANNELS = 1

    MODEL_DIM    = 32       # CPU-safe
    TIMESTEPS    = 500
    DDIM_STEPS   = 20

    # Beta schedule fallback (used if cosine disabled)
    BETA_START   = 1e-4
    BETA_END     = 0.02

    USE_COSINE_SCHEDULE = True

    # ─────────────────────────────────────────────
    # Probability head
    # ─────────────────────────────────────────────
    MC_SAMPLES   = 10       # lower for CPU
    DROPOUT_RATE = 0.15

    # ─────────────────────────────────────────────
    # Crater classification
    # ─────────────────────────────────────────────
    FRESH_THRESH    = 0.75
    DEGRADED_THRESH = 0.40
    OVERLAP_IOU     = 0.15

    # ─────────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────────
    BATCH_SIZE    = 4
    EPOCHS        = 50
    LR            = 1e-4
    WEIGHT_DECAY  = 1e-5
    WARMUP_STEPS  = 200
    GRAD_CLIP     = 1.0
    SEED          = 42

    # Dice loss weighting
    DICE_WEIGHT   = 0.5
    BCE_WEIGHT    = 0.3
    MSE_WEIGHT    = 1.0

    # ─────────────────────────────────────────────
    # Hardware
    # ─────────────────────────────────────────────
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # IMPORTANT:
    # Keep 0 for Windows/basic laptop + Zip safety
    NUM_WORKERS = 0

    PIN_MEMORY = torch.cuda.is_available()

    # ─────────────────────────────────────────────
    # Inference
    # ─────────────────────────────────────────────
    MIN_CRATER_PIXELS = 20
    SLIDING_WINDOW_STRIDE = 64

    # Save every N epochs
    SAVE_VIS_EVERY = 5

    # ─────────────────────────────────────────────
    # Utility
    # ─────────────────────────────────────────────
    @staticmethod
    def print_config():
        print("── Config ─────────────────────────")
        print(f"Device           : {Config.DEVICE}")
        print(f"Patch Size       : {Config.PATCH_SIZE}")
        print(f"Batch Size       : {Config.BATCH_SIZE}")
        print(f"Timesteps        : {Config.TIMESTEPS}")
        print(f"DDIM Steps       : {Config.DDIM_STEPS}")
        print(f"Cosine Schedule  : {Config.USE_COSINE_SCHEDULE}")
        print(f"Checkpoint Dir   : {Config.CHECKPOINT_DIR}")
        print(f"Output Dir       : {Config.OUTPUT_DIR}")
        print("──────────────────────────────────")