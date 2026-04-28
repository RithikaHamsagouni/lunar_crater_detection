"""
config.py — Central configuration for lunar south pole crater detection
using diffusion models with probabilistic outputs.
"""

import torch

class Config:
    # ── Data ──────────────────────────────────────────────────────────────
    DATA_CSV      = "ser_portal_luna_lroc_pds_nac_edrcdr_260406.csv"
    IMAGE_DIR     = "images/"
    PATCH_SIZE    = 128                # 128 for CPU, 256 for GPU
    STRIDE        = 64
    RESOLUTION_THRESH = 1.5
    QUALITY_THRESH    = 0

    # ── Model ─────────────────────────────────────────────────────────────
    IN_CHANNELS   = 1                 # grayscale NAC images
    OUT_CHANNELS  = 1                 # binary crater mask
    MODEL_DIM     = 32                # base channel width (64 for GPU, 32 for CPU)
    TIMESTEPS     = 500               # noise schedule steps (1000 for GPU, 500 for CPU)
    BETA_START    = 1e-4
    BETA_END      = 0.02
    DDIM_STEPS    = 20                # inference steps (50 for GPU, 20 for CPU)

    # ── Probability head ──────────────────────────────────────────────────
    MC_SAMPLES    = 20                # Monte Carlo dropout passes
    DROPOUT_RATE  = 0.15

    # ── Crater classifier ─────────────────────────────────────────────────
    # Degradation thresholds (0–1 score from morphology features)
    FRESH_THRESH     = 0.75           # P(crater) >= 0.75 → fresh
    DEGRADED_THRESH  = 0.40           # 0.40 <= P < 0.75 → degraded
    # Below DEGRADED_THRESH → heavily eroded / uncertain
    OVERLAP_IOU      = 0.15           # IoU threshold to flag overlap

    # ── Training ──────────────────────────────────────────────────────────
    BATCH_SIZE    = 4                  # 4 for CPU, 8-16 for GPU
    EPOCHS        = 50
    LR            = 1e-4
    WEIGHT_DECAY  = 1e-5
    WARMUP_STEPS  = 200
    GRAD_CLIP     = 1.0
    VAL_SPLIT     = 0.15
    SEED          = 42

    # ── Hardware ──────────────────────────────────────────────────────────
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 0                   # 0 = main process only (safe on Windows)

    # ── Output ────────────────────────────────────────────────────────────
    CHECKPOINT_DIR = "checkpoints/"
    OUTPUT_DIR     = "outputs/"
    LOG_DIR        = "runs/"