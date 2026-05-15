import torch


class Config:

    # ─────────────────────────────────────────
    # Device
    # ─────────────────────────────────────────
    DEVICE = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # ─────────────────────────────────────────
    # Reproducibility
    # ─────────────────────────────────────────
    SEED = 42

    # ─────────────────────────────────────────
    # Dataset
    # ─────────────────────────────────────────
    PATCH_SIZE = 128

    STRIDE = 64

    BATCH_SIZE = 8

    NUM_WORKERS = 2

    PIN_MEMORY = True

    # ─────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────
    EPOCHS = 40

    LR = 1e-4

    WEIGHT_DECAY = 1e-5

    GRAD_CLIP = 1.0

    # ─────────────────────────────────────────
    # Diffusion
    # ─────────────────────────────────────────
    TIMESTEPS = 50

    BETA_START = 1e-4

    BETA_END = 0.015

    DDIM_STEPS = 25

    # ─────────────────────────────────────────
    # UNet / Model
    # ─────────────────────────────────────────
    IN_CHANNELS = 1

    MASK_CHANNELS = 1

    BASE_CHANNELS = 64

    TIME_DIM = 256

    DROPOUT_RATE = 0.10

    # ─────────────────────────────────────────
    # Probability Head
    # ─────────────────────────────────────────
    MC_SAMPLES = 6

    # ─────────────────────────────────────────
    # Output Paths
    # ─────────────────────────────────────────
    CHECKPOINT_DIR = "checkpoints"

    OUTPUT_DIR = "outputs"

    LOG_DIR = "logs"

    # ─────────────────────────────────────────
    # Crater filtering
    # ─────────────────────────────────────────
    MIN_CRATER_RADIUS = 3

    MAX_CRATER_RADIUS = 64

    MIN_CRATER_AREA = 8

    # ─────────────────────────────────────────
    # Detection thresholds
    # ─────────────────────────────────────────
    DETECTION_THRESHOLD = 0.20

    CONFIDENCE_THRESHOLD = 0.15

    # ─────────────────────────────────────────
    # Visualization
    # ─────────────────────────────────────────
    SAVE_VISUALIZATIONS = True

    # ─────────────────────────────────────────
    # Debug
    # ─────────────────────────────────────────
    DEBUG = False
