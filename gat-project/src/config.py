"""Project-wide configuration. Single source of truth for paths and constants."""

from pathlib import Path

import torch


def get_device() -> torch.device:
    """Pick the best available device: CUDA > MPS (Apple Silicon) > CPU.

    Set GAT_DEVICE=cpu (or cuda/mps) to override.
    For MPS, set PYTORCH_ENABLE_MPS_FALLBACK=1 if you hit unsupported ops.
    """
    import os

    override = os.environ.get("GAT_DEVICE")
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

NUM_NODES = 500
CONFIDENCE_THRESHOLD = 200

PROCESSED_DATASET_PATH = Path("./data/processed/")

# Active dataset. Switch this to point training/eval/visualization at a different
# processed file without touching every script.
DATASET_FILE = PROCESSED_DATASET_PATH / f"toil_skin_{NUM_NODES}_{CONFIDENCE_THRESHOLD}.pt"

CLASS_NAMES = {0: "Primary Tumor", 1: "Metastasis", 2: "Normal Tissue"}
