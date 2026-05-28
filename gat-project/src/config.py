"""Project-wide configuration. Single source of truth for paths and constants."""

from pathlib import Path

NUM_NODES = 500
CONFIDENCE_THRESHOLD = 500

PROCESSED_DATASET_PATH = Path("./data/processed/")

# Active dataset. Switch this to point training/eval/visualization at a different
# processed file without touching every script.
DATASET_FILE = PROCESSED_DATASET_PATH / f"toil_skin_{NUM_NODES}_{CONFIDENCE_THRESHOLD}.pt"

CLASS_NAMES = {0: "Primary Tumor", 1: "Metastasis", 2: "Normal Tissue"}
