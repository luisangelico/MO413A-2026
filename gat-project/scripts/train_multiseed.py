#!/usr/bin/env python3
"""
Train N independent GATv2 models with different random seeds.

Same dataset, same hyperparameters, same bundled split — only the seed
differs. Used downstream by `scripts.biomarker_report` to find attention
edges that are consistently high across seeds (i.e. stable biomarkers,
not noise from a single training run).

Usage:
    python -m scripts.train_multiseed              # default: 5 seeds
    python -m scripts.train_multiseed --seeds 3
    python -m scripts.train_multiseed --seeds 5 --epochs 300

Outputs:
    data/processed/multiseed_<timestamp>/
        seed_0/best_model.pt + .json + splits.npz + history.npz
        seed_1/...
        ...
        config.json   (top-level: which dataset, hparams, seed list)
"""

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import DATASET_FILE, PROCESSED_DATASET_PATH, get_device
from src.model import GATv2Classifier, save_model, load_model

parser = argparse.ArgumentParser()
parser.add_argument("--seeds", type=int, default=5)
parser.add_argument("--epochs", type=int, default=500)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight-decay", type=float, default=5e-4)
parser.add_argument("--patience", type=int, default=50)
parser.add_argument("--hidden", type=int, default=128)
args = parser.parse_args()

device = get_device()
print(f"Device: {device}")
print(f"Dataset: {DATASET_FILE.name}")

dataset = torch.load(DATASET_FILE, weights_only=False)
NUM_CLASSES = 3

BUNDLED_SPLITS = DATASET_FILE.with_suffix(".splits.npz")
if not BUNDLED_SPLITS.exists():
    raise SystemExit(f"Bundled splits not found at {BUNDLED_SPLITS}. Re-run scripts.download_toil.")
s = np.load(BUNDLED_SPLITS)
train_idx, val_idx, test_idx = s["train"], s["val"], s["test"]
print(f"Splits: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

train_set = [dataset[i] for i in train_idx]
val_set = [dataset[i] for i in val_idx]
test_set = [dataset[i] for i in test_idx]

train_counts = Counter(int(d.y.item()) for d in train_set)
sample_weights_template = [1.0 / train_counts[int(d.y.item())] for d in train_set]

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_ROOT = PROCESSED_DATASET_PATH / f"multiseed_{timestamp}"
RUN_ROOT.mkdir(parents=True, exist_ok=True)

config = {
    "dataset": DATASET_FILE.name,
    "seeds": list(range(args.seeds)),
    "epochs": args.epochs,
    "batch_size": args.batch_size,
    "lr": args.lr,
    "weight_decay": args.weight_decay,
    "patience": args.patience,
    "hidden_channels": args.hidden,
    "num_classes": NUM_CLASSES,
}
(RUN_ROOT / "config.json").write_text(json.dumps(config, indent=2))
print(f"Multi-seed run dir: {RUN_ROOT}\n")

results = []
for seed in range(args.seeds):
    print("=" * 70)
    print(f"SEED {seed}/{args.seeds - 1}")
    print("=" * 70)

    torch.manual_seed(seed)
    np.random.seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    sampler = WeightedRandomSampler(
        sample_weights_template, num_samples=len(train_set), replacement=True, generator=g
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)

    model = GATv2Classifier(hidden_channels=args.hidden, num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
    )
    criterion = torch.nn.CrossEntropyLoss()

    SEED_DIR = RUN_ROOT / f"seed_{seed}"
    SEED_DIR.mkdir(exist_ok=True)
    np.savez(SEED_DIR / "splits.npz", train=train_idx, val=val_idx, test=test_idx)

    def run_epoch(loader, train: bool):
        model.train(train)
        total_loss = 0.0
        correct = 0
        n = 0
        for data in loader:
            data = data.to(device)
            if train:
                optimizer.zero_grad()
            with torch.set_grad_enabled(train):
                out = model(data.x, data.edge_index, data.batch)
                loss = criterion(out, data.y)
                if train:
                    loss.backward()
                    optimizer.step()
            total_loss += loss.item() * data.num_graphs
            correct += (out.argmax(dim=1) == data.y).sum().item()
            n += data.num_graphs
        return total_loss / n, correct / n

    best_val_loss = float("inf")
    best_epoch = 0
    patience_ctr = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(train_loader, train=True)
        val_loss, val_acc = run_epoch(val_loader, train=False)
        scheduler.step(val_loss)
        history.append((epoch, train_loss, train_acc, val_loss, val_acc))

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_ctr = 0
            save_model(model, SEED_DIR / "best_model.pt")
        else:
            patience_ctr += 1

        if epoch % 25 == 0 or epoch == 1 or improved:
            flag = " *" if improved else ""
            print(
                f"  Epoch {epoch:4d} | train {train_loss:.4f}/{train_acc*100:5.2f}% "
                f"| val {val_loss:.4f}/{val_acc*100:5.2f}%{flag}"
            )
        if patience_ctr >= args.patience:
            print(f"  Early stop at epoch {epoch} (best epoch {best_epoch})")
            break

    np.savez(SEED_DIR / "history.npz",
             history=np.array(history, dtype=float),
             columns=np.array(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]))

    best_model, _ = load_model(SEED_DIR / "best_model.pt", device=device)
    model = best_model
    test_loss, test_acc = run_epoch(test_loader, train=False)
    print(f"  Seed {seed} test: loss={test_loss:.4f} acc={test_acc*100:.2f}%\n")
    results.append({"seed": seed, "best_epoch": best_epoch,
                    "val_loss": best_val_loss, "test_acc": test_acc})

(RUN_ROOT / "results.json").write_text(json.dumps(results, indent=2))
mean_acc = np.mean([r["test_acc"] for r in results]) * 100
std_acc = np.std([r["test_acc"] for r in results]) * 100
print("=" * 70)
print(f"Done. Test accuracy across seeds: {mean_acc:.2f}% ± {std_acc:.2f}%")
print(f"Results: {RUN_ROOT}")
