#!/usr/bin/env python3
"""
Train the GAT model on TCGA melanoma data.

- Stratified train/val/test split (70/15/15)
- Drops the Normal class (only 1 sample — unlearnable)
- Best model selected by validation loss, evaluated every epoch
- Early stopping on val loss
"""

from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

from config import DATASET_FILE, PROCESSED_DATASET_PATH
from model import GATv2Classifier, save_model

BATCH_SIZE = 32
NUM_EPOCHS = 500
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-4
EARLY_STOP_PATIENCE = 30
SEED = 42
# TOIL dataset has ~1300 GTEx normal-skin samples, so the Normal class is learnable.
DROP_NORMAL_CLASS = False

torch.manual_seed(SEED)
np.random.seed(SEED)

print("=" * 60)
print("GAT Model Training")
print("=" * 60)

# Load dataset
dataset = torch.load(DATASET_FILE, weights_only=False)
print(f"Loaded {len(dataset)} samples from {DATASET_FILE.name}")

if DROP_NORMAL_CLASS:
    dataset = [d for d in dataset if int(d.y.item()) != 2]
    NUM_CLASSES = 2
    class_names = {0: "Primary Tumor", 1: "Metastasis"}
    print(f"Dropped Normal class → {len(dataset)} patients, binary classification")
else:
    NUM_CLASSES = 3
    class_names = {0: "Primary Tumor", 1: "Metastasis", 2: "Normal Tissue"}

labels = np.array([int(d.y.item()) for d in dataset])

# Stratified 70/15/15 split
idx = np.arange(len(dataset))
train_idx, temp_idx = train_test_split(idx, test_size=0.30, stratify=labels, random_state=SEED)
val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.50, stratify=labels[temp_idx], random_state=SEED
)
train_set = [dataset[i] for i in train_idx]
val_set = [dataset[i] for i in val_idx]
test_set = [dataset[i] for i in test_idx]

print(f"Split: train={len(train_set)} | val={len(val_set)} | test={len(test_set)}")
print(f"  train classes: {Counter(int(d.y.item()) for d in train_set)}")
print(f"  val classes:   {Counter(int(d.y.item()) for d in val_set)}")
print(f"  test classes:  {Counter(int(d.y.item()) for d in test_set)}")

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GATv2Classifier(hidden_channels=64, num_classes=NUM_CLASSES).to(device)
print(f"Device: {device}")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = PROCESSED_DATASET_PATH / f"run_{timestamp}"
RUN_DIR.mkdir(parents=True, exist_ok=True)
print(f"Run dir: {RUN_DIR}")

# Save split indices for reproducibility / honest evaluation
np.savez(RUN_DIR / "splits.npz", train=train_idx, val=val_idx, test=test_idx)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Class weights from training set only (avoid leakage)
train_counts = Counter(int(d.y.item()) for d in train_set)
total = sum(train_counts.values())
weights = torch.tensor(
    [total / (NUM_CLASSES * train_counts[c]) for c in range(NUM_CLASSES)],
    dtype=torch.float,
).to(device)
print(f"Class weights (from train): {weights.tolist()}")
criterion = torch.nn.CrossEntropyLoss(weight=weights)


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


# Training loop
print(f"\nTraining for up to {NUM_EPOCHS} epochs (early stop patience={EARLY_STOP_PATIENCE})\n")

best_val_loss = float("inf")
best_epoch = 0
patience = 0
history = []

for epoch in range(1, NUM_EPOCHS + 1):
    train_loss, train_acc = run_epoch(train_loader, train=True)
    val_loss, val_acc = run_epoch(val_loader, train=False)
    history.append((epoch, train_loss, train_acc, val_loss, val_acc))

    improved = val_loss < best_val_loss
    if improved:
        best_val_loss = val_loss
        best_epoch = epoch
        patience = 0
        save_model(model, RUN_DIR / "best_model.pt")
    else:
        patience += 1

    if epoch % 10 == 0 or epoch == 1 or improved:
        flag = " *" if improved else ""
        print(
            f"Epoch {epoch:4d} | train loss {train_loss:.4f} acc {train_acc*100:5.2f}% "
            f"| val loss {val_loss:.4f} acc {val_acc*100:5.2f}%{flag}"
        )

    if patience >= EARLY_STOP_PATIENCE:
        print(f"\nEarly stop at epoch {epoch} (best epoch {best_epoch}, val loss {best_val_loss:.4f})")
        break

save_model(model, RUN_DIR / "modelo_final.pt")

# Save history
np.savez(
    RUN_DIR / "history.npz",
    history=np.array(history, dtype=float),
    columns=np.array(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]),
)

# Evaluate best model on test set
print(f"\n{'='*60}\nEvaluating best model on held-out test set\n{'='*60}")
from model import load_model

best_model, _ = load_model(RUN_DIR / "best_model.pt", device=device)
model = best_model
test_loss, test_acc = run_epoch(test_loader, train=False)
print(f"Test loss: {test_loss:.4f} | Test accuracy: {test_acc*100:.2f}%")

print(f"\nBest model: {RUN_DIR / 'best_model.pt'}")
print(f"Final model: {RUN_DIR / 'modelo_final.pt'}")
print(f"Splits saved: {RUN_DIR / 'splits.npz'}")
