#!/usr/bin/env python3
"""
Evaluate the trained GAT model on its held-out test split and save results.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix

from src.config import DATASET_FILE, PROCESSED_DATASET_PATH, get_device
from src.model import load_model

print("=" * 70)
print("Evaluating Trained GAT Model")
print("=" * 70)

# Find latest run
model_dirs = sorted(PROCESSED_DATASET_PATH.glob("run_*"), key=lambda p: p.stat().st_mtime)
if not model_dirs:
    print("No trained models found.")
    raise SystemExit(1)

run_dir = model_dirs[-1]
model_path = run_dir / "best_model.pt"
if not model_path.exists():
    model_path = run_dir / "modelo_final.pt"

device = get_device()
model, config = load_model(model_path, device=device)
NUM_CLASSES = config["num_classes"]
print(f"Loaded {model_path.name} (num_classes={NUM_CLASSES})")

# Load dataset
dataset = torch.load(DATASET_FILE, weights_only=False)

if NUM_CLASSES == 2:
    dataset = [d for d in dataset if int(d.y.item()) != 2]
    class_names = {0: "Primary Tumor", 1: "Metastasis"}
else:
    from src.config import CLASS_NAMES
    class_names = dict(CLASS_NAMES)

# Load test split
splits_path = run_dir / "splits.npz"
if splits_path.exists():
    splits = np.load(splits_path)
    test_idx = splits["test"]
    print(f"Using held-out test split: {len(test_idx)} patients")
    eval_set = [dataset[i] for i in test_idx]
    eval_label = "test"
else:
    print("WARNING: no splits.npz found — falling back to full-dataset eval (training data leakage).")
    eval_set = dataset
    eval_label = "all"

# Predict
class_label_list = [class_names[i] for i in range(NUM_CLASSES)]
preds, trues, confs, ids = [], [], [], []

with torch.no_grad():
    for data in eval_set:
        data = data.to(device)
        out = model(
            data.x,
            data.edge_index,
            torch.zeros(data.x.size(0), dtype=torch.long, device=device),
        )
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
        p = int(probs.argmax())
        preds.append(p)
        trues.append(int(data.y.item()))
        confs.append(float(probs[p]))
        ids.append(getattr(data, "paciente_id", "unknown"))

preds_arr = np.array(preds)
trues_arr = np.array(trues)
acc = float((preds_arr == trues_arr).mean())
print(f"\n{eval_label} accuracy: {acc*100:.2f}%")

cm = confusion_matrix(trues_arr, preds_arr, labels=list(range(NUM_CLASSES)))
print("\nConfusion matrix:")
print(cm)

report = classification_report(
    trues_arr, preds_arr, target_names=class_label_list, output_dict=True, zero_division=0
)
for name in class_label_list:
    m = report[name]
    print(f"  {name:18s} P={m['precision']*100:5.1f}% R={m['recall']*100:5.1f}% F1={m['f1-score']*100:5.1f}%")

# Save predictions
df = pd.DataFrame(
    {
        "patient_id": ids,
        "true_label": [class_names[t] for t in trues_arr],
        "predicted_label": [class_names[p] for p in preds_arr],
        "confidence": confs,
        "correct": preds_arr == trues_arr,
        "split": eval_label,
    }
)
out_csv = run_dir / "predictions.csv"
df.to_csv(out_csv, index=False)
print(f"\nSaved predictions: {out_csv}")

# Plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    ax=axes[0],
    xticklabels=class_label_list,
    yticklabels=class_label_list,
)
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("True")
axes[0].set_title(f"Confusion Matrix ({eval_label})")

f1s = [report[n]["f1-score"] for n in class_label_list]
axes[1].bar(class_label_list, f1s, color="steelblue", edgecolor="black")
axes[1].set_ylim(0, 1)
axes[1].set_ylabel("F1")
axes[1].set_title("Per-class F1")
for i, v in enumerate(f1s):
    axes[1].text(i, v + 0.02, f"{v:.2f}", ha="center")

plt.tight_layout()
out_png = run_dir / "model_performance.png"
plt.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"Saved plot: {out_png}")
