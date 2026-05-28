#!/usr/bin/env python3
"""
Simple script to make predictions with your trained GAT model.
Usage: python3 predict.py
"""

import torch
from pathlib import Path

from config import DATASET_FILE
from model import load_model

print("="*70)
print("🧬 GAT Model Predictor")
print("="*70)

# ============================================================================
# Load Model
# ============================================================================
print("\n📦 Loading trained model...")

DATASET_PATH = DATASET_FILE

# Find the most recent run
model_dirs = sorted(Path('data/processed/').glob('run_*'), key=lambda p: p.stat().st_mtime)
if not model_dirs:
    print("❌ No trained models found!")
    exit(1)

latest_dir = model_dirs[-1]
MODEL_PATH = latest_dir / 'best_model.pt'
if not MODEL_PATH.exists():
    MODEL_PATH = latest_dir / 'modelo_final.pt'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, model_config = load_model(MODEL_PATH, device=device)
NUM_CLASSES = model_config["num_classes"]

print(f"   ✓ Model loaded from: {MODEL_PATH}")
print(f"   ✓ Device: {device}")
print(f"   ✓ Classes: {NUM_CLASSES}")

# ============================================================================
# Load Dataset
# ============================================================================
print("\n📊 Loading dataset...")

if not DATASET_PATH.exists():
    print(f"❌ Dataset not found at: {DATASET_PATH}")
    exit(1)

dataset = torch.load(DATASET_PATH, weights_only=False)
print(f"   ✓ Dataset loaded: {len(dataset)} patients")

# If model is binary, drop the Normal class so indices match training
if NUM_CLASSES == 2:
    dataset = [d for d in dataset if int(d.y.item()) != 2]
    print(f"   ✓ Filtered to {len(dataset)} patients (binary model: Primary/Metastasis)")

# ============================================================================
# Interactive Mode
# ============================================================================
_all_class_names = {0: 'Primary Tumor', 1: 'Metastasis', 2: 'Normal Tissue'}
_all_class_colors = {0: '🔴', 1: '🟠', 2: '🟢'}
class_names = {i: _all_class_names[i] for i in range(NUM_CLASSES)}
class_colors = {i: _all_class_colors[i] for i in range(NUM_CLASSES)}

print("\n" + "="*70)
print("🎯 Ready to Make Predictions!")
print("="*70)

while True:
    print("\n" + "-"*70)
    print("Choose an option:")
    print("  1. Predict a specific patient by index (0-471)")
    print("  2. Predict a random patient")
    print("  3. Predict first 5 patients")
    print("  4. Show model statistics")
    print("  5. Exit")
    print("-"*70)

    choice = input("\nEnter your choice (1-5): ").strip()

    if choice == '5':
        print("\n👋 Goodbye!")
        break

    elif choice == '1':
        try:
            idx = int(input(f"Enter patient index (0-{len(dataset)-1}): "))
            if idx < 0 or idx >= len(dataset):
                print(f"❌ Invalid index! Must be between 0 and {len(dataset)-1}")
                continue

            patient_indices = [idx]
        except ValueError:
            print("❌ Invalid input! Please enter a number.")
            continue

    elif choice == '2':
        import random
        patient_indices = [random.randint(0, len(dataset)-1)]

    elif choice == '3':
        patient_indices = list(range(min(5, len(dataset))))

    elif choice == '4':
        print("\n📈 Model Statistics:")
        print(f"   Architecture: GATv2 with 2 layers")
        print(f"   Layer 1: 4-head attention (1 → 256 features)")
        print(f"   Layer 2: 1-head attention (256 → 64 features)")
        print(f"   Output: 3 classes (Primary/Metastasis/Normal)")
        print(f"   Total parameters: ~45,000")
        print(f"   Model size: 145 KB")
        continue

    else:
        print("❌ Invalid choice!")
        continue

    # Make predictions
    print("\n" + "="*70)

    for idx in patient_indices:
        patient = dataset[idx].to(device)

        with torch.no_grad():
            # Get prediction
            out = model(patient.x, patient.edge_index,
                       torch.zeros(patient.x.size(0), dtype=torch.long, device=device))

            probs = torch.softmax(out, dim=1).cpu().numpy()[0]
            pred_class = int(probs.argmax())
            true_class = int(patient.y.item())

            # Get attention weights
            out_attn, edge_index_attn, alpha = model(
                patient.x, patient.edge_index,
                torch.zeros(patient.x.size(0), dtype=torch.long, device=device),
                return_attn=True
            )
            attention_weights = alpha.mean(dim=1).cpu().numpy()

        # Display results
        correct = "✓ CORRECT" if pred_class == true_class else "✗ INCORRECT"

        print(f"\n🔬 Patient {idx}: {patient.paciente_id}")
        print(f"   {'─'*66}")
        print(f"   True Label:      {class_colors[true_class]} {class_names[true_class]}")
        print(f"   Predicted:       {class_colors[pred_class]} {class_names[pred_class]} {correct}")
        print(f"   {'─'*66}")
        print(f"   Confidence Scores:")

        for i, (name, prob) in enumerate(zip([class_names[i] for i in range(NUM_CLASSES)], probs)):
            bar_length = int(prob * 30)
            bar = "█" * bar_length + "░" * (30 - bar_length)
            indicator = " ← PREDICTED" if i == pred_class else ""
            print(f"      {class_colors[i]} {name:20s} {bar} {prob*100:5.1f}%{indicator}")

        # Top genes by expression
        if hasattr(patient, 'gene_names'):
            expression = patient.x.squeeze().cpu().numpy()
            top_indices = expression.argsort()[-3:][::-1]

            print(f"\n   Top 3 Expressed Genes:")
            for rank, idx in enumerate(top_indices, 1):
                gene = patient.gene_names[idx]
                expr = expression[idx]
                print(f"      {rank}. {gene:15s} (expression: {expr:.3f})")

        # Top attention edges
        top_attn_indices = attention_weights.argsort()[-3:][::-1]
        print(f"\n   Top 3 Attention Edges (model focuses on):")
        for rank, idx in enumerate(top_attn_indices, 1):
            edge = edge_index_attn[:, idx].cpu().numpy()
            attn = attention_weights[idx]

            if hasattr(patient, 'gene_names'):
                gene1 = patient.gene_names[edge[0]]
                gene2 = patient.gene_names[edge[1]]

                if edge[0] == edge[1]:
                    print(f"      {rank}. {gene1:15s} (self-attention: {attn:.4f})")
                else:
                    print(f"      {rank}. {gene1} ↔ {gene2} (attention: {attn:.4f})")
            else:
                print(f"      {rank}. Gene {edge[0]} ↔ Gene {edge[1]} (attention: {attn:.4f})")

    print("\n" + "="*70)

print("\n💡 Tip: For interactive visualization, run:")
print("   streamlit run explore_data.py")
print("\n📄 For detailed guide, see: MODEL_USAGE_GUIDE.md")
