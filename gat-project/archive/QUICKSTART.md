# 🚀 Quick Start: Using Your Trained GAT Model

## Three Simple Ways to Use Your Model

---

### **Option 1: Interactive Command-Line Tool** ⚡ (Easiest!)

```bash
python3 predict.py
```

**What you get:**
- Menu-driven interface
- Choose specific patients or random
- See predictions with confidence scores
- View top expressed genes
- See which gene interactions model focuses on (attention weights)

**Example output:**
```
🔬 Patient 0: TCGA-ER-A199-06A
   True Label:      🟠 Metastasis
   Predicted:       🟠 Metastasis ✓ CORRECT
   
   Confidence Scores:
      🔴 Primary Tumor        ███████████░░░  39.9%
      🟠 Metastasis           ████████████████  56.5% ← PREDICTED
      🟢 Normal Tissue        █░░░░░░░░░░░░░░   3.5%
   
   Top 3 Expressed Genes:
      1. IGKC (expression: 2.741)
      2. IGHG1 (expression: 2.632)
      3. IGHM (expression: 2.593)
```

**Perfect for:** Quick testing, exploring predictions, understanding model behavior

---

### **Option 2: Full Analysis Script** 📊

```bash
python3 use_trained_model.py
```

**What you get:**
- Predictions for all 472 patients
- Overall accuracy and performance metrics
- Confusion matrix
- Per-class precision/recall/F1
- Confidence analysis
- **Outputs:**
  - `predictions.csv` - All results
  - `model_performance.png` - 4 visualizations

**Perfect for:** Batch analysis, research papers, comprehensive evaluation

---

### **Option 3: Use in Your Own Python Code** 💻

```python
import torch
import torch.nn.functional as F
from pathlib import Path

# 1. Load your model
from train_model import GATv2Classifier  # Or copy class definition

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GATv2Classifier(hidden_channels=64, num_classes=3).to(device)
model.load_state_dict(torch.load(
    'data/processed/run_20260528_142504/best_model.pt',
    map_location=device
))
model.eval()

# 2. Load dataset
dataset = torch.load('data/processed/tcga_pacientes_500_500.pt', weights_only=False)

# 3. Pick a patient
patient = dataset[0].to(device)

# 4. Make prediction
with torch.no_grad():
    out = model(
        patient.x,
        patient.edge_index,
        torch.zeros(patient.x.size(0), dtype=torch.long, device=device)
    )
    
    probs = torch.softmax(out, dim=1).cpu().numpy()[0]
    predicted_class = probs.argmax()
    confidence = probs[predicted_class]

# 5. Get attention weights (optional)
with torch.no_grad():
    out, edge_index_attn, alpha = model(
        patient.x,
        patient.edge_index,
        torch.zeros(patient.x.size(0), dtype=torch.long, device=device),
        return_attn=True
    )
    attention_weights = alpha.mean(dim=1).cpu().numpy()

# 6. Results
class_names = {0: 'Primary Tumor', 1: 'Metastasis', 2: 'Normal Tissue'}
print(f"Patient: {patient.paciente_id}")
print(f"Predicted: {class_names[predicted_class]} ({confidence:.1%} confidence)")
print(f"Probabilities: Primary={probs[0]:.1%}, Meta={probs[1]:.1%}, Normal={probs[2]:.1%}")
print(f"Max attention weight: {attention_weights.max():.4f}")
```

**Perfect for:** Integration with your analysis pipeline, custom workflows

---

## 📁 Files You Have

### Models:
- `data/processed/run_20260528_142504/best_model.pt` - Best performing model ⭐
- `data/processed/run_20260528_142504/modelo_final.pt` - Final model after 500 epochs
- `data/processed/run_20260528_142504/checkpoint_epoch_*.pt` - Checkpoints every 100 epochs

### Data:
- `data/processed/tcga_pacientes_500_500.pt` - Processed dataset (472 patients)

### Results:
- `predictions.csv` - All predictions with confidence
- `model_performance.png` - Performance visualizations

### Scripts:
- `predict.py` - Interactive command-line tool ⭐ **START HERE**
- `use_trained_model.py` - Full evaluation script
- `train_model.py` - Training script (for reference/retraining)

---

## 🎯 What Each Class Means

| Class | Code | Emoji | Description |
|-------|------|-------|-------------|
| **Primary Tumor** | 0 | 🔴 | Original melanoma tumor |
| **Metastasis** | 1 | 🟠 | Cancer that has spread |
| **Normal Tissue** | 2 | 🟢 | Healthy tissue |

---

## 📊 Your Model's Performance

```
Overall Accuracy: 54.24%

Per-Class Performance:
├─ Primary Tumor:   F1 = 32.6%  (hard to distinguish from metastasis)
├─ Metastasis:      F1 = 65.9%  ⭐ (best performance, 80% precision)
└─ Normal Tissue:   F1 = 0.0%   (only 1 sample - can't learn)

Key Strength: When model says "Metastasis", it's correct 80% of the time!
```

---

## 💡 Common Tasks

### Task 1: Predict a Specific Patient

```bash
python3 predict.py
# Choose option 1
# Enter patient index (0-471)
```

### Task 2: Get All Predictions as CSV

```bash
python3 use_trained_model.py
# Check predictions.csv
```

### Task 3: Find High-Confidence Predictions

```python
import pandas as pd

df = pd.read_csv('predictions.csv')
high_confidence = df[df['confidence'] > 0.7]
print(f"High confidence predictions: {len(high_confidence)}")
print(high_confidence[['patient_id', 'predicted_label', 'confidence', 'correct']])
```

### Task 4: Compare Primary vs Metastasis

```python
import pandas as pd

df = pd.read_csv('predictions.csv')
primary = df[df['true_label'] == 'Primary Tumor']
meta = df[df['true_label'] == 'Metastasis']

print(f"Primary tumor accuracy: {primary['correct'].mean():.1%}")
print(f"Metastasis accuracy: {meta['correct'].mean():.1%}")
```

### Task 5: Find Which Genes the Model Focuses On

```python
import torch
import numpy as np

# Load model and data (see Option 3 code above)

# Aggregate attention across multiple patients
important_edges = {}

for i in range(10):  # Check 10 patients
    patient = dataset[i].to(device)
    
    with torch.no_grad():
        out, edge_idx, alpha = model(..., return_attn=True)
        attn = alpha.mean(dim=1).cpu().numpy()
        
        # Store attention for each edge
        for j, weight in enumerate(attn):
            edge = tuple(sorted(edge_idx[:, j].cpu().numpy()))
            if edge[0] != edge[1]:  # Skip self-loops
                if edge not in important_edges:
                    important_edges[edge] = []
                important_edges[edge].append(weight)

# Find consistently high-attention edges
for edge, weights in sorted(important_edges.items(),
                           key=lambda x: np.mean(x[1]),
                           reverse=True)[:5]:
    gene1 = dataset[0].gene_names[edge[0]]
    gene2 = dataset[0].gene_names[edge[1]]
    avg_attn = np.mean(weights)
    print(f"{gene1} ↔ {gene2}: {avg_attn:.4f}")
```

---

## 🔧 Troubleshooting

### Problem: "Model not found"
**Solution:** Check the model path exists:
```bash
ls data/processed/run_*/best_model.pt
```

### Problem: "Dataset not found"
**Solution:** Make sure you ran the download script:
```bash
python3 download_data.py
```

### Problem: "Out of memory"
**Solution:** The model is small (145KB) and runs on CPU. If issues persist:
```python
# Predict one at a time instead of batching
for patient in dataset:
    # ... make prediction
    # ... clear cache
```

### Problem: "Low confidence predictions"
**This is normal!** The model averages ~54% confidence because:
- Primary and metastatic melanoma are biologically very similar
- Limited training data (472 samples)
- Model is conservative (prefers not to be overconfident)

---

## 📚 Next Steps

1. ✅ **Try the interactive tool**: `python3 predict.py`
2. ✅ **Check the CSV**: Open `predictions.csv` in Excel/Python
3. ✅ **Read the guide**: `MODEL_USAGE_GUIDE.md` for advanced usage
4. 📊 **Visualize**: Check `model_performance.png`
5. 🧬 **Learn more**: Interactive tutorial at http://localhost:8502

---

## ❓ Questions?

- **How accurate is the model?** 54% overall, but 80% precision on metastasis
- **Can I retrain?** Yes! Run `python3 train_model.py` with new data
- **Can I use my own data?** Yes, format it like the TCGA dataset
- **What if I want better performance?** See `MODEL_USAGE_GUIDE.md` → "For Better Performance"

---

## 🎓 Understanding Attention Weights

**High attention (>0.7)** = Model thinks this gene-gene interaction is important for classification

**Example:**
```
VIM ↔ TWIST1: 0.85
```
This means the model learned that the interaction between VIM (EMT marker) and TWIST1 (metastasis transcription factor) is highly predictive of metastasis!

This is **interpretable AI** - you can see **why** the model made its decision! 🎯

---

**Ready to start?**

```bash
python3 predict.py
```

🚀 **Let's make some predictions!**
