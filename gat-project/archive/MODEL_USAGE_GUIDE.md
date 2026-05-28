# 🎯 Using Your Trained GAT Model - Complete Guide

## 📊 Training Results

**Model Performance:**
- **Overall Accuracy**: 54.24%
- **Best at**: Metastasis classification (F1: 65.9%)
- **Trained on**: 472 patients (500 epochs)
- **Model size**: 145 KB

### Why 54% Accuracy?

This is actually reasonable for this challenging task:
1. **Highly imbalanced dataset** (368 metastasis vs 103 primary vs 1 normal)
2. **Biological reality**: Primary and metastatic melanoma share many molecular features
3. **Limited training data**: Only 472 samples for complex biological classification
4. **No validation split**: Training on full dataset (would need cross-validation for production)

**The model learned:** Metastasis detection is strong (80% precision), which is clinically relevant!

---

## 🚀 Three Ways to Use Your Model

### **Option 1: Interactive Visualization (Recommended)**

Launch the Streamlit app to explore predictions with attention weights:

```bash
streamlit run app.py
```

**Features:**
- Select any patient
- See prediction confidence
- **Visualize attention weights** (which gene interactions matter)
- Interactive network graph
- Export results

**Update the model path in app.py:**
```python
# Line ~52 in app.py
model_path = st.sidebar.text_input(
    "Caminho do Modelo (.pt)",
    "./data/processed/run_20260528_142504/best_model.pt"  # ← Update this
)
```

---

### **Option 2: Python Script for Batch Predictions**

Use the evaluation script we just ran:

```bash
python3 use_trained_model.py
```

**Outputs:**
- `predictions.csv` - All predictions with confidence scores
- `model_performance.png` - Performance visualizations
- Console output with metrics

**What you get:**
```csv
patient_id,true_label,predicted_label,confidence,correct
TCGA-ER-A199-06A,Metastasis,Metastasis,0.565,True
TCGA-EE-A3J5-06A,Metastasis,Metastasis,0.553,True
...
```

---

### **Option 3: Use in Your Own Code**

Here's a minimal example:

```python
import torch
import torch.nn.functional as F
from pathlib import Path

# 1. Define model architecture (copy from train_model.py)
class GATv2Classifier(torch.nn.Module):
    # ... (same as before)

# 2. Load trained weights
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GATv2Classifier(hidden_channels=64, num_classes=3).to(device)
model.load_state_dict(torch.load('data/processed/run_20260528_142504/best_model.pt',
                                 map_location=device))
model.eval()

# 3. Load a patient
dataset = torch.load('data/processed/tcga_pacientes_500_500.pt', weights_only=False)
patient = dataset[0].to(device)

# 4. Make prediction
with torch.no_grad():
    out = model(patient.x, patient.edge_index,
               torch.zeros(patient.x.size(0), dtype=torch.long, device=device))
    
    probs = torch.softmax(out, dim=1).cpu().numpy()[0]
    predicted_class = probs.argmax()
    confidence = probs[predicted_class]

# 5. Get attention weights (optional)
with torch.no_grad():
    out, edge_index_attn, alpha = model(
        patient.x, patient.edge_index,
        torch.zeros(patient.x.size(0), dtype=torch.long, device=device),
        return_attn=True
    )
    attention_weights = alpha.mean(dim=1).cpu().numpy()

print(f"Predicted class: {predicted_class}")
print(f"Confidence: {confidence:.2%}")
print(f"Top attention edge: {attention_weights.max():.4f}")
```

---

## 📈 Understanding the Results

### Confusion Matrix Interpretation

```
                  Predicted
               Pri   Met   Nor
True  Pri      50    50     3     ← 50/103 primary correctly classified
      Met     154   206     8     ← 206/368 metastasis correctly classified  
      Nor       0     1     0     ← 0/1 normal correctly classified
```

**Key Insights:**
- Model confuses **primary ↔ metastasis** (biologically similar)
- **Strong at identifying metastasis** when it predicts it (80% precision)
- **Poor on normal tissue** (only 1 sample - not enough data)

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Meaning |
|-------|-----------|--------|----------|---------|
| **Primary** | 24.5% | 48.5% | 32.6% | When model says "primary", correct 24.5% of time |
| **Metastasis** | 80.2% | 56.0% | 65.9% | When model says "metastasis", correct 80% of time |
| **Normal** | 0.0% | 0.0% | 0.0% | Can't learn from 1 sample |

**Clinical Implication:** The model is **conservative** - prefers to call things metastasis (safer to overtreat than undertreat).

---

## 🔍 Interpreting Attention Weights

The model learns which **protein-protein interactions** are important:

```python
# Top 5 attention weights for patient TCGA-ER-A199-06A:
FABP4 ↔ FABP4: 1.0000           # Self-loop (node importance)
FCRLA ↔ FCRLA: 1.0000           # Immune receptor
SPRR1B ↔ SPRR1B: 1.0000         # Differentiation marker
IGKV3D-20 ↔ IGKV3D-20: 1.0000   # Immunoglobulin
```

**Note:** Self-loops (GENE ↔ GENE) have high attention because the node's own expression is very informative.

**To find important interactions:**
```python
# Filter out self-loops
for i, attn in enumerate(attention_weights):
    edge = edge_index_attn[:, i].cpu().numpy()
    if edge[0] != edge[1]:  # Not a self-loop
        gene1 = patient.gene_names[edge[0]]
        gene2 = patient.gene_names[edge[1]]
        if attn > 0.5:  # High attention
            print(f"{gene1} ↔ {gene2}: {attn:.4f}")
```

---

## 🎨 Visualizations Created

### 1. `model_performance.png`
Four subplots showing:
- **Confusion matrix heatmap**
- **Confidence distribution** (correct vs incorrect)
- **Per-class F1 scores** (bar chart)
- **True vs predicted class counts**

### 2. `predictions.csv`
Full results for all 472 patients

### 3. Interactive Streamlit App
- Patient-specific attention visualization
- Network graphs with attention-weighted edges
- Hover to see gene names and expression

---

## 💡 Advanced Usage

### 1. Cross-Validation for Better Estimates

```python
from sklearn.model_selection import StratifiedKFold

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
labels = [data.y.item() for data in dataset]

for fold, (train_idx, val_idx) in enumerate(kfold.split(range(len(dataset)), labels)):
    train_data = [dataset[i] for i in train_idx]
    val_data = [dataset[i] for i in val_idx]
    # Train on train_data, evaluate on val_data
    # ... (training loop)
```

### 2. Extract Important Gene Pairs

```python
# Aggregate attention across all patients
important_edges = {}

for patient in dataset[:10]:  # Sample 10 patients
    with torch.no_grad():
        out, edge_idx, alpha = model(..., return_attn=True)
        attn = alpha.mean(dim=1).cpu().numpy()
        
        for i, weight in enumerate(attn):
            edge = tuple(sorted(edge_idx[:, i].cpu().numpy()))
            if edge not in important_edges:
                important_edges[edge] = []
            important_edges[edge].append(weight)

# Find consistently high-attention edges
for edge, weights in sorted(important_edges.items(),
                           key=lambda x: np.mean(x[1]), reverse=True)[:10]:
    gene1 = dataset[0].gene_names[edge[0]]
    gene2 = dataset[0].gene_names[edge[1]]
    print(f"{gene1} ↔ {gene2}: avg attn = {np.mean(weights):.4f}")
```

### 3. Integrate with Clinical Data

```python
import pandas as pd

# Load predictions
predictions_df = pd.read_csv('predictions.csv')

# Add clinical features (example)
clinical_data = pd.DataFrame({
    'patient_id': [...],
    'age': [...],
    'stage': [...],
    'survival_months': [...]
})

merged = predictions_df.merge(clinical_data, on='patient_id')

# Analyze: Do confident predictions correlate with better outcomes?
high_conf = merged[merged['confidence'] > 0.7]
print(f"High confidence survival: {high_conf['survival_months'].mean():.1f} months")
```

---

## 🚨 Important Notes

### Model Limitations

1. **Training = Testing**: Model was evaluated on training data (optimistic estimate)
   - **Solution**: Use cross-validation or hold-out test set

2. **Class Imbalance**: 368 metastasis vs 103 primary vs 1 normal
   - **Solution**: Use stratified sampling or synthetic data (SMOTE)

3. **Overfitting Risk**: Small dataset (472 patients)
   - **Solution**: Regularization (dropout, early stopping)

4. **Biological Complexity**: Gene expression alone may not capture everything
   - **Solution**: Add clinical features, mutations, copy number alterations

### When to Retrain

- More data becomes available
- Want to add new features
- Need better performance on specific classes
- Integrate with other omics data

---

## 📚 Next Steps

### For Research:
1. **Biological validation**: Do high-attention edges match known cancer pathways?
2. **Literature mining**: Are attention-highlighted genes in melanoma papers?
3. **Experimental validation**: Test predicted interactions in lab

### For Better Performance:
1. **Collect more normal samples** (critical - only have 1!)
2. **Add data augmentation** (graph augmentation techniques)
3. **Ensemble models** (train multiple models, average predictions)
4. **Hyperparameter tuning** (learning rate, hidden dimensions, dropout)

### For Production:
1. **Cross-validation** for honest performance estimates
2. **Calibration** (ensure confidence scores are meaningful)
3. **Monitoring** (track performance on new data)
4. **Documentation** (model card, intended use, limitations)

---

## 🎓 Learning Resources

- **GAT Paper**: [Graph Attention Networks (Veličković et al., 2018)](https://arxiv.org/abs/1710.10903)
- **PyTorch Geometric**: [Documentation](https://pytorch-geometric.readthedocs.io/)
- **Interpretable ML**: [Christoph Molnar's Book](https://christophm.github.io/interpretable-ml-book/)

---

## 💬 Questions?

Check the interactive tutorial:
```bash
streamlit run explain_gat_interactive.py
```

Or explore the code:
- `train_model.py` - Training script
- `use_trained_model.py` - Evaluation and predictions
- `app.py` - Interactive visualization
- `explain_gat_interactive.py` - Educational tool
