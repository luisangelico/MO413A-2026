# 🧬 Understanding Graph Attention Networks for Cancer Classification

## The Core Concept

Think of your data as a **social network of genes**:
- Each **gene** is a person (node)
- **Protein interactions** are friendships (edges)
- **Expression levels** are how "active" each person is
- The model learns which "friendships" matter most for identifying cancer

## Architecture Breakdown

### Layer 1: Multi-Head Attention (4 heads)

**Input:** 500 genes, each with 1 feature (log-transformed expression)

**What it does:**
```
For each gene, the model asks:
"Which of my protein partners should I pay attention to?"
```

**Example:**
- Gene A (highly expressed in metastasis) interacts with Genes B, C, D
- The attention mechanism learns: "When classifying metastasis, Gene C's expression matters more than B or D"
- This happens simultaneously across 4 different "attention heads" - like 4 experts looking at different patterns

**Output:** 256 features per gene (64 features × 4 heads)

### Why Multiple Heads?

Each head can specialize:
- **Head 1**: Might focus on immune response genes
- **Head 2**: Might focus on cell division genes  
- **Head 3**: Might focus on melanocyte-specific genes
- **Head 4**: Might focus on metastasis markers

### Layer 2: Single Attention Head

**What it does:**
Combines the 4 perspectives from Layer 1 into a unified understanding

**Output:** 64 features per gene

### Global Pooling

**Problem:** We have 500 genes but need 1 prediction per patient

**Solution:** Average all gene features into a single patient-level representation
```
Patient representation = Average(all 500 gene embeddings)
```

### Final Classifier

**Output:** 3 probabilities
- P(Primary Tumor)
- P(Metastasis)  
- P(Normal Tissue)

## What Makes This Powerful?

### 1. **Biological Context**
Traditional ML: "Gene X has high expression → probably cancer"
GAT: "Gene X has high expression AND its protein partners Y and Z are also dysregulated → strong metastasis signal"

### 2. **Attention Weights Show Interpretability**

After training, we can see which interactions the model focuses on:
```
High Attention Weight = "This protein interaction is important for classification"
Low Attention Weight = "This interaction doesn't help distinguish tumor types"
```

### 3. **Handles Imbalanced Data**

Your dataset has:
- 368 metastasis samples
- 103 primary tumor samples  
- 1 normal sample

The model uses **class weights** to avoid just predicting "metastasis" for everything:
```python
weights = [1.53, 0.43, 157.33]  # Heavily weight the rare normal class
```

## Real Example: What the Model Learns

### Scenario 1: Primary Melanoma
```
S100B (melanocyte marker) ──high attention──> MITF (melanocyte TF)
                          ──high attention──> TYR (melanin synthesis)

Pattern: Strong melanocyte lineage signatures
→ Prediction: Primary Tumor (78% confidence)
```

### Scenario 2: Metastatic Melanoma
```
IGKC (immune) ──high attention──> HLA-DRA (antigen presentation)
VIM (EMT marker) ──high attention──> TWIST1 (metastasis TF)

Pattern: Immune infiltration + epithelial-mesenchymal transition
→ Prediction: Metastasis (92% confidence)
```

## Key Biological Insights the Model Captures

### 1. **Immune Gene Modules**
In your data, top expressed genes include:
- IGKC, IGHG1, IGHM (immunoglobulins)
- HLA-DRA (MHC class II)

The model learns: "These genes cluster together in the PPI network, and their combined expression pattern indicates tumor-immune interactions"

### 2. **Melanoma-Specific Markers**
- S100B (known melanoma biomarker)
- APOD (associated with tumor microenvironment)

The model learns: "S100B's importance changes based on its network neighbors' states"

### 3. **Network Topology Matters**

Genes with high **degree** (many connections) in the PPI network:
- Act as "hubs" that aggregate information from many neighbors
- The attention mechanism often weights these hub genes more heavily
- Biological interpretation: Master regulators get more attention

## Training Dynamics

### What Happens Over 500 Epochs:

**Epochs 1-50:** Random weights → Learning basic patterns
- "High expression in immune genes → probably metastasis"

**Epochs 51-200:** Refining attention weights
- "IGKC matters when HLA-DRA is also high"
- "Ignore this APOD→IGKV connection for primary tumors"

**Epochs 201-500:** Fine-tuning
- Adjusting subtle patterns
- Balancing overfitting vs. generalization
- Optimizing rare class (normal tissue) prediction

### The Loss Curve

```
Epoch 1:   Loss = 1.5 (basically guessing)
Epoch 100: Loss = 0.3 (learned major patterns)
Epoch 500: Loss = 0.1 (well-optimized)
```

## After Training: The Visualization

When you run `streamlit run app.py`, you'll see:

### 1. **Attention Heatmap**
- Bright edges = "Model pays attention to this interaction"
- Dim edges = "This interaction doesn't help classification"

### 2. **Biological Validation**
You can check if high-attention edges correspond to:
- Known cancer pathways (e.g., MAPK, PI3K-AKT)
- Immune checkpoints (e.g., PD-1/PD-L1 axis)
- Metastasis regulators (e.g., EMT factors)

### 3. **Per-Patient Analysis**
Select different patients and see:
- How does the attention pattern differ for primary vs. metastatic tumors?
- Which genes drive each prediction?

## Why This Matters for Melanoma Research

### Traditional Approach:
"List of differentially expressed genes between primary and metastatic melanoma"
→ Hard to interpret, misses interactions

### GAT Approach:
"Network modules and specific interactions that distinguish tumor types"
→ Reveals functional mechanisms
→ Can identify therapeutic targets (e.g., proteins in high-attention subnetworks)

## Technical Details You're Learning

1. **Message Passing**: Each gene aggregates information from neighbors
2. **Attention Coefficients**: Learned importance weights for each edge
3. **Multi-Head Design**: Captures diverse biological processes simultaneously
4. **Graph Pooling**: Summarizes patient-level patterns from gene-level signals

## Next Steps After Training

1. **Evaluate Performance**: Check accuracy, confusion matrix
2. **Inspect Attention**: Which PPIs are most important?
3. **Biological Validation**: Do high-attention subnetworks match known biology?
4. **Potential Publication**: "Network-based classification of melanoma subtypes using graph attention"
