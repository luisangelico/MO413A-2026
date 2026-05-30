# Skin Cancer and its Types: An Analysis of Gene Expression Profiles in Networks

A Graph Attention Network (GAT) that classifies skin samples as
**Primary Tumor / Metastasis / Normal Skin** from gene expression, and
explains its predictions by inspecting attention over a protein–protein
interaction (PPI) graph.

## Abstract

Each sample is a graph: nodes are top-variable genes carrying their
log₂(TPM+0.001) expression as a feature, edges are STRING physical PPIs.
Data come from harmonized TCGA-SKCM and GTEx skin cohorts processed through
the UCSC Xena **TOIL recompute** to remove cross-cohort batch effects. A
three-layer **GATv2** classifier is trained on this dataset, and per-sample
edge attention is extracted from every layer to derive class-level node
rankings. We benchmark these rankings against curated gene sets (melanocyte
lineage, melanoma progression, immune infiltrate, cornified-envelope/keratin)
and against a **node-degree baseline**. Class-contrast rankings (e.g.
*Metastasis − Primary*) recover known skin/melanoma biology with high
significance and outperform the degree baseline — evidence that the GAT
learns class-specific gene importance, not just PPI hub structure. Outputs
include static figures and **interactive PPI subgraphs** embedded in the
per-run static site.

## Quickstart

```bash
pip install -r requirements.txt

python -m scripts.download_toil       # build dataset (~3 GB cached)
python -m src.train                   # stratified split + early stopping
python -m src.evaluate                # held-out test metrics
python -m src.interpret               # attention rankings + gene-set recovery
python -m src.interpret_viz --html    # static + interactive figures
python -m scripts.build_static_site   # site with predictions + interpret/
```

Run commands from `gat-project/` so `src.config` paths resolve.

## Layout

```
src/        config, model, train, evaluate, predict, interpret, interpret_viz
scripts/    download_toil, build_static_site, explore_streamlit, visualize_dataset
notebooks/  explain_gat_interactive (Streamlit)
site/       generated static site (predictions + interpret/)
docs/       GitHub Pages output
data/       gitignored, produced by download_toil
archive/    superseded scripts (read-only)
```

## Single source of truth

`src/config.py`:

```python
DATASET_FILE = PROCESSED_DATASET_PATH / f"toil_skin_{NUM_NODES}_{CONFIDENCE_THRESHOLD}.pt"
```

Change this line and every script picks it up. `src/model.py` is the only
place `GATv2Classifier` is defined; checkpoints are self-describing via a
sidecar `.json`.

## Methodology highlights

- Stratified 70/15/15 split, seeded, saved to `splits.npz`
- Class weights from train set only (no leakage)
- Best model selected on validation loss, early stopping (patience 30)
- `forward(..., return_all_attn=True)` exposes `(edge_index, alpha)` per layer
- Per-class aggregation uses **correctly classified** test samples only
- Gene-set recovery reported with hypergeometric p-values; degree baseline
  included so attention must beat hub structure to count

Reference gene sets and curation rationale live in `src/interpret.py`.
Mutation drivers (BRAF, NRAS…) are intentionally absent from the default
sets — they are not highly variable in expression and so don't appear in a
top-variability node panel.

## Notes

- **TOIL values are already log2(TPM+0.001)** — do not re-apply `log1p`.
  The legacy `archive/download_data.py` (GDC-Hub TPM) does; outputs are not
  interchangeable.
- SSL behind a corporate proxy: set `REQUESTS_CA_BUNDLE=/path/to/bundle.pem`,
  or `GAT_INSECURE_SSL=1` as a last resort.
