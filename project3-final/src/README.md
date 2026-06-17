# `src/` — Reproduction Package

This directory contains everything needed to reproduce the results reported in
`project3-final/README.md`: a Graph Attention Network (GATv2) classifier over
protein–protein interaction (PPI) graphs of skin samples (Benign Nevus / Primary
Tumor / Metastasis), plus all interpretation, pseudotime and stability analyses.

## Layout

```
src/
├── README.md             <- this file
├── requirements.txt      <- pinned Python dependencies
├── src/                  <- importable package (config, model, train, ...)
│   ├── config.py         <- single source of truth (paths, dataset, classes)
│   ├── model.py          <- GATv2Classifier
│   ├── train.py          <- training loop + early stopping
│   ├── evaluate.py       <- held-out test metrics
│   ├── predict.py        <- interactive CLI
│   ├── interpret.py      <- attention aggregation + gene-set recovery
│   ├── interpret_viz.py  <- per-class subgraph plots
│   ├── pseudotime.py     <- Nevus → Primary → Metastasis trajectory
│   ├── multiseed_stability.py  <- multi-seed Jaccard / Spearman / pseudotime
│   ├── embeddings.py     <- UMAP/PCA of final embeddings
│   ├── i18n.py           <- EN/PT-BR translation strings
│   ├── translate_site.py <- bilingual report helpers
│   ├── about_gat_page.py / dataset_page.py / site_header.py
│   └── __init__.py
├── scripts/              <- entry-point scripts (run as modules)
│   ├── download_toil.py        <- builds the dataset from TOIL + GSE112509
│   ├── train_multiseed.py      <- 5-seed reproducibility runner
│   ├── biomarker_report.py     <- consensus top-K across seeds
│   ├── build_static_site.py    <- exports HTML report (optional)
│   ├── visualize_dataset.py    <- dataset overview plots
│   ├── explore_streamlit.py    <- interactive dataset explorer
│   ├── predict_streamlit.py    <- interactive prediction UI
│   └── rebuild_site.sh
└── notebooks/
    └── explain_gat_interactive.py   <- Streamlit GAT tutorial
```

The package layout matches the original `gat-project/`: every module is imported
as `src.<name>` (e.g. `from src.config import DATASET_FILE`). All commands are
run as Python modules **from this `src/` directory** so that `src.<name>` and
`scripts.<name>` resolve correctly.

## Environment

- Python 3.10+ (tested on 3.10 / 3.11)
- A GPU is recommended but optional. The code auto-selects CUDA → MPS → CPU
  (override with `GAT_DEVICE=cpu|cuda|mps`).
- ~3 GB free disk for the cached raw TOIL/GEO files; ~80 MB for processed
  artifacts.

```bash
cd project3-final/src
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

If you are behind a corporate proxy:

```bash
export REQUESTS_CA_BUNDLE=/path/to/your/ca-bundle.pem    # preferred
# or, as a last resort:
export GAT_INSECURE_SSL=1
```

## Data

`src/config.py` is the single source of truth for paths. By default it expects:

```
project3-final/
├── data/
│   ├── raw/         <- TOIL TPM matrix, phenotype, GSE112509 counts, gencode probemap
│   └── processed/   <- generated .pt PyG datasets, splits, run_*/ outputs
└── src/             <- (this directory)
```

The repository ships with `data/raw/` and `data/processed/` already populated, so
you can skip the download step. To rebuild the dataset from scratch:

```bash
cd project3-final/src
python -m scripts.download_toil      # ~3 GB cached, builds skcm_nevi_1000_200.pt
```

The active dataset filename is `skcm_nevi_{NUM_NODES}_{CONFIDENCE_THRESHOLD}.pt`
where the defaults are `NUM_NODES=1000` (top-variable genes) and
`CONFIDENCE_THRESHOLD=200` (STRING combined-score cutoff). Edit `src/config.py`
to change them — every script picks it up.

## Reproduce the headline results

Run from `project3-final/src/`:

```bash
# 1. Train + evaluate a single seed (sanity check)
python -m src.train             # writes best_model.pt + splits.npz
python -m src.evaluate          # held-out test metrics

# 2. Multi-seed run (the one quoted in the README — 5 seeds)
python -m scripts.train_multiseed --seeds 1 2 3 4 5

# 3. Interpretation: attention rankings + gene-set recovery
python -m src.interpret         # CSVs in data/processed/run_<ts>/
python -m src.interpret_viz     # static + interactive subgraph plots

# 4. Pseudotime trajectory
python -m src.pseudotime

# 5. Multi-seed stability (Jaccard@K, Spearman, pseudotime band)
python -m src.multiseed_stability --runs run_multiseed_seed1 run_multiseed_seed2 ...

# 6. Biomarker consensus across seeds
python -m scripts.biomarker_report
```

All outputs land under `data/processed/run_*/` (single-seed) or
`data/processed/multiseed_*/` (consolidated). The figures referenced in
`project3-final/README.md` (`stability_accuracy.png`, `stability_spearman.png`,
`stability_knn_jaccard.png`, `pseudotime_heatmap.png`, `pseudotime_lines.png`)
are produced by `multiseed_stability.py` and `pseudotime.py`.

## Interactive exploration (optional)

```bash
streamlit run scripts/explore_streamlit.py            # dataset explorer
streamlit run scripts/predict_streamlit.py            # prediction UI
streamlit run notebooks/explain_gat_interactive.py    # GAT tutorial
```

## Reproducibility notes

- **Seed.** `src/train.py` sets `SEED=42`; the multi-seed script overrides it
  per run and saves a separate `splits.npz` per seed (no leakage across seeds).
- **Splits.** Stratified 70/15/15 train/val/test, seeded, persisted to
  `splits.npz`. `src/evaluate.py` reads `splits.npz` and warns loudly if it
  is missing — that warning means data leakage; retrain.
- **Class weights** are computed from the **train set only**.
- **Model selection** is on validation loss with early stopping (`EARLY_STOP_PATIENCE = 50` in `src/train.py`).
- **Per-class aggregation** of attention uses **only correctly classified test
  samples** (see `src.interpret.aggregate_per_class`).
- **TOIL caveat.** TOIL expression values are *already* `log2(TPM+0.001)`. Do
  not apply `log1p` again — the archived GDC-Hub pipeline did, and the two
  outputs are not interchangeable.

## Single source of truth

- **Dataset path** lives in `src/config.py` as `DATASET_FILE`. Don't hardcode
  paths anywhere else.
- **Model architecture** lives in `src/model.py` as `GATv2Classifier`.
  Checkpoints are self-describing via a sidecar `.json` written by
  `save_model()` / read by `load_model()`.

## Class labels

```
0 = Primary Tumor       (TCGA-SKCM, sample type "Primary Tumor")
1 = Metastasis          (TCGA-SKCM, sample type "Metastatic")
2 = Benign Nevus        (GSE112509)
```

Class 2 is benign melanocytic nevi (GSE112509), replacing the legacy
GTEx-skin Class 2. GTEx skin is bulk epidermis (mostly keratinocytes), which
made keratinization genes the dominant — and biologically misleading —
contrast against melanoma. Nevi are also melanocytic, so the current contrast
reflects malignant vs benign melanocyte transcriptomes.
