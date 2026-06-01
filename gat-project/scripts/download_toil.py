#!/usr/bin/env python3
"""
Build the melanoma-vs-nevus dataset.

Two cohorts:
  1. TCGA-SKCM tumor samples — UCSC Xena TOIL recompute, log2(TPM+0.001)
       Labels 0 (Primary Tumor) and 1 (Metastasis).
  2. Benign melanocytic nevi — GEO GSE112509 (Hartmann et al., 2018),
       laser-microdissected nevi, DESeq2-normalized counts from the GEO
       supplementary file.
       Label 2 (Benign Nevus).

Why nevi instead of GTEx skin?
    Melanocytes do not keratinize — keratinization is a keratinocyte function.
    GTEx "skin" is bulk epidermis (overwhelmingly keratinocytes), so a model
    contrasting melanoma with GTEx skin learns a melanocyte-vs-keratinocyte
    signature: keratinization genes (KRT*, LOR, FLG, IVL) dominate. That is a
    tissue-composition artifact, not melanocyte biology. Nevi are benign
    melanocytic lesions — same lineage as melanoma — so the contrast reflects
    malignant vs benign melanocyte transcriptomes.

Cross-cohort harmonization (TOIL vs GSE112509)
    The two cohorts come from different pipelines and different units:
        TOIL:       log2(TPM + 0.001)         (Xena recompute)
        GSE112509:  DESeq2 size-factor counts (GEO supp)
    To put them on comparable scales we use TCGA-train as the reference frame:
        1. log2(x + 1)-transform the GSE112509 counts.
        2. Compute per-gene mean & std from TCGA-SKCM TRAIN samples only.
        3. Standardize BOTH cohorts with these TCGA-reference statistics.
    Why a reference frame instead of per-cohort z-scoring? Cohort and class 2
    (Benign Nevus) are perfectly confounded — all nevi live in GSE112509,
    no tumors do — so per-cohort z-scoring forces nevi to mean 0/std 1 with
    no tumor reference, erasing the very offset that makes nevi distinguishable
    from tumors. Reference-based standardization preserves it: the TCGA mean
    becomes the origin, and a nevus that is consistently below tumor expression
    on (say) cell-cycle genes ends up with negative values for those genes,
    which is the biologically meaningful signal.
    This still does NOT remove all batch effects (joint reprocessing from
    FASTQs would be the gold standard), and we are explicit about that on
    the dataset page.

Top-variable gene selection is computed on the TRAIN split only to avoid
leakage into the test set.
"""

import io
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

from src.config import NUM_NODES, CONFIDENCE_THRESHOLD, PROCESSED_DATASET_PATH

SPLIT_SEED = 42
TEST_FRAC = 0.15
VAL_FRAC = 0.15
RAW_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw_toil"
PROCESSED_DATASET_PATH.mkdir(parents=True, exist_ok=True)
RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)

# TOIL recompute (TCGA-SKCM)
TOIL_BASE = "https://toil-xena-hub.s3.us-east-1.amazonaws.com/download/"
URL_EXPR = TOIL_BASE + "TcgaTargetGtex_rsem_gene_tpm.gz"
URL_PHENO = TOIL_BASE + "TcgaTargetGTEX_phenotype.txt.gz"
URL_PROBEMAP = TOIL_BASE + "probeMap%2Fgencode.v23.annotation.gene.probemap"

EXPR_LOCAL = RAW_DATA_PATH / "TcgaTargetGtex_rsem_gene_tpm.gz"
PHENO_LOCAL = RAW_DATA_PATH / "TcgaTargetGTEX_phenotype.txt.gz"
PROBEMAP_LOCAL = RAW_DATA_PATH / "gencode.v23.annotation.gene.probemap"

# GSE112509 (nevi)
URL_GSE_COUNTS = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE112nnn/GSE112509/"
    "suppl/GSE112509_DESeq2_normalized_counts.txt.gz"
)
GSE_LOCAL = RAW_DATA_PATH / "GSE112509_DESeq2_normalized_counts.txt.gz"

OUTPUT_FILE = PROCESSED_DATASET_PATH / f"skcm_nevi_{NUM_NODES}_{CONFIDENCE_THRESHOLD}.pt"

CA_BUNDLE = Path("./proxy-fix-bundle/allCAbundle.pem")
INSECURE_SSL = os.environ.get("GAT_INSECURE_SSL") == "1"
if INSECURE_SSL:
    warnings.filterwarnings("ignore", message="Unverified HTTPS request")
    import urllib3

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    VERIFY = False
    print("WARNING: GAT_INSECURE_SSL=1 — disabling certificate verification.")
elif CA_BUNDLE.exists():
    VERIFY = str(CA_BUNDLE)
    print(f"Using CA bundle: {VERIFY}")
else:
    VERIFY = True


def download_to(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  cached: {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return
    print(f"  downloading {url}")
    with requests.get(url, verify=VERIFY, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        if total:
            print(f"    size: {total / 1e6:.1f} MB")
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    print(f"  saved: {dest}")


def zscore_per_gene(df: pd.DataFrame) -> pd.DataFrame:
    """Gene-wise z-score across samples (axis=1). Constant genes -> 0."""
    mu = df.mean(axis=1)
    sd = df.std(axis=1).replace(0, 1.0)
    return df.sub(mu, axis=0).div(sd, axis=0)


def standardize_with(df: pd.DataFrame, mu: pd.Series, sd: pd.Series) -> pd.DataFrame:
    """Apply (x - mu)/sd using externally-supplied per-gene stats."""
    sd_safe = sd.replace(0, 1.0)
    return df.sub(mu, axis=0).div(sd_safe, axis=0)


print("=" * 70)
print("TCGA-SKCM (TOIL) + GSE112509 nevi")
print("=" * 70)

if OUTPUT_FILE.exists():
    print(f"\nDataset already exists: {OUTPUT_FILE}")
    ds = torch.load(OUTPUT_FILE, weights_only=False)
    print(f"  {len(ds)} samples")
    raise SystemExit(0)

# ----------------------------------------------------------------------------
# 1. TCGA-SKCM via TOIL
# ----------------------------------------------------------------------------
print("\n[1/7] Downloading TOIL metadata...")
download_to(URL_PHENO, PHENO_LOCAL)
download_to(URL_PROBEMAP, PROBEMAP_LOCAL)

print("\n[2/7] Selecting TCGA-SKCM samples from TOIL phenotype...")
pheno = pd.read_csv(
    PHENO_LOCAL, sep="\t", compression="gzip", low_memory=False, encoding="latin-1"
)
pheno.columns = [c.strip() for c in pheno.columns]

study_col = "_study" if "_study" in pheno.columns else "study"
sample_type_col = "_sample_type" if "_sample_type" in pheno.columns else "sample_type"
disease_col = next(
    c for c in ("primary disease or tissue", "_primary_disease", "primary_disease")
    if c in pheno.columns
)

is_tcga = pheno[study_col].astype(str).str.upper() == "TCGA"
skcm = pheno[is_tcga & pheno[disease_col].astype(str).str.contains(
    "skin cutaneous melanoma", case=False, na=False
)].copy()


def label_skcm(row):
    st = str(row[sample_type_col]).lower()
    if "primary" in st:
        return 0
    if "metastatic" in st or "metastasis" in st:
        return 1
    return None


skcm["y"] = skcm.apply(label_skcm, axis=1)
skcm = skcm.dropna(subset=["y"])
skcm["y"] = skcm["y"].astype(int)
print(f"  TCGA-SKCM Primary:    {(skcm['y'] == 0).sum()}")
print(f"  TCGA-SKCM Metastasis: {(skcm['y'] == 1).sum()}")
skcm_ids = set(skcm["sample"].tolist())

print("\n[3/7] Loading TOIL expression matrix (large, ~3 GB)...")
download_to(URL_EXPR, EXPR_LOCAL)
header = pd.read_csv(EXPR_LOCAL, sep="\t", compression="gzip", nrows=0)
gene_col = list(header.columns)[0]
keep_cols = [gene_col] + [c for c in header.columns[1:] if c in skcm_ids]
print(f"  matched TCGA samples: {len(keep_cols) - 1} / {len(skcm_ids)}")

print("  reading expression...")
toil = pd.read_csv(
    EXPR_LOCAL, sep="\t", compression="gzip", usecols=keep_cols, index_col=0
)
print(f"  TOIL expression shape: {toil.shape}  (genes × samples)")

# Map versioned Ensembl → gene symbol via probemap
probemap = pd.read_csv(
    PROBEMAP_LOCAL, sep="\t", usecols=["id", "gene"]
).set_index("id")
toil = toil.join(probemap, how="inner").dropna(subset=["gene"])
toil = toil.set_index("gene")
toil = toil[~toil.index.duplicated(keep="first")]
print(f"  after gene-symbol mapping: {toil.shape}")

# ----------------------------------------------------------------------------
# 4. GSE112509 nevi
# ----------------------------------------------------------------------------
print("\n[4/7] Downloading GSE112509 (nevi) supplementary counts...")
download_to(URL_GSE_COUNTS, GSE_LOCAL)

print("  reading DESeq2-normalized counts...")
gse = pd.read_csv(GSE_LOCAL, sep="\t", compression="gzip", index_col=0)
print(f"  GSE112509 raw shape: {gse.shape}  (genes × samples)")

# Sample naming: '_N' suffix = nevus, '_M' suffix = melanoma. We only want nevi.
nevus_cols = [c for c in gse.columns if c.endswith("_N")]
print(f"  nevus samples (suffix _N): {len(nevus_cols)}")
gse = gse[nevus_cols]

# Gene IDs are versioned Ensembl, like the probemap. Map → symbol.
gse = gse.join(probemap, how="inner").dropna(subset=["gene"])
gse = gse.set_index("gene")
gse = gse[~gse.index.duplicated(keep="first")]
print(f"  after gene-symbol mapping: {gse.shape}")

# Harmonize units: log2(x + 1) brings DESeq2 counts to a log scale comparable
# in spread (not offset) to TOIL's log2(TPM + 0.001).
gse_log = np.log2(gse.astype(float) + 1.0)

# ----------------------------------------------------------------------------
# 5. Intersect genes across cohorts (still on the log-scale, NOT yet z-scored)
# ----------------------------------------------------------------------------
print("\n[5/7] Intersecting genes across cohorts...")
shared_genes = toil.index.intersection(gse_log.index)
print(f"  shared genes: {len(shared_genes)}")
toil = toil.loc[shared_genes]
gse_log = gse_log.loc[shared_genes]

# Combined log-scale matrix — used ONLY for top-variable gene selection.
# Selecting on z-scored values would give every gene unit variance per cohort,
# making the ranking noise-dominated (lncRNAs/pseudogenes win arbitrary ties).
expr_log = pd.concat([toil, gse_log], axis=1)
print(f"  combined log-expression: {expr_log.shape}  (genes × samples)")

# Build label table aligned to expression columns
y_lookup = {row["sample"]: int(row["y"]) for _, row in skcm.iterrows()}
for col in nevus_cols:
    y_lookup[col] = 2

sample_ids = [c for c in expr_log.columns if c in y_lookup]
labels = np.array([y_lookup[s] for s in sample_ids], dtype=int)
print(f"  total samples: {len(sample_ids)}")
print(f"    Primary:    {(labels == 0).sum()}")
print(f"    Metastasis: {(labels == 1).sum()}")
print(f"    Nevus:      {(labels == 2).sum()}")

# ----------------------------------------------------------------------------
# 6. Stratified split → top-variable genes (on log scale, train only) →
#    per-cohort z-score on the chosen panel → STRING PPI
# ----------------------------------------------------------------------------
idx = np.arange(len(sample_ids))
train_idx, temp_idx = train_test_split(
    idx, test_size=TEST_FRAC + VAL_FRAC, stratify=labels, random_state=SPLIT_SEED
)
val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=TEST_FRAC / (TEST_FRAC + VAL_FRAC),
    stratify=labels[temp_idx],
    random_state=SPLIT_SEED,
)
train_samples = [sample_ids[i] for i in train_idx]
print(
    f"  split: train={len(train_idx)} | val={len(val_idx)} | test={len(test_idx)} "
    f"(seed={SPLIT_SEED})"
)

# Restrict to genes with a known HGNC-ish symbol (drops lncRNA/pseudogene
# names like RP11-* / AC0* / CTD-* that are absent from STRING anyway). The
# remaining set still covers ~20k genes — far more than NUM_NODES.
def _is_protein_coding_symbol(sym: str) -> bool:
    if not isinstance(sym, str):
        return False
    bad_prefixes = ("RP11-", "RP1-", "RP3-", "RP4-", "RP5-", "RP6-", "AC0",
                    "AL", "AP00", "CTD-", "CTC-", "LINC", "MIR", "SNOR",
                    "XLOC", "LOC")
    return not (sym.startswith(bad_prefixes) or "." in sym)

protein_coding = [g for g in expr_log.index if _is_protein_coding_symbol(g)]
expr_log_pc = expr_log.loc[protein_coding]
print(f"  filtered to protein-coding-ish symbols: {len(protein_coding)}")

print(f"\n[6/7] Selecting top {NUM_NODES} variable genes (log scale, train only)...")
# Use TCGA TRAIN samples for variance ranking. Cohort and class 2 are
# perfectly confounded, so any train-only ranking that includes nevi would
# pick up genes that just separate the two cohorts (a platform signature),
# not biology.
tcga_train_samples = [s for s in train_samples if s in toil.columns]
top_genes = (
    expr_log_pc[tcga_train_samples].var(axis=1).nlargest(NUM_NODES).index
)
toil_top = toil.loc[top_genes]
gse_top = gse_log.loc[top_genes]

# Cross-cohort harmonization on the chosen panel.
#
# Problem: TCGA is log2(TPM+0.001), GSE is log2(DESeq2 count+1). The two scales
# differ by a global library-size/normalization offset. Per-gene per-cohort
# z-scoring would over-correct: it would force every gene's nevus mean to
# match its TCGA mean, erasing the very signal that lets the model tell nevi
# apart from tumors.
#
# Instead: remove a SINGLE GLOBAL offset per cohort (the median across all
# genes-and-samples in the panel) so the two scales overlap on average, then
# standardize jointly by TCGA-train per-gene mean & std (the reference frame).
# This preserves per-gene biology between cohorts.
toil_offset = float(toil_top[tcga_train_samples].stack().median())
gse_offset = float(gse_top.stack().median())
print(f"  global offsets — TCGA-train: {toil_offset:.3f}  GSE: {gse_offset:.3f}")
gse_top = gse_top - (gse_offset - toil_offset)  # shift GSE onto TCGA scale

ref_mu = toil_top[tcga_train_samples].mean(axis=1)
ref_sd = toil_top[tcga_train_samples].std(axis=1)
toil_top = standardize_with(toil_top, ref_mu, ref_sd)
gse_top = standardize_with(gse_top, ref_mu, ref_sd)

expr_top = pd.concat([toil_top, gse_top], axis=1)
genes_list = expr_top.index.tolist()

print(f"  Querying STRING for PPI (confidence >= {CONFIDENCE_THRESHOLD})...")
r = requests.post(
    "https://string-db.org/api/tsv/network",
    data={
        "identifiers": "\r".join(genes_list),
        "species": 9606,
        "required_score": CONFIDENCE_THRESHOLD,
        "network_type": "physical",
    },
    verify=VERIFY,
)
r.raise_for_status()
df_ppi = pd.read_csv(io.StringIO(r.text), sep="\t")
print(f"  found {len(df_ppi)} PPI interactions")

gene_to_idx = {g: i for i, g in enumerate(genes_list)}
edges = [
    [gene_to_idx[a], gene_to_idx[b]]
    for a, b in zip(df_ppi["preferredName_A"], df_ppi["preferredName_B"])
    if a in gene_to_idx and b in gene_to_idx
]
if edges:
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
else:
    edge_index = torch.empty((2, 0), dtype=torch.long)
print(f"  edge_index shape: {tuple(edge_index.shape)}")

# ----------------------------------------------------------------------------
# 7. Build per-sample graphs
# ----------------------------------------------------------------------------
print("\n[7/7] Building patient graphs...")
dataset = []
class_counts = {0: 0, 1: 0, 2: 0}
for sample_id in sample_ids:
    y = int(y_lookup[sample_id])
    x = torch.tensor(expr_top[sample_id].values, dtype=torch.float).unsqueeze(1)
    if torch.isnan(x).any():
        x = torch.nan_to_num(x, nan=0.0)
    data = Data(x=x, edge_index=edge_index, y=torch.tensor([y], dtype=torch.long))
    data.paciente_id = sample_id
    data.gene_names = genes_list
    dataset.append(data)
    class_counts[y] += 1

print(f"  built {len(dataset)} graphs")
print(f"    Primary:    {class_counts[0]}")
print(f"    Metastasis: {class_counts[1]}")
print(f"    Nevus:      {class_counts[2]}")

torch.save(dataset, OUTPUT_FILE)
SPLITS_FILE = OUTPUT_FILE.with_suffix(".splits.npz")
np.savez(SPLITS_FILE, train=train_idx, val=val_idx, test=test_idx, seed=SPLIT_SEED)
print(f"\nSaved: {OUTPUT_FILE}")
print(f"Saved: {SPLITS_FILE}")
