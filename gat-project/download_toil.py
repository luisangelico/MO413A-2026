#!/usr/bin/env python3
"""
Download TCGA-SKCM (melanoma) + GTEx skin samples from the UCSC Xena TOIL
recompute, where both are processed through the same pipeline (avoiding
batch effects between TCGA and GTEx).

Builds patient-level graphs with PPI edges from STRING.

Labels:
    0 = Primary Tumor       (TCGA-SKCM, sample_type 'Primary Tumor')
    1 = Metastasis          (TCGA-SKCM, sample_type 'Metastatic')
    2 = Normal skin         (GTEx, primary_site 'Skin')

Notes on TOIL:
    - Expression values are already log2(TPM + 0.001). We do NOT apply log1p again.
    - The expression matrix is large (~3 GB compressed, ~60k genes × ~20k samples).
      We download it once, then filter to the columns we need.
    - Gene IDs are versioned Ensembl IDs (ENSG...). The probemap maps them to symbols.
"""

import io
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
from torch_geometric.data import Data

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
NUM_NODES = 500
CONFIDENCE_THRESHOLD = 500
PROCESSED_DATASET_PATH = Path("./data/processed/")
RAW_DATA_PATH = Path("./data/raw_toil/")
PROCESSED_DATASET_PATH.mkdir(parents=True, exist_ok=True)
RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)

# TOIL recompute URLs on Xena
TOIL_BASE = "https://toil-xena-hub.s3.us-east-1.amazonaws.com/download/"
URL_EXPR = TOIL_BASE + "TcgaTargetGtex_rsem_gene_tpm.gz"
URL_PHENO = TOIL_BASE + "TcgaTargetGTEX_phenotype.txt.gz"
URL_PROBEMAP = TOIL_BASE + "probeMap%2Fgencode.v23.annotation.gene.probemap"

EXPR_LOCAL = RAW_DATA_PATH / "TcgaTargetGtex_rsem_gene_tpm.gz"
PHENO_LOCAL = RAW_DATA_PATH / "TcgaTargetGTEX_phenotype.txt.gz"
PROBEMAP_LOCAL = RAW_DATA_PATH / "gencode.v23.annotation.gene.probemap"

OUTPUT_FILE = PROCESSED_DATASET_PATH / f"toil_skin_{NUM_NODES}_{CONFIDENCE_THRESHOLD}.pt"

# SSL handling — same pattern as download_data.py
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


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# 1. Download files
# ----------------------------------------------------------------------------
print("=" * 70)
print("TOIL recompute: TCGA-SKCM + GTEx skin")
print("=" * 70)

if OUTPUT_FILE.exists():
    print(f"\nDataset already exists: {OUTPUT_FILE}")
    ds = torch.load(OUTPUT_FILE, weights_only=False)
    print(f"  {len(ds)} samples")
    raise SystemExit(0)

print("\n[1/6] Downloading metadata files...")
download_to(URL_PHENO, PHENO_LOCAL)
download_to(URL_PROBEMAP, PROBEMAP_LOCAL)

# ----------------------------------------------------------------------------
# 2. Filter phenotype to skin samples we want
# ----------------------------------------------------------------------------
print("\n[2/6] Selecting skin samples from phenotype...")
pheno = pd.read_csv(PHENO_LOCAL, sep="\t", compression="gzip", low_memory=False)

# Phenotype columns of interest:
#   sample, _study, _primary_site, _sample_type, primary disease or tissue
# Exact column names vary by dump; normalize.
pheno.columns = [c.strip() for c in pheno.columns]
print(f"  phenotype rows: {len(pheno)}")
print(f"  columns: {list(pheno.columns)[:12]}...")

study_col = "_study" if "_study" in pheno.columns else "study"
site_col = "_primary_site" if "_primary_site" in pheno.columns else "primary_site"
sample_type_col = (
    "_sample_type" if "_sample_type" in pheno.columns else "sample_type"
)

# TCGA-SKCM tumor samples — disease is the canonical filter; SKCM = skin cutaneous melanoma
disease_col = None
for c in ("primary disease or tissue", "_primary_disease", "primary_disease"):
    if c in pheno.columns:
        disease_col = c
        break

is_tcga = pheno[study_col].astype(str).str.upper() == "TCGA"
is_gtex = pheno[study_col].astype(str).str.upper() == "GTEX"

tcga_skcm_mask = is_tcga & pheno[disease_col].astype(str).str.contains(
    "skin cutaneous melanoma", case=False, na=False
)
gtex_skin_mask = is_gtex & pheno[site_col].astype(str).str.contains(
    "skin", case=False, na=False
)

skcm_pheno = pheno[tcga_skcm_mask].copy()
gtex_pheno = pheno[gtex_skin_mask].copy()


def label_skcm(row):
    st = str(row[sample_type_col]).lower()
    if "primary" in st:
        return 0
    if "metastatic" in st or "metastasis" in st:
        return 1
    return None


skcm_pheno["y"] = skcm_pheno.apply(label_skcm, axis=1)
skcm_pheno = skcm_pheno.dropna(subset=["y"])
gtex_pheno["y"] = 2

selected = pd.concat(
    [skcm_pheno[["sample", "y"]], gtex_pheno[["sample", "y"]]], ignore_index=True
)
selected["y"] = selected["y"].astype(int)
print(f"  TCGA-SKCM primary:    {(skcm_pheno['y'] == 0).sum()}")
print(f"  TCGA-SKCM metastasis: {(skcm_pheno['y'] == 1).sum()}")
print(f"  GTEx skin (normal):   {len(gtex_pheno)}")
print(f"  total selected:       {len(selected)}")

selected_ids = set(selected["sample"].tolist())

# ----------------------------------------------------------------------------
# 3. Download expression matrix and read only the columns we need
# ----------------------------------------------------------------------------
print("\n[3/6] Downloading expression matrix (large, ~3 GB)...")
download_to(URL_EXPR, EXPR_LOCAL)

print("\n[4/6] Reading expression — peeking header to get available columns...")
header = pd.read_csv(EXPR_LOCAL, sep="\t", compression="gzip", nrows=0)
all_cols = list(header.columns)
gene_col = all_cols[0]  # usually 'sample' (genes-as-rows in TOIL)
keep_cols = [gene_col] + [c for c in all_cols[1:] if c in selected_ids]
print(f"  matching sample columns found: {len(keep_cols) - 1} / {len(selected)}")

print("  loading expression for those columns (this can take a few minutes)...")
expr = pd.read_csv(
    EXPR_LOCAL,
    sep="\t",
    compression="gzip",
    usecols=keep_cols,
    index_col=0,
)
print(f"  expression shape: {expr.shape}  (genes × samples)")

# Map Ensembl IDs → gene symbols
probemap = pd.read_csv(PROBEMAP_LOCAL, sep="\t", usecols=["id", "gene"]).set_index("id")
expr = expr.join(probemap, how="inner").dropna(subset=["gene"])
expr = expr.set_index("gene")
expr = expr[~expr.index.duplicated(keep="first")]
print(f"  after gene-symbol mapping: {expr.shape}")

# ----------------------------------------------------------------------------
# 5. Top-variable gene selection + STRING PPI
# ----------------------------------------------------------------------------
print(f"\n[5/6] Selecting top {NUM_NODES} variable genes...")
top_genes = expr.var(axis=1).nlargest(NUM_NODES).index
expr_top = expr.loc[top_genes]
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
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
print(f"  edge_index shape: {tuple(edge_index.shape)}")

# ----------------------------------------------------------------------------
# 6. Build per-sample graphs
# ----------------------------------------------------------------------------
print("\n[6/6] Building patient graphs...")
y_lookup = dict(zip(selected["sample"], selected["y"]))

dataset = []
class_counts = {0: 0, 1: 0, 2: 0}
for sample_id in expr_top.columns:
    if sample_id not in y_lookup:
        continue
    y = int(y_lookup[sample_id])
    # TOIL values are already log2(TPM+0.001) — do NOT log-transform again.
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
print(f"    Normal:     {class_counts[2]}")

torch.save(dataset, OUTPUT_FILE)
print(f"\nSaved: {OUTPUT_FILE}")
print(
    "\nNext: update train_model.py to point at this file:\n"
    f"  dataset_path = Path('{OUTPUT_FILE}')"
)
