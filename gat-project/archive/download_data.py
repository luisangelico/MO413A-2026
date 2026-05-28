#!/usr/bin/env python3
"""
Script to download and process TCGA melanoma data for GAT analysis.
Run this script first to create the processed dataset.
"""

import os
import pandas as pd
import numpy as np
import torch
import requests
import io
import warnings
from pathlib import Path
from torch_geometric.data import Data

# SSL: prefer CA bundle from proxy-fix-bundle/ if present; allow opt-in to insecure mode
CA_BUNDLE = Path('./proxy-fix-bundle/allCAbundle.pem')
INSECURE_SSL = os.environ.get('GAT_INSECURE_SSL') == '1'

if INSECURE_SSL:
    warnings.filterwarnings('ignore', message='Unverified HTTPS request')
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    VERIFY = False
    print("WARNING: GAT_INSECURE_SSL=1 — disabling certificate verification.")
elif CA_BUNDLE.exists():
    VERIFY = str(CA_BUNDLE)
    print(f"Using CA bundle: {VERIFY}")
else:
    VERIFY = True

# Configuration
NUM_NODES = 500
CONFIDENCE_THRESHOLD = 500
PROCESSED_DATASET_PATH = Path('./data/processed/')

print("=" * 60)
print("TCGA Melanoma Data Download and Processing")
print("=" * 60)

output_file = PROCESSED_DATASET_PATH / f'tcga_pacientes_{NUM_NODES}_{CONFIDENCE_THRESHOLD}.pt'

if output_file.exists():
    print(f"\nDataset already exists: {output_file}")
    print("Loading to verify...")
    dataset_pacientes = torch.load(output_file, weights_only=False)
    print(f"✓ Dataset loaded: {len(dataset_pacientes)} patients")
else:
    print("\nStarting download and processing...")

    # URLs
    url_exp = "https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-SKCM.star_tpm.tsv.gz"
    url_map = "https://gdc-hub.s3.us-east-1.amazonaws.com/download/gencode.v36.annotation.gtf.gene.probemap"

    # Download gene expression data
    print("\n[1/5] Downloading TCGA gene expression data (~88MB)...")
    response_exp = requests.get(url_exp, verify=VERIFY, stream=True)
    total_size = int(response_exp.headers.get('content-length', 0))
    print(f"      File size: {total_size / 1024 / 1024:.1f} MB")

    # Download gene mapping
    print("\n[2/5] Downloading gene mapping data...")
    response_map = requests.get(url_map, verify=VERIFY)

    # Process data
    print("\n[3/5] Processing gene expression matrix...")
    df_exp = pd.read_csv(io.BytesIO(response_exp.content), sep='\t', index_col=0, compression='gzip')
    df_map = pd.read_csv(io.StringIO(response_map.text), sep='\t', usecols=['id', 'gene']).set_index('id')

    df_exp = df_exp.join(df_map).dropna(subset=['gene']).set_index('gene')
    df_exp = df_exp[~df_exp.index.duplicated(keep='first')]
    print(f"      Total genes: {len(df_exp)}")
    print(f"      Total samples: {len(df_exp.columns)}")

    # Select top variable genes
    print(f"\n[4/5] Selecting top {NUM_NODES} most variable genes...")
    top_genes = df_exp.var(axis=1).nlargest(NUM_NODES).index
    df_filtered = df_exp.loc[top_genes]
    genes_list = df_filtered.index.tolist()

    # Query STRING PPI
    print(f"\n[5/5] Querying STRING database for protein interactions...")
    print(f"      Confidence threshold: {CONFIDENCE_THRESHOLD}")
    r = requests.post("https://string-db.org/api/tsv/network",
                      data={"identifiers": "\r".join(genes_list), "species": 9606,
                            "required_score": CONFIDENCE_THRESHOLD, "network_type": "physical"},
                      verify=VERIFY)
    df_ppi = pd.read_csv(io.StringIO(r.text), sep='\t')
    print(f"      Found {len(df_ppi)} interactions")

    # Build graph structure
    gene_to_idx = {g: i for i, g in enumerate(genes_list)}
    edges = [[gene_to_idx[a], gene_to_idx[b]] for a, b in zip(df_ppi['preferredName_A'], df_ppi['preferredName_B'])
             if a in gene_to_idx and b in gene_to_idx]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # Create patient graphs
    print("\nBuilding individual patient graphs...")
    dataset_pacientes = []
    sample_counts = {'primary': 0, 'metastasis': 0, 'normal': 0}

    for paciente_id in df_filtered.columns:
        barcode_parts = paciente_id.split('-')
        if len(barcode_parts) >= 4:
            sample_type = barcode_parts[3][:2]
            if sample_type == '01':
                y = 0  # Primary
                sample_counts['primary'] += 1
            elif sample_type == '06':
                y = 1  # Metastasis
                sample_counts['metastasis'] += 1
            elif sample_type == '11':
                y = 2  # Normal
                sample_counts['normal'] += 1
            else:
                continue

            x = torch.tensor(df_filtered[paciente_id].values, dtype=torch.float)
            x = torch.log1p(x).unsqueeze(1)
            data = Data(x=x, edge_index=edge_index, y=torch.tensor([y], dtype=torch.long))
            data.paciente_id = paciente_id
            data.gene_names = genes_list
            dataset_pacientes.append(data)

    # Save dataset
    print(f"\nSaving processed dataset...")
    PROCESSED_DATASET_PATH.mkdir(parents=True, exist_ok=True)
    torch.save(dataset_pacientes, output_file)

    print("\n" + "=" * 60)
    print("✓ Dataset created successfully!")
    print("=" * 60)
    print(f"Total patients: {len(dataset_pacientes)}")
    print(f"  - Primary tumors: {sample_counts['primary']}")
    print(f"  - Metastatic tumors: {sample_counts['metastasis']}")
    print(f"  - Normal tissue: {sample_counts['normal']}")
    print(f"\nSaved to: {output_file}")
    print("\nYou can now run the Jupyter notebook cells.")
