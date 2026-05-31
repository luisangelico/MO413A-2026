"""Attention-based interpretability for the trained GATv2 classifier.

Pipeline:
    per-sample edge attention (all 3 layers, mean over heads)
        -> per-sample node importance (sum of incoming attention)
        -> per-class mean node importance (correctly classified samples only)
        -> per-class rankings + class contrasts (e.g. Metastasis - Primary)
        -> driver-gene recovery metrics (top-K hit rate, hypergeometric p)

Run as a module:
    python -m src.interpret              # uses latest run_* dir
    python -m src.interpret --run RUNDIR
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import hypergeom

from src.config import CLASS_NAMES, DATASET_FILE, PROCESSED_DATASET_PATH, get_device
from src.model import load_model


# Reference gene sets for attention validation.
# Note: the top-variable-expression panel does NOT contain canonical mutation drivers
# (BRAF/NRAS/CDKN2A...) — those are not highly variable in expression. Instead we
# validate against expression-level signatures the model could plausibly learn.
GENE_SETS = {
    # Melanocyte lineage TFs / pigmentation — should be HIGH in melanoma vs normal skin
    "melanocyte_lineage": [
        "TYR", "MLANA", "PMEL", "DCT", "TYRP1", "SOX10", "MITF",
        "MC1R", "OCA2", "GPNMB", "PAX3", "EDNRB",
    ],
    # Invasion / metastasis / EMT — should separate Metastasis from Primary
    "melanoma_progression": [
        "S100B", "S100A1", "S100A4", "AXL", "MET", "EGFR", "VIM",
        "CDH2", "MMP2", "MMP9", "SPP1", "VEGFA", "BIRC7", "POSTN",
    ],
    # Immune infiltrate — varies across TCGA melanoma samples
    "immune_infiltrate": [
        "CD8A", "CD8B", "CD3D", "CD3E", "GZMB", "PRF1", "IFNG",
        "PDCD1", "CD274", "CTLA4", "LAG3", "FOXP3", "CXCL9", "CXCL10",
    ],
    # Keratinocyte / cornified envelope — should be HIGH in Normal skin (GTEx)
    "keratin_skin_normal": [
        "KRT1", "KRT5", "KRT10", "KRT14", "LOR", "FLG", "IVL",
        "LCE3D", "DSG1", "DSG3",
    ],
}

# Back-compat alias
MELANOMA_DRIVERS = GENE_SETS["melanocyte_lineage"] + GENE_SETS["melanoma_progression"]


@dataclass
class SampleAttention:
    sample_idx: int
    sample_id: str
    y_true: int
    y_pred: int
    confidence: float
    node_importance: np.ndarray  # [num_nodes]


def _per_sample_node_importance(model, data, device) -> tuple[np.ndarray, int, float]:
    """Run model with attention; return (node_importance[num_nodes], y_pred, confidence).

    Node importance = sum over layers of (sum of incoming edge attention to that node),
    averaged over heads. Self-loops are kept (PyG adds them by default for GATv2Conv);
    they amplify hub signal and are part of the model's actual computation.
    """
    model.eval()
    num_nodes = data.x.size(0)
    batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
    with torch.no_grad():
        logits, attn_layers = model(
            data.x.to(device), data.edge_index.to(device), batch, return_all_attn=True
        )
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    y_pred = int(probs.argmax())
    conf = float(probs[y_pred])

    importance = np.zeros(num_nodes, dtype=np.float64)
    for ei, alpha in attn_layers:
        # alpha: [E, heads] -> [E]
        a = alpha.mean(dim=-1).detach().cpu().numpy()
        dst = ei[1].detach().cpu().numpy()
        # accumulate incoming attention onto destination nodes
        np.add.at(importance, dst, a)
    return importance, y_pred, conf


def collect_sample_attention(model, dataset, indices, device) -> list[SampleAttention]:
    out = []
    for i, idx in enumerate(indices):
        data = dataset[int(idx)]
        importance, y_pred, conf = _per_sample_node_importance(model, data, device)
        out.append(SampleAttention(
            sample_idx=int(idx),
            sample_id=getattr(data, "paciente_id", f"sample_{idx}"),
            y_true=int(data.y.item()),
            y_pred=y_pred,
            confidence=conf,
            node_importance=importance,
        ))
        if (i + 1) % 50 == 0:
            print(f"  attention extracted: {i + 1}/{len(indices)}")
    return out


def aggregate_per_class(samples: list[SampleAttention], num_classes: int,
                       correct_only: bool = True) -> np.ndarray:
    """Return [num_classes, num_nodes] mean node importance per class."""
    num_nodes = samples[0].node_importance.shape[0]
    sums = np.zeros((num_classes, num_nodes), dtype=np.float64)
    counts = np.zeros(num_classes, dtype=np.int64)
    for s in samples:
        if correct_only and s.y_pred != s.y_true:
            continue
        sums[s.y_true] += s.node_importance
        counts[s.y_true] += 1
    counts_safe = np.where(counts == 0, 1, counts)
    return sums / counts_safe[:, None], counts


def driver_recovery(rankings: np.ndarray, gene_names: list[str],
                    drivers: list[str], top_ks=(10, 25, 50, 100)) -> dict:
    """Given a 1-D importance ranking, report top-K hit rates and hypergeometric p-values.

    rankings: [num_nodes] importance scores (higher = more attended)
    """
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    drivers_in_panel = [g for g in drivers if g in gene_to_idx]
    driver_idx = np.array([gene_to_idx[g] for g in drivers_in_panel], dtype=int)

    order = np.argsort(-rankings)  # descending
    rank_of = np.empty_like(order)
    rank_of[order] = np.arange(len(order))  # rank 0 = top
    driver_ranks = {g: int(rank_of[gene_to_idx[g]]) for g in drivers_in_panel}

    N = len(gene_names)
    K = len(driver_idx)
    out = {
        "drivers_in_panel": drivers_in_panel,
        "driver_ranks": driver_ranks,
        "panel_size": N,
        "topk": {},
    }
    for k in top_ks:
        topk_set = set(order[:k].tolist())
        hits = int(sum(1 for d in driver_idx if int(d) in topk_set))
        # hypergeometric: P(X >= hits) when drawing k from N with K successes
        if K > 0 and k <= N:
            pval = float(hypergeom.sf(hits - 1, N, K, k))
        else:
            pval = float("nan")
        out["topk"][k] = {"hits": hits, "of_drivers": K, "pvalue": pval}
    return out


def degree_baseline(edge_index: torch.Tensor, num_nodes: int) -> np.ndarray:
    """Trivial baseline: node in-degree from the static PPI graph."""
    dst = edge_index[1].cpu().numpy()
    deg = np.bincount(dst, minlength=num_nodes).astype(np.float64)
    return deg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default=None,
                    help="run_* directory; default = latest")
    ap.add_argument("--split", choices=["test", "all"], default="all",
                    help="which split to interpret (default: all)")
    ap.add_argument("--correct-only", action="store_true", default=True)
    args = ap.parse_args()

    if args.run:
        run_dir = Path(args.run)
    else:
        runs = sorted(PROCESSED_DATASET_PATH.glob("run_*"), key=lambda p: p.stat().st_mtime)
        if not runs:
            raise SystemExit("No run_* directories found.")
        run_dir = runs[-1]
    print(f"Run: {run_dir}")

    model_path = run_dir / "best_model.pt"
    if not model_path.exists():
        model_path = run_dir / "modelo_final.pt"
    device = get_device()
    model, config = load_model(model_path, device=device)
    num_classes = config["num_classes"]

    dataset = torch.load(DATASET_FILE, weights_only=False)
    if num_classes == 2:
        # mirror evaluate.py
        keep = [i for i, d in enumerate(dataset) if int(d.y.item()) != 2]
        dataset = [dataset[i] for i in keep]

    splits_path = run_dir / "splits.npz"
    if args.split == "test" and splits_path.exists():
        idx = np.load(splits_path)["test"]
        print(f"Interpreting held-out test split: {len(idx)} samples")
    else:
        idx = np.arange(len(dataset))
        print(f"WARNING: using all {len(idx)} samples (no split filtering)")

    gene_names = list(dataset[0].gene_names)
    num_nodes = len(gene_names)

    print("Extracting per-sample attention...")
    samples = collect_sample_attention(model, dataset, idx, device)

    per_class_mean, counts = aggregate_per_class(samples, num_classes,
                                                 correct_only=args.correct_only)
    print("Samples aggregated per class (correct-only=%s): %s"
          % (args.correct_only, dict(zip(range(num_classes), counts.tolist()))))

    # Save per-class importance table
    df_imp = pd.DataFrame(per_class_mean.T, index=gene_names,
                          columns=[CLASS_NAMES[i] for i in range(num_classes)])
    # Class contrasts
    if num_classes >= 2:
        df_imp["Metastasis - Primary"] = df_imp[CLASS_NAMES[1]] - df_imp[CLASS_NAMES[0]]
    if num_classes == 3:
        tumor_mean = (df_imp[CLASS_NAMES[0]] + df_imp[CLASS_NAMES[1]]) / 2
        df_imp["Tumor - Normal"] = tumor_mean - df_imp[CLASS_NAMES[2]]

    out_csv = run_dir / "attention_per_class.csv"
    df_imp.sort_values(CLASS_NAMES[0], ascending=False).to_csv(out_csv)
    print(f"Saved: {out_csv}")

    # Driver recovery per ranking
    rankings_to_test = {CLASS_NAMES[i]: per_class_mean[i] for i in range(num_classes)}
    if "Metastasis - Primary" in df_imp.columns:
        rankings_to_test["Metastasis - Primary"] = df_imp["Metastasis - Primary"].values
    if "Tumor - Normal" in df_imp.columns:
        rankings_to_test["Tumor - Normal"] = df_imp["Tumor - Normal"].values

    # Degree baseline (same graph for all samples — use first sample's edge_index)
    deg = degree_baseline(dataset[int(idx[0])].edge_index, num_nodes)
    rankings_to_test["[baseline] node degree"] = deg

    print("\nGene-set recovery (top-K hits | hypergeometric p):")
    rows = []
    for set_name, gene_list in GENE_SETS.items():
        in_panel = [g for g in gene_list if g in gene_names]
        print(f"\n  [{set_name}] {len(in_panel)}/{len(gene_list)} genes in panel: {in_panel}")
        if not in_panel:
            continue
        for name, rk in rankings_to_test.items():
            rep = driver_recovery(rk, gene_names, in_panel)
            line = f"    {name:30s}"
            for k, info in rep["topk"].items():
                line += f"  K={k}:{info['hits']}/{info['of_drivers']}(p={info['pvalue']:.2g})"
            print(line)
            for k, info in rep["topk"].items():
                rows.append({"gene_set": set_name, "ranking": name, "k": k,
                             "hits": info["hits"],
                             "set_size_in_panel": info["of_drivers"],
                             "pvalue": info["pvalue"]})

    df_rec = pd.DataFrame(rows)
    out_rec = run_dir / "attention_gene_set_recovery.csv"
    df_rec.to_csv(out_rec, index=False)
    print(f"\nSaved: {out_rec}")

    # Top-20 genes per ranking (the actual story)
    print("\nTop-20 attended genes per ranking:")
    top_rows = []
    for name, rk in rankings_to_test.items():
        order = np.argsort(-rk)[:20]
        top_genes = [gene_names[i] for i in order]
        print(f"  {name:30s} {top_genes}")
        for rank, i in enumerate(order):
            top_rows.append({"ranking": name, "rank": rank,
                             "gene": gene_names[i], "score": float(rk[i])})
    out_top = run_dir / "attention_top20.csv"
    pd.DataFrame(top_rows).to_csv(out_top, index=False)
    print(f"Saved: {out_top}")


if __name__ == "__main__":
    main()
