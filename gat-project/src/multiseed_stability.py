"""Multi-seed stability analysis for the GAT.

For a `multiseed_*` directory containing `seed_{N}/best_model.pt` checkpoints,
this script reports:

  - Per-seed test accuracy + 95% CI (normal approx)
  - Per-seed per-class F1 means and stds
  - Spearman rank correlation of per-class node-importance vectors across seeds
    (the "are the top-attended genes the same across seeds?" question)
  - Top-K Jaccard overlap of attended genes per class across seeds
  - Mean attention rank table across seeds (consensus rankings)

Outputs (under <multiseed_dir>/):
  - stability.json                   summary metrics
  - stability_spearman.csv           NxN Spearman matrix per ranking
  - stability_consensus_top20.csv    consensus top-20 across seeds
  - figures/stability_spearman.png   heatmap per ranking
  - figures/stability_accuracy.png   boxplot of per-seed accuracies / F1s

Run:
    python -m src.multiseed_stability                       # latest multiseed_*
    python -m src.multiseed_stability --dir <multiseed_dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import f1_score

from sklearn.metrics.pairwise import cosine_similarity

from src.config import CLASS_NAMES, DATASET_FILE, PROCESSED_DATASET_PATH, get_device
from src.interpret import _per_sample_node_importance, aggregate_per_class
from src.interpret import SampleAttention
from src.model import load_model


def _resolve_dir(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    runs = sorted(PROCESSED_DATASET_PATH.glob("multiseed_*"),
                  key=lambda p: p.stat().st_mtime)
    if not runs:
        raise SystemExit("No multiseed_* directories found.")
    return runs[-1]


def collect_seed_results(seed_dir: Path, dataset, num_classes: int, device):
    """Load one seed's model; run held-out test; extract per-class attention,
    embeddings (correctly classified only), pseudotime per sample.
    """
    model_path = seed_dir / "best_model.pt"
    splits_path = seed_dir / "splits.npz"
    model, _ = load_model(model_path, device=device)
    test_idx = np.load(splits_path)["test"]

    samples = []
    y_true, y_pred = [], []
    embeddings = []
    sample_ids = []
    for idx in test_idx:
        data = dataset[int(idx)]
        # Attention per sample
        importance, pred, conf = _per_sample_node_importance(model, data, device)
        # Embedding (separate forward — keeps interpret API untouched)
        with torch.no_grad():
            batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)
            _, emb = model(
                data.x.to(device), data.edge_index.to(device), batch,
                return_embedding=True,
            )
        embeddings.append(emb.cpu().numpy()[0])
        samples.append(SampleAttention(
            sample_idx=int(idx),
            sample_id=getattr(data, "paciente_id", f"sample_{idx}"),
            y_true=int(data.y.item()),
            y_pred=pred,
            confidence=conf,
            node_importance=importance,
        ))
        y_true.append(int(data.y.item()))
        y_pred.append(pred)
        sample_ids.append(getattr(data, "paciente_id", f"sample_{int(idx)}"))

    y_true = np.array(y_true); y_pred = np.array(y_pred)
    Z = np.stack(embeddings)
    sample_ids = np.array(sample_ids)

    per_class_attn, _ = aggregate_per_class(samples, num_classes, correct_only=True)
    test_acc = float((y_true == y_pred).mean())
    f1s = f1_score(y_true, y_pred, average=None,
                   labels=list(range(num_classes)), zero_division=0)

    # Pseudotime on this seed's embeddings (correctly classified only)
    pseudotime = np.full(len(Z), np.nan, dtype=np.float64)
    correct = y_true == y_pred
    if num_classes >= 3 and correct.sum() > 4:
        is_normal_c = correct & (y_true == 2)
        is_meta_c = correct & (y_true == 1)
        if is_normal_c.sum() >= 2 and is_meta_c.sum() >= 2:
            c_normal = Z[is_normal_c].mean(axis=0)
            c_meta = Z[is_meta_c].mean(axis=0)
            axis = c_meta - c_normal
            axis_norm = axis / (np.linalg.norm(axis) + 1e-9)
            proj = (Z - c_normal) @ axis_norm
            # Normalize using only correct samples' span (consistent with src.pseudotime)
            pmin, pmax = proj[correct].min(), proj[correct].max()
            pseudotime = (proj - pmin) / max(pmax - pmin, 1e-9)

    return {
        "test_acc": test_acc,
        "per_class_f1": f1s,
        "per_class_attn": per_class_attn,
        "embeddings": Z,
        "sample_ids": sample_ids,
        "y_true": y_true,
        "y_pred": y_pred,
        "pseudotime": pseudotime,
        "test_idx": np.array(test_idx),
    }


# ---------------------------------------------------------------------------
# Embedding-geometry stability
# ---------------------------------------------------------------------------

def common_sample_alignment(per_seed: list[dict]) -> tuple[list[str], list[np.ndarray]]:
    """Across seeds with potentially-different test splits, return the set of
    sample IDs present in every seed's test split, plus per-seed row indices
    aligned to that common set."""
    id_sets = [set(r["sample_ids"].tolist()) for r in per_seed]
    common = sorted(set.intersection(*id_sets))
    aligned_indices = []
    for r in per_seed:
        id_to_row = {sid: i for i, sid in enumerate(r["sample_ids"])}
        aligned_indices.append(np.array([id_to_row[s] for s in common]))
    return common, aligned_indices


def knn_jaccard_stability(per_seed: list[dict], aligned_indices: list[np.ndarray],
                          k: int = 10) -> tuple[float, float, np.ndarray]:
    """Per-sample Jaccard of top-k cosine neighbors across seeds.

    Returns (mean over samples and seed-pairs, std, per-sample-mean array)."""
    n_seeds = len(per_seed)
    n_common = len(aligned_indices[0])
    knn_sets = []
    for s, idx in enumerate(aligned_indices):
        Z = per_seed[s]["embeddings"][idx]
        sim = cosine_similarity(Z)
        np.fill_diagonal(sim, -np.inf)
        knn = np.argsort(-sim, axis=1)[:, :k]
        knn_sets.append([set(row.tolist()) for row in knn])

    per_sample = np.zeros(n_common)
    pair_count = 0
    for i in range(n_seeds):
        for j in range(i + 1, n_seeds):
            pair_count += 1
            for r in range(n_common):
                inter = len(knn_sets[i][r] & knn_sets[j][r])
                union = len(knn_sets[i][r] | knn_sets[j][r])
                per_sample[r] += inter / union if union else 0.0
    per_sample /= pair_count
    return float(per_sample.mean()), float(per_sample.std()), per_sample


def procrustes_distance(per_seed: list[dict],
                        aligned_indices: list[np.ndarray]) -> float:
    """Mean pairwise procrustes disparity across seeds (0 = identical up to rotation)."""
    from scipy.spatial import procrustes
    n = len(per_seed)
    Zs = [per_seed[s]["embeddings"][aligned_indices[s]] for s in range(n)]
    disps = []
    for i in range(n):
        for j in range(i + 1, n):
            _, _, d = procrustes(Zs[i], Zs[j])
            disps.append(d)
    return float(np.mean(disps))


# ---------------------------------------------------------------------------
# Pseudotime stability
# ---------------------------------------------------------------------------

def pseudotime_stability(per_seed: list[dict], common_ids: list[str],
                         aligned_indices: list[np.ndarray]) -> pd.DataFrame:
    """Per-sample mean ± std pseudotime across seeds.

    Aligns all seeds to a consistent direction (Normal-low, Metastasis-high)
    by Spearman-correlating against the first seed and flipping if needed.
    """
    n_seeds = len(per_seed)
    pt = np.full((n_seeds, len(common_ids)), np.nan)
    for s in range(n_seeds):
        pt[s] = per_seed[s]["pseudotime"][aligned_indices[s]]
    # Reference seed = first; flip others if anti-correlated
    ref = pt[0]
    if np.isfinite(ref).all():
        for s in range(1, n_seeds):
            if np.isfinite(pt[s]).all():
                rho, _ = spearmanr(ref, pt[s])
                if rho < 0:
                    pt[s] = 1.0 - pt[s]

    means = np.nanmean(pt, axis=0)
    stds = np.nanstd(pt, axis=0)

    # Class label per common sample (use seed-0's labels)
    s0 = per_seed[0]
    s0_id_to_row = {sid: i for i, sid in enumerate(s0["sample_ids"])}
    labels = [CLASS_NAMES[s0["y_true"][s0_id_to_row[c]]] for c in common_ids]

    df = pd.DataFrame({
        "sample_id": common_ids,
        "true_label": labels,
        "pseudotime_mean": means,
        "pseudotime_std": stds,
    }).sort_values("pseudotime_mean").reset_index(drop=True)
    return df


def spearman_matrix(rankings: list[np.ndarray]) -> np.ndarray:
    n = len(rankings)
    M = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            r, _ = spearmanr(rankings[i], rankings[j])
            M[i, j] = M[j, i] = r
    return M


def topk_jaccard_matrix(rankings: list[np.ndarray], k: int = 50) -> np.ndarray:
    tops = [set(np.argsort(-r)[:k].tolist()) for r in rankings]
    n = len(tops)
    M = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            inter = len(tops[i] & tops[j])
            union = len(tops[i] | tops[j])
            M[i, j] = M[j, i] = inter / union if union else 0.0
    return M


def consensus_topk(rankings: list[np.ndarray], gene_names: list[str],
                   k: int = 20) -> pd.DataFrame:
    """Average rank across seeds, return top-k by mean rank."""
    n_genes = len(gene_names)
    ranks = np.zeros((len(rankings), n_genes))
    for s, r in enumerate(rankings):
        order = np.argsort(-r)
        for rank_pos, idx in enumerate(order):
            ranks[s, idx] = rank_pos
    mean_rank = ranks.mean(axis=0)
    std_rank = ranks.std(axis=0)
    order = np.argsort(mean_rank)[:k]
    return pd.DataFrame({
        "gene": [gene_names[i] for i in order],
        "mean_rank": mean_rank[order],
        "std_rank": std_rank[order],
    })


def plot_spearman_heatmap(matrices_per_ranking: dict[str, np.ndarray],
                          out_path: Path) -> None:
    n = len(matrices_per_ranking)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.6 * rows))
    axes = np.atleast_2d(axes).flatten()
    for i, (name, M) in enumerate(matrices_per_ranking.items()):
        ax = axes[i]
        im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
        ax.set_title(name, fontsize=10)
        ax.set_xticks(range(M.shape[0]))
        ax.set_yticks(range(M.shape[0]))
        ax.set_xticklabels([f"s{j}" for j in range(M.shape[0])], fontsize=8)
        ax.set_yticklabels([f"s{j}" for j in range(M.shape[0])], fontsize=8)
        for r in range(M.shape[0]):
            for c in range(M.shape[0]):
                ax.text(c, r, f"{M[r, c]:.2f}", ha="center", va="center",
                        fontsize=7,
                        color="white" if M[r, c] < 0.6 else "black")
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.colorbar(im, ax=axes.tolist(), shrink=0.7, label="Spearman ρ")
    fig.suptitle("Cross-seed Spearman correlation of per-class attention rankings",
                 fontsize=12)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pseudotime_stability(pt_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    color_map = {"Primary Tumor": "#d62728", "Metastasis": "#ff7f0e",
                 "Normal Tissue": "#2ca02c"}
    x = np.arange(len(pt_df))
    means = pt_df["pseudotime_mean"].values
    stds = pt_df["pseudotime_std"].values
    colors = [color_map.get(c, "#888") for c in pt_df["true_label"]]
    ax.errorbar(x, means, yerr=stds, fmt="none", ecolor="#bbb",
                elinewidth=0.8, alpha=0.7, zorder=1)
    ax.scatter(x, means, c=colors, s=14, zorder=2, edgecolor="white", linewidth=0.3)
    ax.set_xlabel("samples (sorted by mean pseudotime)")
    ax.set_ylabel("pseudotime  (Normal → Metastasis)")
    ax.set_title("Per-sample pseudotime: mean ± std across seeds\n"
                 f"mean σ = {stds.mean():.3f}  ·  median σ = {np.median(stds):.3f}")
    handles = [plt.Line2D([0], [0], marker="o", linestyle="",
                          markerfacecolor=v, markeredgecolor="white",
                          markersize=7, label=k)
               for k, v in color_map.items()]
    ax.legend(handles=handles, fontsize=9, loc="upper left")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_knn_jaccard(per_sample_jaccard: np.ndarray, labels: list[str],
                     out_path: Path) -> None:
    color_map = {"Primary Tumor": "#d62728", "Metastasis": "#ff7f0e",
                 "Normal Tissue": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    by_cls = {}
    for j, lbl in zip(per_sample_jaccard, labels):
        by_cls.setdefault(lbl, []).append(j)
    classes = list(by_cls.keys())
    data = [by_cls[c] for c in classes]
    parts = ax.boxplot(data, labels=classes, widths=0.5, patch_artist=True)
    for patch, c in zip(parts["boxes"], classes):
        patch.set_facecolor(color_map.get(c, "#bbb"))
        patch.set_alpha(0.6)
    ax.set_ylabel("kNN Jaccard (mean over seed pairs)")
    ax.set_title(f"Per-sample neighborhood stability  "
                 f"(overall mean = {per_sample_jaccard.mean():.3f})")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_box(accs: list[float], f1s_per_class: np.ndarray,
                      class_names: list[str], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.boxplot([accs], labels=["test acc"], widths=0.4)
    ax.scatter(np.ones(len(accs)) + np.random.RandomState(0).uniform(-0.05, 0.05, len(accs)),
               accs, color="#1f77b4", zorder=3)
    ax.set_ylim(min(accs) - 0.02, 1.0)
    ax.set_title(f"Test accuracy across seeds (n={len(accs)})\n"
                 f"mean = {np.mean(accs)*100:.2f}%, std = {np.std(accs)*100:.2f}%")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    data = [f1s_per_class[:, c] for c in range(f1s_per_class.shape[1])]
    ax.boxplot(data, labels=class_names, widths=0.5)
    for c in range(f1s_per_class.shape[1]):
        ax.scatter(np.full(len(accs), c + 1) + np.random.RandomState(c).uniform(-0.05, 0.05, len(accs)),
                   f1s_per_class[:, c], color="#ff7f0e", zorder=3)
    ax.set_ylim(0, 1.0)
    ax.set_title("Per-class F1 across seeds")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default=None)
    ap.add_argument("--top-k", type=int, default=50,
                    help="Top-K for Jaccard overlap and consensus table")
    args = ap.parse_args()

    ms_dir = _resolve_dir(args.dir)
    print(f"Multi-seed dir: {ms_dir}")
    fig_dir = ms_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    config_path = ms_dir / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    num_classes = int(config.get("num_classes", 3))

    seed_dirs = sorted([p for p in ms_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")])
    if len(seed_dirs) < 2:
        raise SystemExit(f"Need >= 2 seeds, found {len(seed_dirs)}")
    print(f"Found {len(seed_dirs)} seeds: {[p.name for p in seed_dirs]}")

    device = get_device()
    dataset = torch.load(DATASET_FILE, weights_only=False)
    if num_classes == 2:
        dataset = [d for d in dataset if int(d.y.item()) != 2]
    gene_names = list(dataset[0].gene_names)

    per_seed = []
    for sd in seed_dirs:
        print(f"\n[{sd.name}]")
        r = collect_seed_results(sd, dataset, num_classes, device)
        print(f"  test_acc={r['test_acc']:.4f}  f1={r['per_class_f1']}")
        per_seed.append(r)

    accs = [r["test_acc"] for r in per_seed]
    f1s = np.stack([r["per_class_f1"] for r in per_seed])  # [seeds, classes]
    attn_stack = np.stack([r["per_class_attn"] for r in per_seed])  # [seeds, classes, genes]

    # Build rankings to compare: per-class + Tumor-Normal contrast
    rankings_by_name: dict[str, list[np.ndarray]] = {}
    for c in range(num_classes):
        rankings_by_name[CLASS_NAMES[c]] = [attn_stack[s, c] for s in range(len(per_seed))]
    if num_classes == 3:
        tumor_minus_normal = [
            (attn_stack[s, 0] + attn_stack[s, 1]) / 2 - attn_stack[s, 2]
            for s in range(len(per_seed))
        ]
        rankings_by_name["Tumor - Normal"] = tumor_minus_normal
        meta_minus_primary = [attn_stack[s, 1] - attn_stack[s, 0]
                              for s in range(len(per_seed))]
        rankings_by_name["Metastasis - Primary"] = meta_minus_primary

    # Spearman matrices and Jaccard
    spearman_mats = {}
    jaccard_mats = {}
    summary = {
        "n_seeds": len(per_seed),
        "test_accuracy": {
            "mean": float(np.mean(accs)),
            "std": float(np.std(accs)),
            "min": float(np.min(accs)),
            "max": float(np.max(accs)),
            "ci95_half_width": float(1.96 * np.std(accs) / np.sqrt(len(accs))),
            "values": [float(a) for a in accs],
        },
        "per_class_f1": {
            CLASS_NAMES[c]: {
                "mean": float(f1s[:, c].mean()),
                "std": float(f1s[:, c].std()),
            } for c in range(num_classes)
        },
        "rankings": {},
    }

    print("\nCross-seed stability:")
    print("-" * 72)
    for name, ranks in rankings_by_name.items():
        sm = spearman_matrix(ranks)
        jm = topk_jaccard_matrix(ranks, k=args.top_k)
        spearman_mats[name] = sm
        jaccard_mats[name] = jm
        # off-diagonal averages
        n = sm.shape[0]
        mask = ~np.eye(n, dtype=bool)
        sp_mean = float(sm[mask].mean()); sp_std = float(sm[mask].std())
        jc_mean = float(jm[mask].mean()); jc_std = float(jm[mask].std())
        summary["rankings"][name] = {
            "spearman_mean": sp_mean, "spearman_std": sp_std,
            f"top{args.top_k}_jaccard_mean": jc_mean,
            f"top{args.top_k}_jaccard_std": jc_std,
        }
        print(f"  {name:30s} Spearman ρ = {sp_mean:.3f} ± {sp_std:.3f}   "
              f"top-{args.top_k} Jaccard = {jc_mean:.3f} ± {jc_std:.3f}")

        # Save per-ranking matrices
        safe = name.replace(" ", "_").replace("-", "minus")
        pd.DataFrame(sm, columns=[f"s{i}" for i in range(n)],
                     index=[f"s{i}" for i in range(n)]).to_csv(
            ms_dir / f"stability_spearman_{safe}.csv")

    # Consensus top-K (uses Tumor-Normal if available, else Primary)
    pick = "Tumor - Normal" if "Tumor - Normal" in rankings_by_name else CLASS_NAMES[0]
    consensus = consensus_topk(rankings_by_name[pick], gene_names, k=20)
    consensus.to_csv(ms_dir / "stability_consensus_top20.csv", index=False)
    print(f"\nConsensus top-20 ({pick}):")
    for _, row in consensus.iterrows():
        print(f"  {row['gene']:12s}  mean rank {row['mean_rank']:6.1f} ± {row['std_rank']:5.1f}")

    # Embedding-geometry stability + pseudotime stability (need common samples)
    common_ids, aligned = common_sample_alignment(per_seed)
    print(f"\nCommon samples across all seeds' test splits: {len(common_ids)}")
    if len(common_ids) >= 5:
        knn_mean, knn_std, per_sample_jc = knn_jaccard_stability(per_seed, aligned, k=10)
        proc_disp = procrustes_distance(per_seed, aligned)
        print(f"  kNN Jaccard (k=10) across seed pairs: {knn_mean:.3f} ± {knn_std:.3f}")
        print(f"  Mean Procrustes disparity: {proc_disp:.4f}  (0 = identical up to rotation)")
        summary["embedding"] = {
            "knn_jaccard_k10_mean": knn_mean,
            "knn_jaccard_k10_std": knn_std,
            "procrustes_mean_disparity": proc_disp,
            "n_common_samples": len(common_ids),
        }

        # Per-class label list aligned to common_ids (use seed 0)
        s0 = per_seed[0]
        s0_id_to_row = {sid: i for i, sid in enumerate(s0["sample_ids"])}
        common_labels = [CLASS_NAMES[s0["y_true"][s0_id_to_row[c]]] for c in common_ids]

        plot_knn_jaccard(per_sample_jc, common_labels,
                         fig_dir / "stability_knn_jaccard.png")
        print(f"Saved: {fig_dir / 'stability_knn_jaccard.png'}")

        # Pseudotime stability
        if num_classes >= 3:
            pt_df = pseudotime_stability(per_seed, common_ids, aligned)
            pt_df.to_csv(ms_dir / "stability_pseudotime.csv", index=False)
            plot_pseudotime_stability(pt_df, fig_dir / "stability_pseudotime.png")
            summary["pseudotime"] = {
                "mean_std": float(pt_df["pseudotime_std"].mean()),
                "median_std": float(pt_df["pseudotime_std"].median()),
                "max_std": float(pt_df["pseudotime_std"].max()),
            }
            print(f"  Pseudotime per-sample σ: mean={pt_df['pseudotime_std'].mean():.3f}, "
                  f"median={pt_df['pseudotime_std'].median():.3f}")
            print(f"Saved: {fig_dir / 'stability_pseudotime.png'}")
            print(f"Saved: {ms_dir / 'stability_pseudotime.csv'}")
    else:
        print("  Skipping embedding/pseudotime stability (need >=5 common samples).")

    # Figures
    plot_spearman_heatmap(spearman_mats, fig_dir / "stability_spearman.png")
    plot_accuracy_box(accs, f1s,
                      [CLASS_NAMES[c] for c in range(num_classes)],
                      fig_dir / "stability_accuracy.png")

    (ms_dir / "stability.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved: {ms_dir / 'stability.json'}")
    print(f"Saved: {fig_dir / 'stability_spearman.png'}")
    print(f"Saved: {fig_dir / 'stability_accuracy.png'}")

    # Site page
    site_dir = Path(__file__).resolve().parent.parent / "site"
    site_dir.mkdir(parents=True, exist_ok=True)
    write_site_page(ms_dir, summary, site_dir / "stability.html")
    print(f"Saved: {site_dir / 'stability.html'}")


def write_site_page(ms_dir: Path, summary: dict, out_path: Path) -> None:
    run_name = ms_dir.name
    acc = summary["test_accuracy"]
    rk = summary.get("rankings", {})
    emb = summary.get("embedding", {})
    pt = summary.get("pseudotime", {})

    rk_rows = "".join(
        f"<tr><td>{name}</td>"
        f"<td>{v.get('spearman_mean',float('nan')):.3f} ± {v.get('spearman_std',float('nan')):.3f}</td>"
        f"<td>{v.get('top50_jaccard_mean',float('nan')):.3f} ± {v.get('top50_jaccard_std',float('nan')):.3f}</td></tr>"
        for name, v in rk.items()
    )

    emb_block = ""
    if emb:
        emb_block = f"""
<h2>2. Embedding geometry across seeds</h2>
<p>
  Even when two seeds learn slightly different 64-dimensional spaces, the
  question is whether the <em>shape</em> of the cloud is the same — does
  each sample land near the same neighbors regardless of seed?
</p>
<ul>
  <li><strong>kNN Jaccard (k=10):</strong>
      {emb['knn_jaccard_k10_mean']:.3f} ± {emb['knn_jaccard_k10_std']:.3f}
      across seed pairs over {emb['n_common_samples']} samples present in
      every seed's test split. <em>1.0 = identical neighborhoods, 0.0 = no overlap.</em></li>
  <li><strong>Procrustes disparity:</strong>
      {emb['procrustes_mean_disparity']:.4f} (0 = identical up to rotation/scale).</li>
</ul>
<img src="../data/processed/{run_name}/figures/stability_knn_jaccard.png"
     alt="Per-sample kNN Jaccard across seeds, by class">
"""

    pt_block = ""
    if pt:
        pt_block = f"""
<h2>3. Pseudotime stability</h2>
<p>
  Each seed produces its own Normal → Metastasis pseudotime axis. We
  align the direction across seeds (flip if anti-correlated), then look
  at how much each sample's pseudotime varies from seed to seed. Tight
  error bars mean the model agrees on where this sample sits along the
  progression axis; large bars mean the sample is ambiguous.
</p>
<ul>
  <li><strong>Per-sample σ:</strong> mean = {pt['mean_std']:.3f},
      median = {pt['median_std']:.3f}, max = {pt['max_std']:.3f}
      (on a 0–1 axis).</li>
</ul>
<img src="../data/processed/{run_name}/figures/stability_pseudotime.png"
     alt="Per-sample pseudotime across seeds">
"""

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Multi-seed stability</title>
<style>
  body {{ font-family:-apple-system,system-ui,sans-serif; max-width:920px;
          margin:1.5em auto; padding:0 1em; color:#222; line-height:1.6; }}
  .back-btn {{ display:inline-block; padding:0.4em 0.9em; margin-bottom:0.6em;
               background:#1f77b4; color:white !important; border-radius:5px;
               text-decoration:none; font-size:0.9em; }}
  .back-btn:hover {{ background:#155a8a; }}
  h1 {{ font-size:1.5em; margin-bottom:0.1em; }}
  h2 {{ font-size:1.15em; margin-top:1.6em; }}
  .meta {{ color:#666; font-size:0.9em; }}
  .plain {{ background:#f6fbf7; border-left:3px solid #2ca02c;
            padding:0.7em 1em; margin:1em 0; }}
  .callout {{ background:#fffbf2; border-left:3px solid #e0a020;
              padding:0.7em 1em; margin:1em 0; }}
  table {{ border-collapse:collapse; font-size:0.92em; margin:0.8em 0; }}
  th, td {{ padding:0.4em 0.7em; border-bottom:1px solid #eee; text-align:left; }}
  th {{ background:#f4f4f4; }}
  img {{ width:100%; max-width:880px; border:1px solid #eee;
         border-radius:6px; margin:0.6em 0; }}
  code {{ background:#f4f4f4; padding:0 0.3em; border-radius:3px; }}
</style></head><body>

<a href="index.html" class="back-btn">&larr; back to predictions site</a>

<h1>Multi-seed stability — are the findings real?</h1>
<p class="meta">Multi-seed run: <code>{run_name}</code> · seeds = {summary['n_seeds']}</p>

<div class="plain">
  <strong>The question.</strong> If we retrain the same architecture from
  scratch with a different random seed, do we recover the same answers?
  If yes, the project's findings are reproducible. If no, any single-seed
  result is anecdotal and should be reported as a consensus.
</div>

<h2>1. Performance variance</h2>
<p>
  Five seeds, identical architecture and data, different initializations:
</p>
<ul>
  <li><strong>Test accuracy:</strong> {acc['mean']*100:.2f}% ± {acc['std']*100:.2f}%
      (range {acc['min']*100:.2f}–{acc['max']*100:.2f}%, 95% CI half-width
      ≈ {acc['ci95_half_width']*100:.2f}%).</li>
</ul>
<img src="../data/processed/{run_name}/figures/stability_accuracy.png"
     alt="Per-seed accuracy and per-class F1">

<h2>2. Attention-ranking stability</h2>
<p>
  We computed each seed's per-class node-importance vector (1000 genes)
  and asked: do the seeds rank genes in the same order?
</p>
<ul>
  <li><strong>Spearman ρ</strong> compares full rankings (1.0 = identical
    order, 0.0 = unrelated).</li>
  <li><strong>Top-50 Jaccard</strong> compares which 50 genes each seed
    flagged as most important (1.0 = same set, 0.0 = no overlap).</li>
</ul>
<table>
<tr><th>ranking</th><th>Spearman ρ</th><th>top-50 Jaccard</th></tr>
{rk_rows}
</table>
<img src="../data/processed/{run_name}/figures/stability_spearman.png"
     alt="Cross-seed Spearman heatmap per ranking">

<div class="callout">
  <strong>Reading guide.</strong>
  ρ ≥ 0.8 means rankings are highly stable — single-seed top-K results
  are defensible. 0.5–0.8 means moderately stable; report
  <em>consensus</em> rankings (mean rank across seeds) rather than any
  one seed's. Below 0.5 means the rankings shuffle from seed to seed and
  conclusions need to be tempered.
  <br><br>
  Difference rankings (e.g. <em>Metastasis − Primary</em>) are
  intrinsically noisier than per-class rankings — noise compounds when you
  subtract two estimates — so expect lower numbers there. The
  <code>stability_consensus_top20.csv</code> file alongside this page
  lists the top-20 genes by mean rank across seeds, with the standard
  deviation of each gene's rank — that is the publishable list.
</div>
{emb_block}
{pt_block}

<h2>What this means</h2>
<p>
  Reproducibility is not a bonus check — it is the difference between an
  anecdote and a finding. A model that lands on the same gene rankings,
  the same neighborhood structure, and the same per-sample progression
  position regardless of initialization is one whose conclusions can be
  cited. Numbers above tell you, per dimension of the analysis, exactly
  how much weight each finding bears.
</p>

</body></html>
"""
    out_path.write_text(html)


if __name__ == "__main__":
    main()
