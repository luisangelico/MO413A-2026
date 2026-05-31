"""Cohort pseudotime: order samples along a Normal -> Metastasis axis in embedding
space, then track which genes the model attends to progressively.

This combines the embedding pipeline (src.embeddings) and the attention pipeline
(src.interpret) into one analysis: a heatmap of mean attention per gene per
pseudotime decile, plus a line plot of representative genes.

Important framing (also baked into the HTML page): TCGA samples are not a time
course. "Pseudotime" here is a *cohort ordering* by how tumor-like the trained
model sees each sample. It describes the model's view, not real biology.

Run after `src.train`, `src.embeddings`, `src.interpret`:
    python -m src.pseudotime
    python -m src.pseudotime --run <run_dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.config import CLASS_NAMES, DATASET_FILE, PROCESSED_DATASET_PATH, get_device
from src.interpret import GENE_SETS, _per_sample_node_importance
from src.model import load_model


GENE_SET_COLORS = {
    "melanocyte_lineage": "#d62728",
    "melanoma_progression": "#ff7f0e",
    "immune_infiltrate": "#2ca02c",
    "keratin_skin_normal": "#1f77b4",
    "other": "#bdbdbd",
}


def gene_to_set(gene: str) -> str:
    for name, members in GENE_SETS.items():
        if gene in members:
            return name
    return "other"


def _resolve_run(run_arg: str | None) -> Path:
    if run_arg:
        return Path(run_arg)
    runs = sorted(PROCESSED_DATASET_PATH.glob("run_*"), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise SystemExit("No run_* directories found.")
    return runs[-1]


def compute_pseudotime(Z: np.ndarray, meta: pd.DataFrame) -> np.ndarray:
    """Project each embedding onto the Normal -> Metastasis centroid axis,
    normalize to [0, 1]. Pseudotime = 0 at the normal-skin centroid,
    1 at the metastasis centroid."""
    is_normal = (meta["true_label"] == CLASS_NAMES[2]).values
    is_meta = (meta["true_label"] == CLASS_NAMES[1]).values
    if is_normal.sum() < 2 or is_meta.sum() < 2:
        raise SystemExit("Need at least 2 Normal and 2 Metastasis samples to root pseudotime.")

    c_normal = Z[is_normal].mean(axis=0)
    c_meta = Z[is_meta].mean(axis=0)
    axis = c_meta - c_normal
    axis_norm = axis / (np.linalg.norm(axis) + 1e-9)
    proj = (Z - c_normal) @ axis_norm  # scalar per sample
    pmin, pmax = proj.min(), proj.max()
    return (proj - pmin) / max(pmax - pmin, 1e-9)


def attention_per_decile(model, dataset, indices, device, gene_names: list[str],
                         pseudotime: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Returns [n_bins, num_genes] mean attention per decile."""
    num_genes = len(gene_names)
    bin_sums = np.zeros((n_bins, num_genes), dtype=np.float64)
    bin_counts = np.zeros(n_bins, dtype=np.int64)
    edges = np.linspace(0, 1, n_bins + 1)
    edges[-1] = 1.0 + 1e-9  # right-open final bin

    for k, idx in enumerate(indices):
        bin_id = int(np.searchsorted(edges, pseudotime[k], side="right") - 1)
        bin_id = max(0, min(n_bins - 1, bin_id))
        importance, _, _ = _per_sample_node_importance(model, dataset[int(idx)], device)
        bin_sums[bin_id] += importance
        bin_counts[bin_id] += 1
        if (k + 1) % 100 == 0:
            print(f"  attention extracted: {k + 1}/{len(indices)}")

    counts_safe = np.where(bin_counts == 0, 1, bin_counts)
    return bin_sums / counts_safe[:, None], bin_counts


def select_top_genes(attn_by_bin: np.ndarray, gene_names: list[str],
                     top_k: int = 40, mode: str = "variable") -> list[int]:
    """Pick the K genes whose attention varies most across pseudotime."""
    if mode == "variable":
        var = attn_by_bin.var(axis=0)
        return np.argsort(-var)[:top_k].tolist()
    elif mode == "mean":
        return np.argsort(-attn_by_bin.mean(axis=0))[:top_k].tolist()
    else:
        raise ValueError(mode)


def plot_heatmap(attn_by_bin: np.ndarray, gene_names: list[str],
                 picks: list[int], bin_counts: np.ndarray,
                 out_path: Path) -> None:
    matrix = attn_by_bin[:, picks].T  # [genes, bins]
    # row-normalize to highlight when each gene peaks (visual)
    rmax = matrix.max(axis=1, keepdims=True)
    rnorm = matrix / np.where(rmax == 0, 1, rmax)

    # order genes by which bin they peak in (early peakers at top)
    peak_bin = rnorm.argmax(axis=1)
    order = np.argsort(peak_bin)
    rnorm = rnorm[order]
    labels = [gene_names[picks[i]] for i in order]
    label_colors = [GENE_SET_COLORS[gene_to_set(g)] for g in labels]

    fig, ax = plt.subplots(figsize=(9, max(6, 0.22 * len(labels))))
    im = ax.imshow(rnorm, aspect="auto", cmap="magma")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    for tick, col in zip(ax.get_yticklabels(), label_colors):
        tick.set_color(col)
    n_bins = matrix.shape[1]
    ax.set_xticks(range(n_bins))
    ax.set_xticklabels([f"{i + 1}\n(n={bin_counts[i]})" for i in range(n_bins)],
                       fontsize=8)
    ax.set_xlabel("pseudotime decile  (Normal-like  →  Metastasis-like)")
    ax.set_title("Per-decile mean attention (row-normalized)\n"
                 "Each row peaks once; rows ordered by when their attention peaks")
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("attention (row-normalized)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_gene_trajectories(attn_by_bin: np.ndarray, gene_names: list[str],
                           picks_by_set: dict[str, list[str]],
                           out_path: Path) -> None:
    name_to_idx = {g: i for i, g in enumerate(gene_names)}
    fig, ax = plt.subplots(figsize=(9, 5.5))
    n_bins = attn_by_bin.shape[0]
    x = np.arange(1, n_bins + 1)
    for set_name, genes in picks_by_set.items():
        color = GENE_SET_COLORS[set_name]
        for g in genes:
            if g not in name_to_idx:
                continue
            y = attn_by_bin[:, name_to_idx[g]]
            ax.plot(x, y, marker="o", color=color, alpha=0.85, label=g, linewidth=1.6)
    ax.set_xlabel("pseudotime decile  (Normal-like  →  Metastasis-like)")
    ax.set_ylabel("mean attention")
    ax.set_title("How attention to selected genes shifts along pseudotime")
    ax.legend(fontsize=8, ncol=2, loc="best")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_html(meta: pd.DataFrame, pseudotime: np.ndarray, bin_counts: np.ndarray,
               out_path: Path, run_name: str) -> None:
    # class composition per decile (for the explainer)
    bins = np.minimum((pseudotime * 10).astype(int), 9)
    comp = pd.crosstab(bins, meta["true_label"])
    for cls in CLASS_NAMES.values():
        if cls not in comp.columns:
            comp[cls] = 0
    comp = comp[[CLASS_NAMES[2], CLASS_NAMES[0], CLASS_NAMES[1]]]
    comp_rows = "".join(
        f"<tr><td>{i + 1}</td><td>{int(comp.iloc[i, 0])}</td>"
        f"<td>{int(comp.iloc[i, 1])}</td><td>{int(comp.iloc[i, 2])}</td>"
        f"<td>{int(bin_counts[i])}</td></tr>"
        for i in range(len(comp))
    )

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Pseudotime — gene attention along tumor-likeness</title>
<style>
  body {{ font-family:-apple-system,system-ui,sans-serif; max-width:900px;
          margin:1.5em auto; padding:0 1em; color:#222; line-height:1.6; }}
  .back-btn {{ display:inline-block; padding:0.4em 0.9em; margin-bottom:0.6em;
               background:#1f77b4; color:white !important; border-radius:5px;
               text-decoration:none; font-size:0.9em; }}
  .back-btn:hover {{ background:#155a8a; }}
  h1 {{ font-size:1.5em; }} h2 {{ font-size:1.1em; margin-top:1.6em; }}
  .meta {{ color:#666; font-size:0.9em; }}
  img {{ width:100%; max-width:880px; border:1px solid #eee; border-radius:6px;
         margin:0.6em 0; }}
  .callout {{ background:#fffbf2; border-left:3px solid #e0a020;
              padding:0.7em 1em; margin:1em 0; }}
  .plain {{ background:#f6fbf7; border-left:3px solid #2ca02c;
            padding:0.7em 1em; margin:1em 0; }}
  table {{ border-collapse:collapse; font-size:0.9em; margin:0.6em 0; }}
  th,td {{ padding:0.35em 0.7em; border-bottom:1px solid #eee; text-align:right; }}
  th {{ background:#f4f4f4; }}
  .legend {{ display:flex; gap:1em; flex-wrap:wrap; font-size:0.9em; margin:0.5em 0; }}
  .swatch {{ display:inline-block; width:12px; height:12px; border-radius:3px;
            margin-right:0.3em; vertical-align:middle; border:1px solid #999; }}
</style></head><body>

<a href="index.html" class="back-btn">&larr; back to predictions site</a>

<h1>Pseudotime — how the model's focus shifts from healthy skin to metastasis</h1>
<p class="meta">Run: <code>{run_name}</code></p>

<div class="plain">
  <strong>In one sentence.</strong> We sort every sample on a "tumor-likeness"
  axis and watch which genes the model relies on at each step. Genes whose
  importance rises as samples become more tumor-like are candidate signals of
  melanoma progression — as <em>seen by the model</em>.
</div>

<h2>What is an "embedding"?</h2>
<p>
  Inside the trained network, each sample is summarized as a list of ~64
  numbers — its <strong>embedding</strong>. Think of it as the model's
  fingerprint of the sample: similar embeddings = samples the model thinks
  look alike. We never look at this fingerprint directly during prediction;
  the network just uses it to decide a class. But we <em>can</em> look at it,
  and that's where this analysis starts.
</p>

<h2>What is "pseudotime" here?</h2>
<p>
  We computed two anchors in embedding space: the average embedding of the
  Normal-skin samples (call it the "normal anchor") and the average
  embedding of the Metastasis samples (the "metastasis anchor"). Then we
  drew an imaginary line between the two and projected every sample onto
  that line. Each sample gets a number between 0 and 1:
</p>
<ul>
  <li><strong>0</strong> = embedding sits at the normal-skin anchor</li>
  <li><strong>1</strong> = embedding sits at the metastasis anchor</li>
  <li>values in between = "somewhere along the way"</li>
</ul>
<p>
  We split this 0→1 axis into <strong>10 equal-width bins</strong>
  (deciles), and for each bin we average the model's attention scores
  across all samples that fell in that bin. The result is a picture of
  <em>which genes the model focuses on at each step along the
  Normal → Metastasis axis</em>.
</p>

<div class="callout">
  <strong>Honest caveat.</strong> TCGA samples are not a time course — there
  is no patient followed across years. "Pseudotime" here is a <em>cohort
  ordering</em> built from how the trained model sees the data. It describes
  the model's view, not the biology of any single tumor evolving. Genes
  whose attention rises along this axis are <em>hypotheses</em> for
  progression markers, not confirmed ones.
</div>

<h2>Composition of each pseudotime decile</h2>
<p>
  Sanity check: the early bins should be mostly Normal samples, the late
  bins should be mostly Metastasis. If they are, the axis is doing what we
  expect. (Primary tumors typically sit in the middle.)
</p>
<table>
<tr><th>decile</th><th>Normal</th><th>Primary</th><th>Metastasis</th><th>total</th></tr>
{comp_rows}
</table>

<h2>Heatmap: gene attention across pseudotime</h2>
<p>
  Rows = genes whose attention varies most across the axis. Columns = the
  10 pseudotime deciles, ordered Normal-like → Metastasis-like.
  Each row is normalized by its own peak (so bright = "this gene's
  attention peaks <em>here</em>"). Rows are reordered so genes that peak
  early appear at the top, genes that peak late at the bottom.
</p>
<p>
  Reading the figure: a row that's bright on the left and dark on the
  right means "the model paid attention to this gene mostly when looking
  at normal-like samples, and stopped caring as samples became more
  tumor-like." The reverse pattern points the other way.
</p>
<img src="pseudotime_heatmap.png"
     alt="Per-decile mean attention heatmap">

<h2>Selected gene trajectories</h2>
<p>
  A few representative genes from each reference category, plotted as
  attention vs. pseudotime decile. This is the same data as the heatmap
  but easier to read for a small set of genes.
</p>
<div class="legend">
  <span><span class="swatch" style="background:#d62728"></span>melanocyte lineage</span>
  <span><span class="swatch" style="background:#ff7f0e"></span>melanoma progression</span>
  <span><span class="swatch" style="background:#2ca02c"></span>immune infiltrate</span>
  <span><span class="swatch" style="background:#1f77b4"></span>keratin / skin normal</span>
</div>
<img src="pseudotime_lines.png"
     alt="Selected gene trajectories">

<h2>Why does this matter?</h2>
<p>
  This is the project's most ambitious figure: it ties together the
  <em>embedding</em> (the model's internal summary) and the
  <em>attention</em> (which genes the model uses) into one progression
  story. If keratin and skin-normal genes dominate the early deciles and
  fade in the late deciles, while melanocyte-lineage and melanoma-
  progression genes do the opposite, the model has internalized a coherent
  picture of how skin transitions toward melanoma — purely from
  expression and PPI topology, with no progression labels in training.
</p>

</body></html>
"""
    out_path.write_text(html)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default=None)
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--top-k", type=int, default=40,
                    help="Top-K most-variable genes for the heatmap")
    ap.add_argument("--site-dir", type=str, default=None,
                    help="Default: <repo>/site")
    args = ap.parse_args()

    run_dir = _resolve_run(args.run)
    print(f"Run: {run_dir}")
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    emb_path = run_dir / "embeddings.npy"
    meta_path = run_dir / "embeddings.csv"
    if not emb_path.exists() or not meta_path.exists():
        raise SystemExit("Run `python -m src.embeddings` first to produce embeddings.")
    Z = np.load(emb_path)
    meta = pd.read_csv(meta_path)
    print(f"Loaded {len(Z)} embeddings (shape {Z.shape})")

    # Restrict to correctly classified samples for a cleaner trajectory
    correct_mask = meta["correct"].values
    Z_c = Z[correct_mask]
    meta_c = meta[correct_mask].reset_index(drop=True)
    print(f"Using {len(Z_c)} correctly classified samples")

    pseudotime = compute_pseudotime(Z_c, meta_c)
    meta_c["pseudotime"] = pseudotime
    meta_c.to_csv(run_dir / "pseudotime.csv", index=False)
    print(f"Saved: {run_dir / 'pseudotime.csv'}")

    # Reload model + dataset to extract attention per sample
    model_path = run_dir / "best_model.pt"
    if not model_path.exists():
        model_path = run_dir / "modelo_final.pt"
    device = get_device()
    model, _ = load_model(model_path, device=device)
    dataset = torch.load(DATASET_FILE, weights_only=False)
    gene_names = list(dataset[0].gene_names)

    print(f"Computing per-decile attention ({args.bins} bins)...")
    indices = meta_c["idx"].values
    attn_by_bin, bin_counts = attention_per_decile(
        model, dataset, indices, device, gene_names, pseudotime, n_bins=args.bins,
    )
    np.save(run_dir / "pseudotime_attn_per_bin.npy", attn_by_bin)
    print(f"Bin counts: {bin_counts.tolist()}")

    print("Plotting heatmap...")
    picks = select_top_genes(attn_by_bin, gene_names, top_k=args.top_k, mode="variable")
    plot_heatmap(attn_by_bin, gene_names, picks, bin_counts,
                 fig_dir / "pseudotime_heatmap.png")
    print(f"Saved: {fig_dir / 'pseudotime_heatmap.png'}")

    print("Plotting line trajectories...")
    # Pick 2 representative genes per gene-set, only those present in panel
    in_panel = set(gene_names)
    picks_by_set = {}
    for set_name, genes in GENE_SETS.items():
        present = [g for g in genes if g in in_panel][:2]
        if present:
            picks_by_set[set_name] = present
    plot_gene_trajectories(attn_by_bin, gene_names, picks_by_set,
                           fig_dir / "pseudotime_lines.png")
    print(f"Saved: {fig_dir / 'pseudotime_lines.png'}")

    site_dir = Path(args.site_dir) if args.site_dir else (
        Path(__file__).resolve().parent.parent / "site")
    site_dir.mkdir(parents=True, exist_ok=True)
    out_html = site_dir / "pseudotime.html"
    write_html(meta_c, pseudotime, bin_counts, out_html, run_dir.name)
    print(f"Saved: {out_html}")

    # Copy figures next to the HTML so paths are self-contained.
    import shutil
    for fname in ("pseudotime_heatmap.png", "pseudotime_lines.png"):
        src = fig_dir / fname
        if src.exists():
            shutil.copyfile(src, site_dir / fname)


if __name__ == "__main__":
    main()
