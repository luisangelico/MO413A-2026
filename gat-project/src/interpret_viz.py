"""Visualizations for attention-based interpretability.

Reads CSVs produced by src.interpret and writes figures to <run_dir>/figures/:
    - top20_<ranking>.png        : per-ranking top-20 bar charts, colored by gene set
    - top20_grid.png             : combined grid of all rankings
    - gene_set_heatmap.png       : -log10(p-value) heatmap, rankings x gene sets
    - subgraph_<class>.png       : induced PPI subgraph of top attended nodes

With --html, also writes interactive subgraphs (pyvis) to site/interpret/:
    - subgraph_<class>.html
    - index.html              (gallery linking to the three class views)

Run:
    python -m src.interpret_viz                    # latest run
    python -m src.interpret_viz --run <run_dir>
    python -m src.interpret_viz --html             # also emit interactive HTML
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch

from src.config import CLASS_NAMES, DATASET_FILE, PROCESSED_DATASET_PATH
from src.interpret import GENE_SETS


GENE_SET_COLORS = {
    "melanocyte_lineage": "#d62728",   # red
    "melanoma_progression": "#ff7f0e", # orange
    "immune_infiltrate": "#2ca02c",    # green
    "keratin_skin_normal": "#1f77b4",  # blue
    "other": "#bdbdbd",                # grey
}


def gene_to_set(gene: str) -> str:
    for set_name, members in GENE_SETS.items():
        if gene in members:
            return set_name
    return "other"


def _resolve_run(run_arg: str | None) -> Path:
    if run_arg:
        return Path(run_arg)
    runs = sorted(PROCESSED_DATASET_PATH.glob("run_*"), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise SystemExit("No run_* directories found.")
    return runs[-1]


def plot_top20_bars(df_top: pd.DataFrame, out_dir: Path) -> None:
    rankings = df_top["ranking"].unique().tolist()

    # Individual figures
    for r in rankings:
        sub = df_top[df_top["ranking"] == r].sort_values("rank")
        colors = [GENE_SET_COLORS[gene_to_set(g)] for g in sub["gene"]]
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.barh(range(len(sub)), sub["score"], color=colors, edgecolor="black", linewidth=0.4)
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(sub["gene"])
        ax.invert_yaxis()
        ax.set_xlabel("attention importance")
        ax.set_title(f"Top-20 attended genes — {r}")
        # legend
        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in GENE_SET_COLORS.values()]
        ax.legend(handles, GENE_SET_COLORS.keys(), loc="lower right", fontsize=8)
        plt.tight_layout()
        safe = r.replace(" ", "_").replace("-", "minus")
        plt.savefig(out_dir / f"top20_{safe}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Grid figure
    n = len(rankings)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = np.atleast_2d(axes).flatten()
    for i, r in enumerate(rankings):
        ax = axes[i]
        sub = df_top[df_top["ranking"] == r].sort_values("rank")
        colors = [GENE_SET_COLORS[gene_to_set(g)] for g in sub["gene"]]
        ax.barh(range(len(sub)), sub["score"], color=colors, edgecolor="black", linewidth=0.3)
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(sub["gene"], fontsize=8)
        ax.invert_yaxis()
        ax.set_title(r, fontsize=10)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in GENE_SET_COLORS.values()]
    fig.legend(handles, GENE_SET_COLORS.keys(),
               loc="lower center", ncol=len(GENE_SET_COLORS), fontsize=9)
    fig.suptitle("Top-20 attended genes by ranking", fontsize=13)
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    plt.savefig(out_dir / "top20_grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_gene_set_heatmap(df_rec: pd.DataFrame, out_dir: Path, k: int = 100) -> None:
    sub = df_rec[df_rec["k"] == k].copy()
    if sub.empty:
        print(f"  no rows for k={k}, skipping heatmap")
        return
    sub["neg_log10_p"] = -np.log10(sub["pvalue"].clip(lower=1e-10))
    pivot = sub.pivot(index="ranking", columns="gene_set", values="neg_log10_p")
    # Order rankings sensibly
    order = [r for r in [
        CLASS_NAMES[0], CLASS_NAMES[1], CLASS_NAMES[2],
        "Metastasis - Primary", "Tumor - Normal", "[baseline] node degree",
    ] if r in pivot.index]
    pivot = pivot.reindex(order)

    fig, ax = plt.subplots(figsize=(1.8 * len(pivot.columns) + 2, 0.5 * len(pivot) + 2))
    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        color="white" if v > pivot.values[~np.isnan(pivot.values)].mean() else "black",
                        fontsize=8)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f"-log10(p), top-{k}")
    ax.set_title(f"Gene-set enrichment in top-{k} attended genes")
    plt.tight_layout()
    plt.savefig(out_dir / "gene_set_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_class_subgraphs(df_top: pd.DataFrame, df_imp: pd.DataFrame,
                         out_dir: Path, edge_index: torch.Tensor,
                         gene_names: list[str], top_n: int = 30) -> None:
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    ei = edge_index.cpu().numpy()
    G_full = nx.Graph()
    G_full.add_edges_from(zip(ei[0].tolist(), ei[1].tolist()))

    # Use class-differential score for coloring (Tumor - Normal if available)
    diff_col = "Tumor - Normal" if "Tumor - Normal" in df_imp.columns else df_imp.columns[0]
    diff = df_imp[diff_col]

    rankings = [CLASS_NAMES[0], CLASS_NAMES[1], CLASS_NAMES[2]]
    rankings = [r for r in rankings if r in df_top["ranking"].unique()]

    for r in rankings:
        top_genes = df_top[df_top["ranking"] == r].sort_values("rank")["gene"].head(top_n).tolist()
        nodes = [gene_to_idx[g] for g in top_genes if g in gene_to_idx]
        H = G_full.subgraph(nodes).copy()
        if H.number_of_nodes() == 0:
            continue
        labels = {n: gene_names[n] for n in H.nodes()}
        node_color = [diff.get(gene_names[n], 0.0) for n in H.nodes()]

        # Score-driven node size
        scores = df_top[df_top["ranking"] == r].set_index("gene")["score"]
        smin, smax = scores.min(), scores.max()
        sizes = [200 + 800 * (scores.get(gene_names[n], smin) - smin)
                 / max(smax - smin, 1e-9) for n in H.nodes()]

        fig, ax = plt.subplots(figsize=(10, 8))
        pos = nx.spring_layout(H, seed=42, k=1.2 / np.sqrt(max(H.number_of_nodes(), 1)))
        nx.draw_networkx_edges(H, pos, ax=ax, alpha=0.4, width=0.8)
        nodes_drawn = nx.draw_networkx_nodes(
            H, pos, ax=ax, node_color=node_color, node_size=sizes,
            cmap="coolwarm", edgecolors="black", linewidths=0.5,
        )
        nx.draw_networkx_labels(H, pos, labels=labels, ax=ax, font_size=8)
        cbar = plt.colorbar(nodes_drawn, ax=ax, shrink=0.7)
        cbar.set_label(diff_col)
        ax.set_title(f"PPI subgraph of top-{top_n} attended nodes — {r}")
        ax.axis("off")
        plt.tight_layout()
        safe = r.replace(" ", "_")
        plt.savefig(out_dir / f"subgraph_{safe}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_class_subgraphs_html(df_top: pd.DataFrame, df_imp: pd.DataFrame,
                              out_dir: Path, edge_index: torch.Tensor,
                              gene_names: list[str], top_n: int = 30) -> list[Path]:
    """Interactive pyvis subgraphs. Returns list of written HTML paths."""
    from pyvis.network import Network
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm

    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    ei = edge_index.cpu().numpy()
    G_full = nx.Graph()
    G_full.add_edges_from(zip(ei[0].tolist(), ei[1].tolist()))

    diff_col = "Tumor - Normal" if "Tumor - Normal" in df_imp.columns else df_imp.columns[0]
    diff = df_imp[diff_col]
    vmin, vmax = float(diff.min()), float(diff.max())
    norm = mcolors.TwoSlopeNorm(vmin=min(vmin, -1e-9), vcenter=0.0,
                                vmax=max(vmax, 1e-9))
    cmap = cm.get_cmap("coolwarm")

    rankings = [CLASS_NAMES[0], CLASS_NAMES[1], CLASS_NAMES[2]]
    rankings = [r for r in rankings if r in df_top["ranking"].unique()]
    written = []

    out_dir.mkdir(parents=True, exist_ok=True)
    for r in rankings:
        sub = df_top[df_top["ranking"] == r].sort_values("rank").head(top_n)
        top_genes = sub["gene"].tolist()
        nodes = [gene_to_idx[g] for g in top_genes if g in gene_to_idx]
        H = G_full.subgraph(nodes).copy()
        if H.number_of_nodes() == 0:
            continue

        scores = sub.set_index("gene")["score"]
        smin, smax = float(scores.min()), float(scores.max())

        isolated = [gene_names[n] for n in H.nodes() if H.degree(n) == 0]
        n_iso = len(isolated)
        n_total = H.number_of_nodes()

        net = Network(height="650px", width="100%", bgcolor="#ffffff",
                      font_color="#222", notebook=False, directed=False)
        net.toggle_physics(True)

        for n in H.nodes():
            gene = gene_names[n]
            d = float(diff.get(gene, 0.0))
            r_, g_, b_, _ = cmap(norm(d))
            color = mcolors.to_hex((r_, g_, b_))
            s = float(scores.get(gene, smin))
            size = 10 + 25 * (s - smin) / max(smax - smin, 1e-9)
            gset = gene_to_set(gene)
            title = (
                f"{gene}\n"
                f"attention score: {s:.2f}\n"
                f"{diff_col}: {d:+.2f}\n"
                f"gene set: {gset}"
            )
            net.add_node(int(n), label=gene, title=title, color=color, size=size)
        for u, v in H.edges():
            net.add_edge(int(u), int(v), color="#cccccc", width=1)

        net.set_options("""
        {
          "physics": {
            "barnesHut": {"gravitationalConstant": -8000, "springLength": 120},
            "minVelocity": 0.5
          },
          "interaction": {"hover": true, "tooltipDelay": 100}
        }
        """)

        safe = r.replace(" ", "_").replace("-", "minus")
        out_path = out_dir / f"subgraph_{safe}.html"
        net.write_html(str(out_path), open_browser=False, notebook=False)

        # Inject an explanatory header above the pyvis canvas
        try:
            raw = out_path.read_text()
            header = f"""
<div style="font-family:-apple-system,system-ui,sans-serif;max-width:900px;margin:1em auto;padding:0 1em;color:#222;line-height:1.55;">
  <p style="margin:0 0 0.4em;"><a href="index.html" style="color:#1f77b4;text-decoration:none;">&larr; back to overview</a></p>
  <h2 style="margin:0.2em 0;">Top-30 attended PPI subgraph — {r}</h2>
  <p style="color:#555;font-size:0.95em;margin:0.4em 0;">
    Genes the GAT attended to most when classifying <strong>{r}</strong> samples
    in the held-out test split. Node size = attention importance.
    Node color = Tumor − Normal differential attention
    (<span style="color:#b40426;">red</span> = tumor-leaning,
     <span style="color:#3b4cc0;">blue</span> = normal-leaning).
    Edges are STRING physical PPIs. Drag nodes to explore; hover for details.
  </p>
  <p style="color:#666;font-size:0.88em;margin:0.4em 0;">
    <strong>Attention score</strong> = sum of incoming softmax-normalized
    edge-attention coefficients onto this gene, averaged over the 4 GAT heads,
    summed over 3 layers, then averaged over correctly classified
    <strong>{r}</strong> samples. Per-layer α(i→j) lies in [0, 1] and sums to 1
    over j's incoming neighbors, so the final score is roughly in [0, ~3].
    Only relative differences between genes matter — a gene at 0.04 attracted
    ~4× the attention of one at 0.01.
  </p>
  <p style="color:#888;font-size:0.85em;margin:0.4em 0 0;">
    <strong>Note on isolated nodes</strong> ({n_iso} of {n_total} here):
    this is the <em>induced</em> subgraph over the top-30 attended genes only.
    A gene appears isolated when its STRING PPI partners exist in the full
    graph but were not themselves in the top-30 for this class — the model
    attended to the gene, but its neighborhood was not uniformly attended.
  </p>
</div>
"""
            raw = raw.replace("<body>", "<body>" + header, 1)
            out_path.write_text(raw)
        except Exception:
            pass

        written.append(out_path)
    return written


def write_subgraph_index(out_dir: Path, html_paths: list[Path], run_name: str) -> Path:
    cards = "\n".join(
        f'''  <li>
    <a href="{p.name}"><strong>{p.stem.replace("subgraph_", "").replace("_", " ")}</strong></a>
    <span class="hint">— top-30 attended genes induced subgraph</span>
  </li>'''
        for p in html_paths
    )
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>GAT attention — interpretability</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 820px;
          margin: 2em auto; color: #222; padding: 0 1em; line-height: 1.55; }}
  h1 {{ font-size: 1.5em; margin-bottom: 0.2em; }}
  h2 {{ font-size: 1.1em; margin-top: 1.8em; color: #333; }}
  .meta {{ color: #666; font-size: 0.9em; }}
  ul {{ line-height: 1.9; }} a {{ color: #1f77b4; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  .hint {{ color: #888; font-size: 0.9em; }}
  .legend {{ display: flex; gap: 1.2em; flex-wrap: wrap; margin: 0.6em 0 1.2em; }}
  .swatch {{ display: inline-block; width: 14px; height: 14px; border-radius: 3px;
            vertical-align: middle; margin-right: 0.4em; border: 1px solid #999; }}
  code {{ background: #f4f4f4; padding: 0 0.3em; border-radius: 3px; }}
  .callout {{ background: #f7faff; border-left: 3px solid #1f77b4;
              padding: 0.7em 1em; margin: 1em 0; font-size: 0.95em; }}
  .back-btn {{ display: inline-block; padding: 0.4em 0.9em; margin-bottom: 0.6em;
               background: #1f77b4; color: white !important; border-radius: 5px;
               text-decoration: none; font-size: 0.9em; }}
  .back-btn:hover {{ background: #155a8a; text-decoration: none; }}
</style></head><body>

<a href="../index.html" class="back-btn">&larr; back to predictions site</a>

<h1>GAT attention — top-30 PPI subgraphs by class</h1>
<p class="meta">Run: <code>{run_name}</code></p>

<h2>What you're looking at</h2>
<p>
  Each page below shows the protein–protein interaction (PPI) subgraph induced by
  the <strong>top-30 most-attended genes</strong> for one class, as seen by the
  trained Graph Attention Network on the held-out test split. Per-sample edge
  attention from all three GAT layers is averaged across heads, summed onto
  destination nodes, then averaged across <em>correctly classified</em> samples in
  the class.
</p>

<h2>What "attention score" means</h2>
<p>
  Inside each GAT layer, every directed edge <em>i → j</em> in the PPI graph
  carries an <strong>attention coefficient</strong> α(i→j) — a number the model
  learns that says <em>how much node j should listen to node i</em> when
  updating its representation. For a fixed destination node j, the
  α(i→j) values are softmax-normalized across j's incoming neighbors
  (including a self-loop), so each individual α lies in <strong>[0, 1]</strong>
  and the values incoming to j sum to 1 (per attention head, per layer).
</p>
<p>
  We turn this into a per-gene importance score in three steps:
</p>
<ol>
  <li><strong>Per sample, per layer:</strong> mean α across the 4 attention
    heads, then sum the incoming α onto each gene (its destination).</li>
  <li><strong>Across the 3 GAT layers:</strong> sum the per-layer values
    (so a gene that is consistently attended deep in the network scores
    higher than one only briefly attended in layer 1).</li>
  <li><strong>Across samples:</strong> average over <em>correctly classified</em>
    test-set samples within the class.</li>
</ol>
<p>
  The resulting score is unitless and approximately on a <strong>[0, ~3]</strong>
  scale (3 layers × softmax-normalized incoming attention). Only relative
  differences between genes matter — a gene with score 0.04 versus 0.01
  attracted four times the attention, even though both are "small" numbers.
  All scores reported here are these per-class averages.
</p>

<h2>How to read the visualization</h2>
<ul>
  <li><strong>Node size</strong> = per-class mean attention importance
    (bigger = the model relied on this gene more when classifying samples in
    this class).</li>
  <li><strong>Node color</strong> = <em>Tumor − Normal</em> differential attention
    (red = preferentially attended in tumor samples, blue = preferentially in
    normal skin, white ≈ neutral).</li>
  <li><strong>Edges</strong> = STRING physical PPIs (confidence ≥ 200) between
    these genes. Layout is force-directed (Barnes-Hut); you can drag nodes.</li>
  <li><strong>Hover</strong> a node for: gene symbol, attention score,
    Tumor − Normal score, and gene-set membership.</li>
</ul>

<div class="legend">
  <span><span class="swatch" style="background:#d62728"></span>melanocyte lineage</span>
  <span><span class="swatch" style="background:#ff7f0e"></span>melanoma progression</span>
  <span><span class="swatch" style="background:#2ca02c"></span>immune infiltrate</span>
  <span><span class="swatch" style="background:#1f77b4"></span>keratin / skin normal</span>
  <span class="hint">(reference gene-set categories — used in the bar charts and heatmap)</span>
</div>

<h2>Why this matters</h2>
<p>
  A GAT trained only on gene expression and PPI topology can be inspected for
  <em>which genes</em> drove each prediction. If the top-attended genes recover
  known melanoma biology (melanocyte lineage markers, immune infiltrate signature,
  cornified-envelope genes for normal skin), the model is learning interpretable
  signal rather than spurious patterns. The <em>contrast</em> rankings
  (Metastasis − Primary, Tumor − Normal) are the most informative — they isolate
  what attention does <strong>differently</strong> across classes, beyond the
  static PPI hub structure.
</p>

<div class="callout">
  <strong>Sanity check.</strong> A node-degree baseline ranking is included in the
  companion bar charts and heatmap. Where attention beats degree
  (e.g. <em>Metastasis − Primary</em> recovering keratin/skin genes at p ≈ 2×10⁻⁵
  vs. p ≈ 5×10⁻³ for degree alone), the GAT is genuinely learning class-specific
  importance — not just attending to PPI hubs.
</div>

<div class="callout" style="background:#fffbf2;border-left-color:#e0a020;">
  <strong>Why some nodes look disconnected.</strong> Each subgraph is the
  <em>induced</em> subgraph over the top-30 attended genes. An isolated node
  means the gene was attended to, but its STRING PPI partners were not
  themselves in the top-30 for that class — its edges still exist in the full
  graph, just not within this filtered view.
</div>

<h2>Class views</h2>
<ul>
{cards}
</ul>

<p class="hint">Companion static figures (top-20 bar charts per ranking,
gene-set enrichment heatmap, matplotlib subgraphs) are saved alongside the model
in <code>data/processed/{run_name}/figures/</code>.</p>

</body></html>
"""
    idx = out_dir / "index.html"
    idx.write_text(html)
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default=None)
    ap.add_argument("--top-n-subgraph", type=int, default=30)
    ap.add_argument("--html", action="store_true",
                    help="Also emit interactive pyvis subgraphs to site/interpret/")
    ap.add_argument("--site-dir", type=str, default=None,
                    help="Override site directory (default: <repo>/site/interpret)")
    args = ap.parse_args()

    run_dir = _resolve_run(args.run)
    print(f"Run: {run_dir}")

    fig_dir = run_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    top_csv = run_dir / "attention_top20.csv"
    rec_csv = run_dir / "attention_gene_set_recovery.csv"
    imp_csv = run_dir / "attention_per_class.csv"
    for p in (top_csv, rec_csv, imp_csv):
        if not p.exists():
            raise SystemExit(f"Missing {p} — run `python -m src.interpret` first.")

    df_top = pd.read_csv(top_csv)
    df_rec = pd.read_csv(rec_csv)
    df_imp = pd.read_csv(imp_csv, index_col=0)

    print("Plotting top-20 bars...")
    plot_top20_bars(df_top, fig_dir)

    print("Plotting gene-set heatmap...")
    plot_gene_set_heatmap(df_rec, fig_dir, k=100)

    print("Plotting class subgraphs...")
    dataset = torch.load(DATASET_FILE, weights_only=False)
    gene_names = list(dataset[0].gene_names)
    edge_index = dataset[0].edge_index
    plot_class_subgraphs(df_top, df_imp, fig_dir, edge_index, gene_names,
                         top_n=args.top_n_subgraph)

    print(f"\nFigures saved to: {fig_dir}")
    for p in sorted(fig_dir.glob("*.png")):
        print(f"  {p.name}")

    if args.html:
        if args.site_dir:
            site_out = Path(args.site_dir)
        else:
            site_out = Path(__file__).resolve().parent.parent / "site" / "interpret"
        print(f"\nWriting interactive subgraphs to: {site_out}")
        html_paths = plot_class_subgraphs_html(
            df_top, df_imp, site_out, edge_index, gene_names,
            top_n=args.top_n_subgraph,
        )
        idx = write_subgraph_index(site_out, html_paths, run_dir.name)
        print(f"  index: {idx}")
        for p in html_paths:
            print(f"  {p.name}")


if __name__ == "__main__":
    main()
