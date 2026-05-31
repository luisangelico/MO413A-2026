"""Dataset analysis page for the static site.

Builds:
  - figures: dataset_overview.png, dataset_ppi_graph.png (under <run_dir>/figures/)
  - site/dataset.html: a non-technical explainer of the data sources (TCGA-SKCM
    + GTEx skin via the UCSC Xena TOIL recompute, STRING physical PPIs) and the
    structure of the per-sample graphs the GAT consumes

Run:
    python -m src.dataset_page
    python -m src.dataset_page --run <run_dir> --site-dir docs
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from sklearn.decomposition import PCA

from src.config import CLASS_NAMES, DATASET_FILE, PROCESSED_DATASET_PATH
from src.site_header import HEADER_CSS, render_header


COLORS = {0: "#d62728", 1: "#ff7f0e", 2: "#2ca02c"}


def _resolve_run(run_arg: str | None) -> Path:
    if run_arg:
        return Path(run_arg)
    runs = sorted(PROCESSED_DATASET_PATH.glob("run_*"), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise SystemExit("No run_* directories found.")
    return runs[-1]


def make_overview_figure(dataset, X: np.ndarray, labels: np.ndarray,
                         out_path: Path) -> dict:
    classes = sorted(set(labels.tolist()))
    n_nodes = dataset[0].x.shape[0]
    n_edges = dataset[0].edge_index.shape[1]
    deg = np.bincount(dataset[0].edge_index[0].numpy(), minlength=n_nodes)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(f"Dataset overview — {DATASET_FILE.name}",
                 fontsize=14, fontweight="bold")

    # 1. Class balance
    ax = axes[0, 0]
    counts = Counter(labels.tolist())
    bars = ax.bar([CLASS_NAMES[c] for c in classes],
                  [counts[c] for c in classes],
                  color=[COLORS[c] for c in classes], edgecolor="black")
    for b, c in zip(bars, classes):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), str(counts[c]),
                ha="center", va="bottom", fontweight="bold")
    ax.set_title("Class balance")
    ax.set_ylabel("Number of samples")

    # 2. Expression distribution per class
    ax = axes[0, 1]
    for c in classes:
        vals = X[labels == c].ravel()
        if len(vals) > 50000:
            vals = np.random.default_rng(0).choice(vals, 50000, replace=False)
        ax.hist(vals, bins=80, density=True, alpha=0.45,
                label=CLASS_NAMES[c], color=COLORS[c])
    ax.set_title("Expression value distribution")
    ax.set_xlabel("Expression (log2 TPM+0.001)")
    ax.set_ylabel("Density")
    ax.legend()

    # 3. PCA scatter
    ax = axes[1, 0]
    pca = PCA(n_components=2, random_state=0)
    Z = pca.fit_transform(X)
    for c in classes:
        mask = labels == c
        ax.scatter(Z[mask, 0], Z[mask, 1], c=COLORS[c], label=CLASS_NAMES[c],
                   alpha=0.6, s=18, edgecolor="black", linewidth=0.3)
    var = float(pca.explained_variance_ratio_.sum() * 100)
    ax.set_title(f"PCA of expression  (var explained: {var:.1f}%)")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.legend()

    # 4. Degree distribution
    ax = axes[1, 1]
    ax.hist(deg, bins=40, color="steelblue", edgecolor="black")
    ax.set_title("Node degree distribution (PPI graph)")
    ax.set_xlabel("Degree"); ax.set_ylabel("Number of genes")
    stats = (f"genes:            {n_nodes}\n"
             f"edges (directed): {n_edges}\n"
             f"avg degree:       {deg.mean():.2f}\n"
             f"max degree:       {deg.max()}\n"
             f"isolated genes:   {int((deg == 0).sum())}")
    ax.text(0.97, 0.97, stats, transform=ax.transAxes, ha="right", va="top",
            family="monospace", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray"))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "n_samples": int(len(dataset)),
        "n_nodes": int(n_nodes),
        "n_edges": int(n_edges),
        "avg_degree": float(deg.mean()),
        "max_degree": int(deg.max()),
        "isolated": int((deg == 0).sum()),
        "class_counts": {CLASS_NAMES[c]: int(counts[c]) for c in classes},
        "pca_var_explained": var,
    }


def make_ppi_figure(dataset, X: np.ndarray, out_path: Path) -> None:
    sample0 = dataset[0]
    n_nodes = sample0.x.shape[0]
    gene_names = getattr(sample0, "gene_names", [str(i) for i in range(n_nodes)])
    mean_expr = X.mean(axis=0)

    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    edges_np = sample0.edge_index.numpy()
    for k in range(edges_np.shape[1]):
        s, d = int(edges_np[0, k]), int(edges_np[1, k])
        if s != d:
            G.add_edge(s, d)
    G.remove_nodes_from([n for n in G.nodes() if G.degree(n) == 0])

    fig, ax = plt.subplots(figsize=(12, 12))
    if len(G) == 0:
        ax.text(0.5, 0.5, "No edges in PPI graph", ha="center", va="center")
    else:
        pos = nx.spring_layout(G, k=0.3, iterations=80, seed=42)
        nodes = list(G.nodes())
        degrees = np.array([G.degree(n) for n in nodes])
        node_expr = np.array([mean_expr[n] for n in nodes])
        sizes = 20 + 200 * (degrees / max(degrees.max(), 1))

        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.25, width=0.6, edge_color="gray")
        nc = nx.draw_networkx_nodes(
            G, pos, ax=ax, nodelist=nodes, node_size=sizes,
            node_color=node_expr, cmap="viridis",
            edgecolors="black", linewidths=0.4, alpha=0.9,
        )
        plt.colorbar(nc, ax=ax, shrink=0.6, label="Mean expression")
        # Label only the top-N hubs
        top = np.argsort(-degrees)[:25]
        labels = {nodes[i]: gene_names[nodes[i]] for i in top}
        nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=8)

    ax.set_title(f"PPI graph — {len(G)} connected genes "
                 f"(top-25 hubs labeled, color = mean expression across samples)")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_ppi_interactive(dataset, X: np.ndarray, out_path: Path,
                         min_degree: int = 1, label_top: int = 50) -> int:
    """Build an interactive pyvis PPI graph. Returns number of nodes rendered.

    Drops nodes with degree < min_degree to keep the layout legible.
    Only the top `label_top` hubs are labeled by default — others show their
    name on hover.
    """
    from pyvis.network import Network
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm

    sample0 = dataset[0]
    n_nodes = sample0.x.shape[0]
    gene_names = getattr(sample0, "gene_names", [str(i) for i in range(n_nodes)])
    mean_expr = X.mean(axis=0)

    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    edges_np = sample0.edge_index.numpy()
    for k in range(edges_np.shape[1]):
        s, d = int(edges_np[0, k]), int(edges_np[1, k])
        if s != d:
            G.add_edge(s, d)
    # Drop low-degree nodes for legibility
    G.remove_nodes_from([n for n in G.nodes() if G.degree(n) < min_degree])
    if G.number_of_nodes() == 0:
        out_path.write_text("<html><body>Empty graph.</body></html>")
        return 0

    nodes = list(G.nodes())
    degrees = np.array([G.degree(n) for n in nodes])
    node_expr = np.array([mean_expr[n] for n in nodes])

    # Color = mean expression, viridis colormap
    expr_norm = mcolors.Normalize(vmin=float(node_expr.min()),
                                  vmax=float(node_expr.max()))
    cmap = cm.get_cmap("viridis")

    # Top-N hubs get visible labels
    top_idx = set(int(i) for i in np.argsort(-degrees)[:label_top])

    net = Network(height="700px", width="100%", bgcolor="#ffffff",
                  font_color="#222", notebook=False, directed=False)
    net.toggle_physics(True)

    smin, smax = float(degrees.min()), float(degrees.max())
    for i, n in enumerate(nodes):
        gene = gene_names[n]
        d = float(node_expr[i])
        deg = int(degrees[i])
        r_, g_, b_, _ = cmap(expr_norm(d))
        color = mcolors.to_hex((r_, g_, b_))
        size = 6 + 30 * (deg - smin) / max(smax - smin, 1e-9)
        title = (f"{gene}\n"
                 f"degree (PPI partners): {deg}\n"
                 f"mean expression: {d:.2f}")
        # Show label only for hubs; others get blank label but full hover info
        label = gene if i in top_idx else ""
        net.add_node(int(n), label=label, title=title,
                     color=color, size=size)
    for u, v in G.edges():
        net.add_edge(int(u), int(v), color="#cccccc", width=0.6)

    net.set_options("""
    {
      "physics": {
        "barnesHut": {"gravitationalConstant": -3000, "springLength": 90,
                      "centralGravity": 0.2, "damping": 0.4},
        "minVelocity": 0.75,
        "stabilization": {"iterations": 200}
      },
      "interaction": {"hover": true, "tooltipDelay": 100, "navigationButtons": true}
    }
    """)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    net.write_html(str(out_path), open_browser=False, notebook=False)

    # Inject a small explanatory header
    raw = out_path.read_text()
    header = f"""
<div style="font-family:-apple-system,system-ui,sans-serif;max-width:1100px;
            margin:0.6em auto;padding:0.4em 1em;color:#444;font-size:0.88em;line-height:1.4;">
  <strong>{G.number_of_nodes()} genes</strong>, {G.number_of_edges()} STRING physical PPIs
  (confidence ≥ 200). Drag nodes; scroll to zoom; hover for gene name, degree,
  and mean expression. Top {label_top} hubs are pre-labeled; others appear on hover.
  Node size = number of PPI partners. Node color = mean expression across the cohort.
</div>
"""
    raw = raw.replace("<body>", "<body>" + header, 1)
    out_path.write_text(raw)
    return G.number_of_nodes()


def write_html(stats: dict, out_path: Path, run_name: str,
               figures_rel_prefix: str) -> None:
    cls = stats["class_counts"]
    cls_rows = "".join(
        f"<tr><td>{name}</td><td>{count}</td></tr>" for name, count in cls.items()
    )
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Dataset — Skin Cancer Gene-Network Analysis</title>
<style>
{HEADER_CSS}
  body {{ font-family:-apple-system,system-ui,sans-serif; margin:0; color:#222;
          line-height:1.6; }}
  .page {{ max-width:920px; margin:0 auto; padding:0 1em 2em; }}
  h1 {{ font-size:1.5em; margin-top:0.4em; }} h2 {{ font-size:1.15em; margin-top:1.6em; }}
  .meta {{ color:#666; font-size:0.9em; }}
  .plain {{ background:#f6fbf7; border-left:3px solid #2ca02c;
            padding:0.7em 1em; margin:1em 0; }}
  .callout {{ background:#fffbf2; border-left:3px solid #e0a020;
              padding:0.7em 1em; margin:1em 0; }}
  table {{ border-collapse:collapse; font-size:0.92em; margin:0.6em 0; }}
  th,td {{ padding:0.4em 0.7em; border-bottom:1px solid #eee; text-align:left; }}
  th {{ background:#f4f4f4; }}
  img {{ width:100%; max-width:880px; border:1px solid #eee;
         border-radius:6px; margin:0.6em 0; }}
  code {{ background:#f4f4f4; padding:0 0.3em; border-radius:3px; }}
  ul {{ line-height:1.8; }}
  .sources {{ display:grid; grid-template-columns:1fr 1fr; gap:1em; }}
  @media (max-width:760px) {{ .sources {{ grid-template-columns:1fr; }} }}
  .src-card {{ background:#f6f8fc; border:1px solid #d8def0;
               border-radius:8px; padding:0.9em 1em; }}
  .src-card h3 {{ margin:0 0 0.3em 0; font-size:1em; }}
</style></head><body>
{render_header("dataset", subtitle=f"Dataset file: <code>{DATASET_FILE.name}</code> · run: <code>{run_name}</code>")}
<div class="page">

<h1>Where the data comes from</h1>

<div class="plain">
  <strong>In one sentence.</strong> Every sample in this project is a
  patient's gene-expression measurement, paired with the protein–protein
  interaction (PPI) network of the genes we measured, packaged together as
  a single graph the GAT can consume.
</div>

<h2>The two data sources</h2>
<div class="sources">
  <div class="src-card">
    <h3>1. Gene expression — UCSC Xena TOIL recompute</h3>
    <p>
      Skin tumor samples come from <strong>TCGA-SKCM</strong> (The Cancer
      Genome Atlas, skin cutaneous melanoma) and healthy skin from
      <strong>GTEx</strong> (Genotype-Tissue Expression). Both are
      reprocessed together by the <strong>UCSC Xena TOIL pipeline</strong>,
      which applies the same alignment, quantification, and normalization
      to every sample. This matters: comparing two cohorts processed by
      different pipelines introduces "batch effects" that often dominate
      any biological signal. TOIL eliminates that confounder by design.
    </p>
    <p>
      Values are reported as <code>log2(TPM + 0.001)</code>: roughly
      "how much each gene is being expressed, on a log scale."
    </p>
  </div>
  <div class="src-card">
    <h3>2. Protein interactions — STRING</h3>
    <p>
      Edges between genes in each sample's graph come from
      <strong>STRING-DB</strong>, a database of known
      protein–protein interactions. We use only <em>physical</em>
      interactions (proteins that actually bind) at confidence
      threshold ≥ 200 — a moderately permissive cutoff that keeps
      well-supported edges and trims speculative ones.
    </p>
    <p>
      The same edge set is used for every sample; only the node
      <em>features</em> (expression values) change patient to patient.
    </p>
  </div>
</div>

<h2>Three classes, one task</h2>
<table>
<tr><th>Label</th><th>Samples</th></tr>
{cls_rows}
</table>
<p>
  The model is trained to look at a sample's expression-on-PPI graph and
  decide which of these three categories it belongs to. The Normal-skin
  class comes from GTEx; both tumor classes come from TCGA-SKCM.
</p>

<h2>What each sample looks like</h2>
<ul>
  <li><strong>{stats['n_nodes']} genes (nodes)</strong> per sample — the
      genes whose expression varies most across the cohort. Less variable
      genes are dropped because they carry little discriminative information.</li>
  <li><strong>{stats['n_edges']} directed edges</strong> per sample
      (≈ {stats['n_edges'] // 2} unique interactions, both directions kept
      so message-passing can flow either way).</li>
  <li><strong>Average degree {stats['avg_degree']:.2f}</strong>; max degree
      {stats['max_degree']}; <strong>{stats['isolated']} genes</strong>
      have no PPI partners in this panel and act as isolated nodes — the
      model still uses their expression directly via self-loops.</li>
</ul>

<h2>Snapshot — class balance, expression distributions, PCA, degree</h2>
<p>
  Four sanity-check views of the dataset before any modeling:
</p>
<ul>
  <li><strong>Top-left:</strong> how many samples per class.</li>
  <li><strong>Top-right:</strong> distribution of log-expression values
    across all genes in each class. Tumor and metastasis distributions
    look broadly similar; normal skin is offset.</li>
  <li><strong>Bottom-left:</strong> PCA of raw expression (no model
    involved). Even a simple linear projection separates Normal from
    Tumor cleanly — the harder task is splitting Primary vs Metastasis.
    PC1 + PC2 explain {stats['pca_var_explained']:.1f}% of variance.</li>
  <li><strong>Bottom-right:</strong> degree distribution of the PPI
    graph. A few hub genes have many partners; most have a handful.</li>
</ul>
<img src="{figures_rel_prefix}dataset_overview.png" alt="Dataset overview">

<h2>The PPI graph — interactive</h2>
<p>
  Force-directed layout of the same PPI network used in every sample's
  graph. <strong>Drag</strong> nodes to explore; <strong>scroll</strong> to zoom;
  <strong>hover</strong> a node for gene name, degree, and mean expression.
  Node color = average expression across the cohort (dark → bright);
  node size = number of PPI partners. The biggest hubs are pre-labeled.
</p>
<div class="callout">
  <strong>Why hubs matter for interpretability.</strong> Genes with many
  PPI partners are easy targets for any attention mechanism — they
  collect information from a lot of neighbors. That's why every
  attention finding in this project is benchmarked against a
  <em>node-degree baseline</em> (see the
  <a href="interpret/index.html">attention interpretability</a> page).
  An attention ranking that beats degree is doing something the
  static graph alone can't.
</div>
<iframe src="dataset_ppi_graph.html" style="width:100%;height:760px;
        border:1px solid #ddd;border-radius:6px;" loading="lazy"></iframe>
<details style="margin-top:0.8em;">
  <summary style="cursor:pointer;color:#666;font-size:0.92em;">
    Static rendering (PNG) for printing / fallback
  </summary>
  <img src="{figures_rel_prefix}dataset_ppi_graph.png" alt="PPI graph (static)">
</details>

<h2>Provenance summary</h2>
<ul>
  <li><strong>Source:</strong> TCGA-SKCM + GTEx skin, jointly reprocessed
    by the UCSC Xena TOIL pipeline.</li>
  <li><strong>Quantification:</strong> log2(TPM + 0.001).</li>
  <li><strong>Gene panel:</strong> top-{stats['n_nodes']} most-variable
    protein-coding genes.</li>
  <li><strong>Graph:</strong> STRING physical PPIs, confidence ≥ 200.</li>
  <li><strong>Per-sample graph:</strong> {stats['n_nodes']} nodes,
    {stats['n_edges']} directed edges (shared topology, per-sample
    expression features).</li>
  <li><strong>Total samples:</strong> {stats['n_samples']}
    ({" / ".join(f"{v} {k}" for k, v in cls.items())}).</li>
</ul>
</div>
</body></html>
"""
    out_path.write_text(html)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default=None)
    ap.add_argument("--site-dir", type=str, default=None,
                    help="Default: <repo>/site")
    args = ap.parse_args()

    run_dir = _resolve_run(args.run)
    print(f"Run: {run_dir}")
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    print(f"Loading {DATASET_FILE}...")
    dataset = torch.load(DATASET_FILE, weights_only=False)
    print(f"  {len(dataset)} samples")

    labels = np.array([int(d.y.item()) for d in dataset])
    X = np.stack([d.x.squeeze().numpy() for d in dataset])

    print("Building overview figure...")
    stats = make_overview_figure(dataset, X, labels,
                                 fig_dir / "dataset_overview.png")
    print(f"  saved: {fig_dir / 'dataset_overview.png'}")

    print("Building PPI graph figure (static)...")
    make_ppi_figure(dataset, X, fig_dir / "dataset_ppi_graph.png")
    print(f"  saved: {fig_dir / 'dataset_ppi_graph.png'}")

    print("Building PPI graph (interactive)...")

    site_dir = Path(args.site_dir) if args.site_dir else (
        Path(__file__).resolve().parent.parent / "site")
    site_dir.mkdir(parents=True, exist_ok=True)

    # Always copy figures next to the page — keeps paths self-contained
    # regardless of where the site is published.
    import shutil
    for fname in ("dataset_overview.png", "dataset_ppi_graph.png"):
        shutil.copyfile(fig_dir / fname, site_dir / fname)
    prefix = ""

    n_rendered = make_ppi_interactive(dataset, X,
                                      site_dir / "dataset_ppi_graph.html")
    print(f"  saved: {site_dir / 'dataset_ppi_graph.html'} ({n_rendered} nodes)")

    out_html = site_dir / "dataset.html"
    write_html(stats, out_html, run_dir.name, figures_rel_prefix=prefix)
    print(f"Saved: {out_html}")


if __name__ == "__main__":
    main()
