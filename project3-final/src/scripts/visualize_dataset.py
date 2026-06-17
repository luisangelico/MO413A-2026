#!/usr/bin/env python3
"""
Static dataset overview plot.

Produces a 4-panel PNG summarizing the active dataset (DATASET_FILE in
src.config):
  1. Class balance
  2. Expression-value distribution per class (KDE)
  3. PCA of per-sample mean expression vectors, colored by class
  4. Graph stats (n_nodes, n_edges, avg degree)

Usage:
    python -m scripts.visualize_dataset
"""

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from sklearn.decomposition import PCA

from src.config import CLASS_NAMES, DATASET_FILE

OUT_PATH = Path("dataset_overview.png")
GRAPH_OUT_PATH = Path("dataset_ppi_graph.png")
COLORS = {0: "#d62728", 1: "#ff7f0e", 2: "#2ca02c"}

print(f"Loading {DATASET_FILE}...")
dataset = torch.load(DATASET_FILE, weights_only=False)
print(f"  {len(dataset)} samples")

labels = np.array([int(d.y.item()) for d in dataset])
classes = sorted(set(labels.tolist()))
class_label_names = [CLASS_NAMES[c] for c in classes]

# Per-sample expression vector (flattened mean across nodes for PCA)
X = np.stack([d.x.squeeze().numpy() for d in dataset])  # (N, num_genes)

# Graph stats (assumes shared edge_index across samples — true for this pipeline)
sample0 = dataset[0]
n_nodes = sample0.x.shape[0]
n_edges = sample0.edge_index.shape[1]
deg = np.bincount(sample0.edge_index[0].numpy(), minlength=n_nodes)

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle(f"Dataset overview — {DATASET_FILE.name}", fontsize=14, fontweight="bold")

# 1. Class balance
ax = axes[0, 0]
counts = Counter(labels.tolist())
bars = ax.bar(
    [CLASS_NAMES[c] for c in classes],
    [counts[c] for c in classes],
    color=[COLORS[c] for c in classes],
    edgecolor="black",
)
for b, c in zip(bars, classes):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height(), str(counts[c]),
            ha="center", va="bottom", fontweight="bold")
ax.set_title("Class balance")
ax.set_ylabel("Number of samples")

# 2. Expression distribution per class (KDE on flattened values)
ax = axes[0, 1]
for c in classes:
    vals = X[labels == c].ravel()
    # subsample for speed
    if len(vals) > 50000:
        vals = np.random.default_rng(0).choice(vals, 50000, replace=False)
    ax.hist(
        vals, bins=80, density=True, alpha=0.45,
        label=CLASS_NAMES[c], color=COLORS[c],
    )
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
    ax.scatter(
        Z[mask, 0], Z[mask, 1],
        c=COLORS[c], label=CLASS_NAMES[c], alpha=0.6, s=18, edgecolor="black", linewidth=0.3,
    )
ax.set_title(f"PCA of expression  (var explained: {pca.explained_variance_ratio_.sum()*100:.1f}%)")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
ax.legend()

# 4. Graph stats
ax = axes[1, 1]
ax.hist(deg, bins=40, color="steelblue", edgecolor="black")
ax.set_title("Node degree distribution (PPI graph)")
ax.set_xlabel("Degree")
ax.set_ylabel("Number of genes")
stats = (
    f"genes:           {n_nodes}\n"
    f"edges (directed): {n_edges}\n"
    f"avg degree:       {deg.mean():.2f}\n"
    f"max degree:       {deg.max()}\n"
    f"isolated genes:   {int((deg == 0).sum())}"
)
ax.text(
    0.97, 0.97, stats, transform=ax.transAxes, ha="right", va="top",
    family="monospace", fontsize=9,
    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray"),
)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved: {OUT_PATH}")

# ----------------------------------------------------------------------------
# Standalone PPI graph figure (shared across all samples)
# ----------------------------------------------------------------------------
print("Building PPI graph figure...")
gene_names = getattr(sample0, "gene_names", [str(i) for i in range(n_nodes)])
mean_expr = X.mean(axis=0)

G = nx.Graph()
G.add_nodes_from(range(n_nodes))
edges_np = sample0.edge_index.numpy()
for k in range(edges_np.shape[1]):
    s, d = int(edges_np[0, k]), int(edges_np[1, k])
    if s != d:
        G.add_edge(s, d)

# Drop isolated nodes for legibility
isolated = [n for n in G.nodes() if G.degree(n) == 0]
G.remove_nodes_from(isolated)

fig2, ax2 = plt.subplots(figsize=(12, 12))
if len(G) == 0:
    ax2.text(0.5, 0.5, "No edges in PPI graph", ha="center", va="center")
else:
    pos = nx.spring_layout(G, k=0.3, iterations=80, seed=42)
    nodes = list(G.nodes())
    degrees = np.array([G.degree(n) for n in nodes])
    node_expr = np.array([mean_expr[n] for n in nodes])

    nx.draw_networkx_edges(G, pos, ax=ax2, alpha=0.25, width=0.6, edge_color="gray")
    nc = nx.draw_networkx_nodes(
        G, pos, nodelist=nodes, ax=ax2,
        node_size=20 + 8 * degrees,
        node_color=node_expr, cmap="viridis",
        edgecolors="black", linewidths=0.3,
    )

    # Label only top-degree hubs
    top_hub_idx = np.argsort(degrees)[::-1][:20]
    labels = {nodes[i]: gene_names[nodes[i]] for i in top_hub_idx}
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax2, font_size=8)

    plt.colorbar(nc, ax=ax2, shrink=0.6, label="Mean expression")
    ax2.set_title(
        f"PPI graph — {len(G)} connected genes / {n_nodes} total · "
        f"{G.number_of_edges()} edges (top 20 hubs labeled)"
    )
    ax2.axis("off")

plt.tight_layout()
plt.savefig(GRAPH_OUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved: {GRAPH_OUT_PATH}")
