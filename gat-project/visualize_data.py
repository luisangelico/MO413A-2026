#!/usr/bin/env python3
"""
Quick visualization of the TCGA melanoma graph dataset.
"""

import torch
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pathlib import Path

from config import DATASET_FILE

# Load dataset
print("Loading dataset...")
dataset = torch.load(DATASET_FILE, weights_only=False)

print(f"\n{'='*60}")
print(f"Dataset Overview")
print(f"{'='*60}")
print(f"Total patients: {len(dataset)}")

# Class distribution
labels = [data.y.item() for data in dataset]
class_names = {0: 'Primary Tumor', 1: 'Metastasis', 2: 'Normal Tissue'}
class_counts = {label: labels.count(label) for label in [0, 1, 2]}

print(f"\nClass Distribution:")
for label, name in class_names.items():
    print(f"  {name}: {class_counts[label]} ({100*class_counts[label]/len(dataset):.1f}%)")

# Sample patient info
sample = dataset[0]
print(f"\nGraph Structure (per patient):")
print(f"  Nodes (genes): {sample.x.size(0)}")
print(f"  Node features: {sample.x.size(1)}")
print(f"  Edges (interactions): {sample.edge_index.size(1)}")
print(f"  Gene names available: {hasattr(sample, 'gene_names')}")

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('TCGA Melanoma Dataset Visualization', fontsize=16, fontweight='bold')

# 1. Class distribution pie chart
ax = axes[0, 0]
colors = ['#ff6b6b', '#4ecdc4', '#95e1d3']
ax.pie([class_counts[i] for i in [0, 1, 2]],
       labels=[class_names[i] for i in [0, 1, 2]],
       autopct='%1.1f%%',
       colors=colors,
       startangle=90)
ax.set_title('Sample Type Distribution', fontweight='bold')

# 2. Gene expression distribution (first patient)
ax = axes[0, 1]
expression = sample.x.squeeze().numpy()
ax.hist(expression, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
ax.set_xlabel('Log-transformed Expression', fontweight='bold')
ax.set_ylabel('Frequency', fontweight='bold')
ax.set_title(f'Gene Expression Distribution\n(Patient: {sample.paciente_id})', fontweight='bold')
ax.grid(True, alpha=0.3)

# 3. Network graph visualization (subset)
ax = axes[1, 0]
G = nx.Graph()
edge_list = sample.edge_index.t().numpy()
# Use only first 100 nodes for visualization clarity
subset_size = min(100, sample.x.size(0))
subset_edges = [(u, v) for u, v in edge_list if u < subset_size and v < subset_size]
G.add_edges_from(subset_edges)

# Node colors by expression level (only for nodes actually in the graph)
nodes_in_graph = list(G.nodes())
node_colors = [expression[n] for n in nodes_in_graph]
pos = nx.spring_layout(G, k=0.3, seed=42)

nx.draw_networkx_nodes(G, pos,
                       node_color=node_colors,
                       node_size=50,
                       cmap='viridis',
                       ax=ax)
nx.draw_networkx_edges(G, pos,
                       alpha=0.3,
                       width=0.5,
                       ax=ax)

ax.set_title(f'PPI Network (First {subset_size} genes)', fontweight='bold')
ax.axis('off')

# Add colorbar
if len(node_colors) > 0:
    sm = plt.cm.ScalarMappable(cmap='viridis',
                               norm=plt.Normalize(vmin=min(node_colors),
                                                vmax=max(node_colors)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Expression Level', fontweight='bold')

# 4. Degree distribution
ax = axes[1, 1]
degrees = [G.degree(n) for n in G.nodes()]
ax.hist(degrees, bins=20, color='coral', edgecolor='black', alpha=0.7)
ax.set_xlabel('Node Degree', fontweight='bold')
ax.set_ylabel('Frequency', fontweight='bold')
ax.set_title('Network Degree Distribution', fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('data_visualization.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Visualization saved as: data_visualization.png")

# Show top expressed genes
if hasattr(sample, 'gene_names'):
    print(f"\nTop 10 Most Expressed Genes in Sample Patient:")
    expr_with_names = list(zip(sample.gene_names, expression))
    expr_with_names.sort(key=lambda x: x[1], reverse=True)
    for i, (gene, expr) in enumerate(expr_with_names[:10], 1):
        print(f"  {i:2d}. {gene:15s} - Expression: {expr:.3f}")

plt.show()
