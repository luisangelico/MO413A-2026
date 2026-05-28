#!/usr/bin/env python3
"""
Interactive visualization of the TCGA melanoma graph dataset using Plotly.
Opens in your web browser with zoom, pan, and hover capabilities.
"""

import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import numpy as np
from pathlib import Path

from config import DATASET_FILE

# Load dataset
print("Loading dataset...")
dataset = torch.load(DATASET_FILE, weights_only=False)

print(f"\nDataset loaded: {len(dataset)} patients")
print("Creating interactive visualizations...")

# Get data
labels = [data.y.item() for data in dataset]
class_names = {0: 'Primary Tumor', 1: 'Metastasis', 2: 'Normal Tissue'}
class_counts = [labels.count(i) for i in [0, 1, 2]]

# Sample patient for network viz
sample = dataset[0]
expression = sample.x.squeeze().numpy()

# Create subplots
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Class Distribution',
                    'Gene Expression Distribution',
                    'Protein-Protein Interaction Network',
                    'Network Degree Distribution'),
    specs=[[{'type': 'pie'}, {'type': 'histogram'}],
           [{'type': 'scatter'}, {'type': 'histogram'}]],
    row_heights=[0.5, 0.5],
    vertical_spacing=0.12,
    horizontal_spacing=0.1
)

# 1. Class distribution pie chart
colors = ['#ff6b6b', '#4ecdc4', '#95e1d3']
fig.add_trace(
    go.Pie(
        labels=[class_names[i] for i in [0, 1, 2]],
        values=class_counts,
        marker=dict(colors=colors),
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>',
        textposition='inside',
        textinfo='percent+label'
    ),
    row=1, col=1
)

# 2. Gene expression histogram
fig.add_trace(
    go.Histogram(
        x=expression,
        nbinsx=50,
        marker=dict(color='steelblue', line=dict(color='black', width=1)),
        hovertemplate='Expression: %{x:.3f}<br>Count: %{y}<extra></extra>',
        name='Expression'
    ),
    row=1, col=2
)

# 3. Network graph visualization
G = nx.Graph()
edge_list = sample.edge_index.t().numpy()
G.add_edges_from(edge_list)

# Use spring layout
pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

# Create edge traces
edge_x = []
edge_y = []
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

fig.add_trace(
    go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(width=0.5, color='rgba(125,125,125,0.3)'),
        hoverinfo='none',
        showlegend=False
    ),
    row=2, col=1
)

# Create node traces with gene names
node_x = []
node_y = []
node_colors = []
node_text = []

for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    node_colors.append(expression[node])
    gene_name = sample.gene_names[node] if hasattr(sample, 'gene_names') else f"Gene {node}"
    node_text.append(f"<b>{gene_name}</b><br>Expression: {expression[node]:.3f}<br>Degree: {G.degree(node)}")

fig.add_trace(
    go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        marker=dict(
            size=8,
            color=node_colors,
            colorscale='Viridis',
            colorbar=dict(
                title="Expression",
                x=0.46,
                len=0.4,
                y=0.22
            ),
            line=dict(width=0.5, color='white')
        ),
        text=node_text,
        hovertemplate='%{text}<extra></extra>',
        showlegend=False
    ),
    row=2, col=1
)

# 4. Degree distribution
degrees = [G.degree(n) for n in G.nodes()]
fig.add_trace(
    go.Histogram(
        x=degrees,
        nbinsx=20,
        marker=dict(color='coral', line=dict(color='black', width=1)),
        hovertemplate='Degree: %{x}<br>Count: %{y}<extra></extra>',
        name='Degree'
    ),
    row=2, col=2
)

# Update layout
fig.update_xaxes(title_text="Log Expression", row=1, col=2)
fig.update_yaxes(title_text="Frequency", row=1, col=2)

fig.update_xaxes(showticklabels=False, showgrid=False, row=2, col=1)
fig.update_yaxes(showticklabels=False, showgrid=False, row=2, col=1)

fig.update_xaxes(title_text="Node Degree", row=2, col=2)
fig.update_yaxes(title_text="Frequency", row=2, col=2)

fig.update_layout(
    title=dict(
        text=f'<b>TCGA Melanoma Dataset - Interactive Visualization</b><br>'
             f'<sub>Sample Patient: {sample.paciente_id} | {len(dataset)} total patients</sub>',
        x=0.5,
        xanchor='center',
        font=dict(size=20)
    ),
    height=900,
    showlegend=False,
    hovermode='closest',
    plot_bgcolor='white'
)

# Save and open
output_file = 'interactive_visualization.html'
fig.write_html(output_file)
print(f"\n✓ Interactive visualization saved as: {output_file}")
print("\nOpening in your default web browser...")
print("\nFeatures:")
print("  • Hover over elements to see details")
print("  • Zoom and pan on the network graph")
print("  • Click and drag to explore")
print("  • Double-click to reset view")

import webbrowser
webbrowser.open('file://' + str(Path(output_file).absolute()))
