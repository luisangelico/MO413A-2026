#!/usr/bin/env python3
"""
Interactive Streamlit app for inspecting trained-model predictions and the
GATv2 attention weights on the PPI graph.

Run with:
    streamlit run scripts/predict_streamlit.py
"""

import sys
from pathlib import Path

# Streamlit runs this script as a top-level program, not as a package module,
# so make sure *this* project's root wins over any other "src" packages on
# sys.path (e.g. unrelated projects in site-packages or other working dirs).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import networkx as nx
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

from src.config import CLASS_NAMES, DATASET_FILE, PROCESSED_DATASET_PATH, get_device
from src.model import load_model

st.set_page_config(page_title="GAT Predictor", layout="wide")
st.title("🧬 GAT Predictor — attention-weighted predictions")

# ----------------------------------------------------------------------------
# Sidebar: model + dataset selection
# ----------------------------------------------------------------------------
st.sidebar.header("Model & data")

run_dirs = sorted(PROCESSED_DATASET_PATH.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
if not run_dirs:
    st.error(f"No trained model runs found in {PROCESSED_DATASET_PATH}/run_*/. Train a model first.")
    st.stop()

run_label = st.sidebar.selectbox(
    "Training run", run_dirs, index=0, format_func=lambda p: p.name
)
weights_choice = st.sidebar.radio("Checkpoint", ["best_model.pt", "modelo_final.pt"], index=0)
weights_path = run_label / weights_choice
if not weights_path.exists():
    st.error(f"{weights_path} does not exist.")
    st.stop()

st.sidebar.caption(f"Dataset: `{DATASET_FILE.name}`")
st.sidebar.caption(f"Weights: `{weights_path.relative_to(PROCESSED_DATASET_PATH)}`")


@st.cache_resource
def load_data_and_model(weights_path_str):
    device = get_device()
    dataset = torch.load(DATASET_FILE, weights_only=False)
    model, config = load_model(Path(weights_path_str), device=device)
    return dataset, model, config, device


dataset, model, config, device = load_data_and_model(str(weights_path))
num_classes = config["num_classes"]
class_names = {i: CLASS_NAMES[i] for i in range(num_classes)}

# Load splits if available so we can mark train/val/test membership
splits_file = run_label / "splits.npz"
sample_split = {}
if splits_file.exists():
    splits = np.load(splits_file)
    for k in ("train", "val", "test"):
        for i in splits[k].tolist():
            sample_split[int(i)] = k

# If model is binary, mirror the train-time filter
if num_classes == 2:
    valid_indices = [i for i, d in enumerate(dataset) if int(d.y.item()) != 2]
else:
    valid_indices = list(range(len(dataset)))

st.sidebar.markdown("---")
st.sidebar.metric("Total samples", len(valid_indices))
st.sidebar.metric("Classes", num_classes)
st.sidebar.metric("Device", str(device))

# ----------------------------------------------------------------------------
# Sample selection
# ----------------------------------------------------------------------------
st.sidebar.header("Sample")
filter_class = st.sidebar.multiselect(
    "Filter by true class",
    options=list(class_names.values()),
    default=list(class_names.values()),
)
filter_split = st.sidebar.multiselect(
    "Filter by split",
    options=["train", "val", "test", "(none)"],
    default=["test"] if sample_split else ["(none)"],
)


def passes_filters(idx):
    cls = class_names[int(dataset[idx].y.item())]
    if cls not in filter_class:
        return False
    sp = sample_split.get(idx, "(none)")
    if sp not in filter_split:
        return False
    return True


candidate_indices = [i for i in valid_indices if passes_filters(i)]
if not candidate_indices:
    st.warning("No samples match the current filters.")
    st.stop()

selected_idx = st.sidebar.selectbox(
    "Sample",
    candidate_indices,
    format_func=lambda i: f"{i}: {getattr(dataset[i], 'paciente_id', '?')[:24]} "
    f"[{class_names[int(dataset[i].y.item())]}]"
    + (f" ({sample_split[i]})" if i in sample_split else ""),
)

# ----------------------------------------------------------------------------
# Predict
# ----------------------------------------------------------------------------
sample = dataset[selected_idx].to(device)
batch = torch.zeros(sample.x.size(0), dtype=torch.long, device=device)

with torch.no_grad():
    logits, edge_index_attn, alpha = model(sample.x, sample.edge_index, batch, return_attn=True)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_class = int(probs.argmax())

true_class = int(sample.y.item())
attn = alpha.mean(dim=1).cpu().numpy()
edges_attn = edge_index_attn.cpu().numpy()
expression = sample.x.squeeze().cpu().numpy()
gene_names = getattr(sample, "gene_names", [str(i) for i in range(len(expression))])

# ----------------------------------------------------------------------------
# Header: prediction summary
# ----------------------------------------------------------------------------
correct = pred_class == true_class
emoji = "✅" if correct else "❌"
top_col1, top_col2, top_col3 = st.columns([2, 2, 3])
top_col1.metric("True class", class_names[true_class])
top_col2.metric(f"Predicted {emoji}", class_names[pred_class], f"{probs[pred_class]*100:.1f}%")
with top_col3:
    st.markdown("**Class probabilities**")
    fig_probs = go.Figure(
        go.Bar(
            x=[class_names[i] for i in range(num_classes)],
            y=probs * 100,
            marker_color=["#d62728", "#ff7f0e", "#2ca02c"][:num_classes],
            text=[f"{p*100:.1f}%" for p in probs],
            textposition="auto",
        )
    )
    fig_probs.update_layout(height=180, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="%")
    st.plotly_chart(fig_probs, use_container_width=True)

# ----------------------------------------------------------------------------
# Attention graph
# ----------------------------------------------------------------------------
st.subheader("Attention-weighted PPI graph")

ctrl1, ctrl2, ctrl3, ctrl4 = st.columns(4)
hide_self_loops = ctrl1.checkbox("Hide self-loops", value=True)
attn_threshold = ctrl2.slider("Min attention", 0.0, 1.0, 0.0, 0.01)
hide_isolated = ctrl3.checkbox("Hide isolated nodes", value=True)
show_labels = ctrl4.checkbox("Show top-attention labels", value=True)

# Build a NetworkX graph from the attention edges
G = nx.Graph()
edge_records = []
for k in range(edges_attn.shape[1]):
    src, dst = int(edges_attn[0, k]), int(edges_attn[1, k])
    a = float(attn[k])
    if hide_self_loops and src == dst:
        continue
    if a < attn_threshold:
        continue
    edge_records.append((src, dst, a))
    G.add_edge(src, dst, weight=a)

# Add isolated nodes if requested
if not hide_isolated:
    for node in range(len(expression)):
        if node not in G:
            G.add_node(node)

if len(G) == 0:
    st.info("No edges pass the current filters. Lower the threshold.")
else:
    # Layout — spring is decent and seeded for stability
    pos = nx.spring_layout(G, k=0.5, iterations=60, seed=42, weight="weight")

    # Edge traces (one trace per edge so we can color by attention)
    max_attn = max(a for _, _, a in edge_records) if edge_records else 1.0
    edge_x, edge_y, edge_color = [], [], []
    edge_traces = []
    for src, dst, a in edge_records:
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        # Map attention -> color (orange→red) and width
        norm = a / max_attn if max_attn else 0
        color = f"rgba({int(200 + 55*norm)},{int(120 - 100*norm)},0,{0.3 + 0.7*norm:.2f})"
        edge_traces.append(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(width=0.5 + 4 * norm, color=color),
                hoverinfo="text",
                hovertext=f"{gene_names[src]} ↔ {gene_names[dst]}<br>attn = {a:.4f}",
                showlegend=False,
            )
        )

    # Node trace
    nodes = list(G.nodes())
    node_x = [pos[n][0] for n in nodes]
    node_y = [pos[n][1] for n in nodes]
    node_expr = [expression[n] for n in nodes]
    node_text = [
        f"{gene_names[n]}<br>expr = {expression[n]:.3f}<br>degree = {G.degree(n)}"
        for n in nodes
    ]

    # Top-attention gene labels (just the genes incident to the top-K edges)
    label_text = [""] * len(nodes)
    if show_labels and edge_records:
        sorted_edges = sorted(edge_records, key=lambda r: r[2], reverse=True)
        top_genes = set()
        for src, dst, _ in sorted_edges[:8]:
            top_genes.add(src)
            top_genes.add(dst)
        for i, n in enumerate(nodes):
            if n in top_genes:
                label_text[i] = gene_names[n]

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        marker=dict(
            size=10,
            color=node_expr,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Expression"),
            line=dict(width=0.5, color="black"),
        ),
        text=label_text,
        textposition="top center",
        textfont=dict(size=10),
        hovertext=node_text,
        hoverinfo="text",
        showlegend=False,
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        height=620,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="white",
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
    )
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------
# Top tables
# ----------------------------------------------------------------------------
left, right = st.columns(2)
with left:
    st.subheader("Top attention edges")
    rows = []
    for src, dst, a in sorted(edge_records, key=lambda r: r[2], reverse=True)[:20]:
        rows.append(
            {
                "gene_a": gene_names[src],
                "gene_b": gene_names[dst],
                "attention": round(a, 4),
                "self_loop": src == dst,
            }
        )
    if rows:
        st.dataframe(rows, use_container_width=True, height=420)
    else:
        st.info("No edges to display.")

with right:
    st.subheader("Top expressed genes (this sample)")
    order = np.argsort(expression)[::-1][:20]
    st.dataframe(
        [
            {"gene": gene_names[i], "expression": round(float(expression[i]), 3)}
            for i in order
        ],
        use_container_width=True,
        height=420,
    )
