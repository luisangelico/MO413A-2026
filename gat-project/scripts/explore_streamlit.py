#!/usr/bin/env python3
"""
Interactive Streamlit app to explore the TCGA melanoma dataset.
Run with: streamlit run explore_data.py
"""

import streamlit as st
import torch
import plotly.graph_objects as go
import networkx as nx
import pandas as pd
import numpy as np
from pathlib import Path

from config import DATASET_FILE

st.set_page_config(page_title="TCGA Melanoma Dataset Explorer", layout="wide")

# Title
st.title("🧬 TCGA Melanoma Dataset Explorer")
st.markdown("Interactive exploration of the Graph Attention Network dataset")

# Load dataset
@st.cache_resource
def load_dataset():
    return torch.load(DATASET_FILE, weights_only=False)

dataset = load_dataset()

# Sidebar - Dataset statistics
st.sidebar.header("📊 Dataset Overview")
labels = [data.y.item() for data in dataset]
class_names = {0: 'Primary Tumor', 1: 'Metastasis', 2: 'Normal Tissue'}
class_counts = {label: labels.count(label) for label in [0, 1, 2]}

st.sidebar.metric("Total Patients", len(dataset))
st.sidebar.metric("Genes per Patient", dataset[0].x.size(0))
st.sidebar.metric("PPI Interactions", dataset[0].edge_index.size(1))

st.sidebar.markdown("---")
st.sidebar.markdown("**Class Distribution:**")
for label, name in class_names.items():
    pct = 100 * class_counts[label] / len(dataset)
    st.sidebar.markdown(f"- {name}: **{class_counts[label]}** ({pct:.1f}%)")

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs(["📈 Statistics", "🕸️ Network View", "👤 Patient Browser", "🧪 Gene Analysis"])

# Tab 1: Statistics
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Class Distribution")
        fig_pie = go.Figure(data=[go.Pie(
            labels=[class_names[i] for i in [0, 1, 2]],
            values=[class_counts[i] for i in [0, 1, 2]],
            marker=dict(colors=['#ff6b6b', '#4ecdc4', '#95e1d3']),
            hole=0.3
        )])
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.subheader("Network Statistics")
        sample = dataset[0]
        G = nx.Graph()
        G.add_edges_from(sample.edge_index.t().numpy())

        # Calculate network metrics
        metrics = {
            "Nodes": len(G.nodes()),
            "Edges": len(G.edges()),
            "Avg Degree": f"{sum(dict(G.degree()).values()) / len(G.nodes()):.2f}",
            "Density": f"{nx.density(G):.4f}",
            "Connected Components": nx.number_connected_components(G)
        }

        for metric, value in metrics.items():
            st.metric(metric, value)

# Tab 2: Network View
with tab2:
    st.subheader("Protein-Protein Interaction Network")

    # Select patient
    selected_idx = st.slider("Select Patient Index", 0, len(dataset)-1, 0)
    patient_data = dataset[selected_idx]

    col1, col2 = st.columns([3, 1])

    with col2:
        st.markdown(f"**Patient ID:** {patient_data.paciente_id}")
        st.markdown(f"**Class:** {class_names[patient_data.y.item()]}")

        # Layout options
        layout_type = st.selectbox("Layout Algorithm",
                                   ["Spring", "Circular", "Kamada-Kawai"])
        node_size = st.slider("Node Size", 2, 20, 8)
        show_labels = st.checkbox("Show Gene Labels", False)

    with col1:
        # Build graph
        G = nx.Graph()
        G.add_edges_from(patient_data.edge_index.t().numpy())

        # Choose layout
        if layout_type == "Spring":
            pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
        elif layout_type == "Circular":
            pos = nx.circular_layout(G)
        else:
            pos = nx.kamada_kawai_layout(G)

        # Create plotly figure
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        expression = patient_data.x.squeeze().numpy()
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]

        fig = go.Figure()

        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(width=0.5, color='rgba(125,125,125,0.3)'),
            hoverinfo='none',
            showlegend=False
        ))

        # Add nodes
        node_text = []
        for node in G.nodes():
            gene_name = patient_data.gene_names[node] if hasattr(patient_data, 'gene_names') else f"Gene {node}"
            node_text.append(f"{gene_name}<br>Expr: {expression[node]:.3f}<br>Degree: {G.degree(node)}")

        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text' if show_labels else 'markers',
            marker=dict(
                size=node_size,
                color=[expression[node] for node in G.nodes()],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Expression")
            ),
            text=[patient_data.gene_names[node][:6] if hasattr(patient_data, 'gene_names') else "" for node in G.nodes()] if show_labels else None,
            textposition="top center",
            hovertext=node_text,
            hoverinfo='text',
            showlegend=False
        ))

        fig.update_layout(
            height=600,
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            hovermode='closest',
            plot_bgcolor='white'
        )

        st.plotly_chart(fig, use_container_width=True)

# Tab 3: Patient Browser
with tab3:
    st.subheader("Browse Individual Patients")

    # Create patient table
    patient_info = []
    for idx, data in enumerate(dataset):
        patient_info.append({
            'Index': idx,
            'Patient ID': data.paciente_id,
            'Class': class_names[data.y.item()],
            'Mean Expression': float(data.x.mean()),
            'Max Expression': float(data.x.max())
        })

    df = pd.DataFrame(patient_info)

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        class_filter = st.multiselect("Filter by Class",
                                     options=list(class_names.values()),
                                     default=list(class_names.values()))
    with col2:
        expr_range = st.slider("Mean Expression Range",
                              float(df['Mean Expression'].min()),
                              float(df['Mean Expression'].max()),
                              (float(df['Mean Expression'].min()),
                               float(df['Mean Expression'].max())))

    # Apply filters
    filtered_df = df[
        (df['Class'].isin(class_filter)) &
        (df['Mean Expression'] >= expr_range[0]) &
        (df['Mean Expression'] <= expr_range[1])
    ]

    st.dataframe(filtered_df, use_container_width=True, height=400)
    st.caption(f"Showing {len(filtered_df)} of {len(df)} patients")

# Tab 4: Gene Analysis
with tab4:
    st.subheader("Gene Expression Analysis")

    # Get all gene names
    sample = dataset[0]
    if hasattr(sample, 'gene_names'):
        gene_names = sample.gene_names

        col1, col2 = st.columns([1, 2])

        with col1:
            selected_gene_idx = st.selectbox(
                "Select Gene",
                range(len(gene_names)),
                format_func=lambda x: f"{gene_names[x]} ({x})"
            )

            st.markdown("---")
            st.markdown("**Gene Information:**")
            st.markdown(f"- **Name:** {gene_names[selected_gene_idx]}")
            st.markdown(f"- **Index:** {selected_gene_idx}")

            # Calculate statistics
            gene_expressions = [data.x[selected_gene_idx].item() for data in dataset]
            st.markdown(f"- **Mean Expr:** {np.mean(gene_expressions):.3f}")
            st.markdown(f"- **Std Dev:** {np.std(gene_expressions):.3f}")

        with col2:
            # Expression distribution across all patients
            fig = go.Figure()

            for class_label, class_name in class_names.items():
                class_data = [data.x[selected_gene_idx].item()
                            for data in dataset if data.y.item() == class_label]

                fig.add_trace(go.Box(
                    y=class_data,
                    name=class_name,
                    boxmean='sd'
                ))

            fig.update_layout(
                title=f"Expression of {gene_names[selected_gene_idx]} Across Sample Types",
                yaxis_title="Log Expression",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Gene names not available in dataset")

# Footer
st.markdown("---")
st.markdown("💡 **Tip:** After training the model, use `app.py` to visualize attention weights!")
