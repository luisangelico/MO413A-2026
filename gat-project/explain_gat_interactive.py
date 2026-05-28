#!/usr/bin/env python3
"""
Interactive explanation of Graph Attention Networks (GAT)
Run with: streamlit run explain_gat_interactive.py
"""

import streamlit as st
import plotly.graph_objects as go
import networkx as nx
import numpy as np
import torch
from pathlib import Path

st.set_page_config(page_title="GAT Interactive Tutorial", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for better styling
st.markdown("""
<style>
.big-font {
    font-size:20px !important;
    font-weight: bold;
}
.highlight {
    background-color: #ffeb3b;
    padding: 10px;
    border-radius: 5px;
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("🧠 Interactive Guide to Graph Attention Networks")
st.markdown("### Understanding how GAT learns from biological networks")

# Sidebar navigation
st.sidebar.title("📚 Tutorial Navigation")
section = st.sidebar.radio(
    "Choose a topic:",
    ["🏠 Introduction",
     "📊 Traditional vs. Graph ML",
     "🔍 Attention Mechanism",
     "🎯 Multi-Head Attention",
     "🧬 Your Melanoma Data",
     "📈 Training Process",
     "💡 Real Example"]
)

# Load dataset for real examples
@st.cache_resource
def load_dataset():
    try:
        from config import DATASET_FILE
        return torch.load(DATASET_FILE, weights_only=False)
    except:
        return None

dataset = load_dataset()

# ============================================================================
# Section: Introduction
# ============================================================================
if section == "🏠 Introduction":
    st.header("What is a Graph Attention Network?")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### The Problem

        You have **cancer gene expression data**, but genes don't work in isolation - they form complex networks!

        **Traditional Machine Learning:**
        - Treats each gene independently
        - Ignores biological interactions
        - Misses the "big picture"

        **Graph Attention Networks:**
        - Models genes as a network (graph)
        - Learns which interactions matter most
        - Captures biological context
        """)

        st.info("💡 **Key Insight**: GAT learns to **pay attention** to the most important gene interactions for classification.")

    with col2:
        st.markdown("### Quick Stats")
        if dataset:
            st.metric("Patients in Dataset", len(dataset))
            st.metric("Genes per Patient", dataset[0].x.size(0))
            st.metric("Protein Interactions", dataset[0].edge_index.size(1))
        else:
            st.warning("Load dataset to see stats")

    st.markdown("---")
    st.markdown("### 🎓 What You'll Learn")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**🔍 Attention**\n\nHow the model decides which connections are important")
    with col2:
        st.markdown("**🎯 Multi-Head**\n\nWhy using multiple 'experts' helps")
    with col3:
        st.markdown("**🧬 Biology**\n\nHow this relates to real cancer mechanisms")

# ============================================================================
# Section: Traditional vs Graph ML
# ============================================================================
elif section == "📊 Traditional vs. Graph ML":
    st.header("Traditional ML vs. Graph Neural Networks")

    tab1, tab2 = st.tabs(["📉 Traditional Approach", "📈 Graph Approach"])

    with tab1:
        st.subheader("Traditional Machine Learning")

        # Create example data
        st.markdown("### Example: 5 Genes")

        gene_data = {
            'Gene': ['BRAF', 'NRAS', 'CDKN2A', 'TP53', 'PTEN'],
            'Expression': [7.2, 5.8, 3.1, 8.5, 4.2],
            'Traditional ML': ['⚠️ Treats independently'] * 5
        }

        st.dataframe(gene_data, use_container_width=True)

        st.markdown("""
        **How Traditional ML Works:**

        1. Takes gene expression as a **flat vector**: `[7.2, 5.8, 3.1, 8.5, 4.2]`
        2. Feeds into neural network
        3. **Problem**: Ignores that BRAF → NRAS is a known pathway!

        ```python
        # Traditional approach
        X = [gene1_expr, gene2_expr, gene3_expr, ...]
        prediction = neural_network(X)
        ```

        ❌ **Missing**: Biological relationships between genes
        """)

    with tab2:
        st.subheader("Graph Neural Network Approach")

        st.markdown("### Same 5 Genes + Interactions")

        # Create interactive graph
        G = nx.Graph()
        genes = ['BRAF', 'NRAS', 'CDKN2A', 'TP53', 'PTEN']
        expressions = [7.2, 5.8, 3.1, 8.5, 4.2]

        # Add nodes with expression
        for i, gene in enumerate(genes):
            G.add_node(gene, expr=expressions[i])

        # Add edges (known pathway interactions)
        edges = [('BRAF', 'NRAS'), ('BRAF', 'TP53'), ('CDKN2A', 'TP53'),
                 ('TP53', 'PTEN'), ('NRAS', 'PTEN')]
        G.add_edges_from(edges)

        # Layout
        pos = nx.spring_layout(G, seed=42)

        # Create plotly figure
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        fig = go.Figure()

        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(width=2, color='#888'),
            hoverinfo='none',
            showlegend=False
        ))

        # Add nodes
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        node_colors = [G.nodes[node]['expr'] for node in G.nodes()]

        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(
                size=40,
                color=node_colors,
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="Expression"),
                line=dict(width=2, color='black')
            ),
            text=list(G.nodes()),
            textposition="middle center",
            textfont=dict(size=10, color='white', family='Arial Black'),
            hovertext=[f"{gene}<br>Expression: {G.nodes[gene]['expr']}" for gene in G.nodes()],
            hoverinfo='text',
            showlegend=False
        ))

        fig.update_layout(
            title="Gene Interaction Network",
            showlegend=False,
            height=400,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        **How Graph ML Works:**

        1. Each gene considers its neighbors' expressions
        2. Learns which connections are important via **attention**
        3. Captures pathway-level patterns

        ```python
        # Graph approach
        for gene in genes:
            neighbors = get_connected_genes(gene)
            aggregated_info = attention_weighted_sum(neighbors)
            gene_embedding = combine(gene.expr, aggregated_info)

        prediction = classify(gene_embeddings)
        ```

        ✅ **Captures**: BRAF-NRAS pathway, TP53 hub role, etc.
        """)

# ============================================================================
# Section: Attention Mechanism
# ============================================================================
elif section == "🔍 Attention Mechanism":
    st.header("The Attention Mechanism Explained")

    st.markdown("""
    ### 🎯 Core Concept: "Who should I listen to?"

    For each gene, the attention mechanism computes:
    > **"How important is each of my protein partners for this classification task?"**
    """)

    st.markdown("---")

    # Interactive example
    st.subheader("🧪 Interactive Example")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("**Choose a central gene:**")
        central_gene = st.selectbox("Central Gene",
            ["S100B (Melanoma marker)", "IGKC (Immune)", "VIM (EMT marker)"])

        st.markdown("**Adjust neighbor expressions:**")
        neighbor1_expr = st.slider("Neighbor 1 Expression", 0.0, 10.0, 5.0, 0.5)
        neighbor2_expr = st.slider("Neighbor 2 Expression", 0.0, 10.0, 7.0, 0.5)
        neighbor3_expr = st.slider("Neighbor 3 Expression", 0.0, 10.0, 3.0, 0.5)

        central_expr = st.slider("Central Gene Expression", 0.0, 10.0, 6.0, 0.5)

    with col2:
        # Calculate simple attention weights (for demonstration)
        expressions = np.array([neighbor1_expr, neighbor2_expr, neighbor3_expr])
        similarity = np.abs(expressions - central_expr)  # Simple similarity
        attention_weights = np.exp(-similarity) / np.sum(np.exp(-similarity))

        # Create visualization
        fig = go.Figure()

        # Central node
        fig.add_trace(go.Scatter(
            x=[0], y=[0],
            mode='markers+text',
            marker=dict(size=60, color='red', line=dict(width=2, color='black')),
            text=[central_gene.split()[0]],
            textposition='middle center',
            textfont=dict(size=12, color='white'),
            hovertext=f"Expression: {central_expr:.1f}",
            showlegend=False
        ))

        # Neighbor nodes with attention-weighted edges
        angles = [0, 120, 240]
        colors = ['#ff6b6b', '#4ecdc4', '#95e1d3']

        for i, (angle, expr, attn) in enumerate(zip(angles, expressions, attention_weights)):
            rad = np.radians(angle)
            x, y = 1.5 * np.cos(rad), 1.5 * np.sin(rad)

            # Edge with thickness proportional to attention
            fig.add_trace(go.Scatter(
                x=[0, x], y=[0, y],
                mode='lines',
                line=dict(width=1 + attn * 15, color=colors[i]),
                hovertext=f"Attention: {attn:.3f}",
                showlegend=False
            ))

            # Neighbor node
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode='markers+text',
                marker=dict(size=40, color=colors[i], line=dict(width=2, color='black')),
                text=[f"N{i+1}"],
                textposition='middle center',
                textfont=dict(size=10, color='white'),
                hovertext=f"Expression: {expr:.1f}<br>Attention: {attn:.3f}",
                showlegend=False
            ))

        fig.update_layout(
            title=f"Attention Weights (thicker line = higher attention)",
            height=400,
            xaxis=dict(range=[-2, 2], showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(range=[-2, 2], showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Computed Attention Weights:")
        for i, attn in enumerate(attention_weights):
            st.progress(float(attn), text=f"Neighbor {i+1}: {attn:.3f} ({attn*100:.1f}%)")

    st.markdown("---")
    st.markdown("""
    ### 📐 The Math (Simplified)

    ```python
    # For each gene and its neighbor:
    attention_score = dot_product(gene_features, neighbor_features)
    attention_weight = softmax(attention_score)  # Normalize to sum to 1

    # Aggregate information
    aggregated = sum(attention_weight * neighbor_features for all neighbors)
    ```

    **Key Points:**
    - Attention weights sum to 1.0 (like percentages)
    - Higher weight = "I care more about this neighbor"
    - Learned automatically during training!
    """)

# ============================================================================
# Section: Multi-Head Attention
# ============================================================================
elif section == "🎯 Multi-Head Attention":
    st.header("Multi-Head Attention: Multiple Perspectives")

    st.markdown("""
    ### 🤔 Why Multiple Heads?

    Imagine you're diagnosing cancer. You might want to check:
    1. **Immune response** - Is the immune system activated?
    2. **Cell proliferation** - Are cells dividing rapidly?
    3. **Melanocyte markers** - Is this melanoma-specific?
    4. **Metastasis signals** - Is it spreading?

    **Multi-head attention = Having multiple "experts" look at different patterns simultaneously**
    """)

    st.markdown("---")

    # Interactive multi-head visualization
    st.subheader("🎨 4-Head Attention in Action")

    num_heads = st.slider("Number of Attention Heads to Show", 1, 4, 4)

    # Create sample network
    G = nx.Graph()
    genes = ['Gene A', 'Gene B', 'Gene C', 'Gene D', 'Gene E']
    G.add_edges_from([('Gene A', 'Gene B'), ('Gene A', 'Gene C'),
                      ('Gene B', 'Gene D'), ('Gene C', 'Gene E')])

    pos = nx.spring_layout(G, seed=42)

    # Create subplots for each head
    cols = st.columns(num_heads)

    head_focus = [
        ("Immune Response", [[0, 1, 0.9], [1, 3, 0.8]]),
        ("Cell Division", [[0, 2, 0.85], [2, 4, 0.9]]),
        ("Melanocyte", [[1, 3, 0.6], [0, 1, 0.7]]),
        ("EMT/Metastasis", [[0, 2, 0.75], [2, 4, 0.95]])
    ]

    for i, col in enumerate(cols):
        with col:
            st.markdown(f"**Head {i+1}**")
            st.caption(head_focus[i][0])

            fig = go.Figure()

            # Add all edges (light gray)
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                fig.add_trace(go.Scatter(
                    x=[x0, x1], y=[y0, y1],
                    mode='lines',
                    line=dict(width=1, color='lightgray'),
                    hoverinfo='none',
                    showlegend=False
                ))

            # Highlight high-attention edges for this head
            for edge_idx, attn in head_focus[i][1]:
                edge = list(G.edges())[edge_idx]
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                fig.add_trace(go.Scatter(
                    x=[x0, x1], y=[y0, y1],
                    mode='lines',
                    line=dict(width=attn*5, color='red'),
                    hoverinfo='none',
                    showlegend=False
                ))

            # Add nodes
            node_x = [pos[n][0] for n in G.nodes()]
            node_y = [pos[n][1] for n in G.nodes()]

            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                marker=dict(size=20, color='steelblue', line=dict(width=2, color='white')),
                showlegend=False
            ))

            fig.update_layout(
                height=250,
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='white'
            )

            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.success("""
    💡 **The Power of Multi-Head Attention:**

    - Each head specializes in different patterns
    - Combined, they capture comprehensive biological context
    - Your model uses 4 heads in the first layer → 4 × 64 = 256 features!
    """)

# ============================================================================
# Section: Your Melanoma Data
# ============================================================================
elif section == "🧬 Your Melanoma Data":
    st.header("How GAT Works on Your Data")

    if not dataset:
        st.error("Dataset not found! Make sure to run the data download script first.")
        st.stop()

    st.success(f"✅ Dataset loaded: {len(dataset)} patients")

    # Show data statistics
    col1, col2, col3 = st.columns(3)

    labels = [data.y.item() for data in dataset]
    class_names = {0: 'Primary', 1: 'Metastasis', 2: 'Normal'}

    with col1:
        primary_count = labels.count(0)
        st.metric("Primary Tumors", primary_count,
                 delta=f"{100*primary_count/len(dataset):.1f}%")

    with col2:
        meta_count = labels.count(1)
        st.metric("Metastatic Tumors", meta_count,
                 delta=f"{100*meta_count/len(dataset):.1f}%")

    with col3:
        normal_count = labels.count(2)
        st.metric("Normal Tissue", normal_count,
                 delta=f"{100*normal_count/len(dataset):.1f}%")

    st.markdown("---")

    # Select a patient to visualize
    st.subheader("🔬 Explore Individual Patients")

    patient_idx = st.selectbox("Select Patient", range(min(20, len(dataset))),
                               format_func=lambda x: f"Patient {x}: {dataset[x].paciente_id}")

    patient = dataset[patient_idx]

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### Patient Info")
        st.markdown(f"**ID:** {patient.paciente_id}")
        st.markdown(f"**Class:** {class_names[patient.y.item()]}")
        st.markdown(f"**Genes:** {patient.x.size(0)}")
        st.markdown(f"**Interactions:** {patient.edge_index.size(1)}")

        # Top expressed genes
        if hasattr(patient, 'gene_names'):
            expression = patient.x.squeeze().numpy()
            top_indices = np.argsort(expression)[-5:][::-1]

            st.markdown("### Top 5 Expressed Genes")
            for idx in top_indices:
                gene = patient.gene_names[idx]
                expr = expression[idx]
                st.markdown(f"- **{gene}**: {expr:.3f}")

    with col2:
        st.markdown("### Network Structure")

        # Build graph
        G = nx.Graph()
        edge_list = patient.edge_index.t().numpy()

        # Sample subset for visualization
        subset_size = min(50, patient.x.size(0))
        subset_edges = [(u, v) for u, v in edge_list if u < subset_size and v < subset_size]
        G.add_edges_from(subset_edges)

        pos = nx.spring_layout(G, k=0.5, seed=42)

        # Create plot
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(width=0.5, color='gray'),
            hoverinfo='none'
        ))

        expression = patient.x.squeeze().numpy()[:subset_size]
        node_x = [pos[n][0] for n in G.nodes()]
        node_y = [pos[n][1] for n in G.nodes()]

        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            marker=dict(
                size=8,
                color=[expression[n] for n in G.nodes()],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Expression")
            ),
            hovertext=[f"Gene {n}<br>Expr: {expression[n]:.2f}" for n in G.nodes()],
            hoverinfo='text'
        ))

        fig.update_layout(
            height=400,
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )

        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Showing first {subset_size} genes")

# ============================================================================
# Section: Training Process
# ============================================================================
elif section == "📈 Training Process":
    st.header("What Happens During Training?")

    st.markdown("""
    ### 🎓 Learning in 3 Steps

    Each epoch, the model goes through:
    """)

    tab1, tab2, tab3 = st.tabs(["1️⃣ Forward Pass", "2️⃣ Calculate Loss", "3️⃣ Update Weights"])

    with tab1:
        st.subheader("Forward Pass: Making Predictions")
        st.markdown("""
        ```python
        # For each patient graph:

        # Layer 1: Multi-head attention
        for gene in genes:
            for head in [1, 2, 3, 4]:
                attention = compute_attention(gene, neighbors)
                aggregated = weighted_sum(neighbors, attention)
            gene_features = combine_all_heads(aggregated)

        # Layer 2: Single attention head
        gene_features = attention_layer_2(gene_features)

        # Pooling: Summarize to patient level
        patient_features = average(all_gene_features)

        # Classification
        probabilities = [P(primary), P(metastasis), P(normal)]
        ```
        """)

        # Visualize with animation
        st.markdown("### Flow Through the Network")

        layers = [
            ("Input", "500 genes × 1 feature (expression)"),
            ("GAT Layer 1", "500 genes × 256 features (4 heads × 64)"),
            ("GAT Layer 2", "500 genes × 64 features"),
            ("Global Pooling", "1 patient × 64 features"),
            ("Output", "3 class probabilities")
        ]

        for i, (layer, desc) in enumerate(layers):
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown(f"**{layer}**")
            with col2:
                st.info(desc)
            if i < len(layers) - 1:
                st.markdown("↓")

    with tab2:
        st.subheader("Calculate Loss: How Wrong Are We?")

        st.markdown("""
        **Cross-Entropy Loss** measures prediction error:

        ```python
        true_label = "Metastasis"  # What it actually is
        predicted = [0.1, 0.85, 0.05]  # [Primary, Metastasis, Normal]

        loss = -log(predicted[true_label])
        loss = -log(0.85) = 0.16  # Lower is better!
        ```
        """)

        # Interactive loss calculator
        st.markdown("### 🧮 Try It Yourself")

        col1, col2 = st.columns(2)

        with col1:
            true_class = st.selectbox("True Class", ["Primary", "Metastasis", "Normal"])
            st.markdown("**Predicted Probabilities:**")
            p_primary = st.slider("Primary", 0.0, 1.0, 0.1, 0.05)
            p_meta = st.slider("Metastasis", 0.0, 1.0, 0.85, 0.05)
            p_normal = 1.0 - p_primary - p_meta
            st.metric("Normal (computed)", f"{p_normal:.2f}")

        with col2:
            # Calculate loss
            probs = {"Primary": p_primary, "Metastasis": p_meta, "Normal": p_normal}
            pred_prob = probs[true_class]

            if pred_prob > 0:
                loss = -np.log(pred_prob)
                st.metric("Cross-Entropy Loss", f"{loss:.4f}")

                if loss < 0.5:
                    st.success("✅ Great prediction! (Loss < 0.5)")
                elif loss < 1.0:
                    st.warning("⚠️ Okay prediction (0.5 < Loss < 1.0)")
                else:
                    st.error("❌ Poor prediction (Loss > 1.0)")

                st.markdown(f"""
                **Interpretation:**
                - Predicted **{true_class}** with {pred_prob*100:.1f}% confidence
                - Loss penalizes confident wrong predictions more
                """)

    with tab3:
        st.subheader("Update Weights: Learn from Mistakes")

        st.markdown("""
        **Backpropagation + Gradient Descent:**

        ```python
        # Calculate how to adjust each weight
        gradients = compute_gradients(loss, model_weights)

        # Update weights (move in direction that reduces loss)
        for weight in model_weights:
            weight -= learning_rate * gradient[weight]
        ```

        **Learning Rate** = How big each step is
        - Too large → Unstable, might overshoot
        - Too small → Slow training
        - Your model: 0.001 (good default)
        """)

        # Visualize gradient descent
        st.markdown("### 📉 Gradient Descent Visualization")

        x = np.linspace(-2, 2, 100)
        y = x**2  # Simple loss landscape

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Loss Landscape',
                                line=dict(color='blue', width=2)))

        # Show steps
        lr = st.slider("Learning Rate", 0.01, 0.5, 0.1, 0.01)
        current_x = 1.5
        steps = []

        for _ in range(10):
            steps.append((current_x, current_x**2))
            gradient = 2 * current_x
            current_x -= lr * gradient

        steps_x = [s[0] for s in steps]
        steps_y = [s[1] for s in steps]

        fig.add_trace(go.Scatter(x=steps_x, y=steps_y, mode='markers+lines',
                                name='Training Steps',
                                marker=dict(size=10, color='red')))

        fig.update_layout(
            title="How the Model Learns (Moving Toward Lower Loss)",
            xaxis_title="Weight Value",
            yaxis_title="Loss",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# Section: Real Example
# ============================================================================
else:  # Real Example
    st.header("🔬 Real Example: Classifying a Patient")

    if not dataset:
        st.error("Dataset not found!")
        st.stop()

    st.markdown("""
    Let's walk through how the trained model classifies a real patient step-by-step.
    """)

    # Check if model exists
    model_dirs = list(Path('data/processed/').glob('run_*'))

    if model_dirs:
        latest_dir = max(model_dirs, key=lambda p: p.stat().st_mtime)
        model_path = latest_dir / 'best_model.pt'

        if model_path.exists():
            st.success(f"✅ Found trained model: {model_path}")

            # Select patient
            patient_idx = st.selectbox("Select Patient to Classify",
                                      range(min(10, len(dataset))),
                                      format_func=lambda x: f"{dataset[x].paciente_id}")

            patient = dataset[patient_idx]

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Input")
                st.markdown(f"**Patient ID:** {patient.paciente_id}")
                st.markdown(f"**True Label:** {['Primary', 'Metastasis', 'Normal'][patient.y.item()]}")
                st.markdown(f"**Graph:** {patient.x.size(0)} genes, {patient.edge_index.size(1)} interactions")

            with col2:
                st.markdown("### What the Model Sees")
                st.markdown("- Gene expression levels")
                st.markdown("- Protein-protein interaction network")
                st.markdown("- Learns attention weights for each edge")

            st.markdown("---")
            st.info("💡 After training completes, rerun `streamlit run app.py` to see full predictions with attention visualization!")
        else:
            st.warning("Model training in progress... Check back soon!")
    else:
        st.warning("No trained models found yet. Training in progress!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>🧬 Interactive GAT Tutorial | Built with Streamlit |
    <a href='https://arxiv.org/abs/1710.10903'>Original GAT Paper</a></p>
</div>
""", unsafe_allow_html=True)
