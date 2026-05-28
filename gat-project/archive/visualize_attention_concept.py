#!/usr/bin/env python3
"""
Visualize the concept of attention in GAT networks.
Shows how attention weights change the importance of different gene interactions.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Example: Gene interaction network
genes = ['S100B\n(Melanoma\nMarker)', 'MITF\n(Melanocyte\nTF)', 'VIM\n(EMT\nMarker)',
         'IGKC\n(Immune)', 'HLA-DRA\n(MHC)']
pos = {
    0: (1, 3),   # S100B
    1: (3, 4),   # MITF
    2: (3, 2),   # VIM
    3: (5, 3.5), # IGKC
    4: (5, 2.5), # HLA-DRA
}

# Define interactions with attention weights
# Format: (from, to, attention_primary, attention_metastasis)
interactions = [
    (0, 1, 0.9, 0.3),  # S100B -> MITF: high attention in primary
    (0, 2, 0.2, 0.8),  # S100B -> VIM: high attention in metastasis
    (2, 3, 0.1, 0.9),  # VIM -> IGKC: high attention in metastasis
    (3, 4, 0.3, 0.95), # IGKC -> HLA-DRA: very high in metastasis
    (1, 2, 0.7, 0.4),  # MITF -> VIM
]

def draw_network(ax, title, attention_idx):
    ax.set_xlim(0, 6)
    ax.set_ylim(1, 5)
    ax.axis('off')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    # Draw edges with attention-based thickness and color
    for from_node, to_node, attn_primary, attn_metastasis in interactions:
        x1, y1 = pos[from_node]
        x2, y2 = pos[to_node]

        attention = attn_primary if attention_idx == 0 else attn_metastasis

        # Color based on attention (red = high, gray = low)
        color = plt.cm.Reds(attention * 0.8 + 0.2)
        width = 0.5 + attention * 4
        alpha = 0.3 + attention * 0.7

        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle='-',
            linewidth=width,
            color=color,
            alpha=alpha,
            zorder=1
        )
        ax.add_patch(arrow)

        # Add attention weight label
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x, mid_y, f'{attention:.2f}',
               fontsize=8, ha='center', va='center',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                        edgecolor='none', alpha=0.8))

    # Draw nodes
    for i, gene in enumerate(genes):
        x, y = pos[i]

        # Node color based on typical expression pattern
        if i in [0, 1]:  # Melanocyte markers
            node_color = '#ff9999'
        elif i == 2:  # EMT marker
            node_color = '#ffcc99'
        else:  # Immune
            node_color = '#99ccff'

        circle = plt.Circle((x, y), 0.4, color=node_color,
                          ec='black', linewidth=2, zorder=2)
        ax.add_patch(circle)
        ax.text(x, y, gene, ha='center', va='center',
               fontsize=9, fontweight='bold', zorder=3)

# Plot 1: Primary Tumor Attention Pattern
draw_network(axes[0], 'Primary Tumor:\nHigh Attention to Melanocyte Program', 0)

# Plot 2: Metastatic Tumor Attention Pattern
draw_network(axes[1], 'Metastatic Tumor:\nHigh Attention to EMT + Immune', 1)

# Add legend
legend_elements = [
    mpatches.Patch(color='#ff9999', label='Melanocyte Markers'),
    mpatches.Patch(color='#ffcc99', label='EMT Markers'),
    mpatches.Patch(color='#99ccff', label='Immune Markers'),
    plt.Line2D([0], [0], color='darkred', linewidth=4, label='High Attention'),
    plt.Line2D([0], [0], color='gray', linewidth=1, label='Low Attention')
]

fig.legend(handles=legend_elements, loc='lower center', ncol=5,
          fontsize=11, frameon=True, bbox_to_anchor=(0.5, -0.05))

plt.suptitle('Graph Attention Network: Learning Different Patterns for Different Tumor Types',
            fontsize=18, fontweight='bold', y=0.98)

plt.tight_layout()
plt.savefig('attention_concept.png', dpi=300, bbox_inches='tight')
print("✓ Concept visualization saved as: attention_concept.png")
print("\nKey Insight:")
print("  The SAME network, but attention weights change based on tumor type")
print("  → Primary: Focuses on melanocyte lineage (S100B ↔ MITF)")
print("  → Metastasis: Focuses on EMT and immune infiltration (VIM ↔ IGKC ↔ HLA-DRA)")
