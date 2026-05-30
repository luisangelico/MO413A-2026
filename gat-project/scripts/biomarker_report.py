#!/usr/bin/env python3
"""
Build a biomarker report from a multi-seed training run.

For each seed's best model, compute mean attention per edge per class.
Edges are "stable biomarkers" if they rank highly *consistently across seeds*,
not just in one lucky run.

Outputs:
    docs/biomarkers.html   — interactive heatmap + overlap table

Usage:
    python -m scripts.biomarker_report
    python -m scripts.biomarker_report --run data/processed/multiseed_<ts>
    python -m scripts.biomarker_report --top-k 50
"""

import argparse
import html as htmllib
import json
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import CLASS_NAMES, DATASET_FILE, PROCESSED_DATASET_PATH, get_device
from src.model import load_model

# Curated melanoma-relevant genes. Mix of MAPK-pathway drivers, melanoma TSGs,
# pigmentation/lineage TFs, immune-checkpoint and proliferation markers.
MELANOMA_GENES = {
    "BRAF", "NRAS", "KIT", "NF1", "MAP2K1", "MAP2K2",
    "CDKN2A", "CDKN2B", "TP53", "PTEN", "RB1", "TERT",
    "MITF", "SOX10", "PAX3", "MLANA", "TYR", "DCT", "TYRP1", "PMEL", "S100B",
    "PDCD1", "CD274", "CTLA4", "LAG3", "IDO1",
    "MYC", "CCND1", "CDK4", "CDK6",
    "ARID2", "ARID1A", "BAP1", "PPP6C", "RAC1", "IDH1",
    "AXL", "EGFR", "MET", "ERBB3",
    "S100A1", "GP100",
}

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=Path, default=None,
                    help="Multi-seed run directory. Defaults to most recent multiseed_* under PROCESSED_DATASET_PATH.")
parser.add_argument("--top-k", type=int, default=40,
                    help="Number of top-spread edges to show in the heatmap.")
parser.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent.parent.parent / "docs" / "biomarkers.html")
args = parser.parse_args()

if args.run is None:
    runs = sorted(PROCESSED_DATASET_PATH.glob("multiseed_*"), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise SystemExit("No multiseed_* runs found. Run scripts.train_multiseed first.")
    run_dir = runs[-1]
else:
    run_dir = args.run

print(f"Run: {run_dir}")
config = json.loads((run_dir / "config.json").read_text())
seed_dirs = sorted([p for p in run_dir.glob("seed_*") if (p / "best_model.pt").exists()])
print(f"Seeds: {len(seed_dirs)}")
if not seed_dirs:
    raise SystemExit("No trained seed models found.")

device = get_device()
dataset = torch.load(DATASET_FILE, weights_only=False)
sample0 = dataset[0]
gene_names = getattr(sample0, "gene_names", [str(i) for i in range(sample0.x.size(0))])
edges_np = sample0.edge_index.cpu().numpy()
src_arr, dst_arr = edges_np[0], edges_np[1]
nonself_mask = src_arr != dst_arr
edge_pairs = list(zip(src_arr[nonself_mask].tolist(), dst_arr[nonself_mask].tolist()))
NUM_CLASSES = config["num_classes"]
class_names = {i: CLASS_NAMES[i] for i in range(NUM_CLASSES)}

splits = np.load(seed_dirs[0] / "splits.npz")
test_idx = splits["test"].tolist()

# Per-seed × per-class mean attention over the test set
# attn_per_seed[seed][class] = vector of length len(edge_pairs)
attn_per_seed = []
for seed_dir in seed_dirs:
    print(f"  Computing attention for {seed_dir.name}...")
    model, _ = load_model(seed_dir / "best_model.pt", device=device)
    model.eval()
    sums = {c: np.zeros(len(edge_pairs), dtype=np.float64) for c in range(NUM_CLASSES)}
    counts = {c: 0 for c in range(NUM_CLASSES)}
    for idx in test_idx:
        sample = dataset[idx].to(device)
        batch = torch.zeros(sample.x.size(0), dtype=torch.long, device=device)
        with torch.no_grad():
            _, edge_index_attn, alpha = model(sample.x, sample.edge_index, batch, return_attn=True)
        attn = alpha.mean(dim=1).cpu().numpy()
        # The attention edge list returned by GATv2Conv matches sample.edge_index.
        # Mask self-loops the same way to keep ordering aligned with edge_pairs.
        ei = edge_index_attn.cpu().numpy()
        m = ei[0] != ei[1]
        attn = attn[m]
        if attn.shape[0] != len(edge_pairs):
            # PyG can append self-loops in-conv; if shapes still differ, skip.
            continue
        c = int(sample.y.item())
        sums[c] += attn
        counts[c] += 1
    per_class = {c: sums[c] / max(counts[c], 1) for c in range(NUM_CLASSES)}
    attn_per_seed.append(per_class)

# Stack: shape (S seeds, C classes, E edges)
S = len(attn_per_seed)
C = NUM_CLASSES
E = len(edge_pairs)
stacked = np.zeros((S, C, E), dtype=np.float64)
for si, pc in enumerate(attn_per_seed):
    for c in range(C):
        stacked[si, c] = pc[c]

# Mean & std across seeds → stability
mean_sc = stacked.mean(axis=0)        # (C, E)
std_sc = stacked.std(axis=0)          # (C, E)
spread = mean_sc.max(axis=0) - mean_sc.min(axis=0)  # how class-discriminative
# Stability score: high spread, low cross-seed std (relative to mean)
rel_std = std_sc.mean(axis=0) / (mean_sc.mean(axis=0) + 1e-9)
score = spread / (1.0 + rel_std)

top_idx = np.argsort(score)[::-1][:args.top_k]

def edge_label(ei, sep="↔"):
    a, b = edge_pairs[ei]
    return f"{gene_names[a]} {sep} {gene_names[b]}"

# Heatmap: classes × top edges, coloring = mean attention across seeds
z_matrix = mean_sc[:, top_idx]
edge_labels = [edge_label(i) for i in top_idx]
heat = go.Figure(go.Heatmap(
    z=z_matrix,
    x=edge_labels,
    y=[class_names[c] for c in range(C)],
    colorscale="Magma",
    colorbar=dict(title="Mean attention"),
    hovertemplate="%{y}<br>%{x}<br>attn = %{z:.4f}<extra></extra>",
))
heat.update_layout(
    height=340,
    margin=dict(l=10, r=10, t=10, b=160),
    xaxis=dict(tickangle=-60, tickfont=dict(size=9)),
    yaxis=dict(autorange="reversed"),
)

# Stability scatter: spread vs rel_std (annotate top-k)
scatter = go.Figure()
scatter.add_trace(go.Scatter(
    x=spread, y=rel_std,
    mode="markers",
    marker=dict(size=4, color="rgba(120,120,120,0.4)"),
    hovertext=[edge_label(i) for i in range(E)],
    hoverinfo="text",
    name="all edges",
))
scatter.add_trace(go.Scatter(
    x=spread[top_idx], y=rel_std[top_idx],
    mode="markers+text",
    marker=dict(size=8, color="#d62728"),
    text=[gene_names[edge_pairs[i][0]] for i in top_idx],
    textposition="top center",
    textfont=dict(size=9),
    hovertext=[edge_label(i) for i in top_idx],
    hoverinfo="text",
    name=f"top {args.top_k} stable",
))
scatter.update_layout(
    height=420,
    margin=dict(l=10, r=10, t=10, b=10),
    xaxis_title="Class spread (max − min mean attention)",
    yaxis_title="Cross-seed relative std (lower = more stable)",
    plot_bgcolor="white",
)

# Overlap with curated melanoma list
def gene_in_list(g):
    return g.upper() in MELANOMA_GENES

rows_table = []
for rank, ei in enumerate(top_idx, start=1):
    a, b = edge_pairs[ei]
    ga, gb = gene_names[a], gene_names[b]
    in_a = gene_in_list(ga)
    in_b = gene_in_list(gb)
    overlap = "both" if in_a and in_b else ("gene_a" if in_a else ("gene_b" if in_b else ""))
    rows_table.append({
        "rank": rank,
        "gene_a": ga,
        "gene_b": gb,
        "overlap": overlap,
        "spread": float(spread[ei]),
        "rel_std": float(rel_std[ei]),
        "mean_per_class": [float(mean_sc[c, ei]) for c in range(C)],
    })

n_overlap = sum(1 for r in rows_table if r["overlap"])
overlap_pct = 100.0 * n_overlap / len(rows_table) if rows_table else 0.0

# Per-class top-5 edges (highest mean attention in that class, across seeds)
per_class_top = {}
for c in range(C):
    order = np.argsort(mean_sc[c])[::-1][:5]
    per_class_top[c] = [
        {"edge": edge_label(ei), "attn": float(mean_sc[c, ei]),
         "rel_std": float(rel_std[ei])}
        for ei in order
    ]

# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------
PAGE_CSS = """
<style>
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       max-width: 1100px; margin: 0 auto; padding: 1.5em; color: #222; }
header { border-bottom: 1px solid #ddd; padding-bottom: 0.6em; margin-bottom: 1em; }
header a { text-decoration: none; color: #06c; }
.metric-row { display: flex; gap: 1em; margin: 1em 0; flex-wrap: wrap; }
.metric { background: #f5f5f7; border-radius: 8px; padding: 0.6em 1em; min-width: 140px; }
.metric .label { font-size: 0.8em; color: #666; }
.metric .value { font-size: 1.2em; font-weight: 600; }
table { border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 0.9em; }
th, td { padding: 0.4em 0.7em; border-bottom: 1px solid #eee; text-align: left; }
th { background: #f5f5f7; }
.hit { background: #fff5d4; }
.cols { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1.2em; }
@media (max-width: 800px) { .cols { grid-template-columns: 1fr; } }
.note { color:#666; font-size: 0.92em; }
</style>
"""

heat_html = heat.to_html(full_html=False, include_plotlyjs="cdn", div_id="heat")
scatter_html = scatter.to_html(full_html=False, include_plotlyjs=False, div_id="scatter")

table_rows = ""
for r in rows_table:
    cls = "hit" if r["overlap"] else ""
    per_class_str = " · ".join(f"{class_names[c][:3]} {r['mean_per_class'][c]:.3f}" for c in range(C))
    table_rows += (
        f"<tr class='{cls}'>"
        f"<td>{r['rank']}</td>"
        f"<td>{htmllib.escape(r['gene_a'])}</td>"
        f"<td>{htmllib.escape(r['gene_b'])}</td>"
        f"<td>{r['overlap']}</td>"
        f"<td>{r['spread']:.4f}</td>"
        f"<td>{r['rel_std']:.3f}</td>"
        f"<td>{per_class_str}</td>"
        f"</tr>"
    )

per_class_blocks = ""
for c in range(C):
    rows = "".join(
        f"<tr><td>{e['edge']}</td><td>{e['attn']:.4f}</td><td>{e['rel_std']:.3f}</td></tr>"
        for e in per_class_top[c]
    )
    per_class_blocks += f"""
<div>
  <h4 style="margin:0">{class_names[c]}</h4>
  <table><tr><th>edge</th><th>attn</th><th>rel std</th></tr>{rows}</table>
</div>"""

results = json.loads((run_dir / "results.json").read_text())
mean_acc = np.mean([r["test_acc"] for r in results]) * 100
std_acc = np.std([r["test_acc"] for r in results]) * 100

page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Biomarker report</title>{PAGE_CSS}</head><body>
<header>
  <a href="index.html">&larr; index</a>
  <h1>Stable attention biomarkers</h1>
  <div>Multi-seed run: <code>{htmllib.escape(run_dir.name)}</code> · {S} seeds · dataset: <code>{htmllib.escape(config['dataset'])}</code></div>
</header>

<div class="metric-row">
  <div class="metric"><div class="label">Seeds</div><div class="value">{S}</div></div>
  <div class="metric"><div class="label">Mean test acc</div><div class="value">{mean_acc:.2f}% ± {std_acc:.2f}%</div></div>
  <div class="metric"><div class="label">Edges scored</div><div class="value">{E}</div></div>
  <div class="metric"><div class="label">Top-{args.top_k} ∩ melanoma list</div><div class="value">{n_overlap} ({overlap_pct:.0f}%)</div></div>
</div>

<h3>Top-{args.top_k} stable, class-discriminative edges</h3>
<p class="note">Each column is one PPI edge that the model attends to differently across the
three classes. Color = mean attention across {S} seeds. Edges are ranked by
<b>class-spread / cross-seed variability</b> — high values mean the model consistently
treats the edge as informative, and treats it differently between sample types.</p>
{heat_html}

<h3>Stability landscape</h3>
<p class="note">All non-self-loop edges. X axis = how much the mean attention differs between classes
(bigger = more discriminative). Y axis = relative cross-seed std (lower = more stable across re-trainings).
Top-{args.top_k} are red and labeled with one of the two endpoint genes.</p>
{scatter_html}

<h3>Top-{args.top_k} table — overlap with curated melanoma genes</h3>
<p class="note">Rows highlighted yellow have at least one endpoint in a curated list of
{len(MELANOMA_GENES)} melanoma-relevant genes (MAPK drivers, TSGs, lineage/pigmentation TFs, checkpoints).
"both" = both endpoints are in the list — strongest signal.</p>
<table>
  <tr><th>rank</th><th>gene a</th><th>gene b</th><th>overlap</th><th>spread</th><th>rel std</th><th>mean attn / class</th></tr>
  {table_rows}
</table>

<h3>Per-class top-5 edges (mean across seeds)</h3>
<div class="cols">
{per_class_blocks}
</div>

<p class="note" style="margin-top:2em">
<b>How to read this.</b> Single-run attention is noisy. Edges that show up at the top across
{S} independently trained models — and discriminate classes — are candidate biomarker interactions
worth biological follow-up. The melanoma overlap is a sanity check, not validation: edges
involving non-listed genes can still be real (the curated list is small and well-known biology).
</p>
</body></html>"""

args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(page)
print(f"Wrote {args.out}")
print(f"Top-{args.top_k} ∩ curated melanoma list: {n_overlap} edges ({overlap_pct:.1f}%)")
