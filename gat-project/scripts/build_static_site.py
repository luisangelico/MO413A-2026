#!/usr/bin/env python3
"""
Build a static site (HTML only) under `site/` showing the trained model's
predictions and attention weights. Suitable for hosting on GitHub Pages.

By default visualizes the **test split** of the latest training run.

Usage:
    python -m scripts.build_static_site
    python -m scripts.build_static_site --split all
    python -m scripts.build_static_site --run data/processed/run_20260528_163559

Layout produced:
    site/
    ├── index.html               (filterable sample table)
    ├── overview.html            (dataset + run summary)
    └── samples/
        └── sample_<idx>.html    (one per sample)
"""

import argparse
import html as htmllib
import json
from pathlib import Path

import networkx as nx
import numpy as np
import plotly.graph_objects as go
import torch
from sklearn.metrics import classification_report

import sys
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import CLASS_NAMES, DATASET_FILE, PROCESSED_DATASET_PATH, get_device
from src.model import load_model

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--run", type=Path, default=None,
                    help="Training run directory. Defaults to most recent under PROCESSED_DATASET_PATH.")
parser.add_argument("--split", choices=["test", "val", "train", "all"], default="all")
parser.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent.parent.parent / "docs",
                    help="Output directory. Defaults to <repo-root>/docs (GitHub Pages source).")
parser.add_argument("--max-samples", type=int, default=None,
                    help="Cap number of per-sample pages (default: no cap).")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Locate run + load
# ---------------------------------------------------------------------------
if args.run is None:
    run_dirs = sorted(PROCESSED_DATASET_PATH.glob("run_*"), key=lambda p: p.stat().st_mtime)
    if not run_dirs:
        raise SystemExit("No training runs found.")
    run_dir = run_dirs[-1]
else:
    run_dir = args.run

weights_path = run_dir / "best_model.pt"
if not weights_path.exists():
    weights_path = run_dir / "modelo_final.pt"

print(f"Run:     {run_dir}")
print(f"Weights: {weights_path.name}")
print(f"Dataset: {DATASET_FILE}")

device = get_device()
dataset = torch.load(DATASET_FILE, weights_only=False)
model, config = load_model(weights_path, device=device)
num_classes = config["num_classes"]
class_names = {i: CLASS_NAMES[i] for i in range(num_classes)}
class_colors = ["#d62728", "#ff7f0e", "#2ca02c"][:num_classes]

# Mirror train-time class filter for binary models
valid_idx = [i for i, d in enumerate(dataset) if num_classes == 3 or int(d.y.item()) != 2]

# Apply split filter
splits_path = run_dir / "splits.npz"
split_of = {}
if splits_path.exists():
    spl = np.load(splits_path)
    for s in ("train", "val", "test"):
        for i in spl[s].tolist():
            split_of[int(i)] = s
else:
    print("WARNING: no splits.npz — using full dataset; numbers will reflect training-data leakage.")

if args.split == "all" or not split_of:
    selected = valid_idx
else:
    selected = [i for i in valid_idx if split_of.get(i) == args.split]

if args.max_samples:
    selected = selected[: args.max_samples]

print(f"Building pages for {len(selected)} samples (split={args.split}).")

# ---------------------------------------------------------------------------
# Pre-compute training-set embeddings for nearest-neighbor lookups
# ---------------------------------------------------------------------------
train_idxs = [i for i in valid_idx if split_of.get(i) == "train"]
train_embeddings = None
if train_idxs:
    from tqdm import tqdm
    print(f"Computing embeddings for {len(train_idxs)} training samples...")
    embs = []
    for i in tqdm(train_idxs, desc="Computing embeddings"):
        s = dataset[i].to(device)
        b = torch.zeros(s.x.size(0), dtype=torch.long, device=device)
        with torch.no_grad():
            _, _, _, e = model(s.x, s.edge_index, b, return_attn=True, return_embedding=True)
        embs.append(e.squeeze(0).cpu().numpy())
    train_embeddings = np.stack(embs)
    train_labels = np.array([int(dataset[i].y.item()) for i in train_idxs])
    train_pids = [getattr(dataset[i], "paciente_id", f"sample_{i}") for i in train_idxs]

# ---------------------------------------------------------------------------
# Output dirs
# ---------------------------------------------------------------------------
OUT = args.out
SAMPLES_DIR = OUT / "samples"
OUT.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

PLOTLY_CDN = "cdn"  # use cdn so each page stays ~50KB

from src.site_header import HEADER_CSS, PROJECT_TITLE, render_header
PROJECT_TITLE_HTML = PROJECT_TITLE

PAGE_CSS = "<style>" + HEADER_CSS + """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       margin: 0; color: #222; }
.page { max-width: 1100px; margin: 0 auto; padding: 0 1.5em 2em; }
.page-header { border-bottom: 1px solid #eee; padding-bottom: 0.6em;
               margin-bottom: 1em; }
.page-header h2 { margin: 0.2em 0 0.3em; font-size: 1.3em; }
.page-header .meta { color: #666; font-size: 0.9em; }
.page-header a { text-decoration: none; color: #06c; }
.metric-row { display: flex; gap: 1em; margin: 1em 0; flex-wrap: wrap; }
.metric { background: #f5f5f7; border-radius: 8px; padding: 0.6em 1em; min-width: 130px; }
.metric .label { font-size: 0.8em; color: #666; }
.metric .value { font-size: 1.2em; font-weight: 600; }
.correct { color: #2ca02c; }
.incorrect { color: #d62728; }
table { border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 0.9em; }
th, td { padding: 0.4em 0.7em; border-bottom: 1px solid #eee; text-align: left; }
th { background: #f5f5f7; cursor: pointer; user-select: none; }
tr:hover { background: #fafafa; }
input.search { padding: 0.5em; width: 240px; font-size: 1em; margin-right: 0.5em; }
select { padding: 0.5em; font-size: 1em; margin-right: 0.5em; }
.cols { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5em; }
@media (max-width: 700px) { .cols { grid-template-columns: 1fr; } }
.pill { display: inline-block; padding: 0.1em 0.6em; border-radius: 999px;
        font-size: 0.8em; color: white; }
</style>
"""


def class_pill(c):
    color = class_colors[c]
    return f'<span class="pill" style="background:{color}">{htmllib.escape(class_names[c])}</span>'


# ---------------------------------------------------------------------------
# Build per-sample pages
# ---------------------------------------------------------------------------
sample_records = []  # for the index
embeddings_by_idx = {}  # idx -> np.ndarray (hidden_channels,)
attn_sum_by_class = {c: None for c in range(num_classes)}  # class -> np.ndarray over edges
attn_count_by_class = {c: 0 for c in range(num_classes)}
shared_edges = None  # captured once

for idx in tqdm(selected, desc="Building per-sample pages"):
    sample = dataset[idx].to(device)
    batch = torch.zeros(sample.x.size(0), dtype=torch.long, device=device)
    with torch.no_grad():
        logits, edge_index_attn, alpha, embedding = model(
            sample.x, sample.edge_index, batch, return_attn=True, return_embedding=True
        )
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred = int(probs.argmax())
    true = int(sample.y.item())
    attn = alpha.mean(dim=1).cpu().numpy()
    edges = edge_index_attn.cpu().numpy()
    embeddings_by_idx[idx] = embedding.squeeze(0).cpu().numpy()
    if shared_edges is None:
        shared_edges = edges
    if shared_edges.shape == edges.shape and attn.shape[0] == shared_edges.shape[1]:
        true_c = int(sample.y.item())
        if attn_sum_by_class[true_c] is None:
            attn_sum_by_class[true_c] = np.zeros(attn.shape[0], dtype=np.float64)
        attn_sum_by_class[true_c] += attn
        attn_count_by_class[true_c] += 1
    expr = sample.x.squeeze().cpu().numpy()
    gene_names = getattr(sample, "gene_names", [str(i) for i in range(len(expr))])
    paciente_id = getattr(sample, "paciente_id", f"sample_{idx}")

    # Build attention graph (drop self-loops; keep all edges otherwise)
    edge_records = []
    for k in range(edges.shape[1]):
        s, d = int(edges[0, k]), int(edges[1, k])
        if s == d:
            continue
        edge_records.append((s, d, float(attn[k])))

    G = nx.Graph()
    for s, d, a in edge_records:
        G.add_edge(s, d, weight=a)
    if len(G) == 0:
        # Add at least one node so the figure isn't empty
        G.add_node(0)

    pos = nx.spring_layout(G, k=0.5, iterations=60, seed=42, weight="weight")
    max_attn = max((a for *_, a in edge_records), default=1.0) or 1.0

    edge_traces = []
    for s, d, a in edge_records:
        x0, y0 = pos[s]; x1, y1 = pos[d]
        norm = a / max_attn
        color = f"rgba({int(200 + 55*norm)},{int(120 - 100*norm)},0,{0.3 + 0.7*norm:.2f})"
        edge_traces.append(go.Scatter(
            x=[x0, x1], y=[y0, y1], mode="lines",
            line=dict(width=0.5 + 4 * norm, color=color),
            hoverinfo="text",
            hovertext=f"{gene_names[s]} ↔ {gene_names[d]}<br>attn = {a:.4f}",
            showlegend=False,
        ))

    nodes = list(G.nodes())
    top_genes = set()
    for s, d, _ in sorted(edge_records, key=lambda r: r[2], reverse=True)[:8]:
        top_genes.update([s, d])
    node_trace = go.Scatter(
        x=[pos[n][0] for n in nodes],
        y=[pos[n][1] for n in nodes],
        mode="markers+text",
        marker=dict(
            size=10,
            color=[expr[n] for n in nodes],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Expression"),
            line=dict(width=0.5, color="black"),
        ),
        text=[gene_names[n] if n in top_genes else "" for n in nodes],
        textposition="top center",
        hovertext=[
            f"{gene_names[n]}<br>expr = {expr[n]:.3f}<br>degree = {G.degree(n)}"
            for n in nodes
        ],
        hoverinfo="text",
        showlegend=False,
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        height=560,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="white",
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
    )
    graph_html = fig.to_html(full_html=False, include_plotlyjs=PLOTLY_CDN, div_id=f"g{idx}")

    # Probability bar
    bar = go.Figure(go.Bar(
        x=[class_names[i] for i in range(num_classes)],
        y=probs * 100,
        marker_color=class_colors,
        text=[f"{p*100:.1f}%" for p in probs],
        textposition="auto",
    ))
    bar.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="%")
    bar_html = bar.to_html(full_html=False, include_plotlyjs=False, div_id=f"b{idx}")

    # Top tables
    top_attn = sorted(edge_records, key=lambda r: r[2], reverse=True)[:15]
    rows_attn = "".join(
        f"<tr><td>{htmllib.escape(gene_names[s])}</td>"
        f"<td>{htmllib.escape(gene_names[d])}</td><td>{a:.4f}</td></tr>"
        for s, d, a in top_attn
    )
    top_expr_idx = np.argsort(expr)[::-1][:15]
    rows_expr = "".join(
        f"<tr><td>{htmllib.escape(gene_names[i])}</td><td>{expr[i]:.3f}</td></tr>"
        for i in top_expr_idx
    )

    # Nearest training neighbors in embedding space
    nn_section = ""
    if train_embeddings is not None and split_of.get(idx) != "train":
        q = embeddings_by_idx[idx]
        # Cosine distance
        q_norm = q / (np.linalg.norm(q) + 1e-9)
        t_norm = train_embeddings / (np.linalg.norm(train_embeddings, axis=1, keepdims=True) + 1e-9)
        sims = t_norm @ q_norm
        top5 = np.argsort(sims)[::-1][:5]
        majority = int(np.bincount(train_labels[top5], minlength=num_classes).argmax())
        nn_rows = "".join(
            f"<tr><td>{train_idxs[j]}</td>"
            f"<td>{htmllib.escape(train_pids[j])}</td>"
            f"<td>{class_pill(int(train_labels[j]))}</td>"
            f"<td>{sims[j]:.3f}</td></tr>"
            for j in top5
        )
        agree = "agrees with prediction" if majority == pred else "disagrees with prediction"
        agree_cls = "correct" if majority == pred else "incorrect"
        nn_section = f"""
<h3>Nearest training neighbors (embedding space)</h3>
<p style="color:#666">5 closest training samples by cosine similarity of pooled GAT embeddings.
Majority vote: {class_pill(majority)} <span class="{agree_cls}">({agree})</span>.</p>
<table><tr><th>train idx</th><th>patient id</th><th>class</th><th>cosine sim</th></tr>{nn_rows}</table>
"""

    correct = pred == true
    correct_class = "correct" if correct else "incorrect"
    correct_label = "✓ correct" if correct else "✗ incorrect"
    sp = split_of.get(idx, "")
    sp_html = f' <span class="pill" style="background:#888">{sp}</span>' if sp else ""

    page = f"""<!doctype html>
<html><head>
<meta charset="utf-8">
<title>Sample {idx} — {htmllib.escape(paciente_id)}</title>
{PAGE_CSS}
</head><body>
{render_header("predictions", subtitle=f"Sample {idx} — {htmllib.escape(paciente_id)}", prefix="../")}
<div class="page">
<div class="page-header">
  <a href="../index.html">&larr; back to predictions</a>
  <h2>Sample {idx}</h2>
  <div class="meta">{htmllib.escape(paciente_id)}{sp_html}</div>
</div>

<div class="metric-row">
  <div class="metric"><div class="label">True class</div><div class="value">{class_pill(true)}</div></div>
  <div class="metric"><div class="label">Predicted</div><div class="value">{class_pill(pred)} <span class="{correct_class}">{correct_label}</span></div></div>
  <div class="metric"><div class="label">Confidence</div><div class="value">{probs[pred]*100:.1f}%</div></div>
</div>

<h3>Class probabilities</h3>
{bar_html}

<h3>Attention-weighted PPI graph</h3>
<p style="color:#666">Edges colored & widened by attention magnitude. Hover for gene names and weights.</p>
{graph_html}

<div class="cols">
  <div>
    <h3>Top attention edges</h3>
    <table><tr><th>gene a</th><th>gene b</th><th>attention</th></tr>{rows_attn}</table>
  </div>
  <div>
    <h3>Top expressed genes</h3>
    <table><tr><th>gene</th><th>expression</th></tr>{rows_expr}</table>
  </div>
</div>
{nn_section}
</div>
</body></html>"""
    (SAMPLES_DIR / f"sample_{idx}.html").write_text(page)

    sample_records.append({
        "idx": idx,
        "paciente_id": paciente_id,
        "true": true,
        "pred": pred,
        "confidence": float(probs[pred]),
        "correct": correct,
        "split": sp,
    })

print(f"Wrote {len(sample_records)} per-sample pages.")

# ---------------------------------------------------------------------------
# Overview page
# ---------------------------------------------------------------------------
trues = np.array([r["true"] for r in sample_records])
preds = np.array([r["pred"] for r in sample_records])
acc = float((trues == preds).mean()) if len(trues) else 0.0

cm = np.zeros((num_classes, num_classes), dtype=int)
for t, p in zip(trues, preds):
    cm[t, p] += 1

cm_fig = go.Figure(go.Heatmap(
    z=cm,
    x=[class_names[i] for i in range(num_classes)],
    y=[class_names[i] for i in range(num_classes)],
    colorscale="Blues",
    text=cm,
    texttemplate="%{text}",
    showscale=False,
))
cm_fig.update_layout(height=400, xaxis_title="Predicted", yaxis_title="True",
                    margin=dict(l=10, r=10, t=30, b=10))
cm_html = cm_fig.to_html(full_html=False, include_plotlyjs=PLOTLY_CDN, div_id="cm")

# Global PPI graph (shared edge_index across samples)
sample0 = dataset[valid_idx[0]] if valid_idx else dataset[0]
gene_names_all = getattr(sample0, "gene_names", [str(i) for i in range(sample0.x.size(0))])
mean_expr_all = np.stack([d.x.squeeze().cpu().numpy() for d in dataset]).mean(axis=0)
edges_np = sample0.edge_index.cpu().numpy()

G_full = nx.Graph()
for k in range(edges_np.shape[1]):
    s, d = int(edges_np[0, k]), int(edges_np[1, k])
    if s != d:
        G_full.add_edge(s, d)

if len(G_full) == 0:
    ppi_html = "<p>No edges in PPI graph.</p>"
else:
    pos_full = nx.spring_layout(G_full, k=0.3, iterations=80, seed=42)
    nodes_full = list(G_full.nodes())
    degrees_full = np.array([G_full.degree(n) for n in nodes_full])

    edge_x, edge_y = [], []
    for s, d in G_full.edges():
        x0, y0 = pos_full[s]; x1, y1 = pos_full[d]
        edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=0.4, color="rgba(120,120,120,0.4)"),
        hoverinfo="none", showlegend=False,
    )

    top_hub_set = set(np.argsort(degrees_full)[::-1][:20].tolist())
    node_labels = [
        gene_names_all[nodes_full[i]] if i in top_hub_set else ""
        for i in range(len(nodes_full))
    ]
    node_trace_full = go.Scatter(
        x=[pos_full[n][0] for n in nodes_full],
        y=[pos_full[n][1] for n in nodes_full],
        mode="markers+text",
        marker=dict(
            size=6 + 1.5 * degrees_full,
            color=[mean_expr_all[n] for n in nodes_full],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Mean expr"),
            line=dict(width=0.4, color="black"),
        ),
        text=node_labels,
        textposition="top center",
        textfont=dict(size=10),
        hovertext=[
            f"{gene_names_all[n]}<br>degree = {G_full.degree(n)}<br>"
            f"mean expr = {mean_expr_all[n]:.2f}"
            for n in nodes_full
        ],
        hoverinfo="text",
        showlegend=False,
    )

    ppi_fig = go.Figure(data=[edge_trace, node_trace_full])
    ppi_fig.update_layout(
        height=640, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white",
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
    )
    ppi_html = ppi_fig.to_html(full_html=False, include_plotlyjs=False, div_id="ppi")

# ---------------------------------------------------------------------------
# Learned embeddings: 2D projection
# ---------------------------------------------------------------------------
emb_idxs = [r["idx"] for r in sample_records]
emb_matrix = np.stack([embeddings_by_idx[i] for i in emb_idxs])
emb_true = np.array([r["true"] for r in sample_records])
emb_pred = np.array([r["pred"] for r in sample_records])

projection_label = "PCA"
emb_2d = None
if len(emb_matrix) >= 3:
    try:
        import umap  # type: ignore
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        emb_2d = reducer.fit_transform(emb_matrix)
        projection_label = "UMAP"
    except ImportError:
        from sklearn.decomposition import PCA
        emb_2d = PCA(n_components=2, random_state=42).fit_transform(emb_matrix)

if emb_2d is None:
    emb_section = "<p>Not enough samples for projection.</p>"
else:
    emb_traces = []
    for c in range(num_classes):
        mask = emb_true == c
        if not mask.any():
            continue
        correct_mask = (emb_pred[mask] == c)
        symbols = np.where(correct_mask, "circle", "x")
        hover = [
            f"idx {sample_records[i]['idx']}<br>{sample_records[i]['paciente_id']}"
            f"<br>true={class_names[sample_records[i]['true']]}"
            f"<br>pred={class_names[sample_records[i]['pred']]}"
            f"<br>conf={sample_records[i]['confidence']*100:.1f}%"
            for i in np.where(mask)[0]
        ]
        emb_traces.append(go.Scatter(
            x=emb_2d[mask, 0], y=emb_2d[mask, 1],
            mode="markers",
            marker=dict(
                size=9, color=class_colors[c], symbol=symbols,
                line=dict(width=0.6, color="black"),
            ),
            name=class_names[c],
            hovertext=hover, hoverinfo="text",
        ))
    emb_fig = go.Figure(data=emb_traces)
    emb_fig.update_layout(
        height=520, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white",
        xaxis=dict(title=f"{projection_label}-1", showgrid=True, gridcolor="#eee", zeroline=False),
        yaxis=dict(title=f"{projection_label}-2", showgrid=True, gridcolor="#eee", zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
    )
    emb_html = emb_fig.to_html(full_html=False, include_plotlyjs=False, div_id="emb")
    emb_section = f"""
<h3>Learned graph embeddings ({projection_label} of post-pool features)</h3>
<p style="color:#666">Each point is one sample's pooled representation just before the classifier head.
Color = true class; ✗ markers are misclassified samples. Tight class clusters mean the GAT learned a
separable representation; mixed regions are where the model struggles.</p>
{emb_html}
"""

# ---------------------------------------------------------------------------
# Per-class attention: top discriminative edges
# ---------------------------------------------------------------------------
attn_section = ""
if shared_edges is not None and all(v is not None for v in attn_sum_by_class.values()):
    mean_attn_by_class = {
        c: attn_sum_by_class[c] / max(attn_count_by_class[c], 1)
        for c in range(num_classes)
    }
    # Drop self-loops
    src_arr, dst_arr = shared_edges[0], shared_edges[1]
    nonself = src_arr != dst_arr
    edge_pairs = list(zip(src_arr[nonself].tolist(), dst_arr[nonself].tolist()))

    # Score: max - min mean attention across classes (i.e. how much classes disagree)
    stacked = np.stack([mean_attn_by_class[c][nonself] for c in range(num_classes)])  # (C, E)
    spread = stacked.max(axis=0) - stacked.min(axis=0)
    top_k = min(40, len(edge_pairs))
    top_order = np.argsort(spread)[::-1][:top_k]

    edge_labels = [
        f"{gene_names_all[edge_pairs[i][0]]} → {gene_names_all[edge_pairs[i][1]]}"
        for i in top_order
    ]
    z_matrix = stacked[:, top_order]  # (C, top_k)

    heat = go.Figure(go.Heatmap(
        z=z_matrix,
        x=edge_labels,
        y=[class_names[c] for c in range(num_classes)],
        colorscale="Magma",
        colorbar=dict(title="Mean attention"),
        hovertemplate="%{y}<br>%{x}<br>attn = %{z:.4f}<extra></extra>",
    ))
    heat.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=10, b=140),
        xaxis=dict(tickangle=-60, tickfont=dict(size=9)),
        yaxis=dict(autorange="reversed"),
    )
    heat_html = heat.to_html(full_html=False, include_plotlyjs=False, div_id="attnheat")
    attn_section = f"""
<h3>Per-class attention — top {top_k} discriminative edges</h3>
<p style="color:#666">Edges ranked by how much the mean attention differs across classes
(max − min over classes). High contrast in a column means the GAT routes information through that
edge differently depending on the sample type — a candidate biomarker interaction.</p>
{heat_html}
"""

n_iso = sample0.x.size(0) - len(G_full)
ppi_section = f"""
<h3>PPI graph</h3>
<p style="color:#666">Shared protein-protein interaction graph (STRING). {len(G_full)} connected genes
({n_iso} isolated, hidden) · {G_full.number_of_edges()} edges · top-20 hubs labeled.
Node color = mean expression across the dataset; node size = degree.</p>
{ppi_html}
"""

report = classification_report(
    trues, preds, target_names=[class_names[i] for i in range(num_classes)],
    output_dict=True, zero_division=0,
)
metric_rows = "".join(
    f"<tr><td>{class_names[i]}</td>"
    f"<td>{report[class_names[i]]['precision']*100:.1f}%</td>"
    f"<td>{report[class_names[i]]['recall']*100:.1f}%</td>"
    f"<td>{report[class_names[i]]['f1-score']*100:.1f}%</td>"
    f"<td>{int(report[class_names[i]]['support'])}</td></tr>"
    for i in range(num_classes)
)

overview = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Run overview — {PROJECT_TITLE_HTML}</title>{PAGE_CSS}</head><body>
{render_header("overview", subtitle=f"Run: {htmllib.escape(run_dir.name)} · weights: {htmllib.escape(weights_path.name)} · split: {args.split}")}
<div class="page">
<div class="page-header">
  <h2>Run overview</h2>
</div>
<div class="metric-row">
  <div class="metric"><div class="label">Samples evaluated</div><div class="value">{len(sample_records)}</div></div>
  <div class="metric"><div class="label">Accuracy</div><div class="value">{acc*100:.2f}%</div></div>
  <div class="metric"><div class="label">Classes</div><div class="value">{num_classes}</div></div>
</div>
<h3>Per-class metrics</h3>
<table><tr><th>class</th><th>precision</th><th>recall</th><th>F1</th><th>support</th></tr>{metric_rows}</table>
<h3>Confusion matrix</h3>
{cm_html}
{emb_section}
{attn_section}
{ppi_section}
</div>
</body></html>"""
(OUT / "overview.html").write_text(overview)
print(f"Wrote overview.html (accuracy {acc*100:.2f}%).")

# ---------------------------------------------------------------------------
# Index page (with client-side filtering / sort)
# ---------------------------------------------------------------------------
records_json = json.dumps([
    {
        "idx": r["idx"],
        "id": r["paciente_id"],
        "true": class_names[r["true"]],
        "pred": class_names[r["pred"]],
        "conf": round(r["confidence"], 4),
        "ok": r["correct"],
        "split": r["split"],
    }
    for r in sample_records
])

index = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Predictions — {PROJECT_TITLE_HTML}</title>{PAGE_CSS}</head><body>
{render_header("predictions", subtitle=f"Run: <code>{htmllib.escape(run_dir.name)}</code> · split: <strong>{args.split}</strong>")}
<div class="page">
<div class="page-header">
  <h2>Model predictions</h2>
  <div class="meta">Click any row to inspect the per-sample graph and nearest training neighbors.</div>
</div>
<div class="metric-row">
  <div class="metric"><div class="label">Samples</div><div class="value">{len(sample_records)}</div></div>
  <div class="metric"><div class="label">Accuracy</div><div class="value">{acc*100:.2f}%</div></div>
</div>
<div>
  <input class="search" id="q" placeholder="search id…">
  <select id="cls"><option value="">all classes</option>{"".join(f'<option>{class_names[i]}</option>' for i in range(num_classes))}</select>
  <select id="ok"><option value="">all</option><option value="true">correct only</option><option value="false">incorrect only</option></select>
  <select id="split">
    <option value="all">all splits</option>
    <option value="train" selected>train only</option>
    <option value="test">test only</option>
    <option value="validation">validation only</option>
  </select>
</div>
<table id="t">
  <thead>
    <tr>
      <th data-k="idx">idx</th>
      <th data-k="id">patient id</th>
      <th data-k="true">true</th>
      <th data-k="pred">predicted</th>
      <th data-k="split">split</th>
      <th data-k="conf">confidence</th>
      <th data-k="ok">correct?</th>
    </tr>
  </thead>
  <tbody></tbody>
</table>
<script>
const data = {records_json};
let sortKey = "idx", sortAsc = true;
const body = document.querySelector("#t tbody");
const q = document.querySelector("#q");
const cls = document.querySelector("#cls");
const ok = document.querySelector("#ok");
const split = document.querySelector("#split");

function render() {{
  const qv = q.value.toLowerCase();
  const clsv = cls.value;
  const okv = ok.value;
  const splitv = split.value;
  let rows = data.filter(r =>
    (!qv || r.id.toLowerCase().includes(qv)) &&
    (!clsv || r.true === clsv) &&
    (!okv || String(r.ok) === okv) &&
    (splitv === "all" || (splitv === "validation" ? r.split === "val" : r.split === splitv))
  );
  rows.sort((a,b) => {{
    const av = a[sortKey], bv = b[sortKey];
    if (av < bv) return sortAsc ? -1 : 1;
    if (av > bv) return sortAsc ? 1 : -1;
    return 0;
  }});
  body.innerHTML = rows.map(r => `
    <tr onclick="location='samples/sample_${{r.idx}}.html'" style="cursor:pointer">
      <td>${{r.idx}}</td>
      <td>${{r.id}}</td>
      <td>${{r.true}}</td>
      <td>${{r.pred}}</td>
      <td>${{r.split === "val" ? "validation" : r.split}}</td>
      <td>${{(r.conf*100).toFixed(1)}}%</td>
      <td>${{r.ok ? '<span style="color:#2ca02c">✓</span>' : '<span style="color:#d62728">✗</span>'}}</td>
    </tr>`).join("");
}}

document.querySelectorAll("th").forEach(th => th.onclick = () => {{
  const k = th.dataset.k;
  if (sortKey === k) sortAsc = !sortAsc; else {{ sortKey = k; sortAsc = true; }}
  render();
}});
[q, cls, ok, split].forEach(el => el.oninput = render);
render();
</script>
</div>
</body></html>"""
(OUT / "index.html").write_text(index)
print(f"Wrote index.html → {OUT}/")
print(f"\nOpen {OUT / 'index.html'} in a browser, or push {OUT}/ to a gh-pages branch.")
