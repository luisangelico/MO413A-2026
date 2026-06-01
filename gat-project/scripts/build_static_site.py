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
sample_attns = {}
sample_exprs = {}

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
    sample_attns[idx] = attn
    sample_exprs[idx] = expr
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

# ---------------------------------------------------------------------------
# Build Exploration Tab
# ---------------------------------------------------------------------------
print("Building Exploration Tab...")
non_self_loop_indices = []
global_edges = []
sample0 = dataset[selected[0]].to(device)
edges_np = sample0.edge_index.cpu().numpy()
for k in range(edges_np.shape[1]):
    s, d = int(edges_np[0, k]), int(edges_np[1, k])
    if s != d:
        non_self_loop_indices.append(k)
        global_edges.append([s, d])

exploration_samples = []
for r in sample_records:
    idx = r["idx"]
    attn = sample_attns[idx]
    expr = sample_exprs[idx]
    attn_list = [round(float(attn[k]), 4) for k in non_self_loop_indices]
    expr_list = [round(float(expr[n]), 2) for n in range(len(expr))]
    
    exploration_samples.append({
        "idx": idx,
        "id": r["paciente_id"],
        "true": class_names[r["true"]],
        "pred": class_names[r["pred"]],
        "conf": float(r["confidence"]),
        "ok": bool(r["correct"]),
        "split": str(r["split"]),
        "expr": expr_list,
        "attn": attn_list
    })

exploration_data = {
    "genes": gene_names_all,
    "edges": global_edges,
    "positions": {int(node): [float(coord) for coord in coords] for node, coords in pos_full.items()} if 'pos_full' in locals() else {},
    "samples": exploration_samples
}
data_js = f"const EXPLORATION_DATA = {json.dumps(exploration_data)};"
(OUT / "exploration_data.js").write_text(data_js)
print(f"Wrote exploration_data.js → {OUT}/")

# Generate exploration.html
exploration_html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Exploration — {PROJECT_TITLE_HTML}</title>
{PAGE_CSS}
<style>
.explore-container {{
    display: grid;
    grid-template-columns: 350px 1fr;
    gap: 1.5em;
    margin-top: 1em;
}}
@media (max-width: 1000px) {{
    .explore-container {{
        grid-template-columns: 1fr;
    }}
}}
.sidebar {{
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 1.2em;
    height: fit-content;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}}
.sidebar h3 {{
    margin-top: 0;
    font-size: 1.1em;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 0.5em;
    color: #1e293b;
}}
.control-group {{
    margin-bottom: 1em;
}}
.control-group label {{
    display: block;
    font-size: 0.82em;
    font-weight: 600;
    color: #475569;
    margin-bottom: 0.3em;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}}
.sidebar input.search, .sidebar select, .sidebar input[type="range"] {{
    width: 100%;
    box-sizing: border-box;
    padding: 0.5em;
    font-size: 0.9em;
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    background: white;
}}
.sidebar input[type="range"] {{
    padding: 0;
}}
.sample-list-container {{
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    height: 250px;
    overflow-y: auto;
    background: white;
    margin-top: 0.5em;
}}
.sample-item {{
    display: flex;
    align-items: center;
    padding: 0.4em 0.6em;
    border-bottom: 1px solid #f1f5f9;
    font-size: 0.85em;
    cursor: pointer;
    transition: background 0.1s;
}}
.sample-item:hover {{
    background: #f1f5f9;
}}
.sample-item input {{
    margin-right: 0.6em;
    cursor: pointer;
}}
.sample-item .meta {{
    margin-left: auto;
    font-size: 0.8em;
    color: #64748b;
}}
.btn-group {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.4em;
    margin-bottom: 1em;
}}
.btn {{
    padding: 0.45em 0.8em;
    font-size: 0.8em;
    font-weight: 500;
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    background: white;
    cursor: pointer;
    text-align: center;
    transition: all 0.15s;
}}
.btn:hover {{
    background: #f1f5f9;
    border-color: #94a3b8;
}}
.btn.primary {{
    background: #1f77b4;
    color: white;
    border-color: #1f77b4;
}}
.btn.primary:hover {{
    background: #155a8a;
    border-color: #155a8a;
}}
.alg-selector {{
    display: flex;
    gap: 0.2em;
    background: #e2e8f0;
    padding: 0.25em;
    border-radius: 6px;
    margin-bottom: 1em;
}}
.alg-btn {{
    flex: 1;
    text-align: center;
    padding: 0.45em 0.2em;
    font-size: 0.82em;
    font-weight: 600;
    border-radius: 4px;
    cursor: pointer;
    background: transparent;
    border: none;
    color: #475569;
    transition: all 0.12s;
}}
.alg-btn.active {{
    background: white;
    color: #1e293b;
    box-shadow: 0 1px 2px rgba(0,0,0,0.1);
}}
.selected-pane {{
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 1.2em;
    margin-top: 1.2em;
}}
.graph-card {{
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 1em;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    margin-bottom: 1.5em;
}}
.sub-select-group {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.5em;
    margin-bottom: 1em;
    background: #f1f5f9;
    padding: 0.6em;
    border-radius: 6px;
    border: 1px solid #e2e8f0;
}}
.sub-select-group label {{
    font-size: 0.78em;
    font-weight: 700;
    color: #475569;
    margin-bottom: 0.2em;
    display: block;
}}
.badge {{
    display: inline-block;
    padding: 0.15em 0.5em;
    border-radius: 999px;
    font-size: 0.75em;
    color: white;
    font-weight: 500;
}}
</style>
<script charset="utf-8" src="https://cdn.plot.ly/plotly-3.5.0.min.js" integrity="sha256-fHbNLP+GlIXN+efbQec78UkemUz3NJp7UmfGxC1tNxs=" crossorigin="anonymous"></script>
<script src="exploration_data.js"></script>
</head><body>
{render_header("exploration", subtitle=f"Run: {htmllib.escape(run_dir.name)}", prefix="")}
<div class="page">
<div class="page-header">
  <h2>Sample Exploration Dashboard</h2>
  <div class="meta">Filter and select samples to visually compare their Graph Attention Network mappings.</div>
</div>

<div class="explore-container">
  <div class="sidebar">
    <h3>🔍 Filters</h3>
    <div class="control-group">
      <label>Search Patient ID</label>
      <input type="text" class="search" id="q" placeholder="search id…">
    </div>
    <div class="control-group">
      <label>True Class</label>
      <select id="cls">
        <option value="">all classes</option>
        {"".join(f'<option>{class_names[i]}</option>' for i in range(num_classes))}
      </select>
    </div>
    <div class="control-group">
      <label>Split</label>
      <select id="split">
        <option value="all">all splits</option>
        <option value="train">train only</option>
        <option value="test">test only</option>
        <option value="val">validation only</option>
      </select>
    </div>
    <div class="control-group">
      <label>Status</label>
      <select id="ok">
        <option value="">all</option>
        <option value="true">correct only</option>
        <option value="false">incorrect only</option>
      </select>
    </div>
    
    <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 1.5em; margin-bottom: 0.3em;">
      <h3 style="margin: 0; border: none; padding: 0; font-size: 1.1em;">👤 Samples</h3>
      <span id="match-count" style="font-size: 0.8em; color: #64748b; font-weight: 500;">0 matched</span>
    </div>
    <div class="btn-group">
      <button class="btn" onclick="selectAll(true)">Select All</button>
      <button class="btn" onclick="selectAll(false)">Clear</button>
    </div>
    <div class="sample-list-container">
      <div id="sample-list"></div>
    </div>
    
    <div style="margin-top: 1.5em;">
      <h3>⚙️ Comparison</h3>
      <div class="control-group">
        <label>Aggregation Algorithm</label>
        <div class="alg-selector">
          <button class="alg-btn active" id="alg-mean" onclick="setAlgorithm('mean')">Mean</button>
          <button class="alg-btn" id="alg-addition" onclick="setAlgorithm('addition')">Addition</button>
          <button class="alg-btn" id="alg-subtraction" onclick="setAlgorithm('subtraction')">Subtraction</button>
        </div>
      </div>
      
      <div class="sub-select-group" id="subtraction-controls" style="display: none;">
        <div>
          <label>Sample A (Minuend)</label>
          <select id="sample-a" onchange="renderGraph()"></select>
        </div>
        <div>
          <label>Sample B (Subtrahend)</label>
          <select id="sample-b" onchange="renderGraph()"></select>
        </div>
      </div>

      <div class="control-group">
        <label>Edge Filter Threshold (<span id="threshold-val">0.02</span>)</label>
        <input type="range" id="threshold" min="0.0" max="0.3" step="0.005" value="0.02" oninput="document.getElementById('threshold-val').innerText=this.value; renderGraph();">
      </div>
    </div>
  </div>
  
  <div class="main-content">
    <div class="graph-card">
      <h3 style="margin-top: 0; font-size: 1.2em; color: #1e293b; border-bottom: 1px solid #f1f5f9; padding-bottom: 0.5em;" id="graph-title">Attention Graph Comparison</h3>
      <div id="plotly-graph" style="height: 600px; width: 100%;"></div>
    </div>
    
    <div class="selected-pane" id="selected-info-pane" style="display: none;">
      <h3 style="margin-top: 0; font-size: 1.1em; color: #1e293b;">Selected Samples Information</h3>
      <table id="selected-table">
        <thead>
          <tr>
            <th>idx</th>
            <th>patient id</th>
            <th>true class</th>
            <th>predicted</th>
            <th>split</th>
            <th>confidence</th>
            <th>correct?</th>
          </tr>
        </thead>
        <tbody></tbody>
      </table>
    </div>

    <div class="graph-card" style="margin-top: 1.5em;">
      <h3 style="margin-top: 0; font-size: 1.1em; color: #1e293b; border-bottom: 1px solid #f1f5f9; padding-bottom: 0.5em;" id="table-title">Top Edge Interactions</h3>
      <table id="interactions-table">
        <thead>
          <tr>
            <th>gene a</th>
            <th>gene b</th>
            <th id="col-val-a">Attention A</th>
            <th id="col-val-b" style="display: none;">Attention B</th>
            <th id="col-val-compare">Aggregated Value</th>
          </tr>
        </thead>
        <tbody></tbody>
      </table>
    </div>
  </div>
</div>

<script>
let selectedSampleIds = new Set([0, 1]); // Default selection
let currentAlgorithm = "mean";

const sampleListEl = document.getElementById("sample-list");
const qEl = document.getElementById("q");
const clsEl = document.getElementById("cls");
const splitEl = document.getElementById("split");
const okEl = document.getElementById("ok");
const subControlsEl = document.getElementById("subtraction-controls");
const sampleAEl = document.getElementById("sample-a");
const sampleBEl = document.getElementById("sample-b");

function init() {{
  // Filter listeners
  [qEl, clsEl, splitEl, okEl].forEach(el => el.addEventListener("input", filterSamples));
  
  // Initialize sample list
  filterSamples();
  
  // Initial render
  updateSelectedPane();
  renderGraph();
}}

function filterSamples() {{
  const qv = qEl.value.toLowerCase();
  const clsv = clsEl.value;
  const splitv = splitEl.value;
  const okv = okEl.value;
  
  const matches = EXPLORATION_DATA.samples.filter(s => 
    (!qv || s.id.toLowerCase().includes(qv)) &&
    (!clsv || s.true === clsv) &&
    (!okv || String(s.ok) === okv) &&
    (splitv === "all" || s.split === splitv)
  );
  
  document.getElementById("match-count").innerText = `${{matches.length}} matched`;
  
  sampleListEl.innerHTML = matches.map(s => {{
    const checked = selectedSampleIds.has(s.idx) ? "checked" : "";
    const color = s.true === "Primary Tumor" ? "#d62728" : (s.true === "Metastasis" ? "#ff7f0e" : "#2ca02c");
    return `
      <div class="sample-item" onclick="toggleSampleCheckbox(${{s.idx}})">
        <input type="checkbox" id="chk-${{s.idx}}" ${{checked}} onclick="event.stopPropagation(); toggleSample(${{s.idx}})">
        <strong>${{s.idx}}</strong> - ${{s.id}}
        <span class="meta"><span class="badge" style="background:${{color}}">${{s.true.substring(0, 7)}}</span></span>
      </div>
    `;
  }}).join("");
}}

function toggleSampleCheckbox(idx) {{
  const chk = document.getElementById(`chk-${{idx}}`);
  if (chk) {{
    chk.checked = !chk.checked;
    toggleSample(idx);
  }}
}}

function toggleSample(idx) {{
  if (selectedSampleIds.has(idx)) {{
    selectedSampleIds.delete(idx);
  }} else {{
    selectedSampleIds.add(idx);
  }}
  updateSelectedPane();
  renderGraph();
}}

function selectAll(checked) {{
  const qv = qEl.value.toLowerCase();
  const clsv = clsEl.value;
  const splitv = splitEl.value;
  const okv = okEl.value;
  
  EXPLORATION_DATA.samples.forEach(s => {{
    const match = (!qv || s.id.toLowerCase().includes(qv)) &&
                  (!clsv || s.true === clsv) &&
                  (!okv || String(s.ok) === okv) &&
                  (splitv === "all" || s.split === splitv);
    if (match) {{
      if (checked) selectedSampleIds.add(s.idx);
      else selectedSampleIds.delete(s.idx);
    }}
  }});
  
  filterSamples();
  updateSelectedPane();
  renderGraph();
}}

function setAlgorithm(alg) {{
  currentAlgorithm = alg;
  document.querySelectorAll(".alg-btn").forEach(b => b.classList.remove("active"));
  document.getElementById(`alg-${{alg}}`).classList.add("active");
  
  if (alg === "subtraction") {{
    subControlsEl.style.display = "grid";
  }} else {{
    subControlsEl.style.display = "none";
  }}
  
  updateSelectedPane();
  renderGraph();
}}

function updateSelectedPane() {{
  const activeIds = Array.from(selectedSampleIds);
  const pane = document.getElementById("selected-info-pane");
  
  // Update subtraction dropdowns
  const prevA = sampleAEl.value;
  const prevB = sampleBEl.value;
  
  const optionsHtml = activeIds.map(idx => {{
    const s = EXPLORATION_DATA.samples.find(sm => sm.idx === idx);
    return `<option value="${{idx}}">idx ${{idx}} - ${{s.id.substring(0, 12)}}...</option>`;
  }}).join("");
  
  sampleAEl.innerHTML = optionsHtml;
  sampleBEl.innerHTML = optionsHtml;
  
  if (activeIds.includes(Number(prevA))) sampleAEl.value = prevA;
  else if (activeIds.length > 0) sampleAEl.value = activeIds[0];
  
  if (activeIds.includes(Number(prevB))) sampleBEl.value = prevB;
  else if (activeIds.length > 1) sampleBEl.value = activeIds[1];
  else if (activeIds.length > 0) sampleBEl.value = activeIds[0];

  if (activeIds.length === 0) {{
    pane.style.display = "none";
    return;
  }}
  
  pane.style.display = "block";
  const tbody = document.querySelector("#selected-table tbody");
  tbody.innerHTML = activeIds.map(idx => {{
    const s = EXPLORATION_DATA.samples.find(sm => sm.idx === idx);
    const color = s.true === "Primary Tumor" ? "#d62728" : (s.true === "Metastasis" ? "#ff7f0e" : "#2ca02c");
    const predColor = s.pred === "Primary Tumor" ? "#d62728" : (s.pred === "Metastasis" ? "#ff7f0e" : "#2ca02c");
    return `
      <tr>
        <td>${{s.idx}}</td>
        <td>${{s.id}}</td>
        <td><span class="badge" style="background:${{color}}">${{s.true}}</span></td>
        <td><span class="badge" style="background:${{predColor}}">${{s.pred}}</span></td>
        <td>${{s.split === "val" ? "validation" : s.split}}</td>
        <td>${{(s.conf * 100).toFixed(1)}}%</td>
        <td>${{s.ok ? '<span style="color:#2ca02c">✓</span>' : '<span style="color:#d62728">✗</span>'}}</td>
      </tr>
    `;
  }}).join("");
}}

function renderGraph() {{
  const activeIds = Array.from(selectedSampleIds);
  const numActive = activeIds.length;
  const graphTitleEl = document.getElementById("graph-title");
  const tableTitleEl = document.getElementById("table-title");
  
  if (numActive === 0) {{
    Plotly.newPlot("plotly-graph", [], {{
      annotations: [{{ text: "Select one or more samples from the sidebar to visualize network attention map", showarrow: false, font: {{ size: 16 }} }}],
      xaxis: {{ visible: false }}, yaxis: {{ visible: false }}
    }});
    document.querySelector("#interactions-table tbody").innerHTML = "<tr><td colspan='5' style='text-align: center; color: #64748b;'>No samples selected</td></tr>";
    return;
  }}

  let edgeAttns = new Array(EXPLORATION_DATA.edges.length).fill(0);
  let nodeExprs = new Array(EXPLORATION_DATA.genes.length).fill(0);
  
  // Calculate average gene expressions for selected nodes
  for (let i = 0; i < EXPLORATION_DATA.genes.length; i++) {{
    let sum = 0;
    for (let idx of activeIds) {{
      const s = EXPLORATION_DATA.samples.find(sm => sm.idx === idx);
      if (s) sum += s.expr[i];
    }}
    nodeExprs[i] = sum / numActive;
  }}

  let subtitle = "";
  let sample1Id = null, sample2Id = null;
  
  if (currentAlgorithm === "mean") {{
    subtitle = `Mean attention of ${{numActive}} sample(s)`;
    for (let i = 0; i < EXPLORATION_DATA.edges.length; i++) {{
      let sum = 0;
      for (let idx of activeIds) {{
        const s = EXPLORATION_DATA.samples.find(sm => sm.idx === idx);
        if (s) sum += s.attn[i];
      }}
      edgeAttns[i] = sum / numActive;
    }}
  }} else if (currentAlgorithm === "addition") {{
    subtitle = `Addition attention of ${{numActive}} sample(s)`;
    for (let i = 0; i < EXPLORATION_DATA.edges.length; i++) {{
      let sum = 0;
      for (let idx of activeIds) {{
        const s = EXPLORATION_DATA.samples.find(sm => sm.idx === idx);
        if (s) sum += s.attn[i];
      }}
      edgeAttns[i] = sum;
    }}
  }} else if (currentAlgorithm === "subtraction") {{
    sample1Id = Number(sampleAEl.value);
    sample2Id = Number(sampleBEl.value);
    if (isNaN(sample1Id) || isNaN(sample2Id)) {{
      Plotly.newPlot("plotly-graph", [], {{
        annotations: [{{ text: "Select two samples to perform subtraction", showarrow: false, font: {{ size: 16 }} }}],
        xaxis: {{ visible: false }}, yaxis: {{ visible: false }}
      }});
      return;
    }}
    const sA = EXPLORATION_DATA.samples.find(s => s.idx === sample1Id);
    const sB = EXPLORATION_DATA.samples.find(s => s.idx === sample2Id);
    subtitle = `Subtraction: Sample ${{sample1Id}} (${{sA.id.substring(0, 10)}}) − Sample ${{sample2Id}} (${{sB.id.substring(0, 10)}})`;
    for (let i = 0; i < EXPLORATION_DATA.edges.length; i++) {{
      if (sA && sB) {{
        edgeAttns[i] = sA.attn[i] - sB.attn[i];
      }}
    }}
  }}
  
  graphTitleEl.innerText = `Attention Network Mappings (${{subtitle}})`;
  tableTitleEl.innerText = `Top Edge Interactions (${{subtitle}})`;

  // Draw edges
  const threshold = parseFloat(document.getElementById("threshold").value) || 0.0;
  let maxAbsAttn = 0.0001;
  for (let i = 0; i < edgeAttns.length; i++) {{
    const absVal = Math.abs(edgeAttns[i]);
    if (absVal > maxAbsAttn) maxAbsAttn = absVal;
  }}

  const edgeTraces = [];
  const topEdgesList = [];
  
  for (let i = 0; i < EXPLORATION_DATA.edges.length; i++) {{
    const edge = EXPLORATION_DATA.edges[i];
    const sNode = edge[0];
    const dNode = edge[1];
    const val = edgeAttns[i];
    const absVal = Math.abs(val);
    
    topEdgesList.push({{
      sName: EXPLORATION_DATA.genes[sNode],
      dName: EXPLORATION_DATA.genes[dNode],
      sAttn: sample1Id !== null ? EXPLORATION_DATA.samples.find(sm => sm.idx === sample1Id).attn[i] : null,
      dAttn: sample2Id !== null ? EXPLORATION_DATA.samples.find(sm => sm.idx === sample2Id).attn[i] : null,
      val: val,
      absVal: absVal
    }});
    
    if (absVal < threshold) continue;
    
    const x0 = EXPLORATION_DATA.positions[sNode][0];
    const y0 = EXPLORATION_DATA.positions[sNode][1];
    const x1 = EXPLORATION_DATA.positions[dNode][0];
    const y1 = EXPLORATION_DATA.positions[dNode][1];
    
    let color, width;
    const norm = absVal / maxAbsAttn;
    
    if (currentAlgorithm === "subtraction") {{
      // Diverging colors: Red for positive (Sample A higher), Blue for negative (Sample B higher)
      if (val > 0) {{
        color = `rgba(${{Math.round(214 + 41*norm)}}, ${{Math.round(39 - 39*norm)}}, ${{Math.round(40 - 40*norm)}}, ${{0.3 + 0.7*norm}})`;
      }} else {{
        color = `rgba(${{Math.round(31 - 31*norm)}}, ${{Math.round(119 - 119*norm)}}, ${{Math.round(180 + 75*norm)}}, ${{0.3 + 0.7*norm}})`;
      }}
      width = 0.5 + 6.0 * norm;
    }} else {{
      // Sequential: Orange-Red
      color = `rgba(230, ${{Math.round(100 - 100*norm)}}, 0, ${{0.3 + 0.7*norm}})`;
      width = 0.5 + 6.0 * norm;
    }}
    
    edgeTraces.push({{
      x: [x0, x1],
      y: [y0, y1],
      mode: 'lines',
      line: {{ width: width, color: color }},
      hoverinfo: 'text',
      hovertext: `${{EXPLORATION_DATA.genes[sNode]}} ↔ ${{EXPLORATION_DATA.genes[dNode]}}<br>Attention: ${{val.toFixed(4)}}`,
      showlegend: false
    }});
  }}

  // Draw nodes
  const nodes = Object.keys(EXPLORATION_DATA.positions).map(Number);
  const nodeX = [], nodeY = [], nodeText = [], nodeColors = [];
  
  for (let n of nodes) {{
    const pos = EXPLORATION_DATA.positions[n];
    nodeX.push(pos[0]);
    nodeY.push(pos[1]);
    nodeText.push(`${{EXPLORATION_DATA.genes[n]}}<br>Mean Expression: ${{nodeExprs[n].toFixed(3)}}`);
    nodeColors.push(nodeExprs[n]);
  }}
  
  const nodeTrace = {{
    x: nodeX,
    y: nodeY,
    mode: 'markers',
    marker: {{
      size: 9,
      color: nodeColors,
      colorscale: 'Viridis',
      showscale: true,
      colorbar: {{ title: 'Expression', thickness: 15, len: 0.6 }},
      line: {{ width: 0.6, color: 'black' }}
    }},
    hovertext: nodeText,
    hoverinfo: 'text',
    showlegend: false
  }};
  
  const layout = {{
    height: 600,
    margin: {{ l: 10, r: 10, t: 10, b: 10 }},
    plot_bgcolor: "white",
    xaxis: {{ showticklabels: false, showgrid: false, zeroline: false }},
    yaxis: {{ showticklabels: false, showgrid: false, zeroline: false }},
    hovermode: 'closest'
  }};
  
  Plotly.newPlot("plotly-graph", [...edgeTraces, nodeTrace], layout, {{ responsive: true }});
  
  // Render Top Interactions Table
  topEdgesList.sort((a, b) => b.absVal - a.absVal);
  const top15 = topEdgesList.slice(0, 15);
  
  const colValAEl = document.getElementById("col-val-a");
  const colValBEl = document.getElementById("col-val-b");
  const colValCompareEl = document.getElementById("col-val-compare");
  
  if (currentAlgorithm === "subtraction") {{
    colValAEl.style.display = "table-cell";
    colValBEl.style.display = "table-cell";
    colValAEl.innerText = `Attn (Sample ${{sample1Id}})`;
    colValBEl.innerText = `Attn (Sample ${{sample2Id}})`;
    colValCompareEl.innerText = "Difference";
  }} else {{
    colValAEl.style.display = "none";
    colValBEl.style.display = "none";
    colValCompareEl.innerText = currentAlgorithm === "mean" ? "Mean Attention" : "Added Attention";
  }}
  
  const tableBody = document.querySelector("#interactions-table tbody");
  tableBody.innerHTML = top15.map(row => {{
    let cells = "";
    if (currentAlgorithm === "subtraction") {{
      cells = `
        <td>${{row.sAttn.toFixed(4)}}</td>
        <td>${{row.dAttn.toFixed(4)}}</td>
      `;
    }}
    
    let compStyle = "";
    if (currentAlgorithm === "subtraction") {{
      compStyle = `style="font-weight: 700; color: ${{row.val > 0 ? '#d62728' : (row.val < 0 ? '#1f77b4' : '#222')}}"`;
    }} else {{
      compStyle = 'style="font-weight: 700;"';
    }}
    
    return `
      <tr>
        <td>${{row.sName}}</td>
        <td>${{row.dName}}</td>
        ${{cells}}
        <td ${{compStyle}}>${{row.val.toFixed(4)}}</td>
      </tr>
    `;
  }}).join("");
}}

window.onload = init;
</script>
</div>
</body></html>"""
(OUT / "exploration.html").write_text(exploration_html)
print(f"Wrote exploration.html → {OUT}/")
print(f"\nOpen {OUT / 'index.html'} in a browser, or push {OUT}/ to a gh-pages branch.")

