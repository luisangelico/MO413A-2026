"""Graph-level embedding analyses for the trained GAT.

Extracts the post-pool embedding for every sample, then writes:
  - embeddings.npy / embeddings.csv : raw vectors + metadata
  - figures/embedding_pca.png       : PCA(2D) colored by class / correctness / confidence
  - figures/embedding_tsne.png      : t-SNE(2D), same colorings
  - knn_neighbors.csv               : top-K cosine neighbors for every sample
  - site/embeddings.html            : interactive scatter (D3) with hover + neighbor list

Run:
    python -m src.embeddings                 # latest run, test split
    python -m src.embeddings --split all
    python -m src.embeddings --run <run_dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

from src.config import CLASS_NAMES, DATASET_FILE, PROCESSED_DATASET_PATH, get_device
from src.model import load_model
from src.site_header import HEADER_CSS, render_header


def _resolve_run(run_arg: str | None) -> Path:
    if run_arg:
        return Path(run_arg)
    runs = sorted(PROCESSED_DATASET_PATH.glob("run_*"), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise SystemExit("No run_* directories found.")
    return runs[-1]


def extract_embeddings(model, dataset, indices, device):
    """Returns (Z [N, D], meta DataFrame)."""
    model.eval()
    embeddings = []
    rows = []
    with torch.no_grad():
        for i, idx in enumerate(indices):
            data = dataset[int(idx)]
            batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)
            logits, emb = model(
                data.x.to(device), data.edge_index.to(device), batch,
                return_embedding=True,
            )
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            y_pred = int(probs.argmax())
            y_true = int(data.y.item())
            embeddings.append(emb.cpu().numpy()[0])
            rows.append({
                "idx": int(idx),
                "sample_id": getattr(data, "paciente_id", f"sample_{idx}"),
                "y_true": y_true,
                "y_pred": y_pred,
                "true_label": CLASS_NAMES[y_true],
                "pred_label": CLASS_NAMES[y_pred],
                "confidence": float(probs[y_pred]),
                "correct": bool(y_pred == y_true),
            })
            if (i + 1) % 100 == 0:
                print(f"  embedded: {i + 1}/{len(indices)}")
    Z = np.stack(embeddings)
    meta = pd.DataFrame(rows)
    return Z, meta


def project_2d(Z: np.ndarray) -> dict[str, np.ndarray]:
    """PCA(2D) and t-SNE(2D). Returns dict of {name: [N,2]}."""
    out = {}
    pca = PCA(n_components=2, random_state=42)
    out["PCA"] = pca.fit_transform(Z)
    perplexity = max(5, min(30, len(Z) // 4))
    tsne = TSNE(n_components=2, random_state=42, init="pca",
                perplexity=perplexity, learning_rate="auto")
    out["t-SNE"] = tsne.fit_transform(Z)
    return out


CLASS_COLORS = {
    "Primary Tumor": "#d62728",
    "Metastasis":    "#ff7f0e",
    "Normal Tissue": "#2ca02c",
}


def plot_projection(coords, meta, name, fig_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # by class
    ax = axes[0]
    for cls, color in CLASS_COLORS.items():
        m = meta["true_label"] == cls
        ax.scatter(coords[m, 0], coords[m, 1], c=color, s=22, alpha=0.75,
                   label=f"{cls} (n={m.sum()})", edgecolor="white", linewidth=0.4)
    ax.set_title(f"{name} — by true class")
    ax.legend(fontsize=8, loc="best")
    ax.set_xlabel(f"{name} 1"); ax.set_ylabel(f"{name} 2")

    # by correctness
    ax = axes[1]
    correct = meta["correct"].values
    ax.scatter(coords[correct, 0], coords[correct, 1], c="#2ca02c", s=22,
               alpha=0.7, label=f"correct (n={correct.sum()})",
               edgecolor="white", linewidth=0.4)
    ax.scatter(coords[~correct, 0], coords[~correct, 1], c="#d62728", s=30,
               alpha=0.85, label=f"misclassified (n={(~correct).sum()})",
               edgecolor="black", linewidth=0.5, marker="X")
    ax.set_title(f"{name} — correct vs misclassified")
    ax.legend(fontsize=8)
    ax.set_xlabel(f"{name} 1"); ax.set_ylabel(f"{name} 2")

    # by confidence
    ax = axes[2]
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=meta["confidence"].values,
                    cmap="viridis", s=22, alpha=0.85,
                    edgecolor="white", linewidth=0.4, vmin=0.33, vmax=1.0)
    plt.colorbar(sc, ax=ax, label="confidence")
    ax.set_title(f"{name} — by prediction confidence")
    ax.set_xlabel(f"{name} 1"); ax.set_ylabel(f"{name} 2")

    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def knn_table(Z: np.ndarray, meta: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    sim = cosine_similarity(Z)
    np.fill_diagonal(sim, -np.inf)
    rows = []
    ids = meta["sample_id"].values
    truths = meta["true_label"].values
    for i in range(len(Z)):
        nn = np.argsort(-sim[i])[:k]
        for rank, j in enumerate(nn):
            rows.append({
                "query_idx": int(meta.iloc[i]["idx"]),
                "query_id": ids[i],
                "query_class": truths[i],
                "rank": rank,
                "neighbor_id": ids[j],
                "neighbor_class": truths[j],
                "cosine": float(sim[i, j]),
                "same_class": bool(truths[i] == truths[j]),
            })
    return pd.DataFrame(rows)


def write_html(coords_dict, meta, knn_df, out_path: Path, run_name: str):
    """Single-file interactive scatter (vanilla JS) with class filter and neighbor list."""
    payload = {
        "run": run_name,
        "samples": [
            {
                "idx": int(r["idx"]),
                "id": r["sample_id"],
                "true": r["true_label"],
                "pred": r["pred_label"],
                "conf": round(float(r["confidence"]), 4),
                "ok": bool(r["correct"]),
                "PCA":  [float(coords_dict["PCA"][i, 0]),  float(coords_dict["PCA"][i, 1])],
                "tSNE": [float(coords_dict["t-SNE"][i, 0]), float(coords_dict["t-SNE"][i, 1])],
            }
            for i, r in meta.reset_index(drop=True).iterrows()
        ],
    }
    # Build neighbor lookup: query_id -> list of (rank, neighbor_id, neighbor_class, cosine)
    nn_lookup: dict[str, list] = {}
    for _, r in knn_df.iterrows():
        nn_lookup.setdefault(r["query_id"], []).append({
            "rank": int(r["rank"]),
            "id": r["neighbor_id"],
            "class": r["neighbor_class"],
            "cos": round(float(r["cosine"]), 4),
        })

    payload_json = json.dumps(payload)
    nn_json = json.dumps(nn_lookup)

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Embeddings — Skin Cancer Gene-Network Analysis</title>
<style>
{HEADER_CSS}
  body {{ font-family:-apple-system,system-ui,sans-serif; margin:0;
          color:#222; line-height:1.5; }}
  .page {{ max-width:1100px; margin:0 auto; padding:0 1em 2em; }}
  h1 {{ font-size:1.4em; margin:0.4em 0 0.1em; }}
  .meta {{ color:#666; font-size:0.9em; }}
  .controls {{ margin:0.8em 0; display:flex; gap:0.8em; align-items:center;
               flex-wrap:wrap; }}
  .controls select, .controls label {{ font-size:0.92em; }}
  .layout {{ display:grid; grid-template-columns: 1.4fr 1fr; gap:1em; }}
  @media (max-width:900px) {{ .layout {{ grid-template-columns:1fr; }} }}
  svg {{ background:#fafafa; border:1px solid #ddd; border-radius:6px; width:100%;
         height:560px; }}
  circle {{ cursor:pointer; }}
  circle.selected {{ stroke:#000; stroke-width:2; }}
  .tooltip {{ position:absolute; background:#222; color:white; padding:6px 10px;
              border-radius:4px; font-size:0.82em; pointer-events:none;
              opacity:0; transition:opacity 0.15s; }}
  .panel {{ background:#f7f7f9; border-radius:6px; padding:0.9em 1em;
            font-size:0.92em; max-height:560px; overflow-y:auto; }}
  .panel h3 {{ margin-top:0; font-size:1.05em; }}
  table {{ width:100%; border-collapse:collapse; font-size:0.88em; }}
  th, td {{ padding:0.3em 0.4em; border-bottom:1px solid #e0e0e0; text-align:left; }}
  .legend span {{ display:inline-block; width:10px; height:10px; border-radius:50%;
                  margin-right:0.3em; vertical-align:middle; }}
  .same {{ color:#2ca02c; font-weight:600; }}
  .diff {{ color:#d62728; }}
</style></head><body>
{render_header("embeddings", subtitle=f"Run: <code>{run_name}</code>")}
<div class="page">

<h1>Graph-level embeddings</h1>
<p class="meta">
  Each point is one sample's post-pooling embedding from the trained GAT,
  projected to 2D. Click a point to see its top-10 nearest neighbors in the
  full embedding space (cosine similarity).
</p>

<details open style="margin:0.8em 0 1.2em; background:#f6fbf7;
        border-left:3px solid #2ca02c; border-radius:4px; padding:0.7em 1em;">
  <summary style="cursor:pointer; font-weight:600;">What does an "embedding" represent here?</summary>

  <p><strong>The short version.</strong> An embedding is the model's
  64-number fingerprint of one sample. Two samples with similar embeddings
  are samples the model thinks look alike — based on both their gene
  expression <em>and</em> how that expression propagates through the
  protein–protein interaction network.</p>

  <p><strong>How it's built.</strong> The model processes each sample's PPI
  graph through three GAT layers. Each of the 1000 genes ends up with a
  64-number vector that mixes its own expression with attention-weighted
  information from its PPI neighbors (and their neighbors, and theirs —
  three hops out). The model then averages those 1000 vectors into one
  <strong>64-dimensional summary of the whole sample</strong>. That
  averaged vector is the embedding.</p>

  <p><strong>What it represents.</strong> Think of it as the model's
  compressed answer to: <em>"if I had to describe this sample with 64
  numbers, capturing both what genes are highly expressed AND how those
  genes relate to each other through known protein interactions, what
  would they be?"</em></p>

  <p>Two samples will have similar embeddings if they share expression
  patterns in the genes the model learned matter, <strong>and</strong> the
  attention flow over the PPI graph reaches the same conclusion about
  them. That's stronger than "similar expression" alone — what matters is
  the expression plus how it propagates through the network.</p>

  <p><strong>What it is <em>not</em>.</strong></p>
  <ul>
    <li><strong>Not the prediction.</strong> The prediction is one more
      step: a linear layer turns the 64-d embedding into 3 class logits.
      The embedding lives upstream of that decision.</li>
    <li><strong>Not interpretable per-dimension.</strong> Each of the 64
      numbers has no individual meaning — only the geometry (distances,
      directions, neighborhoods) is meaningful.</li>
    <li><strong>Not biological coordinates.</strong> There's no axis
      labeled "tumor-ness" or "immune activity" — those are things we
      <em>find</em> by analysis (see the pseudotime page), not things the
      model was told to encode.</li>
  </ul>

  <p><strong>Why we look at them.</strong> The embedding is the most
  compact, lossy-but-faithful fingerprint of the sample as the model sees
  it. The geometry below — clusters, neighborhoods, distances — is the
  model's view of how skin samples relate to each other, with the noise
  in the original 1000-gene vector compressed out by training.</p>
</details>

<div class="controls">
  <label>Projection:
    <select id="proj">
      <option value="PCA">PCA</option>
      <option value="tSNE" selected>t-SNE</option>
    </select>
  </label>
  <label>Color by:
    <select id="color">
      <option value="true">true class</option>
      <option value="ok">correct vs misclassified</option>
      <option value="conf">confidence</option>
    </select>
  </label>
  <span class="legend" id="legend"></span>
</div>

<div class="layout">
  <div style="position:relative;">
    <svg id="scatter"></svg>
    <div class="tooltip" id="tip"></div>
  </div>
  <div class="panel" id="panel">
    <h3>Click a point</h3>
    <p class="meta">A sample's 10 nearest neighbors in embedding space will appear here.</p>
  </div>
</div>

<script>
const DATA = {payload_json};
const NN   = {nn_json};
const CLASS_COLORS = {{
  "Primary Tumor": "#d62728",
  "Metastasis":    "#ff7f0e",
  "Normal Tissue": "#2ca02c"
}};

const svg = document.getElementById("scatter");
const tip = document.getElementById("tip");
const panel = document.getElementById("panel");
const projSel = document.getElementById("proj");
const colorSel = document.getElementById("color");
const legend = document.getElementById("legend");
let selected = null;

function colorFor(s, mode) {{
  if (mode === "true") return CLASS_COLORS[s.true] || "#999";
  if (mode === "ok")   return s.ok ? "#2ca02c" : "#d62728";
  // confidence -> viridis-ish gradient
  const c = Math.max(0, Math.min(1, (s.conf - 0.33) / 0.67));
  // simple blue->yellow ramp
  const r = Math.round(68 + (253 - 68) * c);
  const g = Math.round(1  + (231 - 1) * c);
  const b = Math.round(84 + (37  - 84) * c);
  return `rgb(${{r}},${{g}},${{b}})`;
}}

function updateLegend(mode) {{
  if (mode === "true") {{
    legend.innerHTML = Object.entries(CLASS_COLORS)
      .map(([k,v])=>`<span style="background:${{v}}"></span>${{k}}&nbsp;&nbsp;`).join("");
  }} else if (mode === "ok") {{
    legend.innerHTML = '<span style="background:#2ca02c"></span>correct&nbsp;&nbsp;'
                    + '<span style="background:#d62728"></span>misclassified';
  }} else {{
    legend.innerHTML = 'confidence: <span style="background:rgb(68,1,84)"></span>low '
                    + '<span style="background:rgb(253,231,37)"></span>high';
  }}
}}

function render() {{
  const proj = projSel.value;
  const colorMode = colorSel.value;
  updateLegend(colorMode);

  const W = svg.clientWidth, H = svg.clientHeight, P = 30;
  const xs = DATA.samples.map(s => s[proj][0]);
  const ys = DATA.samples.map(s => s[proj][1]);
  const xmin = Math.min(...xs), xmax = Math.max(...xs);
  const ymin = Math.min(...ys), ymax = Math.max(...ys);
  const sx = x => P + (W - 2*P) * (x - xmin) / (xmax - xmin || 1);
  const sy = y => H - P - (H - 2*P) * (y - ymin) / (ymax - ymin || 1);

  svg.innerHTML = "";
  DATA.samples.forEach((s, i) => {{
    const c = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    c.setAttribute("cx", sx(s[proj][0]));
    c.setAttribute("cy", sy(s[proj][1]));
    c.setAttribute("r", colorMode === "ok" && !s.ok ? 6 : 4.5);
    c.setAttribute("fill", colorFor(s, colorMode));
    c.setAttribute("opacity", "0.8");
    c.setAttribute("stroke", "white");
    c.setAttribute("stroke-width", "0.6");
    c.dataset.idx = s.idx;
    c.addEventListener("mousemove", e => {{
      tip.style.opacity = 1;
      tip.style.left = (e.pageX + 12) + "px";
      tip.style.top  = (e.pageY + 12) + "px";
      tip.innerHTML = `<b>${{s.id}}</b><br>true: ${{s.true}}<br>pred: ${{s.pred}}`
                    + `<br>conf: ${{(s.conf*100).toFixed(1)}}%`;
    }});
    c.addEventListener("mouseleave", () => tip.style.opacity = 0);
    c.addEventListener("click", () => selectSample(s));
    svg.appendChild(c);
  }});
  if (selected) highlight(selected);
}}

function highlight(s) {{
  document.querySelectorAll("#scatter circle").forEach(c => {{
    c.classList.toggle("selected", parseInt(c.dataset.idx) === s.idx);
  }});
}}

function selectSample(s) {{
  selected = s;
  highlight(s);
  const neighbors = NN[s.id] || [];
  const rows = neighbors.map(n => `
    <tr>
      <td>${{n.rank + 1}}</td>
      <td>${{n.id}}</td>
      <td class="${{n.class === s.true ? 'same' : 'diff'}}">${{n.class}}</td>
      <td>${{n.cos.toFixed(3)}}</td>
    </tr>`).join("");
  panel.innerHTML = `
    <h3>${{s.id}}</h3>
    <p class="meta">true: <b>${{s.true}}</b> · pred: <b>${{s.pred}}</b>
      · conf: ${{(s.conf*100).toFixed(1)}}% · ${{s.ok ? '<span class="same">correct</span>' : '<span class="diff">misclassified</span>'}}</p>
    <h3 style="margin-top:1em;">Top-10 nearest neighbors</h3>
    <table>
      <tr><th>#</th><th>sample</th><th>class</th><th>cosine</th></tr>
      ${{rows}}
    </table>`;
}}

projSel.addEventListener("change", render);
colorSel.addEventListener("change", render);
window.addEventListener("resize", render);
render();
</script>
</div>
</body></html>
"""
    out_path.write_text(html)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default=None)
    ap.add_argument("--split", choices=["test", "val", "train", "all"], default="test")
    ap.add_argument("--k", type=int, default=10, help="Nearest-neighbor count")
    ap.add_argument("--site-dir", type=str, default=None,
                    help="Where to write embeddings.html (default: <repo>/site)")
    args = ap.parse_args()

    run_dir = _resolve_run(args.run)
    print(f"Run: {run_dir}")
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    model_path = run_dir / "best_model.pt"
    if not model_path.exists():
        model_path = run_dir / "modelo_final.pt"
    device = get_device()
    model, config = load_model(model_path, device=device)
    num_classes = config["num_classes"]

    dataset = torch.load(DATASET_FILE, weights_only=False)
    if num_classes == 2:
        dataset = [d for d in dataset if int(d.y.item()) != 2]

    splits_path = run_dir / "splits.npz"
    if args.split == "all" or not splits_path.exists():
        idx = np.arange(len(dataset))
        print(f"Embedding all {len(idx)} samples")
    else:
        idx = np.load(splits_path)[args.split]
        print(f"Embedding {args.split} split: {len(idx)} samples")

    print("Extracting embeddings...")
    Z, meta = extract_embeddings(model, dataset, idx, device)
    np.save(run_dir / "embeddings.npy", Z)
    meta.to_csv(run_dir / "embeddings.csv", index=False)
    print(f"Saved: {run_dir / 'embeddings.npy'} (shape {Z.shape})")

    print("Projecting to 2D (PCA + t-SNE)...")
    coords = project_2d(Z)

    plot_projection(coords["PCA"],   meta, "PCA",   fig_dir / "embedding_pca.png")
    plot_projection(coords["t-SNE"], meta, "t-SNE", fig_dir / "embedding_tsne.png")
    print(f"Saved: {fig_dir / 'embedding_pca.png'}")
    print(f"Saved: {fig_dir / 'embedding_tsne.png'}")

    print(f"Computing top-{args.k} cosine neighbors...")
    knn_df = knn_table(Z, meta, k=args.k)
    knn_df.to_csv(run_dir / "knn_neighbors.csv", index=False)
    same_class_rate = knn_df["same_class"].mean()
    print(f"Saved: {run_dir / 'knn_neighbors.csv'}")
    print(f"  kNN same-class rate (k={args.k}): {same_class_rate:.3f}")

    site_dir = Path(args.site_dir) if args.site_dir else (
        Path(__file__).resolve().parent.parent / "site")
    site_dir.mkdir(parents=True, exist_ok=True)
    out_html = site_dir / "embeddings.html"
    write_html(coords, meta, knn_df, out_html, run_dir.name)
    print(f"Saved: {out_html}")


if __name__ == "__main__":
    main()
