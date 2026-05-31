"""Generate the 'What is a GAT?' explainer page.

A non-technical introduction to Graph Attention Networks, what makes them
suitable for this project, and how each step of the model maps onto a
familiar idea (a sample = a graph; attention = a learned vote on which
neighbors matter; embedding = the model's fingerprint of a sample).

Run:
    python -m src.about_gat_page                  # writes to <repo>/site
    python -m src.about_gat_page --site-dir docs
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.site_header import HEADER_CSS, render_header


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>What is a GAT? — Skin Cancer Gene-Network Analysis</title>
<style>
__HEADER_CSS__
  body { font-family:-apple-system,system-ui,sans-serif; margin:0;
         color:#222; line-height:1.65; }
  .page { max-width:860px; margin:0 auto; padding:0 1em 2em; }
  h1 { font-size:1.5em; margin:0.4em 0 0.2em; }
  h2 { font-size:1.15em; margin-top:1.8em; }
  h3 { font-size:1em; margin-top:1.2em; color:#333; }
  .meta { color:#666; font-size:0.9em; }
  .plain { background:#f6fbf7; border-left:3px solid #2ca02c;
           padding:0.7em 1em; margin:1em 0; }
  .callout { background:#fffbf2; border-left:3px solid #e0a020;
             padding:0.7em 1em; margin:1em 0; }
  .grid2 { display:grid; grid-template-columns:1fr 1fr; gap:1em; margin:0.8em 0; }
  @media (max-width:760px) { .grid2 { grid-template-columns:1fr; } }
  .card { background:#f6f8fc; border:1px solid #d8def0; border-radius:8px;
          padding:0.9em 1em; }
  .card h3 { margin:0 0 0.3em 0; font-size:1em; }
  code { background:#f4f4f4; padding:0 0.3em; border-radius:3px; }
  ul, ol { line-height:1.85; }
  .formula { background:#f7f7f9; border-radius:6px; padding:0.7em 1em;
             font-family: ui-monospace, Menlo, monospace; font-size:0.9em;
             margin:0.6em 0; overflow-x:auto; }
  .next a { display:inline-block; padding:0.4em 0.8em; margin:0.3em 0.4em 0 0;
            background:#1f77b4; color:white; border-radius:5px;
            text-decoration:none; font-size:0.9em; }
</style></head><body>
__HEADER__
<div class="page">

<h1>What is a Graph Attention Network?</h1>
<p class="meta">A plain-language tour of the architecture this project uses,
and why graphs (not flat lists of numbers) are the right shape for the
problem.</p>

<div class="plain">
  <strong>In one sentence.</strong> A Graph Attention Network (GAT) is a neural
  network that reads a <em>graph</em> — nodes connected by edges — and learns
  not just what each node looks like on its own, but how much each node
  should listen to each of its neighbors when forming an opinion.
</div>

<h2>Why graphs?</h2>
<p>
  Most machine-learning models eat a flat row of numbers: 1000 gene
  expression values per sample, say. That works, but it throws away
  something important — biologists know that <strong>genes don't act in
  isolation</strong>. They form pathways, complexes, regulatory cascades.
  Two genes that physically bind each other behave very differently from
  two genes that have nothing to do with one another, even if their
  individual expression values look identical.
</p>
<p>
  A graph captures exactly that structure. In this project:
</p>
<ul>
  <li>Each <strong>node</strong> is a gene, carrying its measured expression
    as a feature.</li>
  <li>Each <strong>edge</strong> is a known protein–protein interaction
    (PPI) from STRING — two proteins that physically bind.</li>
  <li>Each <strong>sample</strong> (one patient) becomes one full graph:
    same edges as everyone else, but the per-node expression values
    differ patient to patient.</li>
</ul>
<p>
  The GAT's job: look at this whole graph and decide whether the patient
  is normal skin, primary melanoma, or metastatic melanoma.
</p>

<h2>What does "attention" mean here?</h2>
<p>
  Imagine each gene is a person at a meeting. Their PPI partners are the
  other people they're allowed to talk to. Now ask each person to summarize
  what's going on. They'll listen to their neighbors — but not equally.
  Some neighbors are highly relevant; others, less so. <strong>Attention</strong>
  is a learned weighting that says, for every connection, "how much should
  this person rely on what that person says?"
</p>
<p>
  Mathematically, for a node <code>j</code> updating itself based on its
  neighbors <code>i</code>, attention assigns each incoming edge a weight
  α(i→j) ∈ [0, 1], with all weights summing to 1 across <code>j</code>'s
  neighbors. The new representation of <code>j</code> is a weighted sum
  of its neighbors' messages.
</p>
<div class="formula">
  h<sub>j</sub><sup>new</sup> = σ( Σ<sub>i ∈ neighbors(j)</sub> α(i→j) · W · h<sub>i</sub> )
</div>
<p>
  The crucial part: α is <strong>learned</strong>. The model figures out,
  from training data, which gene→gene connections matter for the task.
  That's the secret sauce, and it's also what makes attention
  <em>interpretable</em>: after training, we can read the α values and ask
  "which connections did the model rely on most?" That's the entire
  premise of the
  <a href="interpret/index.html">interpretability</a> page.
</p>

<h2>How GATv2 differs from "vanilla" GAT</h2>
<p>
  This project uses <strong>GATv2</strong>. The original GAT had a subtle
  flaw: in some configurations its attention couldn't depend on the query
  node, only on the source nodes. GATv2 reorders the operations so
  attention is genuinely <em>dynamic</em> — every node–neighbor pair gets
  its own context-dependent weight. In practice GATv2 trains more
  reliably and performs at least as well on every standard benchmark.
</p>

<h2>The architecture used here</h2>
<p>
  Three GAT layers stacked, with attention heads in parallel:
</p>
<ol>
  <li><strong>Layer 1:</strong> 1 input feature (expression) →
    128 hidden, with 4 attention heads. Each head looks at the same
    graph independently, producing its own view; their outputs are
    concatenated.</li>
  <li><strong>Layer 2:</strong> hidden → hidden, again 4 heads.
    Information now propagates two hops out — a gene "sees"
    its neighbors and its neighbors' neighbors.</li>
  <li><strong>Layer 3:</strong> hidden → hidden, 1 head, three hops out.
    Most of the long-range signal has now mixed in.</li>
  <li><strong>Pooling:</strong> the 1000 per-gene vectors are averaged
    into one 64-number summary of the whole sample — its
    <a href="embeddings.html">embedding</a>.</li>
  <li><strong>Classifier:</strong> a small linear layer turns the
    embedding into 3 probabilities (Primary / Metastasis / Normal).</li>
</ol>

<h2>What the model produces along the way</h2>
<div class="grid2">
  <div class="card">
    <h3>Edge attention</h3>
    <p>
      One number per edge per layer per head. We aggregate these into
      per-gene importance scores and study which genes the model leaned
      on. See the
      <a href="interpret/index.html">interpretability</a> and
      <a href="biomarkers.html">biomarkers</a> pages.
    </p>
  </div>
  <div class="card">
    <h3>Embeddings</h3>
    <p>
      The 64-dimensional sample fingerprints. Geometry in this space —
      clusters, distances, neighborhoods — is the model's view of how
      patients relate. See the
      <a href="embeddings.html">embeddings</a> page.
    </p>
  </div>
  <div class="card">
    <h3>Pseudotime</h3>
    <p>
      An ordering of samples on a Normal → Metastasis axis derived from
      embedding geometry, used to track which genes the model recruits
      as samples become more tumor-like.
      See <a href="pseudotime.html">pseudotime</a>.
    </p>
  </div>
  <div class="card">
    <h3>Predictions and confidence</h3>
    <p>
      The class probabilities themselves, with uncertainty calibration
      visible per sample.
      See <a href="index.html">predictions</a> and
      <a href="overview.html">run overview</a>.
    </p>
  </div>
</div>

<h2>Why a GAT is well-suited to this problem</h2>
<ul>
  <li><strong>The graph structure is real biology.</strong> Treating gene
    expression as a flat 1000-dimensional vector ignores the fact that
    biology operates through interacting protein networks. A GAT can use
    that structure directly.</li>
  <li><strong>It generalizes across patients.</strong> The same edge set
    is used for every patient (the human PPI network doesn't change
    sample to sample). What changes is each gene's expression, the node
    feature. The model learns rules about <em>how expression patterns
    propagate through the network</em>, not patient-specific facts.</li>
  <li><strong>It's interpretable.</strong> Attention weights are not
    abstract internal states — they're per-edge numbers we can read,
    aggregate, and validate against known biology.</li>
  <li><strong>Smaller than the alternative.</strong> A GAT with 128
    hidden channels has far fewer parameters than a fully-connected
    network operating on 1000 genes — the graph constrains how
    information flows, which is a useful inductive bias when training
    data are limited.</li>
</ul>

<div class="callout">
  <strong>What a GAT is <em>not</em>.</strong> It's not magic. It can't
  create signal that isn't in the data. If the input expression doesn't
  distinguish two classes, no graph structure will save it. And
  attention is a tool, not a guarantee — high attention doesn't mean
  "biologically important," only "useful to this model on this data."
  That's why every interpretability claim in this project is benchmarked
  against a node-degree baseline and validated across 5 random seeds
  (see the <a href="stability.html">stability</a> page).
</div>

<h2>Where to go next</h2>
<p class="next">
  <a href="dataset.html">Dataset</a>
  <a href="index.html">Predictions</a>
  <a href="interpret/index.html">Interpretability</a>
  <a href="embeddings.html">Embeddings</a>
  <a href="pseudotime.html">Pseudotime</a>
  <a href="stability.html">Stability</a>
</p>

</div>
</body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site-dir", type=str, default=None,
                    help="Default: <repo>/site")
    args = ap.parse_args()

    site_dir = Path(args.site_dir) if args.site_dir else (
        Path(__file__).resolve().parent.parent / "site")
    site_dir.mkdir(parents=True, exist_ok=True)
    out_path = site_dir / "about-gat.html"

    html = (PAGE
            .replace("__HEADER_CSS__", HEADER_CSS)
            .replace("__HEADER__", render_header("about-gat")))
    out_path.write_text(html)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
