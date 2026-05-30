# `site/` — static site for a training run

Browsable HTML report for a trained GAT run: predictions table, per-sample
detail pages, run overview, and (optionally) attention interpretability.

This folder is **generated**. Do not edit files here by hand — they will be
overwritten on the next site build.

## What gets generated

```
site/
├── index.html              # predictions table (sortable, filterable)
├── overview.html           # run summary (metrics, confusion matrix, etc.)
├── samples/
│   └── sample_<idx>.html   # one page per evaluated sample
└── interpret/              # only if you run interpret_viz with --html
    ├── index.html          # gallery + explanation
    └── subgraph_*.html     # interactive PPI subgraphs (pyvis)
```

## How to build it

From the `gat-project/` directory, after `src.train` and `src.evaluate` have
finished:

```bash
# 1. (Optional but recommended) attention interpretability
python -m src.interpret               # writes CSVs into the run dir
python -m src.interpret_viz --html    # writes site/interpret/*.html

# 2. Predictions site (index.html, overview.html, samples/)
python -m scripts.build_static_site
```

The builder picks the **latest** `run_*` directory under
`data/processed/` by default. Override with flags:

```bash
python -m scripts.build_static_site --run data/processed/run_20260530_122720
python -m scripts.build_static_site --split all          # test|val|train|all
python -m scripts.build_static_site --max-samples 100    # cap sample pages
python -m scripts.build_static_site --out path/to/dir    # custom output dir
```

> The default `--out` is `<repo-root>/docs/` (GitHub Pages source). When you
> run without flags from `gat-project/`, the builder writes to the
> repo-level `docs/` directory; this `site/` folder is the in-tree mirror
> used during development. Keep the two consistent if you want both
> populated.

## Order matters

`scripts.build_static_site` adds a header link to `interpret/index.html`. If
you want that link to resolve, run `src.interpret_viz --html` **before** (or
alongside) the site build so the target exists.

## Regenerating after a new training run

```bash
python -m src.train
python -m src.evaluate
python -m src.interpret
python -m src.interpret_viz --html
python -m scripts.build_static_site
```

That sequence rebuilds every artifact this site depends on.

## Viewing locally

```bash
python -m http.server --directory site 8000
# then open http://localhost:8000
```

Plain `file://` works too, but the interactive subgraphs render more
reliably under an HTTP server.
