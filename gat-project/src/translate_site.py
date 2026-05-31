"""Generate the Portuguese (pt-br) mirror of the static site.

For pages that already render their own bilingual HTML
(about_gat_page, dataset_page if applicable), this script does nothing —
they should be invoked separately with --lang pt-br.

For the remaining pages (predictions, overview, embeddings, pseudotime,
stability, biomarkers, interpret), this script:
  1. Reads the rendered English HTML from <site>/.
  2. Replaces its <header class="site-header"> with a freshly-rendered
     PT header (translated nav labels, language toggle pointing at
     the EN twin, prefix accounting for pt-br/ depth).
  3. Prepends a "Versão em português em construção" callout right
     after the header.
  4. Writes the result to <site>/pt-br/.
  5. Copies all sibling figures/HTML assets the EN page references,
     so relative paths still resolve.

Run after the regular EN rebuild:
    python -m src.translate_site --site docs
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

from src.i18n import t
from src.site_header import HEADER_CSS, render_header


# Map of EN page filename → page slug (matches NAV_LINKS).
PAGES = {
    "index.html":      ("predictions", ""),
    "overview.html":   ("overview",    ""),
    "dataset.html":    ("dataset",     ""),
    "embeddings.html": ("embeddings",  ""),
    "pseudotime.html": ("pseudotime",  ""),
    "stability.html":  ("stability",   ""),
    "biomarkers.html": ("biomarkers",  ""),
    "interpret/index.html": ("interpret", "interpret/"),
}

# Asset filenames (relative to <site>/) to copy alongside the PT pages.
SHARED_ASSETS = [
    "dataset_overview.png", "dataset_ppi_graph.png", "dataset_ppi_graph.html",
    "pseudotime_heatmap.png", "pseudotime_lines.png",
    "stability_accuracy.png", "stability_spearman.png",
    "stability_knn_jaccard.png", "stability_pseudotime.png",
]


HEADER_RE = re.compile(
    r'<header class="site-header">.*?</header>\s*',
    re.DOTALL,
)


def _build_callout() -> str:
    body = t("pt_construction_callout", "pt-br")
    return (
        '<div style="background:#fffbf2;border-left:3px solid #e0a020;'
        'padding:0.7em 1em;margin:1em auto;max-width:1100px;'
        'font-family:-apple-system,system-ui,sans-serif;line-height:1.55;">'
        f'{body}'
        '</div>'
    )


def translate_page(en_path: Path, pt_path: Path, slug: str, sub_prefix: str) -> None:
    """Read EN page, swap header for PT header + callout, write to pt_path."""
    if not en_path.exists():
        print(f"  skip (missing): {en_path}")
        return
    raw = en_path.read_text()

    # prefix from the PT page back to the site root:
    #   pt-br/<slug>.html              -> "../"
    #   pt-br/<sub>/<slug>.html        -> "../../"
    prefix = "../" if sub_prefix == "" else "../../"
    new_header = render_header(slug, lang="pt-br", prefix=prefix)
    callout = _build_callout()

    if HEADER_RE.search(raw):
        raw = HEADER_RE.sub(new_header + callout, raw, count=1)
    else:
        # Fall back: inject after <body>
        raw = raw.replace("<body>", "<body>\n" + new_header + callout, 1)

    # Adjust the <title> tag if present
    raw = re.sub(
        r"<title>([^<]*)</title>",
        lambda m: f"<title>{m.group(1)} (PT)</title>",
        raw, count=1,
    )

    pt_path.parent.mkdir(parents=True, exist_ok=True)
    pt_path.write_text(raw)
    print(f"  wrote: {pt_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", type=str, default="site",
                    help="Site root (where index.html lives). Default: site")
    args = ap.parse_args()

    site_root = Path(args.site).resolve()
    pt_root = site_root / "pt-br"
    pt_root.mkdir(parents=True, exist_ok=True)
    print(f"Site: {site_root}")
    print(f"PT:   {pt_root}")

    # Translate the chrome-only pages
    for filename, (slug, sub) in PAGES.items():
        en = site_root / filename
        pt = pt_root / filename
        translate_page(en, pt, slug, sub)

    # Copy sibling asset files (figures, embedded interactive subgraphs)
    for asset in SHARED_ASSETS:
        src = site_root / asset
        if src.exists():
            dst = pt_root / asset
            shutil.copyfile(src, dst)

    # Copy the per-class subgraph HTML files inside interpret/
    interp_en = site_root / "interpret"
    interp_pt = pt_root / "interpret"
    if interp_en.exists():
        interp_pt.mkdir(exist_ok=True)
        for f in interp_en.glob("subgraph_*.html"):
            shutil.copyfile(f, interp_pt / f.name)

    # samples/ — the PT site reuses the EN per-sample pages by symlink-equivalent
    # (relative href). Easiest is to copy them, but they're 193 files. We instead
    # leave the predictions table linking back to the EN samples/ via "../samples/".
    # Patch the index.html inside pt-br to point at ../samples/ instead of samples/.
    pt_index = pt_root / "index.html"
    if pt_index.exists():
        raw = pt_index.read_text()
        # Replace `'samples/sample_${r.idx}.html'` with `'../samples/...'`
        raw = raw.replace("'samples/sample_${r.idx}.html'",
                          "'../samples/sample_${r.idx}.html'")
        pt_index.write_text(raw)

    print(f"\nDone. {pt_root}/index.html is the PT entry point.")


if __name__ == "__main__":
    main()
