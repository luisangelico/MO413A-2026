"""Shared header / nav for every page in the static site.

One source of truth for the project title, the link bar, and the highlighted
"current page" state. All page generators import render_header() and inject
the result + HEADER_CSS into their templates.
"""

from __future__ import annotations

PROJECT_TITLE = "Skin Cancer Gene-Network Analysis"
PROJECT_TAGLINE = "Graph attention networks over PPI for melanoma classification"

# (slug, label, href). The slug is used to highlight the active page.
NAV_LINKS = [
    ("predictions", "Predictions",     "index.html"),
    ("about-gat",   "What is a GAT?",  "about-gat.html"),
    ("overview",    "Run overview",    "overview.html"),
    ("dataset",     "Dataset",         "dataset.html"),
    ("interpret",   "Interpretability","interpret/index.html"),
    ("embeddings",  "Embeddings",      "embeddings.html"),
    ("pseudotime",  "Pseudotime",      "pseudotime.html"),
    ("stability",   "Stability",       "stability.html"),
    ("biomarkers",  "Biomarkers",      "biomarkers.html"),
]

HEADER_CSS = """
.site-header { background: linear-gradient(180deg, #1a2332 0%, #243047 100%);
               color: white; padding: 1.1em 0 0.7em; margin: 0 0 1.4em 0;
               border-bottom: 3px solid #1f77b4;
               box-shadow: 0 1px 4px rgba(0,0,0,0.08); }
.site-header .inner { max-width: 1100px; margin: 0 auto; padding: 0 1.4em; }
.site-header h1.site-title { font-size: 1.25em; margin: 0; font-weight: 600;
                  letter-spacing: 0.01em; color: white; line-height: 1.2; }
.site-header .tagline { color: #94a3bd; font-size: 0.85em; margin: 0.2em 0 0; }
.site-header .subtitle { color: #cdd6e3; font-size: 0.92em; margin-top: 0.5em; }
.site-nav { display: flex; flex-wrap: wrap; gap: 0.15em; margin-top: 0.85em; }
.site-nav a { color: #cdd6e3; text-decoration: none; padding: 0.45em 0.85em;
              border-radius: 5px; font-size: 0.88em;
              transition: background 0.12s, color 0.12s; }
.site-nav a:hover { background: rgba(255,255,255,0.08); color: white; }
.site-nav a.active { background: #1f77b4; color: white; font-weight: 500; }
@media (max-width: 700px) {
  .site-header h1.site-title { font-size: 1.1em; }
  .site-nav a { padding: 0.35em 0.7em; font-size: 0.82em; }
}
"""


def render_header(current: str, subtitle: str = "", prefix: str = "") -> str:
    """Render the shared site header.

    current: slug of the active page (must match a NAV_LINKS first column,
             or an empty string to render no active link).
    subtitle: optional page-specific subtitle (e.g. dataset name, run id).
    prefix: relative path prefix to reach the docs root from the current
            page. Use "" for top-level pages, "../" for pages inside
            subdirectories like interpret/ or samples/.
    """
    nav_items = []
    for slug, label, href in NAV_LINKS:
        cls = ' class="active"' if slug == current else ""
        nav_items.append(f'    <a href="{prefix}{href}"{cls}>{label}</a>')
    nav_html = "\n".join(nav_items)
    sub_html = f'\n    <div class="subtitle">{subtitle}</div>' if subtitle else ""
    return f"""<header class="site-header">
  <div class="inner">
    <h1 class="site-title">{PROJECT_TITLE}</h1>
    <div class="tagline">{PROJECT_TAGLINE}</div>{sub_html}
    <nav class="site-nav">
{nav_html}
    </nav>
  </div>
</header>
"""
