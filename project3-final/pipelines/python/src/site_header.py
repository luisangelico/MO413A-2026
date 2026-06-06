"""Shared header / nav for every page in the static site.

Bilingual: 'en' (default, served from site root) and 'pt-br' (served from
<site>/pt-br/). Each page generator passes its lang to render_header().
A language-switcher pill in the top-right links to the same slug in the
opposite language.
"""

from __future__ import annotations

from src.i18n import (
    DEFAULT_LANG, NAV_LABEL, NAV_LINKS, PROJECT_TAGLINE, PROJECT_TITLE,
    lang_subdir, opposite, t,
)

# Re-export for backwards compatibility
PROJECT_TITLE_EN = PROJECT_TITLE["en"]


HEADER_CSS = """
.site-header { background: linear-gradient(180deg, #1a2332 0%, #243047 100%);
               color: white; padding: 1.1em 0 0.7em; margin: 0 0 1.4em 0;
               border-bottom: 3px solid #1f77b4;
               box-shadow: 0 1px 4px rgba(0,0,0,0.08); }
.site-header .inner { max-width: 1100px; margin: 0 auto; padding: 0 1.4em;
                       position: relative; }
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
.site-lang-switch { position: absolute; top: 0; right: 1.4em;
                    display: flex; gap: 0.3em; align-items: center;
                    font-size: 0.82em; }
.site-lang-switch a { color: #cdd6e3; text-decoration: none;
                      padding: 0.25em 0.65em; border-radius: 999px;
                      border: 1px solid rgba(255,255,255,0.18);
                      transition: background 0.12s, color 0.12s; }
.site-lang-switch a:hover { background: rgba(255,255,255,0.12); color: white; }
.site-lang-switch .current { color: white;
                              background: rgba(255,255,255,0.12);
                              border-color: rgba(255,255,255,0.3); }
@media (max-width: 700px) {
  .site-header h1.site-title { font-size: 1.1em; }
  .site-nav a { padding: 0.35em 0.7em; font-size: 0.82em; }
  .site-lang-switch { position: static; margin-top: 0.6em; }
}
"""


def _lang_switch_html(lang: str, current_slug: str, prefix: str) -> str:
    """Render the language-switcher pill cluster.

    Both languages link to the same logical page (current_slug). The
    relative URL accounts for the current page's depth (`prefix`) and
    the destination language's directory.
    """
    other = opposite(lang)
    # Find the href for the current slug
    href_for = {slug: href for slug, href in NAV_LINKS}
    target_href = href_for.get(current_slug, "index.html")

    # Self link is not really a link, just a styled current-language pill
    self_label = "EN" if lang == "en" else "PT"
    other_label = "PT" if lang == "en" else "EN"

    # Build href to the same page in the other language.
    # From any page in `lang`, the site root is `prefix`. From there the
    # target language's pages live at `prefix + lang_subdir(other) + target_href`.
    other_url = prefix + lang_subdir(other) + target_href
    self_url = prefix + lang_subdir(lang) + target_href

    return (
        f'<div class="site-lang-switch">'
        f'<a class="current" href="{self_url}" title="{self_label}">{self_label}</a>'
        f'<a href="{other_url}" title="Switch to {other_label}">{other_label}</a>'
        f'</div>'
    )


def render_header(current: str, *, lang: str = DEFAULT_LANG,
                  subtitle: str = "", prefix: str = "") -> str:
    """Render the shared site header.

    current: slug of the active page (must match a NAV_LINKS first column).
    lang:    'en' or 'pt-br'. Drives titles, tagline, link labels.
    subtitle: optional page-specific subtitle.
    prefix:  relative path from the current page back to the site root
             (where the default-language files live). For top-level pages
             in the default language: "". For pages in subdirs (e.g.
             interpret/, samples/): "../". For pt-br top-level pages:
             "../" (one level up to escape pt-br/). For pt-br/interpret/:
             "../../".
    """
    nav_items = []
    for slug, href in NAV_LINKS:
        cls = ' class="active"' if slug == current else ""
        label = NAV_LABEL[slug][lang]
        # Always link inside the current language directory
        url = prefix + lang_subdir(lang) + href
        nav_items.append(f'    <a href="{url}"{cls}>{label}</a>')
    nav_html = "\n".join(nav_items)
    sub_html = f'\n    <div class="subtitle">{subtitle}</div>' if subtitle else ""
    lang_switch = _lang_switch_html(lang, current, prefix)
    return f"""<header class="site-header">
  <div class="inner">
    {lang_switch}
    <h1 class="site-title">{PROJECT_TITLE[lang]}</h1>
    <div class="tagline">{PROJECT_TAGLINE[lang]}</div>{sub_html}
    <nav class="site-nav">
{nav_html}
    </nav>
  </div>
</header>
"""
