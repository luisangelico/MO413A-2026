"""Generate the Portuguese (pt-br) mirror of the static site, with real
machine-translated body text.

Pipeline per page:
  1. Read rendered EN HTML from <site>/.
  2. Walk the DOM: collect every visible text node (skipping <script>,
     <style>, <code>, <pre>, and elements with class="notranslate").
  3. Send the batch to Claude (Haiku) with a system prompt that (a) gives
     the biology glossary so MITF/TYR stay untranslated and "Benign Nevus"
     becomes "Nevo Benigno", and (b) tells the model to return a JSON list
     in the same order. Results are cached on disk by sha256, so re-runs
     are free unless the source text changed.
  4. Splice translated strings back into the DOM.
  5. Replace the EN site-header with a freshly-rendered PT header.
  6. Adjust <title> and HTML <html lang="..."> attribute.
  7. Write to <site>/pt-br/.

Backend (auto-detected, override with TRANSLATE_BACKEND={bedrock,anthropic}):
  - AWS Bedrock if AWS creds are present (AWS_ACCESS_KEY_ID / AWS_PROFILE /
    AWS_BEARER_TOKEN_BEDROCK). Default region: us-east-1 (set AWS_REGION).
  - Anthropic API if ANTHROPIC_API_KEY is set.
  - Otherwise falls back to the previous "shell mirror" behavior (PT header
    + 'page under construction' callout, body left in English) so the build
    still succeeds in offline/CI contexts.

Override the default model with TRANSLATE_MODEL=<id>.

Run after the regular EN rebuild:
    python -m src.translate_site --site site
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from html.parser import HTMLParser
from pathlib import Path

from src.i18n import t
from src.site_header import HEADER_CSS, render_header


PAGES = {
    "index.html":      ("predictions", ""),
    "exploration.html": ("exploration", ""),
    "overview.html":   ("overview",    ""),
    "dataset.html":    ("dataset",     ""),
    "embeddings.html": ("embeddings",  ""),
    "pseudotime.html": ("pseudotime",  ""),
    "stability.html":  ("stability",   ""),
    "biomarkers.html": ("biomarkers",  ""),
    "interpret/index.html": ("interpret", "interpret/"),
}

SHARED_ASSETS = [
    "dataset_overview.png", "dataset_ppi_graph.png", "dataset_ppi_graph.html",
    "pseudotime_heatmap.png", "pseudotime_lines.png",
    "stability_accuracy.png", "stability_spearman.png",
    "stability_knn_jaccard.png", "stability_pseudotime.png",
    "exploration_data.js",
]

# Tags whose text content we never translate.
SKIP_TAGS = {"script", "style", "code", "pre", "title"}

# Per-batch cap (chars) so a single huge page doesn't blow the context window.
BATCH_CHAR_LIMIT = 8000

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "translation_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

HEADER_RE = re.compile(r'<header class="site-header">.*?</header>\s*', re.DOTALL)


# ---------------------------------------------------------------------------
# Text-node extraction
# ---------------------------------------------------------------------------
class TextNodeCollector(HTMLParser):
    """Walks HTML preserving exact source spans of translatable text nodes.

    Produces a list of (start, end, text) triples that index into the
    original source. We then splice translations back by replacing those
    spans in reverse order, leaving every tag, attribute, comment, doctype,
    script, and style block byte-identical.
    """

    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.spans: list[tuple[int, int, str]] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in SKIP_TAGS:
            self._skip_depth += 1
        else:
            # class="notranslate" → skip subtree
            for k, v in attrs:
                if k == "class" and v and "notranslate" in v.split():
                    self._skip_depth += 1
                    self._skip_marker_tag = tag
                    return

    def handle_endtag(self, tag):
        if tag in SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        elif self._skip_depth > 0 and getattr(self, "_skip_marker_tag", None) == tag:
            self._skip_depth -= 1
            self._skip_marker_tag = None

    def handle_data(self, data):
        if self._skip_depth > 0:
            return
        if not data.strip():
            return
        if data.strip().startswith("//") or data.strip().startswith("/*"):
            return
        # rawdata + position from the parser
        offset = self.getpos()
        # HTMLParser doesn't expose absolute offsets directly; reconstruct by
        # finding `data` starting at the previous offset. Reliable because we
        # process whole-document raw text and HTMLParser preserves data verbatim
        # in handle_data when convert_charrefs=False.
        start = self.rawdata.find(data, self._cursor)
        if start == -1:
            return
        end = start + len(data)
        self._cursor = end
        self.spans.append((start, end, data))

    def feed(self, raw):
        self._cursor = 0
        super().feed(raw)


def collect_text_spans(html: str) -> list[tuple[int, int, str]]:
    p = TextNodeCollector()
    p.feed(html)
    p.close()
    return p.spans


# ---------------------------------------------------------------------------
# Translation backend
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You translate technical website copy from English to Brazilian Portuguese (pt-br).

CONTEXT: a research website about a Graph Attention Network (GAT) classifying skin samples (melanoma vs benign nevus) using gene expression data.

GLOSSARY — translate these consistently:
- "Primary Tumor" -> "Tumor Primário"
- "Metastasis" -> "Metástase"
- "Benign Nevus" / "Benign Nevi" -> "Nevo Benigno" / "Nevos Benignos"
- "Normal Tissue" -> "Tecido Normal"
- "gene expression" -> "expressão gênica"
- "protein-protein interaction" / "PPI" -> "interação proteína-proteína" (keep "PPI" abbreviation)
- "attention" (GAT context) -> "atenção"
- "embedding" -> keep as "embedding"
- "pseudotime" -> "pseudotempo"
- "graph neural network" -> "rede neural de grafos"
- "node" -> "nó"; "edge" -> "aresta"
- "training" / "test" / "validation" splits -> "treino" / "teste" / "validação"
- "patient" -> "paciente"
- "dataset" -> "dataset" (or "conjunto de dados")
- "cohort" -> "coorte"
- "keratinization" -> "queratinização"; "keratinocyte" -> "queratinócito"
- "melanocyte" -> "melanócito"

DO NOT TRANSLATE:
- Gene names / symbols (MITF, TYR, KRT5, FLG, SOX10, etc.) — these are biological identifiers, leave verbatim.
- Cohort/database names: TCGA-SKCM, GTEx, GSE112509, STRING-DB, UCSC Xena, TOIL, recount3, Anthropic, Claude, etc.
- Code, file paths, command lines.
- Numbers, percentages, units.

STYLE:
- Match the source's tone (clear, slightly didactic — this is a research site).
- Keep the same punctuation style and sentence boundaries.
- Preserve leading/trailing whitespace exactly (if input begins with a space or newline, output must too).
- Do not add or remove sentences.

INPUT FORMAT: a JSON array of strings (each string is one HTML text node).
OUTPUT FORMAT: a JSON array of the same length, in the same order, with each element being the pt-br translation of the corresponding input string.
Output ONLY the JSON array. No prose, no markdown fences, no commentary."""


def _cache_path(text: str) -> Path:
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]
    return CACHE_DIR / f"{h}.txt"


def _load_cached(texts: list[str]) -> dict[int, str]:
    """Return {index: cached_translation} for any texts already cached."""
    out = {}
    for i, t_ in enumerate(texts):
        p = _cache_path(t_)
        if p.exists():
            out[i] = p.read_text()
    return out


def _save_cached(text: str, translation: str) -> None:
    _cache_path(text).write_text(translation)


def _batch(items: list[str]) -> list[list[int]]:
    """Group indices so each batch stays under BATCH_CHAR_LIMIT chars."""
    batches, cur, cur_chars = [], [], 0
    for i, t_ in enumerate(items):
        n = len(t_)
        if cur and cur_chars + n > BATCH_CHAR_LIMIT:
            batches.append(cur)
            cur, cur_chars = [], 0
        cur.append(i)
        cur_chars += n
    if cur:
        batches.append(cur)
    return batches


def translate_batch(client, texts: list[str]) -> list[str]:
    """Translate a batch via Claude. Returns list aligned with input."""
    payload = json.dumps(texts, ensure_ascii=False)
    model_id = os.environ.get(
        "TRANSLATE_MODEL",
        # Default depends on backend; main() injects the right one below.
        getattr(client, "_default_model", "claude-haiku-4-5-20251001"),
    )
    msg = client.messages.create(
        model=model_id,
        max_tokens=8000,
        system=[
            {"type": "text", "text": SYSTEM_PROMPT,
             "cache_control": {"type": "ephemeral"}},
        ],
        messages=[{"role": "user", "content": payload}],
    )
    raw = msg.content[0].text.strip()
    # tolerate accidental code fences
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw
        if raw.endswith("```"):
            raw = raw[:-3]
    try:
        out = json.loads(raw)
    except json.JSONDecodeError:
        # Try to salvage: extract first [...] block
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if not m:
            raise
        out = json.loads(m.group(0))
    if not isinstance(out, list) or len(out) != len(texts):
        raise ValueError(
            f"translation length mismatch: got {len(out) if isinstance(out, list) else type(out).__name__}"
            f" for {len(texts)} inputs"
        )
    return [str(x) for x in out]


def translate_all(texts: list[str], client) -> list[str]:
    """Returns a list aligned with `texts`, translated. Uses cache."""
    if not texts:
        return []
    out = list(texts)
    cached = _load_cached(texts)
    pending = [i for i in range(len(texts)) if i not in cached]
    for i, v in cached.items():
        out[i] = v
    if not pending:
        return out
    pending_texts = [texts[i] for i in pending]
    for batch_idx in _batch(pending_texts):
        sub = [pending_texts[k] for k in batch_idx]
        translated = translate_batch(client, sub)
        for k, tr in zip(batch_idx, translated):
            global_i = pending[k]
            out[global_i] = tr
            _save_cached(texts[global_i], tr)
    return out


# ---------------------------------------------------------------------------
# Per-page translation
# ---------------------------------------------------------------------------
def _strip_lang_attr(html: str) -> str:
    """Replace lang="en" / lang="..." on <html> tag with lang="pt-br"."""
    return re.sub(
        r'(<html[^>]*?)\slang="[^"]*"',
        r'\1 lang="pt-br"',
        html, count=1,
    ) or html


def _ensure_lang_attr(html: str) -> str:
    if 'lang="pt-br"' in html.lower()[:300]:
        return html
    if re.search(r'<html\b[^>]*lang=', html[:300], re.IGNORECASE):
        return _strip_lang_attr(html)
    return re.sub(r'<html\b', '<html lang="pt-br"', html, count=1)


def translate_html(raw: str, client) -> str:
    """Translate the body text of an EN HTML doc to pt-br."""
    spans = collect_text_spans(raw)
    if not spans:
        return raw
    texts = [s[2] for s in spans]
    translations = translate_all(texts, client)
    # Splice in reverse so earlier offsets stay valid
    out = raw
    for (start, end, _orig), tr in zip(reversed(spans), reversed(translations)):
        # preserve leading/trailing whitespace from original
        lead = len(_orig) - len(_orig.lstrip())
        tail = len(_orig) - len(_orig.rstrip())
        body = tr.strip()
        new_chunk = _orig[:lead] + body + (_orig[len(_orig) - tail:] if tail else "")
        out = out[:start] + new_chunk + out[end:]
    out = _ensure_lang_attr(out)
    # also translate <title>
    out = re.sub(
        r"<title>([^<]*)</title>",
        lambda m: f"<title>{_translate_title(m.group(1), client)}</title>",
        out, count=1,
    )
    return out


def _translate_title(title: str, client) -> str:
    if not title.strip():
        return title
    p = _cache_path("TITLE::" + title)
    if p.exists():
        return p.read_text()
    out = translate_batch(client, [title])[0]
    p.write_text(out)
    return out


# ---------------------------------------------------------------------------
# Fallback (no API key) — old shell-mirror behavior
# ---------------------------------------------------------------------------
def _build_callout() -> str:
    body = t("pt_construction_callout", "pt-br")
    return (
        '<div style="background:#fffbf2;border-left:3px solid #e0a020;'
        'padding:0.7em 1em;margin:1em auto;max-width:1100px;'
        'font-family:-apple-system,system-ui,sans-serif;line-height:1.55;">'
        f'{body}'
        '</div>'
    )


def shell_mirror(raw: str, slug: str, sub_prefix: str) -> str:
    prefix = "../" if sub_prefix == "" else "../../"
    new_header = render_header(slug, lang="pt-br", prefix=prefix)
    callout = _build_callout()
    if HEADER_RE.search(raw):
        raw = HEADER_RE.sub(new_header + callout, raw, count=1)
    else:
        raw = raw.replace("<body>", "<body>\n" + new_header + callout, 1)
    raw = re.sub(
        r"<title>([^<]*)</title>",
        lambda m: f"<title>{m.group(1)} (PT)</title>",
        raw, count=1,
    )
    return raw


def replace_header_with_pt(raw: str, slug: str, sub_prefix: str) -> str:
    """Swap the EN site-header for a PT-rendered one (after body translation)."""
    prefix = "../" if sub_prefix == "" else "../../"
    new_header = render_header(slug, lang="pt-br", prefix=prefix)
    if HEADER_RE.search(raw):
        return HEADER_RE.sub(new_header, raw, count=1)
    return raw.replace("<body>", "<body>\n" + new_header, 1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def _make_client():
    """Build an Anthropic client; return None if not available.

    Backend selection (in priority order):
      1. AWS Bedrock — if AWS_ACCESS_KEY_ID/AWS_PROFILE/AWS creds are present
         (or TRANSLATE_BACKEND=bedrock is forced). Uses anthropic.AnthropicBedrock.
         Default model: us.anthropic.claude-haiku-4-5-20251001-v1:0
         (override with TRANSLATE_MODEL).
      2. Anthropic API — if ANTHROPIC_API_KEY is set.
      3. None (shell-mirror fallback).
    """
    try:
        import anthropic  # type: ignore
    except ImportError:
        print("  anthropic SDK not installed — falling back to shell mirror.")
        print("  pip install anthropic   to enable real translation.")
        return None

    backend = os.environ.get("TRANSLATE_BACKEND", "").lower()
    has_bedrock_creds = bool(
        os.environ.get("AWS_ACCESS_KEY_ID")
        or os.environ.get("AWS_PROFILE")
        or os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
    )
    has_anthropic_key = bool(os.environ.get("ANTHROPIC_API_KEY"))

    use_bedrock = backend == "bedrock" or (backend != "anthropic" and has_bedrock_creds)

    if use_bedrock:
        try:
            client = anthropic.AnthropicBedrock(
                aws_region=os.environ.get("AWS_REGION", "us-east-1"),
            )
        except Exception as e:
            print(f"  failed to init AnthropicBedrock: {e}")
            return None
        # Bedrock model IDs differ from API ones — set a sensible default.
        client._default_model = (
            "us.anthropic.claude-haiku-4-5-20251001-v1:0"
        )
        print(f"  backend: AWS Bedrock (region: {os.environ.get('AWS_REGION', 'us-east-1')})")
        return client

    if has_anthropic_key:
        client = anthropic.Anthropic()
        client._default_model = "claude-haiku-4-5-20251001"
        print("  backend: Anthropic API")
        return client

    print("  no translation credentials found (set ANTHROPIC_API_KEY,")
    print("  or AWS_ACCESS_KEY_ID/AWS_PROFILE for Bedrock) — shell mirror only.")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", type=str, default="site")
    ap.add_argument("--no-translate", action="store_true",
                    help="Force shell-mirror mode even if API key is set.")
    args = ap.parse_args()

    site_root = Path(args.site).resolve()
    pt_root = site_root / "pt-br"
    pt_root.mkdir(parents=True, exist_ok=True)
    print(f"Site: {site_root}")
    print(f"PT:   {pt_root}")

    client = None if args.no_translate else _make_client()
    mode = "translate (Claude Haiku)" if client else "shell-mirror only"
    print(f"Mode: {mode}\n")

    for filename, (slug, sub) in PAGES.items():
        en = site_root / filename
        pt = pt_root / filename
        if not en.exists():
            print(f"  skip (missing): {en}")
            continue
        raw = en.read_text()
        if client is None:
            out = shell_mirror(raw, slug, sub)
        else:
            try:
                out = translate_html(raw, client)
            except Exception as e:
                print(f"  WARN: translation failed for {filename}: {e}")
                print("        falling back to shell mirror for this page.")
                out = shell_mirror(raw, slug, sub)
            else:
                # successful translation: still need PT nav header
                out = replace_header_with_pt(out, slug, sub)
        pt.parent.mkdir(parents=True, exist_ok=True)
        pt.write_text(out)
        print(f"  wrote: {pt}")

    for asset in SHARED_ASSETS:
        src = site_root / asset
        if src.exists():
            shutil.copyfile(src, pt_root / asset)

    interp_en = site_root / "interpret"
    interp_pt = pt_root / "interpret"
    if interp_en.exists():
        interp_pt.mkdir(exist_ok=True)
        for f in interp_en.glob("subgraph_*.html"):
            shutil.copyfile(f, interp_pt / f.name)

    pt_index = pt_root / "index.html"
    if pt_index.exists():
        raw = pt_index.read_text()
        raw = raw.replace("'samples/sample_${r.idx}.html'",
                          "'../samples/sample_${r.idx}.html'")
        pt_index.write_text(raw)

    print(f"\nDone. {pt_root}/index.html is the PT entry point.")
    if client is not None:
        cache_files = list(CACHE_DIR.glob("*.txt"))
        print(f"Translation cache: {len(cache_files)} entries at {CACHE_DIR}")


if __name__ == "__main__":
    main()
