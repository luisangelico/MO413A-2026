"""Internationalization for the static site.

Two languages: 'en' (default, served from <site>/) and 'pt-br' (served
from <site>/pt-br/). All page generators accept --lang and write to the
appropriate subtree.

The translation strategy:
- Header chrome (title, tagline, nav labels, page subtitles) is
  fully translated for both languages.
- Two narrative pages are fully translated: about-gat and dataset.
- The remaining pages keep their dense statistical narrative in
  English with a "Versão em português em construção" callout in
  the PT version, since numbers, tables, and figure labels are
  language-neutral.
"""

from __future__ import annotations

LANGS = ("en", "pt-br")
DEFAULT_LANG = "en"


def lang_subdir(lang: str) -> str:
    """Subdirectory under the site root for a given language. Empty for default."""
    return "" if lang == DEFAULT_LANG else f"{lang}/"


def opposite(lang: str) -> str:
    return "pt-br" if lang == "en" else "en"


PROJECT_TITLE = {
    "en": "Skin Cancer Gene-Network Analysis",
    "pt-br": "Análise de Câncer de Pele em Redes Gênicas",
}
PROJECT_TAGLINE = {
    "en": "Graph attention networks over PPI for melanoma classification",
    "pt-br": "Redes de atenção em grafos sobre PPI para classificação de melanoma",
}

# (slug, href). Hrefs are relative to the language root.
NAV_LINKS = [
    ("predictions", "index.html"),
    ("exploration", "exploration.html"),
    ("about-gat",   "about-gat.html"),
    ("overview",    "overview.html"),
    ("dataset",     "dataset.html"),
    ("interpret",   "interpret/index.html"),
    ("embeddings",  "embeddings.html"),
    ("pseudotime",  "pseudotime.html"),
    ("stability",   "stability.html"),
    ("biomarkers",  "biomarkers.html"),
]

# Per-slug labels for each language.
NAV_LABEL = {
    "predictions": {"en": "Predictions",      "pt-br": "Predições"},
    "exploration": {"en": "Exploration",      "pt-br": "Exploração"},
    "about-gat":   {"en": "What is a GAT?",   "pt-br": "O que é uma GAT?"},
    "overview":    {"en": "Run overview",     "pt-br": "Visão da execução"},
    "dataset":     {"en": "Dataset",          "pt-br": "Base de dados"},
    "interpret":   {"en": "Interpretability", "pt-br": "Interpretabilidade"},
    "embeddings":  {"en": "Embeddings",       "pt-br": "Embeddings"},
    "pseudotime":  {"en": "Pseudotime",       "pt-br": "Pseudotempo"},
    "stability":   {"en": "Stability",        "pt-br": "Estabilidade"},
    "biomarkers":  {"en": "Biomarkers",       "pt-br": "Biomarcadores"},
}


# General-purpose translation table. Keys are stable identifiers used by
# page generators; values are language → string maps. Anything not present
# falls back to the English string via t().
T = {
    "back_to_predictions": {
        "en": "← back to predictions",
        "pt-br": "← voltar para predições",
    },
    "back_to_overview": {
        "en": "← back to overview",
        "pt-br": "← voltar para a visão geral",
    },
    "run_label": {"en": "Run", "pt-br": "Execução"},
    "split_label": {"en": "split", "pt-br": "split"},
    "samples_label": {"en": "Samples", "pt-br": "Amostras"},
    "accuracy_label": {"en": "Accuracy", "pt-br": "Acurácia"},
    "classes_label": {"en": "Classes", "pt-br": "Classes"},
    "test_predictions_h2": {
        "en": "Test-set predictions",
        "pt-br": "Predições no conjunto de teste",
    },
    "test_predictions_meta": {
        "en": "Click any row to inspect the per-sample graph and nearest training neighbors.",
        "pt-br": "Clique em qualquer linha para inspecionar o grafo da amostra e os vizinhos mais próximos no treino.",
    },
    "search_placeholder": {"en": "search id…", "pt-br": "buscar id…"},
    "all_classes": {"en": "all classes", "pt-br": "todas as classes"},
    "all_status": {"en": "all", "pt-br": "todos"},
    "correct_only": {"en": "correct only", "pt-br": "somente corretos"},
    "incorrect_only": {"en": "incorrect only", "pt-br": "somente incorretos"},
    "col_idx": {"en": "idx", "pt-br": "idx"},
    "col_id": {"en": "patient id", "pt-br": "id do paciente"},
    "col_true": {"en": "true", "pt-br": "verdadeiro"},
    "col_pred": {"en": "predicted", "pt-br": "previsto"},
    "col_conf": {"en": "confidence", "pt-br": "confiança"},
    "col_correct": {"en": "correct?", "pt-br": "correto?"},
    "run_overview_h2": {"en": "Run overview", "pt-br": "Visão geral da execução"},
    "weights_label": {"en": "weights", "pt-br": "pesos"},
    "samples_evaluated": {
        "en": "Samples evaluated",
        "pt-br": "Amostras avaliadas",
    },
    "per_class_metrics": {
        "en": "Per-class metrics",
        "pt-br": "Métricas por classe",
    },
    "confusion_matrix": {
        "en": "Confusion matrix",
        "pt-br": "Matriz de confusão",
    },
    "pt_construction_callout": {
        "en": "",  # not shown in English
        "pt-br": (
            "<strong>Versão em português em construção.</strong> O texto narrativo desta "
            "página está atualmente em inglês — números, tabelas e legendas das "
            "figuras são independentes de idioma. As páginas <em>O que é uma GAT?</em> e "
            "<em>Base de dados</em> já têm tradução completa."
        ),
    },
    "class_normal_tissue": {"en": "Normal Tissue", "pt-br": "Tecido Normal"},
    "class_benign_nevus":  {"en": "Benign Nevus",  "pt-br": "Nevo Benigno"},
    "class_primary_tumor": {"en": "Primary Tumor", "pt-br": "Tumor Primário"},
    "class_metastasis":    {"en": "Metastasis",    "pt-br": "Metástase"},
}


# Per-page <title> tags (browser tab titles) for each language.
PAGE_TITLE = {
    "en": {
        "predictions": "Predictions — Skin Cancer Gene-Network Analysis",
        "exploration": "Exploration — Skin Cancer Gene-Network Analysis",
        "about-gat":   "What is a GAT? — Skin Cancer Gene-Network Analysis",
        "overview":    "Run overview — Skin Cancer Gene-Network Analysis",
        "dataset":     "Dataset — Skin Cancer Gene-Network Analysis",
        "interpret":   "Interpretability — Skin Cancer Gene-Network Analysis",
        "embeddings":  "Embeddings — Skin Cancer Gene-Network Analysis",
        "pseudotime":  "Pseudotime — Skin Cancer Gene-Network Analysis",
        "stability":   "Stability — Skin Cancer Gene-Network Analysis",
        "biomarkers":  "Biomarkers — Skin Cancer Gene-Network Analysis",
    },
    "pt-br": {
        "predictions": "Predições — Análise de Câncer de Pele em Redes Gênicas",
        "exploration": "Exploração — Análise de Câncer de Pele em Redes Gênicas",
        "about-gat":   "O que é uma GAT? — Análise de Câncer de Pele em Redes Gênicas",
        "overview":    "Visão da execução — Análise de Câncer de Pele em Redes Gênicas",
        "dataset":     "Base de dados — Análise de Câncer de Pele em Redes Gênicas",
        "interpret":   "Interpretabilidade — Análise de Câncer de Pele em Redes Gênicas",
        "embeddings":  "Embeddings — Análise de Câncer de Pele em Redes Gênicas",
        "pseudotime":  "Pseudotempo — Análise de Câncer de Pele em Redes Gênicas",
        "stability":   "Estabilidade — Análise de Câncer de Pele em Redes Gênicas",
        "biomarkers":  "Biomarcadores — Análise de Câncer de Pele em Redes Gênicas",
    },
}


def t(key: str, lang: str) -> str:
    """Translate a key. Falls back to English if the key has no PT entry."""
    entry = T.get(key)
    if entry is None:
        return key
    return entry.get(lang) or entry.get("en") or key
