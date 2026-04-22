import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def imports():
    import marimo as mo
    import GEOparse
    import pandas as pd
    import numpy as np
    from pathlib import Path
    from scipy import stats

    return GEOparse, Path, mo, np, pd, stats


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Câncer de Pele — Análise de Expressão Gênica em Redes

    **Equipe ALFAK** | MO413A — UNICAMP 2026

    ---

    ## Datasets GEO

    | Dataset | Grupos | Amostras | Descrição |
    |---------|--------|----------|-----------|
    | [GSE4570](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE4570) | Saudável, Melanoma | 8 | Melanócitos normais vs. linhagens de melanoma avançado (Affymetrix HG-U133A) |
    | [GSE2503](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE2503) | Saudável, Não-melanoma | 15 | Pele normal vs. queratose actínica vs. SCC em pacientes transplantados (Affymetrix HG-U133A) |
    | [GSE53462](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE53462) | Saudável, Não-melanoma | 26 | Pele normal vs. BCC vs. SCC — classificação molecular (Illumina HumanHT-12 v4) |
    | [GSE8401](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE8401) | Melanoma | 83 | Melanoma primário vs. metástases — assinatura de agressividade (Affymetrix HG-U133A) |
    | [GSE7553](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE7553) | Melanoma | 87 | Espectro completo: pele normal, melanoma in situ, primário, metastático, BCC, SCC (Affymetrix U133 Plus 2.0) |
    | [GSE45216](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE45216) | Não-melanoma | 40 | Queratose actínica vs. SCC cutâneo — 196 DEGs identificados (Affymetrix U133 Plus 2.0) |

    ---

    ## Pipeline

    - [x] **1. Download dos datasets GEO** — GEOparse, cache local em `data/geo/`
    - [ ] **2. Pré-processamento**
        - [x] Extrair matrizes de expressão (genes × amostras) de cada `.soft.gz` *(feito para GSE7553)*
        - [x] Mapear subgrupo de amostra (ex.: *in situ*, primário, metastático, BCC, SCC, normal) a partir da metadata `characteristics_ch1`
        - [x] Verificar normalização (log2 já aplicado pelo depositante?) e aplicar se necessário *(confirmado raw; mantemos raw para replicar Orange)*
        - [x] Deduplicar sondas por gene (equivalente ao widget `Unique` do Orange) *(feito em `metastatic_unique`)*
        - [ ] *(Depois, ao integrar múltiplos GEOs)* correção de batch effect
    - [ ] **3. Análise de Expressão Diferencial (DEA)** — granularidade fina, por estágio/tipo
        - [ ] Pele normal × Melanoma in situ *(GSE7553)*
        - [ ] Pele normal × Melanoma primário *(GSE7553, GSE4570, GSE8401)*
        - [x] Pele normal × Melanoma metastático *(GSE7553)* — reproduz Orange (99.7 % Jaccard)
        - [ ] Pele normal × Carcinoma basocelular (BCC) *(GSE7553, GSE53462)*
        - [ ] Pele normal × Carcinoma espinocelular (SCC) *(GSE7553, GSE2503, GSE45216, GSE53462)*
        - [ ] *(Opcional)* progressões: in situ → primário → metastático; AK → SCC *(GSE45216)*
        - [x] Dois passes por comparação: completo + filtrado por `|logFC| ≥ 2.3` **e** `p ≤ 0.001` (replica o Orange)
    - [ ] **4. Anotação externa — Open Targets**
        - [x] Baixar alvos associados ao **melanoma** (`EFO_0000756`); BCC/SCC pendente
        - [x] Anexar `oncogeneScore` a cada gene como atributo de nó (0–1, drogabilidade/relevância clínica)
    - [ ] **5. Construção da rede PPI**
        - [ ] Query dos DEGs na API do STRING
        - [ ] Filtrar por `combined_score` (ex.: ≥ 0.4 = confiança média)
    - [ ] **6. Análise de rede** (Cytoscape / Python — NetworkX, igraph)
        - [ ] Hubs (degree, betweenness, closeness, eigenvector)
        - [ ] Detecção de módulos (MCODE / Louvain / Leiden)
    - [ ] **7. Comparação entre redes**
        - [ ] Hubs compartilhados vs. específicos por tipo/estágio
        - [ ] Genes ganhos/perdidos entre estágios
        - [ ] Diferenças topológicas (distribuição de grau, clustering)
    - [ ] **8. Graph Attention Network (GAT)** — análise adicional sobre TCGA-SKCM
    """)
    return


@app.cell(hide_code=True)
def load_datasets(GEOparse, Path, mo, pd):
    from dataclasses import dataclass

    mo.output.append(mo.md("## 1. Download dos Datasets GEO"))

    ACCESSIONS: list[str] = [
        "GSE4570",
        "GSE2503",
        "GSE53462",
        "GSE8401",
        "GSE7553",
        "GSE45216",
    ]

    # Regras de categorização. Ordem importa: primeira substring que casar
    # (case-insensitive) em `characteristics_ch1` vence.
    CATEGORY_RULES: list[tuple[str, str]] = [
        ("metastatic melanoma", "Metastatic Melanoma"),
        ("primary melanoma",    "Primary Melanoma"),
        ("melanoma in situ",    "Melanoma in situ"),
        ("melanocyte",          "Normal"),
        ("normal skin",         "Normal"),
        ("actinic keratosis",   "Actinic Keratosis"),
        ("squamous cell",       "Squamous Cell Carcinoma"),
        ("basal cell",          "Basal Cell Carcinoma"),
    ]

    def categorize(characteristics: list[str]) -> str | None:
        """Retorna a categoria pela primeira regra que casar com algum
        valor de `characteristics_ch1`; `None` se nenhuma casar."""
        text = " · ".join(characteristics).lower()
        for substr, cat in CATEGORY_RULES:
            if substr in text:
                return cat
        return None

    @dataclass
    class Dataset:
        """Agregado por dataset: objeto GEOparse + DataFrame de amostras."""
        accession: str
        gse: GEOparse.GSE           # dataset completo: .gsms, .gpls, .metadata
        samples: pd.DataFrame       # colunas: GSM, Título, Características, Categoria

    # Download + parsing com cache persistente. O parsing custa ~10s/dataset;
    # resultado só muda se ACCESSIONS mudar.
    GEO_DIR = Path("data/geo")
    GEO_DIR.mkdir(parents=True, exist_ok=True)
    with mo.persistent_cache("geo_datasets"):
        gse_objects = {}
        for acc in mo.status.progress_bar(
            ACCESSIONS,
            title="Baixando datasets GEO",
            remove_on_exit=True,
        ):
            gse_objects[acc] = GEOparse.get_GEO(
                geo=acc, destdir=str(GEO_DIR), silent=True,
            )

    # Monta Dataset (gse + samples categorizadas). dtype="string" no
    # construtor garante StringDtype para todas as colunas e missing
    # uniforme como pd.NA — evita mix de None (object) e NaN (string)
    # que o pandas 2.x gera ao inferir dtype por homogeneidade.
    datasets: dict[str, Dataset] = {}
    for acc in ACCESSIONS:
        gse = gse_objects[acc]
        sample_df = pd.DataFrame(
            [
                {
                    "GSM": gsm_id,
                    "Título": gsm.metadata.get("title", [""])[0],
                    "Características": " · ".join(gsm.metadata.get("characteristics_ch1", [])),
                    "Categoria": categorize(gsm.metadata.get("characteristics_ch1", [])),
                }
                for gsm_id, gsm in gse.gsms.items()
            ],
            dtype="string",

        ).set_index("GSM")


        datasets[acc] = Dataset(accession=acc, gse=gse, samples=sample_df)

    # Tabela master: concatena todos os .samples, coluna Dataset como
    # desambiguador. GSMs são globalmente únicos no GEO — preservamos
    # como index (sem ignore_index=True).
    samples = pd.concat(
        [ds.samples.assign(Dataset=acc) for acc, ds in datasets.items()],
    ).loc[:, ["Dataset", "Título", "Características", "Categoria"]]
    return datasets, samples


@app.cell(hide_code=True)
def show_all_samples(mo, samples):
    # Consumidora trivial para garantir que `samples` seja exportado por
    # load_datasets e fique disponível em cells futuras. De brinde, mostra
    # a contagem por (Dataset, Categoria) antes da aba-por-dataset abaixo.
    _pivot = (
        samples
        .assign(Categoria=lambda s: s["Categoria"].fillna("(sem categoria)"))
        .groupby(["Dataset", "Categoria"]).size().unstack(fill_value=0)
    )
    mo.vstack([
        mo.md(f"**Total:** {len(samples)} amostras em {samples['Dataset'].nunique()} dataset(s)"),
        _pivot,
    ])
    return


@app.cell(hide_code=True)
def datasets_summary(datasets: "dict[str, Dataset]", mo):
    def _panel(acc, ds):
        gse = ds.gse
        title = gse.metadata.get("title", [""])[0]
        platform = ", ".join(gse.metadata.get("platform_id", ["?"]))
        submitted = gse.metadata.get("submission_date", ["?"])[0]
        summary_text = gse.metadata.get("summary", [""])[0]
        if len(summary_text) > 500:
            summary_text = summary_text[:500].rstrip() + "…"

        # Contagem por categoria (inclui `None` como "(sem categoria)")
        category_counts = (
            ds.samples["Categoria"]
            .value_counts(dropna=False)
            .reset_index(name="N")
        )
        category_counts["Categoria"] = category_counts["Categoria"].fillna("(sem categoria)")

        header = mo.md(f"""
        |  |  |
        | ---: | :--- |
        | **Accession** | [{acc}](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={acc}) |
        | **Title** | {title} |
        | **Platform** | `{platform}` |
        | **Samples** | {len(gse.gsms)} |
        | **Submitted** | {submitted} |

        {summary_text}
        """)

        return mo.vstack([
            header, 
            mo.md("### Amostras por categoria"),
            category_counts, 
            mo.md("### Amostras"), 
            ds.samples])

    mo.ui.tabs({acc: _panel(acc, ds) for acc, ds in datasets.items()})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 2. Reprodução do pipeline Orange — Metastatic Melanoma

    Esta seção reproduz fielmente `workspace/Metastatic Melanoma/Melanoma.ows` em Python,
    servindo como **baseline de validação** antes de generalizar para os outros subgrupos.

    Passos (espelham o `.ows`):

    1. Filtrar amostras do GSE7553 — **Metastatic Melanoma** × **Normal Skin**.
    2. Construir matriz de expressão (sondas × amostras) e deduplicar sondas.
    3. DEA com Welch's t-test; filtros combinados `|logFC| ≥ 2.3` **e** `p ≤ 0.001`.
    4. Anotar cada gene com `oncogeneScore` do Open Targets (EFO_0000756).
    5. Validar contra `workspace/Metastatic Melanoma/Nodes_cytoscape.csv`.
    """)
    return


@app.cell
def metastatic_samples(datasets: "dict[str, Dataset]"):
    gse7553 = datasets["GSE7553"]
    chars = gse7553.samples["Características"].fillna("").str.lower()

    # Escopo do Orange: sample_substring='Normal Skin, metastatic, Group 3'
    # — match OR por qualquer das 3 substrings em `characteristics_ch1`.
    scope_substrings = ["normal skin", "metastatic", "group 3"]
    in_scope = chars.apply(lambda s: any(x in s for x in scope_substrings))
    scoped = gse7553.samples[in_scope]

    # G1 = Metastatic Melanoma; G2 = complemento dentro do escopo.
    # NB: a DE widget do Orange define G2 = ~G1 (não "a outra classe
    # selecionada"), então o background inclui Normal Skin + Primary
    # Melanoma + Melanocyte — todas que casaram o sample_substring mas
    # não são metastatic.
    metastatic_gsms = scoped.index[scoped["Categoria"] == "Metastatic Melanoma"].tolist()
    other_gsms = [g for g in scoped.index if g not in metastatic_gsms]
    return gse7553, metastatic_gsms, other_gsms


@app.cell(hide_code=True)
def metastatic_samples_show(
    datasets: "dict[str, Dataset]",
    metastatic_gsms,
    mo,
    other_gsms,
):
    # Breakdown por categoria no G2 (complemento)
    cat_breakdown = (
        datasets["GSE7553"].samples
        .loc[other_gsms, "Categoria"]
        .fillna("(sem categoria)")
        .value_counts()
    )
    breakdown_md = "\n".join(f"    - {c}: **{n}**" for c, n in cat_breakdown.items())
    mo.md(f"""
    **Seleção de amostras do GSE7553** (espelha `sample_substring='Normal Skin, metastatic, Group 3'` do Orange):

    - **G1 — Metastatic Melanoma:** {len(metastatic_gsms)} amostras
    - **G2 — Complemento no escopo:** {len(other_gsms)} amostras
    {breakdown_md}

    _A DE widget do Orange compara `G1 × ~G1`, não `G1 × outra_classe`, então G2 inclui todas as não-metastáticas que casaram o filtro do extractor._
    """)
    return


@app.cell
def metastatic_expression(gse7553, metastatic_gsms, other_gsms, pd):
    # Matriz probes × amostras. Probes são globalmente únicos no GPL, e o
    # concat de Series com mesmo index não produz linhas all-NaN — por isso
    # sem dedup de probe nem dropna(how="all"). Valores são brutos: o Orange
    # calcula LogFC como log2(mean_a / mean_b) direto em raw.
    selected = metastatic_gsms + other_gsms
    metastatic_expr = pd.concat(
        [
            gse7553.gse.gsms[gid].table.set_index("ID_REF")["VALUE"].rename(gid)
            for gid in selected
        ],
        axis=1,
    )
    return (metastatic_expr,)


@app.cell
def metastatic_platform(gse7553):
    # GPL570 (HG-U133 Plus 2.0) — mapeia probe ID -> Gene Symbol + Entrez ID.
    # Sondas multi-mapeadas (p.ex. 'DDR1 /// MIR4640') ficam com o primeiro gene
    # só — mesmo comportamento de deduplicação do Orange.
    gpl_id = next(iter(gse7553.gse.gpls))
    gpl_table = gse7553.gse.gpls[gpl_id].table
    probe_map = gpl_table.set_index("ID")[["Gene Symbol", "ENTREZ_GENE_ID"]].copy()
    probe_map["Gene Symbol"] = (
        probe_map["Gene Symbol"].astype(str).str.split(" /// ").str[0].str.strip()
    )
    probe_map["Entrez ID"] = (
        probe_map["ENTREZ_GENE_ID"].astype(str).str.split(" /// ").str[0].str.strip()
    )
    probe_map = probe_map[["Gene Symbol", "Entrez ID"]]
    probe_map = probe_map.replace({"nan": None, "": None}).dropna()
    return (probe_map,)


@app.cell
def metastatic_unique(metastatic_expr, probe_map):
    # Replica o widget "Unique" do Orange: 1 probe por Entrez ID,
    # primeira probe alfabética vence (tiebreaker default).
    # Entrada: ~54 675 probes. Saída: ~21 121 genes.
    #
    # Probes com anotação ausente ficam agrupadas sob "?" (mesma
    # convenção que Orange usa).
    annotated = metastatic_expr.join(probe_map, how="left")
    annotated["Entrez ID"] = annotated["Entrez ID"].fillna("?")
    annotated["Gene Symbol"] = annotated["Gene Symbol"].fillna("?")
    annotated = annotated.sort_index()  # probe ID alfabético
    metastatic_unique_expr = annotated.drop_duplicates(
        subset="Entrez ID", keep="first"
    )
    return (metastatic_unique_expr,)


@app.cell
def metastatic_dea(
    metastatic_gsms,
    metastatic_unique_expr,
    np,
    other_gsms,
    stats,
):
    m_cols = [c for c in metastatic_unique_expr.columns if c in metastatic_gsms]
    n_cols = [c for c in metastatic_unique_expr.columns if c in other_gsms]

    m = metastatic_unique_expr[m_cols].to_numpy()
    n = metastatic_unique_expr[n_cols].to_numpy()

    # LogFC no estilo Orange: log2 da razão entre médias brutas.
    # NB: não equivale a mean(log2(A)) - mean(log2(B)).
    logfc = np.log2(m.mean(axis=1) / n.mean(axis=1))

    # Student's t-test (equal_var=True) em valores brutos — é o que o widget
    # 'Differential Expression' da Orange3-Bioinformatics usa por padrão.
    _t, pval = stats.ttest_ind(m, n, axis=1, equal_var=True, nan_policy="omit")

    # Carrega anotação (Gene Symbol, Entrez ID) que ja veio com a matriz unica
    metastatic_de = metastatic_unique_expr[["Gene Symbol", "Entrez ID"]].copy()
    metastatic_de["LogFC"] = logfc
    metastatic_de["p-value"] = pval
    return (metastatic_de,)


@app.cell
def metastatic_filter(metastatic_de):
    # Filtros combinados |logFC| >= 2.3 E p <= 0.001 (mesmo Melanoma.ows).
    # Dedup por gene ja aconteceu em metastatic_unique — sem precisar aqui.
    mask = (metastatic_de["LogFC"].abs() >= 2.3) & (metastatic_de["p-value"] <= 0.001)
    metastatic_filtered = metastatic_de.loc[mask].copy()
    # Remove entrada "?" (probes sem anotação) se tiver sobrevivido
    metastatic_filtered = metastatic_filtered[
        (metastatic_filtered["Entrez ID"] != "?")
        & (metastatic_filtered["Gene Symbol"] != "?")
    ]
    return (metastatic_filtered,)


@app.cell
def metastatic_opentargets(Path, pd):
    ot_path = Path("workspace/Metastatic Melanoma/OT-EFO_0000756-associated-targets-4_20_2026-v26_03.tsv")
    ot = pd.read_csv(ot_path, sep="\t")
    ot_scores = (
        ot[["symbol", "globalScore"]]
        .dropna(subset=["symbol"])
        .drop_duplicates(subset="symbol", keep="first")
        .rename(columns={"symbol": "Gene Symbol", "globalScore": "oncogeneScore"})
    )
    return (ot_scores,)


@app.cell
def metastatic_final(metastatic_filtered, ot_scores, pd):
    metastatic_nodes = (
        metastatic_filtered
        .merge(ot_scores, on="Gene Symbol", how="left")
        .loc[:, ["Entrez ID", "Gene Symbol", "LogFC", "p-value", "oncogeneScore"]]
        .reset_index(drop=True)
    )
    # Entrez ID no baseline do Orange é inteiro — coage para permitir comparação.
    metastatic_nodes["Entrez ID"] = pd.to_numeric(metastatic_nodes["Entrez ID"], errors="coerce").astype("Int64")

    metastatic_nodes
    return (metastatic_nodes,)


@app.cell(hide_code=True)
def metastatic_validate(Path, metastatic_nodes, mo, pd):
    expected = pd.read_csv(Path("workspace/Metastatic Melanoma/Nodes_cytoscape.csv"))

    got_genes = set(metastatic_nodes["Gene Symbol"].dropna())
    exp_genes = set(expected["Gene Symbol"].dropna())

    common = got_genes & exp_genes
    only_got = got_genes - exp_genes
    only_exp = exp_genes - got_genes

    union = got_genes | exp_genes
    jaccard = len(common) / len(union) if union else 0.0

    mo.md(f"""
    ### Comparação com baseline do Orange

    | Métrica | Valor |
    | --- | --- |
    | Genes no notebook | {len(metastatic_nodes)} |
    | Genes esperados (Orange) | {len(expected)} |
    | Em comum | {len(common)} |
    | Só no notebook | {len(only_got)} |
    | Só no Orange | {len(only_exp)} |
    | Similaridade de Jaccard | {jaccard:.2%} |

    **Status:** pipeline bate **~99.7 %** com o baseline Orange. Os genes só no
    nosso output (tipicamente 1) correspondem a um **bug conhecido do
    `orange3-biosci.geo_soft_extractor`** — ele descarta Entrez IDs com ≤ 2
    dígitos (filtro `len(candidate) > 2` nas linhas 637 e 655), marcando como
    `"?"` e perdendo-os no Unique. Afeta ~139 probes de genes "antigos" (Entrez
    baixo): A2M (2), ACACB (32), ACADL (33), ACAT1 (38), ACP1 (52), etc.

    Não replicamos o bug — nosso output inclui esses genes legitimamente.
    """)
    return


@app.cell
def metastatic_save(Path, metastatic_nodes):
    # Salva a saída do notebook para conferência lado a lado com o Nodes_cytoscape.csv
    out_dir = Path("output/Metastatic Melanoma")
    out_dir.mkdir(parents=True, exist_ok=True)
    metastatic_nodes.to_csv(out_dir / "Nodes_cytoscape.csv", index=False)
    return


if __name__ == "__main__":
    app.run()
