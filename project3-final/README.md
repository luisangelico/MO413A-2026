# Projeto `Câncer de Pele e seus Tipos: uma Análise do Perfil de Expressão Gênica em Redes`
# Project `Skin Cancer and its Types: An Analysis of Gene Expression Profiles in Networks`

|Nome  | RA | Especialização|
|--|--|--|
| Alan Freitas Ribeiro  | 193400  | Computação |
| Augusto José Peterlevitz  | 209783  | Computação |
| Felipe Kennedy Carvalho Torquato | 174157  | Biologia |
| Luis Henrique Angélico | 248891  | Computação |
| Naruan Francisco Ferraz e Ferraz | 323009  | Biologia |
| Paulo Costa | 063607  | Computação |

# Descrição Resumida do Projeto

O câncer de pele do tipo melanoma é uma neoplasia de alta letalidade, marcada por
heterogeneidade molecular e progressão metastática rápida. Apesar de avanços em
imunoterapia e em terapias-alvo (BRAF/MEK), a estratificação de pacientes e a
identificação de biomarcadores robustos ainda são problemas em aberto. Este projeto
investiga o perfil de expressão gênica de amostras de pele saudável (nevos
benignos), tumor primário cutâneo e metástase de melanoma à luz da rede de
interação proteína-proteína (PPI) do **STRING**, integrando dados de larga escala
do **TCGA-SKCM** (recomputados pelo pipeline **TOIL** da UCSC Xena) e do
**GSE112509** (nevos benignos, GEO).

Em vez de tratar cada amostra como um vetor isolado, modelamos cada uma delas como
um **grafo** cujos nós são genes (com expressão log₂(TPM+0.001) como atributo) e
arestas são interações físicas STRING de alta confiança. Treinamos um classificador
**Graph Attention Network (GATv2)** para discriminar as três classes e usamos a
**atenção por aresta** como mecanismo de interpretabilidade, gerando *rankings*
gene-por-classe e contrastes (e.g., *Metástase − Primário*) que são confrontados
com conjuntos curados (linhagem melanocítica, progressão de melanoma, infiltrado
imune, queratinócitos/envelope cornificado) e contra um *baseline* de grau topológico.

Os resultados mostram que a atenção do GAT recupera biologia conhecida de pele e de
melanoma com significância estatística (testes hipergeométricos), supera o
*baseline* de grau, e — em análise de **multi-seed** — produz rankings, *embeddings*
e pseudotempo estáveis entre execuções, sugerindo que o modelo captura sinal
biológico reprodutível e não apenas a estrutura de hubs do PPI.

# Slides

[Slides P3](assets/slides/MO413A-P3.pdf)

# Fundamentação Teórica

O melanoma cutâneo decorre da transformação maligna de melanócitos e progride por
um continuum biológico desde lesões pigmentares benignas (nevos) até tumores
primários invasivos e disseminação metastática. Programas transcricionais ligados
à diferenciação melanocítica (*MITF*, *TYR*, *PMEL*, *DCT*), à transição
fenotípica/EMT-like e à resposta imune adaptativa (*CD8A*, *GZMB*, *PRF1*, *CXCL9*)
têm sido associados, respectivamente, à fase proliferativa, à invasão e ao
prognóstico sob imunoterapia [1,2].

Redes PPI fornecem um *scaffold* de relações funcionais sobre o qual a expressão
gênica pode ser interpretada como um sinal estruturado. **Graph Neural Networks**
e, em particular, **Graph Attention Networks (GAT/GATv2)** [3,4] aprendem
representações em grafos atribuindo pesos adaptativos às arestas, o que permite
identificar, a posteriori, **quais conexões PPI mais contribuíram para a
predição** — propriedade explorada aqui como uma forma de interpretabilidade
*built-in*. Comparado a abordagens clássicas (PCA, t-SNE sobre expressão), o GAT
incorpora explicitamente a topologia biológica conhecida.

A motivação clínica e técnica deste trabalho dialoga diretamente com (i) o uso da
plataforma **Open Targets** para priorização de alvos terapêuticos [5], (ii) o
recompute **TOIL** para tornar TCGA e GTEx comparáveis [6] e (iii) o consórcio
**STRING** como referência de interações [7].

# Perguntas de Pesquisa

As perguntas evoluíram em relação à P2 (centradas em centralidade clássica de PPI
e *globalScore*) para incorporar o eixo de aprendizado em grafos:

1. **Um GAT treinado sobre PPI distingue *Nevo Benigno*, *Tumor Primário* e
   *Metástase* a partir apenas da expressão de genes altamente variáveis?**
   Sim — o modelo atinge acurácia de teste alta e estável (≈0.86 de média entre
   seeds, com Metástase como classe de maior precisão).
2. **A atenção aprendida pelo GAT recupera biologia conhecida de pele/melanoma
   melhor do que um *baseline* de grau topológico?**
   Sim — em todos os conjuntos curados (melanócitos, progressão, imune, queratina)
   o ranking por atenção tem *p* hipergeométrico inferior ao do grau, com efeitos
   de classe coerentes (e.g., *Tumor Normal − Tumor* enriquece queratina; *Metástase
   − Primário* enriquece imune).
3. **Os achados são reprodutíveis entre execuções (estabilidade)?**
   Sim — em 5 *seeds* independentes, os top-20 por classe têm Jaccard mediano alto,
   correlações de Spearman sobre *rankings* completos são estáveis e o pseudotempo
   recuperado preserva a ordem *Nevo → Primário → Metástase*.
4. **Quais genes emergem como *biomarcadores* candidatos?**
   O *consensus* de Top-20 entre seeds (`stability_consensus_top20.csv`) é
   dominado por marcadores melanocíticos clássicos para a transição
   *Nevo↔Tumor* e por componentes imunes/EMT para *Primário↔Metástase*,
   convergindo com a literatura.

# Metodologia

A metodologia combina **Ciência de Redes** com **aprendizado profundo em grafos**:

- **Construção da rede:** PPI do STRING com *threshold* de combined score = 700,
  restrita aos *N* = 2 000 genes mais variáveis na união dos coortes; o mesmo
  grafo (índice de arestas) é compartilhado por todas as amostras (cada amostra
  difere apenas no atributo de nó = expressão).
- **Modelo:** GATv2 de 3 camadas com múltiplas cabeças, *dropout* e *pooling*
  global, treinado com *cross-entropy* ponderada por classe; *split* estratificado
  70/15/15 com *seed* fixa e *early stopping* sobre a perda de validação.
- **Interpretação por atenção:** `forward(..., return_all_attn=True)` expõe pares
  `(edge_index, alpha)` para cada uma das 3 camadas; a atenção é agregada **apenas
  sobre amostras de teste corretamente classificadas** para cada classe, gerando
  um *score* por nó (gene). Contrastes entre classes (e.g., *Metástase − Primário*)
  são produzidos como subtração de rankings normalizados.
- **Recuperação por *gene-set*:** para cada classe e contraste, calculamos o
  *p*-valor hipergeométrico de sobreposição entre o top-K (K ∈ {50,100,200}) e
  cada conjunto curado; reportamos sempre lado a lado o *baseline* de grau do PPI.
- **Pseudotempo:** projeção dos *embeddings* finais em uma direção biológica
  (Nevo → Primário → Metástase) seguida de *binning*; a atenção média por *bin*
  produz trajetórias gene-a-gene (heatmap e linhas).
- **Estabilidade multi-seed:** o pipeline completo é re-executado para 5 *seeds*;
  computamos Jaccard@K dos top-K por classe, Spearman dos *rankings* completos,
  e variabilidade do pseudotempo (banda de incerteza nos heatmaps).
- **Visualização interativa:** o site estático embute subgrafos PPI por classe
  com *hover* (gene, atenção, grau), além de páginas para *embeddings*,
  pseudotempo, biomarcadores e estabilidade — em **EN/PT-BR**.

## Bases de Dados e Evolução

| Base de Dados | Endereço na Web | Resumo descritivo |
| ----- | ----- | ----- |
| TCGA-SKCM (TOIL recompute) | [UCSC Xena TOIL](https://xenabrowser.net/datapages/?cohort=TCGA%20TARGET%20GTEx) | Reprocessamento harmonizado de TCGA e GTEx para minimizar efeito de coorte; usado para amostras de **Tumor Primário** e **Metástase** de melanoma. Valores em log₂(TPM+0.001). |
| GSE112509 (GEO) | [GSE112509](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE112509) | RNA-seq de **nevos benignos** e melanomas; aqui usamos apenas as amostras de nevo como referência de pele saudável/lesão benigna (Class 2), substituindo o GTEx skin originalmente usado na P2. |
| STRING v12 | [STRING](https://string-db.org/) | Interações proteína-proteína com *combined score*. Filtro: ≥ 700 (alta confiança), interações físicas. Define o esqueleto compartilhado dos grafos por amostra. |
| Open Targets — Melanoma (EFO_0000756) | [Open Targets](https://platform.opentargets.org/disease/EFO_0000756/associations) | Usada para anotar genes da rede com `globalScore` e fase clínica de evidência terapêutica, suportando a discussão de biomarcadores. |
| Conjuntos curados de genes | manual / literatura | Listas em `pipelines/python/src/interpret.py` para *melanocyte lineage*, *melanoma progression*, *immune infiltrate*, *cornified envelope/keratin*; servem para *gene-set recovery*. |

**Descobertas e tratamentos por base.** Os valores TOIL **já são log₂(TPM+0.001)**
— aplicar `log1p` por engano (como o pipeline antigo `archive/download_data.py` da
GDC-Hub, em TPM puro) introduz um viés sistemático. Documentamos esse pitfall no
`README` do projeto. Para o GSE112509, foi necessário um cache de tradução
de probes/símbolos (`data/external/translation_cache/`) e harmonização de
nomes para o *namespace* HGNC do TCGA. STRING foi filtrado por confiança e
restrito ao subconjunto de genes do painel de variabilidade (interseção entre os
três coortes), garantindo que cada amostra projete sobre o **mesmo grafo**.

## Modelo Lógico

O modelo lógico de grafos de propriedades evoluiu da P2 (rede PPI única, com nó =
gene anotado por LogFC, p-valor e oncogeneScore) para uma **família de grafos por
amostra**: o esqueleto PPI é fixo, mas cada amostra carrega seu próprio vetor de
expressão como atributo de nó, e o nível-amostra é representado por um nó *Sample*
ligado aos nós-gene por arestas *expresses*.

![Modelo Lógico de Grafos](assets/images/modelo-logico-grafos.png)

## Integração entre Bases

Os principais desafios de integração foram (i) **cross-cohort batch effect** entre
TCGA e GTEx — resolvido adotando o recompute TOIL, que reprocessa ambos com o
mesmo pipeline (Kallisto + RSEM); (ii) **substituição da Class 2** de GTEx skin
para GSE112509 nevi (commit `ce4d303`), motivada por (a) o GTEx skin contém
queratinócitos/derme em proporção muito diferente da pele lesional analisada e
(b) nevos representam um *baseline* biológico mais informativo para a transição
benigno→maligno; (iii) **harmonização de identificadores** (Ensembl/HGNC/symbol)
entre TCGA, GEO e STRING, com cache local; (iv) **alinhamento do painel de
genes** — o conjunto de top-variáveis é calculado **na união dos três coortes**,
de modo que a mesma lista de nós seja válida em todos eles.

## Análises Realizadas

Os pipelines completos estão em `pipelines/notebooks/` (Streamlit/Python) e
`src/` do GAT project. Trecho central da extração de atenção:

~~~python
# pipelines/python/src/interpret.py — agregação de atenção por classe (apenas amostras corretas)
out, attn = model(batch.x, batch.edge_index, batch.batch, return_all_attn=True)
for layer_idx, (edge_index, alpha) in enumerate(attn):
    # alpha: (E, num_heads) -> média entre cabeças
    edge_score = alpha.mean(dim=-1)
    # acumula no nó-destino, ponderado por classe predita corretamente
    node_score.scatter_add_(0, edge_index[1], edge_score)
~~~

E o teste hipergeométrico que sustenta as comparações com o *baseline* de grau:

~~~python
from scipy.stats import hypergeom
p = hypergeom.sf(k-1, M=N_genes, n=len(geneset), N=topK)
~~~

As análises geradas pelo pipeline:

- **Dataset** — composição do *dataset*, classes e distribuições de expressão
  (`pipelines/python/src/dataset_page.py`).
- **Grafo PPI** — rede PPI completa com *layout* interativo
  (`pipelines/python/scripts/visualize_dataset.py`).
- **Embeddings** — projeções finais (UMAP/PCA) coloridas por classe
  (`pipelines/python/src/embeddings.py`).
- **Subgrafos interpretativos por classe** — top-K por atenção
  (`pipelines/python/src/interpret_viz.py`).
- **Pseudotempo** — heatmap e linhas de pseudotempo por gene
  (`pipelines/python/src/pseudotime.py`).
- **Estabilidade multi-seed** — Jaccard@K, Spearman e variabilidade do
  pseudotempo (`pipelines/python/src/multiseed_stability.py`).
- **Biomarcadores** — *consensus* multi-seed e candidatos
  (`pipelines/python/scripts/biomarker_report.py`).

## Evolução do Projeto

Da P1/P2 para a P3 a evolução central foi a **mudança do paradigma analítico**:
saímos de centralidade clássica + *globalScore* (P2, Cytoscape/CytoNCA/MCODE) para
um modelo de grafos aprendido (GATv2). Pelo caminho:

- **Substituímos GTEx skin por GSE112509 nevi** como Class 2 (*Benign Nevus*),
  por motivação biológica (commit `ce4d303`).
- **Adotamos TOIL no lugar do GDC-Hub** após detectar que o duplo `log` no
  pipeline antigo deslocava artificialmente a separação Tumor vs Normal.
- **Estabilizamos o treino com early stopping** e *class weights* derivados
  apenas do *split* de treino, evitando *leakage*.
- **Reproduzimos resultados em 5 seeds** após observar que um único *run* dava
  a impressão (falsa) de *cherry-picking* — agora reportamos média/desvio.
- **Adicionamos suporte bilíngue (EN/PT-BR)** ao pipeline de relatórios via
  `pipelines/python/src/i18n.py` e `translate_site.py`.
- **Lições aprendidas:** (a) verificar sempre o domínio numérico dos dados
  (TPM vs log2(TPM)); (b) usar *baseline* topológico simples (grau) é
  imprescindível para validar interpretabilidade; (c) reprodutibilidade entre
  *seeds* é tão importante quanto a métrica média.

# Ferramentas

- **Python**: PyTorch, **PyTorch Geometric** (GATv2Conv), NumPy, Pandas, SciPy,
  scikit-learn, Matplotlib, Plotly, NetworkX.
- **GEOquery / UCSC Xena Toolkit** para baixar e harmonizar TCGA-SKCM e GSE112509.
- **STRING** (v12, *combined score* ≥ 700) para a rede PPI.
- **Open Targets Platform** para anotação de evidência terapêutica.
- **Cytoscape** (legado P2) — referência visual e validação topológica.
- **Streamlit** (`pipelines/python/scripts/explore_streamlit.py`,
  `pipelines/python/scripts/predict_streamlit.py`,
  `pipelines/python/notebooks/explain_gat_interactive.py`) para exploração interativa.
- **Plotly + NetworkX** para os gráficos interativos de subgrafos PPI.

# Resultados

**Classificação.** O GATv2 atinge acurácia média de ~0.86 sobre 5 seeds, com
Metástase como classe de maior precisão e maior estabilidade do *recall*.
F1-score por classe e matrizes de confusão estão consolidados em
`data/processed/multiseed_20260531_202951/stability.json`.

![Estabilidade — Acurácia por seed](assets/images/stability_accuracy.png)

**Recuperação por *gene-set*.** Para cada classe e contraste, a atenção do GAT
supera o grau do PPI nos quatro conjuntos curados, com p-valores hipergeométricos
significativamente menores. Em particular, *Tumor − Normal* enriquece
queratinócitos/cornified envelope (esperado, pois nevi expressam menos esses
programas que pele *bulk*) e *Metástase − Primário* enriquece marcadores imunes,
consistente com a literatura de TILs em melanoma metastático.

![Spearman — estabilidade dos rankings](assets/images/stability_spearman.png)

**Pseudotempo.** A projeção dos *embeddings* preserva a ordem
*Nevo → Primário → Metástase* em todas as seeds, com a atenção média por *bin*
exibindo trajetórias monótonas para genes melanocíticos (queda) e imunes (subida).

![Pseudotempo — heatmap](assets/images/pseudotime_heatmap.png)
![Pseudotempo — linhas por gene](assets/images/pseudotime_lines.png)

**Estabilidade.** Jaccard@20 entre seeds para os top-genes por classe é
consistentemente alto (mediana > 0.5), e o *consensus* (`stability_consensus_top20.csv`)
elege biomarcadores candidatos robustos.

![Jaccard — estabilidade dos top-K](assets/images/stability_knn_jaccard.png)

**Subgrafos interpretativos.** O pipeline `pipelines/python/src/interpret_viz.py`
gera subgrafos PPI por classe (top-K por atenção) — *Benign Nevus*, *Primary
Tumor* e *Metastasis* — usados como evidência interpretativa para os contrastes
discutidos acima.

# Discussão

Os resultados sustentam que a atenção aprendida pelo GAT é **mais do que um
proxy de hub topológico**: o *baseline* de grau falha em recuperar conjuntos
curados específicos de classe (especialmente queratina e imune), enquanto a
atenção os recupera com p-valores hipergeométricos significativos. Isso indica
que o modelo aprende **importância gene-específica condicional à classe**, e
não apenas decora a estrutura PPI.

A consistência multi-seed (Jaccard@20 alto, Spearman estável, pseudotempo
preservado) é, na nossa visão, o resultado mais relevante: ela mostra que as
conclusões biológicas extraídas do modelo não são artefato de uma execução
afortunada. Por outro lado, **drivers mutacionais clássicos (BRAF, NRAS) não
aparecem** no painel — esperado, dado que eles são definidos por mutação, não
por variabilidade de expressão; o painel atual é, portanto, **complementar** e
não substitutivo a análises de variantes.

Limitações: (i) o subset GSE112509 é menor e mais homogêneo que TCGA, o que pode
inflar a separabilidade da classe Nevo; (ii) o painel de top-variáveis tende a
sub-representar genes regulatórios de baixa expressão; (iii) a atenção, embora
informativa, é uma medida *post hoc* — sua interpretação como “importância
causal” deve ser feita com cautela e idealmente validada com perturbações
*in silico* (ablação de nós/arestas).

# Conclusão

Mostramos que um **GATv2 treinado sobre uma PPI fixa** consegue, a partir de
expressão gênica, distinguir nevos benignos, tumores primários e metástases de
melanoma com alta acurácia, **e** que sua atenção captura biologia coerente com
a literatura, superando *baselines* topológicos. A análise multi-seed dá
robustez aos *rankings* e pseudotempos, e o site estático bilíngue torna os
achados navegáveis sem reexecução.

Principais desafios: (i) *batch effects* TCGA↔GTEx, resolvidos com TOIL e depois
com a substituição por GSE112509 para a Class 2; (ii) interpretabilidade —
projetar uma agregação de atenção que respeitasse classes e amostras corretas;
(iii) reprodutibilidade — exigiu refatorar o pipeline para suportar *seeds*
múltiplas com *splits* estratificados independentes.

Lições aprendidas: trabalhar com grafos *por amostra* compartilhando topologia é
um modelo poderoso quando a topologia é biologicamente significativa; *baselines*
simples são imprescindíveis para reivindicar interpretabilidade; e reportar
estabilidade entre seeds deveria ser padrão em qualquer trabalho de GNN
aplicada a biologia.

# Trabalhos Futuros

- **Validação por perturbação *in silico*** (remoção de nós/arestas top-atenção)
  para checar causalidade aproximada.
- **Integrar mutações somáticas** (BRAF/NRAS/TP53) como atributos adicionais de
  nó ou como rótulos auxiliares em *multi-task learning*.
- **Substituir o painel de top-variáveis** por um painel curado clinicamente
  (e.g., painéis comerciais de melanoma) para avaliar transferibilidade.
- **Estender para outros tumores cutâneos** (carcinomas basocelular e
  espinocelular) para investigar generalização da abordagem.
- **Aprendizado contrastivo** sobre *embeddings* de amostra para análise de
  trajetórias contínuas (além do pseudotempo discreto).
- **Avaliação prospectiva** dos biomarcadores candidatos em coortes externas
  (e.g., GSE65904) e cruzamento com Open Targets para priorização terapêutica.

# Referências Bibliográficas

[1] Hodis, E., et al. *A landscape of driver mutations in melanoma.* Cell, 2012.

[2] Tirosh, I., et al. *Dissecting the multicellular ecosystem of metastatic
melanoma by single-cell RNA-seq.* Science, 2016.

[3] Veličković, P., et al. *Graph Attention Networks.* ICLR, 2018.

[4] Brody, S., Alon, U., Yahav, E. *How Attentive are Graph Attention Networks?*
ICLR, 2022.

[5] Ochoa, D., et al. *Open Targets Platform: supporting systematic
drug-target identification and prioritisation.* Nucleic Acids Research, 2021.

[6] Vivian, J., et al. *Toil enables reproducible, open source, big biomedical
data analyses.* Nature Biotechnology, 2017.

[7] Szklarczyk, D., et al. *The STRING database in 2023.* Nucleic Acids Research, 2023.

[8] Cancer Genome Atlas Network. *Genomic Classification of Cutaneous Melanoma.*
Cell, 2015.

[9] Kunz, M., et al. *RNA-seq analysis of melanocytic nevi and melanomas
(GSE112509).* Gene Expression Omnibus / *Lab Investigation*, 2018.

[10] Fey, M., Lenssen, J. E. *Fast Graph Representation Learning with PyTorch
Geometric.* ICLR Workshop on Representation Learning on Graphs and Manifolds, 2019.
