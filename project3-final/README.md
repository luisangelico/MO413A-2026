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
investiga o perfil de expressão gênica de **nevos melanocíticos benignos**,
**tumor primário cutâneo** e **metástase de melanoma** à luz da rede de
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

> Os slides da apresentação podem ser encontrados [neste link](assets/slides/MO413A-P3.pdf`).

# Fundamentação Teórica


O câncer de pele é o mais recorrente em todo o mundo, com incidência progressivamente crescente nas últimas décadas, configurando um relevante problema de saúde pública. Clinicamente, classifica-se em dois grandes grupos: o melanoma cutâneo e o câncer de pele não-melanoma (CPNM), sendo este último responsável pela expressiva maioria dos casos diagnosticados globalmente (Sung et al., 2021). Segundo dados do GLOBOCAN, a taxa de incidência padronizada por idade para o melanoma foi de 3,4 por 100.000 habitantes em 2020, com taxa de mortalidade de 0,55 por 100.000, e as regiões de maior incidência mundial concentram-se na Austrália e Nova Zelândia (Huang et al., 2023). No Brasil, o câncer de pele é a neoplasia mais frequente, correspondendo a 30% de todos os diagnósticos oncológicos do país, com dados longitudinais evidenciando crescimento expressivo da morbidade, que passou de 14,88 para 27,69 casos por 100.000 habitantes entre 2012 e 2022 (Castro, Martins e Kock, 2024), e 122,90 por 100.000 habitantes em cada ano do triênio de 2026 a 2028 (INCA, 2026). O CPNM, cujos principais subtipos são o carcinoma basocelular (CBC) e o carcinoma espinocelular (CEC), é responsável por mais de 176.000 casos anuais no território nacional, com acometimento predominante da população de pele branca e correlação negativa com a latitude geográfica, sendo as regiões Sul e Sudeste as mais afetadas (Castro, Martins e Kock, 2024). O número de casos estimados para 2026 pode chegar a 263.280 (INCA, 2026).
O melanoma cutâneo origina-se da transformação maligna dos melanócitos, células localizadas predominantemente na camada basal da epiderme e responsáveis pela síntese de melanina. Embora represente uma proporção minoritária dos cânceres de pele em termos de incidência absoluta, é a forma mais letal da doença, responsável pela maioria dos óbitos relacionados a neoplasias cutâneas em virtude de seu elevado potencial de invasão e metástase precoce (Gieniusz, Skrzydlewska e Łuczaj, 2024). A radiação ultravioleta (UV) constitui o principal fator etiológico ambiental, exercendo efeito carcinogênico por meio de danos oxidativos ao DNA, às proteínas e aos lipídios celulares, comprometendo a regulação do ciclo celular e favorecendo a evasão da vigilância imunológica (Gieniusz, Skrzydlewska e Łuczaj, 2024). No plano molecular, a patogênese do melanoma caracteriza-se por acentuada heterogeneidade genômica, com o dano solar cumulativo modulando o perfil mutacional dos tumores; dentre as alterações mais prevalentes, destaca-se a mutação no gene BRAF, particularmente a variante BRAFV600E, cuja ativação constitutiva da via RAF/MEK/ERK (MAPK) promove proliferação celular desregulada e resistência à apoptose, sendo identificada em 40 a 50% dos melanomas cutâneos (Colombino et al., 2024). O prognóstico da doença é fortemente influenciado pelo estadiamento no momento do diagnóstico, com impacto significativo na sobrevida global dos pacientes (Cirillo, 2025).
O câncer de pele não-melanoma compreende um espectro heterogêneo de neoplasias derivadas de células epidérmicas não melanocíticas, dentre as quais o CBC e o CEC representam os subtipos de maior relevância clínica e epidemiológica (Attal et al., 2024). O CBC, responsável por aproximadamente 80% dos casos de CPNM, origina-se das células basais da epiderme e apresenta crescimento lento com baixo potencial metastático, embora possa ocasionar destruição tecidual local significativa na ausência de tratamento oportuno (Castro, Martins e Kock, 2024). Sua patogênese envolve primariamente a ativação aberrante da via de sinalização Hedgehog (HH), frequentemente associada a mutações inativadoras no gene PTCH1 decorrentes da exposição cumulativa à radiação UV (Attal et al., 2024). O CEC, por sua vez, origina-se de queratinócitos com displasia progressiva e apresenta comportamento biologicamente mais agressivo, com maior capacidade de invasão local e risco de disseminação linfática e hematogênica (Didona et al., 2018). Entre os fatores de risco compartilhados por ambos os subtipos, destacam-se a exposição crônica à radiação UV, o fotótipo cutâneo claro, a idade avançada, a imunossupressão e a predisposição genética, sendo o diagnóstico precoce determinante para a redução da morbimortalidade associada a essas neoplasias (Attal et al., 2024; Didona et al., 2018).


Redes PPI fornecem um *scaffold* de relações funcionais sobre o qual a expressão
gênica pode ser interpretada como um sinal estruturado. **Graph Neural Networks**
e, em particular, **Graph Attention Networks (GAT/GATv2)** aprendem
representações em grafos atribuindo pesos adaptativos às arestas, o que permite
identificar, a posteriori, **quais conexões PPI mais contribuíram para a
predição** — propriedade explorada aqui como uma forma de interpretabilidade
*built-in*. Comparado a abordagens clássicas (PCA, t-SNE sobre expressão), o GAT
incorpora explicitamente a topologia biológica conhecida.

Mais especificamente, a equação das _embeddings_  _H_ de uma determinada camada _l_ de uma Graph Neural Network pode ser representada conforme ilustra a figura a seguir, onde se difere de uma rede neural "tradicional" com a utilização de uma matriz adjacente _A_ , representando a relação entre cada nó por meio de uma matriz de 0 e 1, e uma matriz diagonal _D_, que se caracteriza uma matriz com o grau de cada nó e cuja finalidade é normalizar a matriz _A_ . Nas equações \sigma representa a função de ativação e _W_ a matriz de pesos a serem aprendidos no treinamento do modelo.

![Graph Neural Network](assets/images/gat.png)

GAT

A motivação clínica e técnica deste trabalho dialoga diretamente com (i) o uso da
plataforma **Open Targets** para priorização de alvos terapêuticos [5], (ii) o
recompute **TOIL** para tornar TCGA e GTEx comparáveis [6] e (iii) o consórcio
**STRING** como referência de interações [7].

# Perguntas de Pesquisa

As perguntas evoluíram em relação à P2 (centradas em centralidade clássica de PPI
e *globalScore*) para incorporar o eixo de aprendizado em grafos:
1. Como as redes gênicas diferem entre câncer de pele melanoma e não-melanoma?
2. Existem genes “exclusivos” em cada tipo de câncer? 
3. Existem genes compartilhados entre melanoma e não-melanoma?
4. Quais interações são adquiridas ou perdidas ao longo da progressão do melanoma?
5. Podemos identificar módulos (clusters) específicos para cada tipo de câncer?


# Metodologia

A metodologia combina **Ciência de Redes** com **aprendizado profundo em grafos**:

- **Construção da rede:** PPI do STRING com *threshold* de combined score ≥ 200
  (parâmetro `CONFIDENCE_THRESHOLD` em `src/config.py`), restrita aos *N* = 1 000
  genes mais variáveis na união dos coortes; o mesmo
  grafo (índice de arestas) é compartilhado por todas as amostras (cada amostra
  difere apenas no atributo de nó = expressão).
- **Modelo:** GATv2 de 3 camadas (`hidden_channels=128`, 4 cabeças nas duas
  primeiras camadas e 1 na última, `dropout=0.4`, *LayerNorm* + *global mean
  pooling*), treinado com *cross-entropy* ponderada por classe; *split*
  estratificado 70/15/15 com *seed* fixa (`SEED=42`), `EARLY_STOP_PATIENCE=50`
  sobre a perda de validação e *scheduler* `ReduceLROnPlateau`.
- **Interpretação por atenção:** `forward(..., return_all_attn=True)` expõe pares
  `(edge_index, alpha)` para cada uma das 3 camadas; a atenção é agregada **apenas
  sobre amostras de teste corretamente classificadas** para cada classe, gerando
  um *score* por nó (gene). Contrastes entre classes (e.g., *Metástase − Primário*)
  são produzidos como subtração de rankings normalizados.
- **Recuperação por *gene-set*:** para cada classe e contraste, calculamos o
  *p*-valor hipergeométrico de sobreposição entre o top-K (K ∈ {10, 25, 50,
  100}) e cada conjunto curado; reportamos sempre lado a lado o *baseline* de
  grau do PPI.
- **Pseudotempo:** projeção dos *embeddings* finais em uma direção biológica
  (Nevo → Primário → Metástase) seguida de *binning*; a atenção média por *bin*
  produz trajetórias gene-a-gene (heatmap e linhas).
- **Estabilidade multi-seed:** o pipeline completo é re-executado para 5 *seeds*;
  computamos Jaccard@K dos top-K por classe, Spearman dos *rankings* completos,
  e variabilidade do pseudotempo (banda de incerteza nos heatmaps).
- **Visualizações:** subgrafos PPI por classe (top-K por atenção), projeções
  PCA/t-SNE dos *embeddings*, heatmaps e linhas de pseudotempo, gráficos de
  estabilidade — gerados pelos módulos em `pipelines/python/src/`.

## Bases de Dados e Evolução

| Base de Dados | Endereço na Web | Resumo descritivo |
| ----- | ----- | ----- |
| TCGA-SKCM (TOIL recompute) | [UCSC Xena TOIL](https://xenabrowser.net/datapages/?cohort=TCGA%20TARGET%20GTEx) | Reprocessamento harmonizado de TCGA e GTEx para minimizar efeito de coorte; usado para amostras de **Tumor Primário** e **Metástase** de melanoma. Valores em log₂(TPM+0.001). |
| GSE112509 (GEO) | [GSE112509](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE112509) | RNA-seq de **nevos benignos** e melanomas; aqui usamos apenas as 23 amostras de nevo como Class 2 (*Benign Nevus*), substituindo o GTEx skin originalmente usado na P2. |
| STRING v12 | [STRING](https://string-db.org/) | Interações proteína-proteína com *combined score*. Filtro: ≥ 200 (configurável via `CONFIDENCE_THRESHOLD` em `src/config.py`). Define o esqueleto compartilhado dos grafos por amostra. |
| GENCODE / Ensembl v23 (probemap) | [GENCODE v23](https://www.gencodegenes.org/human/release_23.html) | Anotação de referência usada para mapear identificadores Ensembl (formato do TOIL) para símbolos HGNC, e para filtrar o painel a genes *protein-coding*. Arquivo: `gencode.v23.annotation.gene.probemap`. |
| Open Targets — Melanoma (EFO_0000756) | [Open Targets](https://platform.opentargets.org/disease/EFO_0000756/associations) | Usada para anotar genes da rede com `globalScore` e fase clínica de evidência terapêutica, suportando a discussão de biomarcadores. |
| Conjuntos curados de genes | manual / literatura | Listas em `pipelines/python/src/interpret.py` para *melanocyte lineage*, *melanoma progression*, *immune infiltrate*, *cornified envelope/keratin*; servem para *gene-set recovery*. |

**Descobertas e tratamentos por base.** Os valores TOIL **já são log₂(TPM+0.001)**
— aplicar `log1p` por engano (como ocorria no pipeline GDC-Hub, em TPM puro,
hoje arquivado) introduz um viés sistemático que deslocava artificialmente a
separação Tumor vs Normal. Documentamos esse *pitfall* no `src/README.md`.
Para o GSE112509, foi necessário um cache de tradução de identificadores
(`data/external/translation_cache/`) e harmonização de nomes para o
*namespace* HGNC do TCGA. STRING foi filtrado por confiança e restrito ao
subconjunto de genes do painel de variabilidade (interseção entre os coortes),
garantindo que cada amostra projete sobre o **mesmo grafo**.

## Pipeline de Pré-processamento

O diagrama a seguir resume todas as transformações aplicadas aos dados *antes*
do treinamento, do download bruto até o `.pt` consumido pelo GAT
(`scripts/download_toil.py`):

```mermaid
flowchart LR
    %% ========== Sources ==========
    A1["UCSC Xena TOIL<br/>TcgaTargetGtex_rsem_gene_tpm.gz<br/>log2(TPM+0.001)"]:::raw
    A2["TcgaTargetGTEX_phenotype.txt.gz<br/>(rótulos de amostra)"]:::raw
    A3["GEO GSE112509<br/>DESeq2 normalized counts<br/>(nevos benignos)"]:::raw
    A4["gencode.v23.annotation.gene.probemap<br/>(Ensembl ↔ HGNC)"]:::raw
    A5["STRING v12 API<br/>(physical PPI)"]:::raw

    %% ========== TCGA branch ==========
    A1 --> B1["Filtra TCGA-SKCM<br/>Primary Tumor + Metastatic<br/>(Class 0 / Class 1)"]
    A2 --> B1
    A4 --> B1
    B1 --> B2["Mapeia Ensembl → HGNC<br/>via probemap"]

    %% ========== GSE112509 branch ==========
    A3 --> C1["Seleciona apenas amostras<br/>de NEVO (Class 2)"]
    C1 --> C2["log2(counts + 1)<br/>harmoniza escala com TOIL"]

    %% ========== Merge ==========
    B2 --> D1["Concat amostras<br/>TCGA ∪ GSE112509<br/>~492 amostras"]
    C2 --> D1

    D1 --> D2["Mantém genes<br/>protein-coding (HGNC)"]
    D2 --> D3["split estratificado 70/15/15<br/>(SEED=42)"]
    D3 --> D4["Top-variable genes<br/>variância calculada<br/>SOMENTE no train TCGA<br/>NUM_NODES = 1000"]

    %% ========== STRING graph ==========
    A5 --> E1["STRING REST<br/>required_score ≥ 200<br/>identifiers = top-1000 genes"]
    D4 --> E1
    E1 --> E2["edge_index único<br/>compartilhado por todas<br/>as amostras"]

    %% ========== Per-sample graphs ==========
    D4 --> F1["Para cada amostra:<br/>x = vetor de expressão (1000×1)<br/>y = classe ∈ {0,1,2}"]
    E1 --> F1
    F1 --> F2["torch_geometric.data.Data<br/>(x, edge_index, y)"]
    F2 --> G["skcm_nevi_1000_200.pt<br/>(492 grafos)<br/>+ splits.npz"]:::out

    %% ========== Styles ==========
    classDef raw fill:#e8f0fe,stroke:#3367d6,color:#1a3a7a;
    classDef out fill:#e6f4ea,stroke:#137333,color:#0d4d22;
```

**Etapas:**

1. **TOIL TPM** — baixa a matriz `log2(TPM+0.001)` recomputada da UCSC Xena para todas as amostras TCGA + GTEx + TARGET.
2. **Phenotype** — baixa o arquivo de rótulos da Xena com tipo de tecido, *sample type* e coorte de cada amostra.
3. **GSE112509** — baixa as contagens DESeq2 normalizadas do GEO para o estudo de nevos benignos vs melanomas.
4. **GENCODE probemap** — baixa a tabela de equivalência Ensembl ID ↔ símbolo HGNC ↔ tipo de gene (versão v23).
5. **STRING API** — endpoint REST do STRING v12 que devolve interações físicas com *combined score*.
6. **Filtra TCGA-SKCM** — seleciona, no TOIL, apenas amostras do coorte SKCM com *sample type* "Primary Tumor" (Class 0) ou "Metastatic" (Class 1).
7. **Ensembl → HGNC** — substitui IDs Ensembl pelos símbolos HGNC do probemap, descartando linhas sem mapeamento.
8. **Seleciona nevos** — filtra o GSE112509 para reter apenas as 23 amostras de nevo (Class 2), descartando os melanomas do GEO.
9. **log2(counts+1)** — aplica `log2(x+1)` às contagens DESeq2 do GSE para casar a *escala* do TOIL (não o valor absoluto).
10. **Concatena coortes** — junta as matrizes TCGA-SKCM e GSE112509 por gene em comum, formando a matriz combinada (~492 amostras).
11. **Protein-coding** — mantém apenas genes anotados como *protein-coding* no GENCODE, descartando pseudogenes/lncRNAs.
12. **Split 70/15/15** — divisão estratificada por classe com `SEED=42`, persistida em `splits.npz` para evitar *leakage* nas etapas seguintes.
13. **Top-variable genes** — calcula a variância **somente nas amostras de treino do TCGA** e seleciona os `NUM_NODES = 1000` genes mais variáveis.
14. **STRING REST** — consulta o STRING para os 1000 símbolos com `required_score ≥ 200`, recuperando todas as arestas físicas entre eles.
15. **edge_index único** — converte a lista de arestas em um `edge_index` PyG **compartilhado** por todas as amostras (a topologia não muda).
16. **Per-sample x,y** — para cada amostra constrói `x ∈ ℝ^{1000×1}` (vetor de expressão) e `y ∈ {0,1,2}` (rótulo de classe).
17. **Data objects** — empacota cada `(x, edge_index, y)` em um `torch_geometric.data.Data`.
18. **Saída** — serializa a lista de 492 grafos em `skcm_nevi_1000_200.pt`, junto a `splits.npz` para uso reprodutível em treino/validação/teste.

**Pontos-chave que evitam vazamento e garantem comparabilidade:**

- A variância dos genes é calculada **apenas no *split* de treino do TCGA**, de
  forma que o painel top-1000 não é contaminado por amostras de validação,
  teste ou pelo coorte de nevo.
- Os valores TOIL **não recebem `log1p`** (já estão em `log2(TPM+0.001)`). O
  GSE112509 vem em contagens DESeq2, e é trazido para `log2(x+1)` para casar
  o *spread* da escala — não a localização absoluta, que permanece distinta.
- A topologia do grafo (`edge_index` STRING) é **fixa** para todas as
  amostras; cada amostra varia apenas no atributo de nó (expressão).

# Treinamento

O modelo é definido em `pipelines/python/src/model.py` e treinado por
`pipelines/python/src/train.py`. A reprodução multi-seed (5 seeds independentes)
fica em `pipelines/python/scripts/train_multiseed.py` — todos os números
reportados na seção *Resultados* vêm dela.

**Arquitetura (`GATv2Classifier`).**

| Parâmetro | Valor |
| --- | --- |
| Camadas GATv2 | 3 |
| `hidden_channels` | 128 |
| Cabeças por camada | 4, 4, 1 |
| `dropout` | 0.4 |
| Normalização | `LayerNorm` após cada camada |
| Pooling | `global_mean_pool` |
| Cabeça de classificação | `Linear(128 → 3)` |
| Atributo de nó (`in_channels`) | 1 (expressão log) |

**Hiperparâmetros de treino (`src/train.py`).**

| Parâmetro | Valor |
| --- | --- |
| Otimizador | `Adam` |
| `learning_rate` | 1e-3 |
| `weight_decay` | 5e-4 |
| Scheduler | `ReduceLROnPlateau` (factor 0.5, patience 10, min_lr 1e-6) |
| Loss | `CrossEntropyLoss` |
| Balanceamento | `WeightedRandomSampler` por *batch* (train only) |
| `batch_size` | 32 |
| `num_epochs` (máx) | 500 |
| `EARLY_STOP_PATIENCE` | 50 (sobre val loss) |
| `SEED` (single-run) | 42 |
| Split | 70/15/15 estratificado, salvo em `splits.npz` |
| Seleção de modelo | menor val loss observada |
| Device | auto: CUDA → MPS → CPU (env `GAT_DEVICE`) |

**Multi-seed (`scripts/train_multiseed.py`).** O *driver* roda 5 seeds (0–4)
com a mesma arquitetura, mesmo dataset e mesmo *bundled split*, variando apenas
a seed do PyTorch/NumPy. Cada execução grava
`multiseed_<timestamp>/seed_<i>/best_model.pt`, `splits.npz` e `history.npz`,
mais um `results.json` agregado e um `stability.json` com:

- `test_accuracy.{mean, std, ci95_half_width, values}`
- `per_class_f1.{Primary Tumor, Metastasis, Benign Nevus}.{mean, std}`

A consolidação multi-seed final usada no relatório é
`data/processed/multiseed_20260531_202951/`.

**Avaliação (`src/evaluate.py`).** Carrega o `best_model.pt` salvo, lê o
`splits.npz` correspondente e reporta acurácia, F1 por classe e matriz de
confusão **somente sobre o split de teste**. Se `splits.npz` estiver ausente,
o script avisa e cai para o dataset inteiro — esse aviso indica vazamento e
deve ser tratado retreinando.

**Sem *leakage*.** Os pesos por classe (via sampler) e o painel de top-1000
genes são derivados **apenas do treino**; a perda de validação é a única
métrica usada para seleção de modelo; o teste só é tocado em `evaluate.py`.

# Modelo Lógico

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
genes** — o conjunto de top-variáveis é calculado **na união dos coortes**,
de modo que a mesma lista de nós seja válida em todos eles.

## Análises Realizadas

Os pipelines completos estão em `pipelines/python/scripts/` (entry-points
executáveis) e `pipelines/python/src/` (módulos importáveis). Uma cópia
pronta para reprodução também está em `src/` (ver `src/README.md`). Trecho
central da extração de atenção:

~~~python
# pipelines/python/src/interpret.py — importância de nó por amostra
logits, attn_layers = model(data.x, data.edge_index, batch, return_all_attn=True)
importance = np.zeros(num_nodes, dtype=np.float64)
for ei, alpha in attn_layers:
    a = alpha.mean(dim=-1).detach().cpu().numpy()      # média entre cabeças
    dst = ei[1].detach().cpu().numpy()                 # nó-destino de cada aresta
    np.add.at(importance, dst, a)                      # soma atenção que entra em cada nó
# em seguida, aggregate_per_class média a importância apenas sobre amostras
# corretamente classificadas para cada classe predita.
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
- **Expusemos uma camada de internacionalização (EN/PT-BR)** em
  `pipelines/python/src/i18n.py` para textos dos relatórios.
- **Lições aprendidas:** (a) verificar sempre o domínio numérico dos dados
  (TPM vs log2(TPM)); (b) usar *baseline* topológico simples (grau) é
  imprescindível para validar interpretabilidade; (c) reprodutibilidade entre
  *seeds* é tão importante quanto a métrica média.

# Ferramentas

- **Python**: PyTorch, **PyTorch Geometric** (GATv2Conv), NumPy, Pandas, SciPy,
  scikit-learn, Matplotlib, Plotly, NetworkX.
- **GEOquery / UCSC Xena Toolkit** para baixar e harmonizar TCGA-SKCM e GSE112509.
- **STRING** (v12, *combined score* ≥ 200) para a rede PPI.
- **Open Targets Platform** para anotação de evidência terapêutica.
- **Cytoscape** (legado P2) — referência visual e validação topológica.
- **Streamlit** (`pipelines/python/scripts/explore_streamlit.py`,
  `pipelines/python/scripts/predict_streamlit.py`,
  `pipelines/python/notebooks/explain_gat_interactive.py`) para exploração interativa.
- **Plotly + NetworkX** para os gráficos interativos de subgrafos PPI.

# Resultados

**Classificação.** Em 5 *seeds* sobre o *split* de teste (74 amostras), o
GATv2 atinge acurácia média **0.784 ± 0.057** (mín 0.716, máx 0.851). O F1
por classe é desigual: **Metástase 0.87 ± 0.04**, **Nevo Benigno 0.57 ±
0.12**, **Tumor Primário 0.40 ± 0.22** — reflexo do desbalanço acentuado
(367/102/23) e da sobreposição transcricional Primário↔Metástase. Métricas
completas em `data/processed/multiseed_20260531_202951/stability.json` e
`results.json`.

![Estabilidade — Acurácia por seed](assets/images/stability_accuracy.png)

**Recuperação por *gene-set*.** Para cada classe e contraste, a atenção do GAT
supera o grau do PPI nos quatro conjuntos curados, com p-valores hipergeométricos
significativamente menores. Em particular, o contraste *Tumor − Nevo* enriquece
programas melanocíticos/diferenciação, e *Metástase − Primário* enriquece
marcadores imunes, consistente com a literatura de TILs em melanoma metastático.
Os *p*-valores e ranks completos estão em
`data/processed/run_multiseed_seed*/attention_gene_set_recovery.csv`.

![Spearman — estabilidade dos rankings](assets/images/stability_spearman.png)

**Embeddings**. Na imagem abaixo, cada coluna é uma aresta PPI que o modelo trata com atenção diferente entre as três classes. Cor = atenção média sobre 5 seeds. As arestas são ordenadas por amplitude entre classes / variabilidade entre seeds — valores altos indicam que o modelo trata a aresta como informativa de forma consistente e a diferencia entre tipos de amostra.

![Embeddings](assets/images/embeddings.png)

A imagem a seguir ilustra a clusterização por t-SNE, demonstrando uma segregação entre tumor benigno de tumor primário e metasfáse. Essas duas últimas se misturam na projeção, demonstrando uma dificuldade na segregação das classes.


![Embeddings - cluster](assets/images/embeddings_cluster.png)


**Pseudotempo.** A projeção dos *embeddings* preserva a ordem
*Nevo → Primário → Metástase* em todas as seeds, com a atenção média por *bin*
exibindo trajetórias monótonas para genes melanocíticos (queda) e imunes (subida).

![Pseudotempo — heatmap](assets/images/pseudotime_heatmap.png)

![Pseudotempo — linhas por gene](assets/images/pseudotime_lines.png)

**Estabilidade.** Jaccard@K e correlação de Spearman dos rankings completos
entre seeds estão em
`data/processed/multiseed_20260531_202951/stability_spearman_*.csv` e
`figures/stability_knn_jaccard.png`. O *consensus* dos top-20 entre seeds
(`stability_consensus_top20.csv`) elege os candidatos a biomarcador
discutidos abaixo.

![Jaccard — estabilidade dos top-K](assets/images/stability_knn_jaccard.png)

**Subgrafos interpretativos.** O pipeline `pipelines/python/src/interpret_viz.py`
gera subgrafos PPI por classe (top-K por atenção) — *Benign Nevus*, *Primary
Tumor* e *Metastasis* — usados como evidência interpretativa para os contrastes
discutidos acima.

# Discussão
## Graph Attention Networks
Os resultados sustentam que a atenção aprendida pelo GAT é **mais do que um
proxy de hub topológico**: o *baseline* de grau falha em recuperar conjuntos
curados específicos de classe (especialmente queratina e imune), enquanto a
atenção os recupera com p-valores hipergeométricos significativos. Isso indica
que o modelo aprende **importância gene-específica condicional à classe**, e
não apenas decora a estrutura PPI.

A consistência multi-seed (Spearman estável dos rankings, pseudotempo
preservado, *consensus* de top-20 com sobreposição entre seeds) é, na nossa
visão, o resultado mais relevante: ela mostra que as conclusões biológicas
extraídas do modelo não são artefato de uma execução afortunada. Por outro
lado, **drivers mutacionais clássicos (BRAF, NRAS) não aparecem** no painel —
esperado, dado que eles são definidos por mutação, não por variabilidade de
expressão; o painel atual é, portanto, **complementar** e não substitutivo a
análises de variantes.

Limitações: (i) **forte desbalanço de classes** (367 Metástase / 102 Primário
/ 23 Nevo), que se reflete diretamente nos F1 baixos para Primário e Nevo,
mesmo com *class weights*; (ii) **sobreposição transcricional Primário ↔
Metástase**, intrínseca à biologia (subclones metastáticos derivados do
primário), que limita o teto da acurácia; (iii) o subset GSE112509 (Nevo) é
pequeno (n=23), o que pode inflar a separabilidade observada da classe Nevo;
(iv) o painel de top-variáveis tende a sub-representar genes regulatórios de
baixa expressão; (v) a atenção, embora informativa, é uma medida *post hoc*
— sua interpretação como “importância causal” deve ser feita com cautela e
idealmente validada com perturbações *in silico* (ablação de nós/arestas).

## Análise de Graphos (Cytoscape)
As análises mostraram a existencia de genes exclusivos para cada tipo de câncer, como consta na tabela a seguir:

| Câncer | Quantidade de Genes |
|---------|----------------:|
| Melanoma in situ | 583 |
| Melanoma primário | 75 |
| Melanoma metastático | 43 |
| Carcinoma espinocelular | 88 |

A tabela a seguir mostra os genes compartilhados e o tipo de função associada encontrados:

| Genes | Função |
|--------|---------|
| CDK1, CDC45, CHEK1, NCAPH, ZWINT | Controle do ciclo celular |
| RRM2, CHEK1, CDC45 | Replicação e integridade genômica |
| IFI27, HSPB6, ATG12 | Estresse e sobrevivência |
| CCN1 | Remodelamento tumoral |
| MELK, PBK | Proliferação |

Também foi realizada uma análise das interações adiquiridas ou perdidas nao longo da progressão do melanoma (ver figura abaixo, que mostra um exemplo de interação adquirida em melanoma):

- Adquiridas: ERBB2, HRAS, MAPK3, ETS1, HDAC4, PHGDH
- Perdidas: AKT3, ALDH2, AQP1, ABLIM1, ABI2


![Interações adquirida em melanoma](assets/images/interacoes_adquiridas_melanoma.png)

*Exemplo de interação adquirida em melanoma*.

A imagem a seguir mostra um cluster que é do câncer de pele não-melanoma, só aparece nele. Aqui os genes KRT estão associados à queratinização, SPRR1A está associado à diferenciação do queratinócito, e SERPINB são reguladores negativos de apoptose. A cor também mais próxima ao vermelho aponta uma maior expressão em relação à pele saudável.

![Cluster não-melanoma](assets/images/cluster_nao_melanoma.png)

*Cluster do câncer de pele não-melanoma: os genes KRT estão associados à queratinização, SPRR1A está associado à diferenciação do queratinócito, e SERPINB são reguladores negativos de apoptose.*

# Conclusão

Mostramos que um GAT treinado sobre uma rede de interação proteína-proteína (PPI) fixa é capaz de distinguir diferentes estados de progressão do melanoma utilizando apenas perfis de expressão gênica. Em um pipeline reprodutível executado em cinco seeds independentes, combinando amostras do TCGA-SKCM e da coorte GSE112509, o modelo atingiu acurácia média de 78 ± 6%, demonstrando desempenho consistente apesar da variabilidade inerente aos conjuntos de dados biológicos.

Além do desempenho preditivo, observamos que os padrões de atenção aprendidos pelo modelo convergem para sinais biologicamente plausíveis e amplamente descritos na literatura, incluindo genes associados à diferenciação epitelial (família KRT) e à resposta imune tumoral. A concordância entre os genes priorizados pelo modelo e estudos recentes sugere que a atenção captura aspectos relevantes da biologia do melanoma, embora não deva ser interpretada como evidência causal direta.

Os principais desafios encontrados envolveram a integração de dados provenientes de diferentes coortes, a construção de métricas de interpretabilidade adequadas para modelos baseados em atenção e a garantia de reprodutibilidade experimental. A adoção de múltiplas seeds independentes e de um pipeline padronizado permitiu avaliar de forma mais robusta a estabilidade dos resultados.

Também identificamos limitações importantes. Parte do sinal discriminativo aprendido pelo modelo parece refletir diferenças associadas ao sítio anatômico das amostras — particularmente contrastes entre tecido epidérmico e linfonodal — e não exclusivamente mecanismos biológicos de progressão tumoral. Além disso, o desempenho permaneceu desigual entre classes, indicando que a tarefa ainda apresenta desafios de generalização.

Por fim, a evolução do projeto levou a uma reformulação da pergunta científica original. Em vez de comparar diferentes tipos de câncer utilizando métricas clássicas de centralidade em redes biológicas, passamos a investigar a progressão do melanoma por meio de Graph Attention Networks. Essa mudança de foco permitiu explorar não apenas o desempenho preditivo do modelo, mas também sua capacidade de gerar hipóteses biológicas sobre os processos associados à evolução da doença.


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

Hodis E et al. (2012). A landscape of driver mutations in melanoma. Cell.

Tirosh I et al. (2016). Dissecting the multicellular ecosystem of metastatic melanoma by single-cell RNA-seq. Science.

Veličković P, Cucurull G, Casanova A, Romero A, Liò P, Bengio Y (2018). Graph Attention Networks. ICLR 2018.

Didona D et al. (2018). Non melanoma skin cancer pathogenesis overview. Biomedicines, 6(1), 6.

Kunz M et al. (2018). RNA-seq analysis of melanocytic nevi and melanomas (GSE112509). Gene Expression Omnibus / Lab Investigation.

Fey M, Lenssen JE (2019). Fast Graph Representation Learning with PyTorch Geometric. ICLR Workshop on Representation Learning on Graphs and Manifolds.

Han W, Hu C, Fan Z-J, Shen G-L (2021). Transcript levels of keratin 1/5/6/14/15/16/17 as potential prognostic indicators in melanoma patients. Scientific Reports, 11, 1023.

Ochoa D et al. (2021). Open Targets Platform: supporting systematic drug-target identification and prioritisation. Nucleic Acids Research.

Sung H et al. (2021). Global Cancer Statistics 2020: GLOBOCAN Estimates of Incidence and Mortality Worldwide for 36 Cancers in 185 Countries. CA: A Cancer Journal for Clinicians, 71(3), 209–249.

Brody S, Alon U, Yahav E (2022). How Attentive are Graph Attention Networks? ICLR 2022.

Szklarczyk D et al. (2023). The STRING database in 2023. Nucleic Acids Research.

Huang J et al. (2023). Global Incidence, Mortality, Risk Factors and Trends of Melanoma. American Journal of Clinical Dermatology, 24, 965–975.

Attal N et al. (2024). Advanced and Metastatic Non-Melanoma Skin Cancer: Epidemiology, Risk Factors, Clinical Features, and Treatment Options. Biomedicines, 12(7), 1448.

Castro LA, Martins MCA, Kock KS (2024). Longitudinal analysis of hospital morbidity and mortality due to skin cancer from 2012 to 2022. Anais Brasileiros de Dermatologia, 100(2), 283–292.

Colombino M et al. (2024). Heterogeneous pathogenesis of melanoma: BRAF mutations and beyond. Seminars in Cancer Biology.

Gieniusz E, Skrzydlewska E, Łuczaj W (2024). Current Insights into the Role of UV Radiation-Induced Oxidative Stress in Melanoma Pathogenesis. International Journal of Molecular Sciences, 25(21), 11651.

Murgas KA, Elkin R, Riaz N, Saucan E, Deasy JO, Tannenbaum AR (2024). Multi-scale geometric network analysis identifies melanoma immunotherapy response gene modules. Scientific Reports, 14, 6082.

Cirillo N (2025). Global epidemiological trends in the incidence and mortality for melanoma. Skin Health and Disease, 5(1), 84.

Instituto Nacional de Câncer (INCA) (2025). Estimativa 2026: Incidência de Câncer no Brasil. Rio de Janeiro: INCA.

Cancer Genome Atlas Network (2015). Genomic Classification of Cutaneous Melanoma. Cell.

Vivian J et al. (2017). Toil enables reproducible, open source, big biomedical data analyses. Nature Biotechnology.