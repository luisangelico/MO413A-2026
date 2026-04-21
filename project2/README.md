
# Projeto `Câncer de Pele e seus Tipos: uma Análise do Perfil de Expressão Gênica em Redes.`
# Project `Skin Cancer and its Types: An Analysis of Gene Expression Profiles in Networks.`

# Descrição Resumida do Projeto

> Este projeto visa construir e analisar uma rede de interação proteína-proteína derivada de genes diferencialmente expressos em melanoma metastático. A partir de dados de expressão gênica (GSE7553) e de pontuações de relevância oncológica obtidas da plataforma Open Targets, os nós da rede são enriquecidos com atributos clínicos e biológicos. A análise de topologia e centralidade da rede busca identificar genes-chave e potenciais alvos terapêuticos no contexto do melanoma.


# Slides

[Slides P2](assets/MO413A_P2.pdf)

# Fundamentação Teórica

> O melanoma é o tipo de câncer de pele com maior mortalidade, caracterizado por alta heterogeneidade molecular e resistência a tratamentos convencionais. A análise de redes de interação proteína-proteína (PPI) permite identificar genes centrais e módulos biológicos relevantes para a progressão tumoral. A integração com dados de alvos terapêuticos (Open Targets) oferece uma abordagem translacional para priorizar candidatos a biomarcadores e alvos de intervenção clínica.

# Perguntas de Pesquisa

> Quais genes diferencialmente expressos no melanoma metastático são centrais na rede PPI?

> Quais nós centrais da rede apresentam maior relevância oncológica segundo a plataforma Open Targets?

> Existem módulos funcionais enriquecidos com genes de alta pontuação clínica?

> Quais interações proteicas são mais relevantes para a progressão do melanoma metastático?

> É possível identificar alvos terapêuticos prioritários combinando centralidade de rede e score clínico do Open Targets?


# Bases de Dados

> Elencar bases de dados candidatas a serem utilizadas no projeto na forma de tabela:

> Base de Dados | Endereço na Web | Resumo descritivo
> ----- | ----- | -----
> Gene Expression Patterns Involved in the Malignant Transformation and Progression of Metastatic Melanoma | [GSE7553](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE7553) | Este conjunto de dados contém perfis de expressão gênica de tumores cutâneos, incluindo melanoma primário e metastático. Utilizado para obter genes diferencialmente expressos (LogFC e p-valor) que alimentam a construção da rede PPI.
> Open Targets Platform — Melanoma (EFO_0000756) | [Open Targets](https://platform.opentargets.org/disease/EFO_0000756/associations) | Plataforma que integra evidências genômicas, clínicas e funcionais para associar genes/proteínas a doenças. Utilizada para obter o `globalScore` (oncogeneScore) e indicadores de precedência clínica dos genes presentes na rede.
> STRING — Protein-Protein Interaction Networks | [STRING](https://string-db.org/) | Base de dados de interações proteína-proteína com scores de confiança baseados em evidências experimentais, anotações de bancos de dados e mineração de texto. Utilizada para construir a rede de interações entre os genes diferencialmente expressos.

# Modelo Lógico

> Modelo lógico da base de grafos que será construída. Para o modelo de grafos de propriedades, utilize este
> [modelo de base](https://docs.google.com/presentation/d/10RN7bDKUka_Ro2_41WyEE76Wxm4AioiJOrsh6BRY3Kk/edit?usp=sharing) para construir o seu.
> Coloque a imagem do PNG do seu modelo lógico como ilustrado abaixo (a imagem estará na pasta `image`):
>
> ![Modelo Lógico de Grafos](assets/images/modelo_dados.png)

# Metodologia

1. Obtenção e processamento do dataset GSE7553 (Gene Expression Omnibus):
  - Download do arquivo `.soft.gz` com perfis de expressão gênica de melanoma
  - Processamento no Orange (workflow `Melanoma.ows`) para normalização e análise de expressão diferencial
  - Extração de genes com LogFC e p-valor significativos

2. Enriquecimento com dados de alvos terapêuticos (Open Targets):
  - Download dos alvos associados ao melanoma (EFO_0000756) via Open Targets Platform
  - Integração do `globalScore` (oncogeneScore) e indicadores de precedência clínica aos genes diferencialmente expressos

3. Construção da rede PPI via STRING:
  - Mapeamento dos genes para identificadores STRING (`nodes_to_string.csv`)
  - Consulta à API/exportação do STRING para obter interações com scores detalhados (experimental, banco de dados, text mining, combined score)

4. Análise de rede no Cytoscape:
  - Importação de nós (`Nodes_cytoscape.csv`) com atributos: LogFC, p-valor, oncogeneScore
  - Importação de arestas (`Cytoscape_edges.csv`) com combined score do STRING
  - Cálculo de métricas de centralidade: degree, betweenness centrality, closeness centrality, eigenvector centrality (CytoNCA)
  - Identificação de genes centrais com alto oncogeneScore
  - Clusterização da rede (MCODE / clusterMaker2)
  - Visualização e interpretação dos módulos funcionais


# Ferramentas

> Ferramentas a serem utilizadas (com base na visão atual do grupo sobre o projeto).

- Gene Expression Omnibus (GEO)
- Open Targets Platform
- Orange (Data Mining)
- STRING
- Cytoscape
- CytoNCA
- NetworkAnalyzer
- MCODE
- clusterMaker2

# Referências Bibliográficas

> Lista de artigos, links e referências bibliográficas.
>
> Fiquem à vontade para escolher o padrão de referenciamento preferido pelo grupo.
