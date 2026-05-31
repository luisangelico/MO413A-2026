"""Generate the 'What is a GAT?' explainer page.

A non-technical introduction to Graph Attention Networks, what makes them
suitable for this project, and how each step of the model maps onto a
familiar idea (a sample = a graph; attention = a learned vote on which
neighbors matter; embedding = the model's fingerprint of a sample).

Run:
    python -m src.about_gat_page                  # writes to <repo>/site
    python -m src.about_gat_page --site-dir docs
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.i18n import PAGE_TITLE, lang_subdir
from src.site_header import HEADER_CSS, render_header


CSS = """
  body { font-family:-apple-system,system-ui,sans-serif; margin:0;
         color:#222; line-height:1.65; }
  .page { max-width:860px; margin:0 auto; padding:0 1em 2em; }
  h1 { font-size:1.5em; margin:0.4em 0 0.2em; }
  h2 { font-size:1.15em; margin-top:1.8em; }
  h3 { font-size:1em; margin-top:1.2em; color:#333; }
  .meta { color:#666; font-size:0.9em; }
  .plain { background:#f6fbf7; border-left:3px solid #2ca02c;
           padding:0.7em 1em; margin:1em 0; }
  .callout { background:#fffbf2; border-left:3px solid #e0a020;
             padding:0.7em 1em; margin:1em 0; }
  .grid2 { display:grid; grid-template-columns:1fr 1fr; gap:1em; margin:0.8em 0; }
  @media (max-width:760px) { .grid2 { grid-template-columns:1fr; } }
  .card { background:#f6f8fc; border:1px solid #d8def0; border-radius:8px;
          padding:0.9em 1em; }
  .card h3 { margin:0 0 0.3em 0; font-size:1em; }
  code { background:#f4f4f4; padding:0 0.3em; border-radius:3px; }
  ul, ol { line-height:1.85; }
  .formula { background:#f7f7f9; border-radius:6px; padding:0.7em 1em;
             font-family: ui-monospace, Menlo, monospace; font-size:0.9em;
             margin:0.6em 0; overflow-x:auto; }
  .next a { display:inline-block; padding:0.4em 0.8em; margin:0.3em 0.4em 0 0;
            background:#1f77b4; color:white; border-radius:5px;
            text-decoration:none; font-size:0.9em; }
"""


BODY_EN = """
<h1>What is a Graph Attention Network?</h1>
<p class="meta">A plain-language tour of the architecture this project uses,
and why graphs (not flat lists of numbers) are the right shape for the
problem.</p>

<div class="plain">
  <strong>In one sentence.</strong> A Graph Attention Network (GAT) is a neural
  network that reads a <em>graph</em> — nodes connected by edges — and learns
  not just what each node looks like on its own, but how much each node
  should listen to each of its neighbors when forming an opinion.
</div>

<h2>Why graphs?</h2>
<p>
  Most machine-learning models eat a flat row of numbers: 1000 gene
  expression values per sample, say. That works, but it throws away
  something important — biologists know that <strong>genes don't act in
  isolation</strong>. They form pathways, complexes, regulatory cascades.
  Two genes that physically bind each other behave very differently from
  two genes that have nothing to do with one another, even if their
  individual expression values look identical.
</p>
<p>
  A graph captures exactly that structure. In this project:
</p>
<ul>
  <li>Each <strong>node</strong> is a gene, carrying its measured expression
    as a feature.</li>
  <li>Each <strong>edge</strong> is a known protein–protein interaction
    (PPI) from STRING — two proteins that physically bind.</li>
  <li>Each <strong>sample</strong> (one patient) becomes one full graph:
    same edges as everyone else, but the per-node expression values
    differ patient to patient.</li>
</ul>
<p>
  The GAT's job: look at this whole graph and decide whether the patient
  is normal skin, primary melanoma, or metastatic melanoma.
</p>

<h2>What does "attention" mean here?</h2>
<p>
  Imagine each gene is a person at a meeting. Their PPI partners are the
  other people they're allowed to talk to. Now ask each person to summarize
  what's going on. They'll listen to their neighbors — but not equally.
  Some neighbors are highly relevant; others, less so. <strong>Attention</strong>
  is a learned weighting that says, for every connection, "how much should
  this person rely on what that person says?"
</p>
<p>
  Mathematically, for a node <code>j</code> updating itself based on its
  neighbors <code>i</code>, attention assigns each incoming edge a weight
  α(i→j) ∈ [0, 1], with all weights summing to 1 across <code>j</code>'s
  neighbors. The new representation of <code>j</code> is a weighted sum
  of its neighbors' messages.
</p>
<div class="formula">
  h<sub>j</sub><sup>new</sup> = σ( Σ<sub>i ∈ neighbors(j)</sub> α(i→j) · W · h<sub>i</sub> )
</div>
<p>
  The crucial part: α is <strong>learned</strong>. The model figures out,
  from training data, which gene→gene connections matter for the task.
  That's the secret sauce, and it's also what makes attention
  <em>interpretable</em>: after training, we can read the α values and ask
  "which connections did the model rely on most?" That's the entire
  premise of the
  <a href="interpret/index.html">interpretability</a> page.
</p>

<h2>How GATv2 differs from "vanilla" GAT</h2>
<p>
  This project uses <strong>GATv2</strong>. The original GAT had a subtle
  flaw: in some configurations its attention couldn't depend on the query
  node, only on the source nodes. GATv2 reorders the operations so
  attention is genuinely <em>dynamic</em> — every node–neighbor pair gets
  its own context-dependent weight. In practice GATv2 trains more
  reliably and performs at least as well on every standard benchmark.
</p>

<h2>The architecture used here</h2>
<p>
  Three GAT layers stacked, with attention heads in parallel:
</p>
<ol>
  <li><strong>Layer 1:</strong> 1 input feature (expression) →
    128 hidden, with 4 attention heads. Each head looks at the same
    graph independently, producing its own view; their outputs are
    concatenated.</li>
  <li><strong>Layer 2:</strong> hidden → hidden, again 4 heads.
    Information now propagates two hops out — a gene "sees"
    its neighbors and its neighbors' neighbors.</li>
  <li><strong>Layer 3:</strong> hidden → hidden, 1 head, three hops out.
    Most of the long-range signal has now mixed in.</li>
  <li><strong>Pooling:</strong> the 1000 per-gene vectors are averaged
    into one 64-number summary of the whole sample — its
    <a href="embeddings.html">embedding</a>.</li>
  <li><strong>Classifier:</strong> a small linear layer turns the
    embedding into 3 probabilities (Primary / Metastasis / Normal).</li>
</ol>

<h2>What the model produces along the way</h2>
<div class="grid2">
  <div class="card">
    <h3>Edge attention</h3>
    <p>
      One number per edge per layer per head. We aggregate these into
      per-gene importance scores and study which genes the model leaned
      on. See the
      <a href="interpret/index.html">interpretability</a> and
      <a href="biomarkers.html">biomarkers</a> pages.
    </p>
  </div>
  <div class="card">
    <h3>Embeddings</h3>
    <p>
      The 64-dimensional sample fingerprints. Geometry in this space —
      clusters, distances, neighborhoods — is the model's view of how
      patients relate. See the
      <a href="embeddings.html">embeddings</a> page.
    </p>
  </div>
  <div class="card">
    <h3>Pseudotime</h3>
    <p>
      An ordering of samples on a Normal → Metastasis axis derived from
      embedding geometry, used to track which genes the model recruits
      as samples become more tumor-like.
      See <a href="pseudotime.html">pseudotime</a>.
    </p>
  </div>
  <div class="card">
    <h3>Predictions and confidence</h3>
    <p>
      The class probabilities themselves, with uncertainty calibration
      visible per sample.
      See <a href="index.html">predictions</a> and
      <a href="overview.html">run overview</a>.
    </p>
  </div>
</div>

<h2>Why a GAT is well-suited to this problem</h2>
<ul>
  <li><strong>The graph structure is real biology.</strong> Treating gene
    expression as a flat 1000-dimensional vector ignores the fact that
    biology operates through interacting protein networks. A GAT can use
    that structure directly.</li>
  <li><strong>It generalizes across patients.</strong> The same edge set
    is used for every patient (the human PPI network doesn't change
    sample to sample). What changes is each gene's expression, the node
    feature. The model learns rules about <em>how expression patterns
    propagate through the network</em>, not patient-specific facts.</li>
  <li><strong>It's interpretable.</strong> Attention weights are not
    abstract internal states — they're per-edge numbers we can read,
    aggregate, and validate against known biology.</li>
  <li><strong>Smaller than the alternative.</strong> A GAT with 128
    hidden channels has far fewer parameters than a fully-connected
    network operating on 1000 genes — the graph constrains how
    information flows, which is a useful inductive bias when training
    data are limited.</li>
</ul>

<div class="callout">
  <strong>What a GAT is <em>not</em>.</strong> It's not magic. It can't
  create signal that isn't in the data. If the input expression doesn't
  distinguish two classes, no graph structure will save it. And
  attention is a tool, not a guarantee — high attention doesn't mean
  "biologically important," only "useful to this model on this data."
  That's why every interpretability claim in this project is benchmarked
  against a node-degree baseline and validated across 5 random seeds
  (see the <a href="stability.html">stability</a> page).
</div>

<h2>Where to go next</h2>
<p class="next">
  <a href="dataset.html">Dataset</a>
  <a href="index.html">Predictions</a>
  <a href="interpret/index.html">Interpretability</a>
  <a href="embeddings.html">Embeddings</a>
  <a href="pseudotime.html">Pseudotime</a>
  <a href="stability.html">Stability</a>
</p>
"""


BODY_PT = """
<h1>O que é uma Graph Attention Network?</h1>
<p class="meta">Um passeio em linguagem simples pela arquitetura usada
neste projeto, e por que grafos (e não listas planas de números) são a
forma certa para este problema.</p>

<div class="plain">
  <strong>Em uma frase.</strong> Uma Graph Attention Network (GAT) é uma
  rede neural que lê um <em>grafo</em> — nós conectados por arestas — e
  aprende não apenas como cada nó parece sozinho, mas também o quanto
  cada nó deve escutar cada um de seus vizinhos ao formar uma opinião.
</div>

<h2>Por que grafos?</h2>
<p>
  A maioria dos modelos de aprendizado de máquina recebe uma linha plana
  de números: 1000 valores de expressão gênica por amostra, por exemplo.
  Isso funciona, mas joga fora algo importante — biólogos sabem que
  <strong>genes não atuam isoladamente</strong>. Eles formam vias,
  complexos, cascatas regulatórias. Dois genes que se ligam fisicamente
  comportam-se de modo muito diferente de dois genes sem nenhuma relação
  entre si, mesmo que seus valores individuais de expressão sejam
  idênticos.
</p>
<p>
  Um grafo captura exatamente essa estrutura. Neste projeto:
</p>
<ul>
  <li>Cada <strong>nó</strong> é um gene, com sua expressão medida como
    atributo.</li>
  <li>Cada <strong>aresta</strong> é uma interação proteína–proteína
    (PPI) conhecida do STRING — duas proteínas que se ligam fisicamente.</li>
  <li>Cada <strong>amostra</strong> (um paciente) torna-se um grafo
    completo: as mesmas arestas que todos os outros, mas valores de
    expressão por nó que diferem paciente a paciente.</li>
</ul>
<p>
  A tarefa da GAT: olhar para esse grafo inteiro e decidir se o paciente
  é pele saudável, melanoma primário ou metastático.
</p>

<h2>O que significa "atenção" aqui?</h2>
<p>
  Imagine que cada gene é uma pessoa em uma reunião. Seus parceiros PPI
  são as outras pessoas com quem ele pode conversar. Agora peça a cada
  pessoa que resuma o que está acontecendo. Ela vai escutar seus
  vizinhos — mas não igualmente. Alguns vizinhos são muito relevantes;
  outros, menos. <strong>Atenção</strong> é uma ponderação aprendida
  que diz, para cada conexão, "o quanto esta pessoa deve confiar no que
  aquela diz?"
</p>
<p>
  Matematicamente, para um nó <code>j</code> atualizando-se com base em
  seus vizinhos <code>i</code>, a atenção atribui a cada aresta
  recebida um peso α(i→j) ∈ [0, 1], com a soma dos pesos sobre os
  vizinhos de <code>j</code> igual a 1. A nova representação de
  <code>j</code> é uma soma ponderada das mensagens de seus vizinhos.
</p>
<div class="formula">
  h<sub>j</sub><sup>nova</sup> = σ( Σ<sub>i ∈ vizinhos(j)</sub> α(i→j) · W · h<sub>i</sub> )
</div>
<p>
  O ponto crucial: α é <strong>aprendido</strong>. O modelo descobre,
  a partir dos dados de treino, quais conexões gene→gene importam para
  a tarefa. Esse é o segredo, e é também o que torna a atenção
  <em>interpretável</em>: depois do treinamento, podemos ler os
  valores de α e perguntar "em quais conexões o modelo mais se
  apoiou?". Esse é o tema central da página de
  <a href="interpret/index.html">interpretabilidade</a>.
</p>

<h2>Como o GATv2 difere do GAT "original"</h2>
<p>
  Este projeto usa <strong>GATv2</strong>. O GAT original tinha uma
  falha sutil: em algumas configurações, sua atenção não conseguia
  depender do nó-consulta, apenas dos nós-fonte. O GATv2 reordena as
  operações de modo que a atenção é genuinamente <em>dinâmica</em> —
  cada par nó–vizinho recebe seu próprio peso dependente de contexto.
  Na prática, GATv2 treina de maneira mais confiável e tem desempenho
  ao menos tão bom quanto o GAT em todos os benchmarks padrão.
</p>

<h2>A arquitetura usada aqui</h2>
<p>
  Três camadas GAT empilhadas, com cabeças de atenção em paralelo:
</p>
<ol>
  <li><strong>Camada 1:</strong> 1 atributo de entrada (expressão) →
    128 ocultas, com 4 cabeças de atenção. Cada cabeça olha para o mesmo
    grafo independentemente, produzindo sua própria visão; suas saídas
    são concatenadas.</li>
  <li><strong>Camada 2:</strong> oculta → oculta, novamente 4 cabeças.
    A informação agora se propaga dois saltos — um gene "vê" seus
    vizinhos e os vizinhos dos seus vizinhos.</li>
  <li><strong>Camada 3:</strong> oculta → oculta, 1 cabeça, três
    saltos. A maior parte do sinal de longo alcance já se misturou.</li>
  <li><strong>Pooling:</strong> os 1000 vetores por gene são
    promediados em um único resumo de 64 números da amostra inteira —
    seu <a href="embeddings.html">embedding</a>.</li>
  <li><strong>Classificador:</strong> uma pequena camada linear
    converte o embedding em 3 probabilidades
    (Primário / Metástase / Normal).</li>
</ol>

<h2>O que o modelo produz pelo caminho</h2>
<div class="grid2">
  <div class="card">
    <h3>Atenção por aresta</h3>
    <p>
      Um número por aresta por camada por cabeça. Agregamos isso em
      pontuações de importância por gene e estudamos em quais genes o
      modelo se apoiou. Veja
      <a href="interpret/index.html">interpretabilidade</a> e
      <a href="biomarkers.html">biomarcadores</a>.
    </p>
  </div>
  <div class="card">
    <h3>Embeddings</h3>
    <p>
      As impressões digitais de 64 dimensões das amostras. A geometria
      desse espaço — clusters, distâncias, vizinhanças — é a visão do
      modelo sobre como os pacientes se relacionam. Veja
      <a href="embeddings.html">embeddings</a>.
    </p>
  </div>
  <div class="card">
    <h3>Pseudotempo</h3>
    <p>
      Uma ordenação das amostras em um eixo Normal → Metástase derivada
      da geometria do embedding, usada para acompanhar quais genes o
      modelo recruta à medida que as amostras se tornam mais "tumorais".
      Veja <a href="pseudotime.html">pseudotempo</a>.
    </p>
  </div>
  <div class="card">
    <h3>Predições e confiança</h3>
    <p>
      As probabilidades de classe propriamente ditas, com a calibração
      de incerteza visível por amostra. Veja
      <a href="index.html">predições</a> e
      <a href="overview.html">visão geral</a>.
    </p>
  </div>
</div>

<h2>Por que uma GAT é adequada a este problema</h2>
<ul>
  <li><strong>A estrutura do grafo é biologia real.</strong> Tratar
    expressão gênica como um vetor plano de 1000 dimensões ignora o
    fato de que a biologia opera por redes proteicas em interação. Uma
    GAT pode usar essa estrutura diretamente.</li>
  <li><strong>Generaliza entre pacientes.</strong> O mesmo conjunto de
    arestas é usado para todos os pacientes (a rede PPI humana não muda
    de amostra para amostra). O que muda é a expressão de cada gene, o
    atributo do nó. O modelo aprende regras sobre <em>como padrões de
    expressão se propagam pela rede</em>, não fatos paciente-específicos.</li>
  <li><strong>É interpretável.</strong> Os pesos de atenção não são
    estados internos abstratos — são números por aresta que podemos ler,
    agregar e validar contra biologia conhecida.</li>
  <li><strong>Menor que a alternativa.</strong> Uma GAT com 128 canais
    ocultos tem muito menos parâmetros que uma rede totalmente conectada
    operando em 1000 genes — o grafo restringe como a informação flui,
    o que é um inductive bias útil quando os dados de treino são
    limitados.</li>
</ul>

<div class="callout">
  <strong>O que uma GAT <em>não</em> é.</strong> Não é mágica. Não pode
  criar sinal que não está nos dados. Se a expressão de entrada não
  distingue duas classes, nenhuma estrutura de grafo vai resolver. E
  atenção é uma ferramenta, não uma garantia — alta atenção não
  significa "biologicamente importante", apenas "útil para este modelo
  nestes dados". É por isso que toda alegação de interpretabilidade
  neste projeto é comparada com um baseline de grau de nó e validada em
  5 sementes aleatórias (veja a página de
  <a href="stability.html">estabilidade</a>).
</div>

<h2>Para onde ir agora</h2>
<p class="next">
  <a href="dataset.html">Dados</a>
  <a href="index.html">Predições</a>
  <a href="interpret/index.html">Interpretabilidade</a>
  <a href="embeddings.html">Embeddings</a>
  <a href="pseudotime.html">Pseudotempo</a>
  <a href="stability.html">Estabilidade</a>
</p>
"""


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>__TITLE__</title>
<style>
__HEADER_CSS__
__CSS__
</style></head><body>
__HEADER__
<div class="page">
__BODY__
</div>
</body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site-dir", type=str, default=None,
                    help="Default: <repo>/site")
    ap.add_argument("--lang", choices=["en", "pt-br"], default="en")
    args = ap.parse_args()

    site_dir = Path(args.site_dir) if args.site_dir else (
        Path(__file__).resolve().parent.parent / "site")
    out_dir = site_dir / lang_subdir(args.lang) if args.lang != "en" else site_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "about-gat.html"

    body = BODY_EN if args.lang == "en" else BODY_PT
    prefix = "" if args.lang == "en" else "../"
    header = render_header("about-gat", lang=args.lang, prefix=prefix)
    title = PAGE_TITLE[args.lang]["about-gat"]

    html = (PAGE
            .replace("__TITLE__", title)
            .replace("__HEADER_CSS__", HEADER_CSS)
            .replace("__CSS__", CSS)
            .replace("__HEADER__", header)
            .replace("__BODY__", body))
    out_path.write_text(html)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
