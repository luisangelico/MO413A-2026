# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an academic research project for the graduate course **MO413 — Ciência e Visualização de Dados em Saúde** (Science and Data Visualization in Health) at Unicamp (2026.1).

**Research goal**: Compare gene interaction networks derived from melanoma, carcinoma, and healthy tissue samples, analyzing differences in network topology, central genes, and biological modules.

## Repository Structure

```
README.md          # Team info and project summary
project1/
  README.md        # Full project specification (methodology, datasets, tools)
  assets/
    images/        # Figures and diagrams (e.g., logical graph model)
    slides/        # Presentation slides
```

## Research Pipeline

The project follows a 3-phase bioinformatics workflow:

1. **Dataset retrieval** — Download gene expression datasets from GEO (GSE4570, GSE2503, GSE8401, GSE7553, GSE45213)
2. **Differential expression analysis** — Compare: Healthy vs Melanoma, Healthy vs Carcinoma, Melanoma vs Carcinoma
3. **Network analysis** via STRING → Cytoscape:
   - Protein-protein interactions with confidence scores
   - Centrality metrics: Eigenvector (CytoNCA), Degree, Betweenness, Closeness
   - Clustering coefficient (NetworkAnalyzer)
   - Module detection (MCODE / clusterMaker2)
   - Cross-network comparison (gained/lost genes, exclusive hub genes)

## Tools Used

- **GEO** (Gene Expression Omnibus) — data source
- **Excel / Python** — differential expression analysis
- **STRING** — protein interaction database
- **Cytoscape** + plugins: CytoNCA, NetworkAnalyzer, MCODE, clusterMaker2

## Key Links

- Project slides: https://docs.google.com/presentation/d/1Wa21J_w7CIT62xtVCvaYQdo1LmQk3-bqbujdAck-rXY/edit?usp=sharing
- Graph model template: https://docs.google.com/presentation/d/10RN7bDKUka_Ro2_41WyEE76Wxm4AioiJOrsh6BRY3Kk/edit?usp=sharing
- Course GitHub: https://github.com/datasci4health

## Team

| Name | RA | Focus |
|---|---|---|
| Alan Freitas Ribeiro | 193400 | Biology |
| Augusto José Peterlevitz | 209783 | Computer Science |
| Felipe Kennedy Carvalho Torquato | 174157 | Computer Science |
| Luis Henrique Angélico | 248891 | Computer Science |
| Naruan Francisco Ferraz e Ferraz | 123456 | TBD |
| Paulo Costa | 123456 | TBD |
