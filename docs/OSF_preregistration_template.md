# OSF pre-registration (fill and register BEFORE running country comparisons)

Title: Mapping Global AI Governance: A Topic-Modeling Analysis of National AI Policies Using the OECD.AI Policy Database

1. Data: OECD.AI Policy Navigator export downloaded on ____ ; SHA-256 ____ (from results/summary.json, run with --fast is fine for this).
2. Sample rules: as in config/config.yaml (attach the file; record its SHA-256 ____).
3. Topic model: BERTopic settings in config/config.yaml; five seeds; primary seed 42.
4. Topic labels and state roles: attach config/topic_labels.csv (SHA-256 ____), completed before any country comparison was inspected.
5. Expectations:
   - E1: India's share of control-role topics is lower than the pooled EU-27's. Contradicted if the 95% interval for (EU-27 share minus India share) lies at or below zero.
   - E2: India's authority-tool share and binding share (recorded status only) are lower than the pooled EU-27's. Contradicted if either is equal or higher.
6. Inference: smoothed cluster bootstrap (2,000 replicates, organisation clusters, Dirichlet(counts+0.5) per replicate); BH-FDR q = 0.10 within benchmark for theme gaps.
7. Proximity to the EU and the US is descriptive; no directional hypothesis.
8. Deviations will be reported in the paper.
