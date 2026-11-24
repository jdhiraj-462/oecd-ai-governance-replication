# Replication materials

**Mapping Global AI Governance: A Topic-Modeling Analysis of National AI Policies Using the OECD.AI Policy Database**

This repository contains the data collection script, the collected data, all analysis code,
model outputs, supplementary tables, figures, and the simulation study for the article.

## Data

Source: OECD.AI Policy Navigator, https://oecd.ai/en/dashboards/policy-initiatives
(accessed 23 November 2026). Please cite the data as OECD.AI asks:
OECD.AI (2025), OECD.AI Policy Navigator, accessed on 23/11/2026, https://oecd.ai/dashboards.

The Navigator has no bulk export. `scripts/collect_oecd.py` reads the 134 public list pages,
each of which embeds the full records of 20 initiatives as structured JSON, at one request
every 1.5 seconds (the site's robots.txt permits these pages). Staff names and e-mail
addresses present in the records are not kept.

| File | Content |
|---|---|
| `data/raw/oecd_ai_policy_navigator.csv` | 2,675 initiatives as collected |
| `data/raw/collection_log.json` | Date, counts, SHA-256 checksum of the CSV |
| `data/raw/international_orgs.txt` | International organisations listed by the Navigator (excluded) |
| `data/processed/corpus.csv` | 2,446 national initiatives analysed, with masked text, topic and instrument codes |
| `data/processed/corpus_all.csv` | Same, including subnational entries (sensitivity analysis) |

Raw HTML pages are not included because of their size; running the collector recreates them.

## Reproducing the results

Python 3.11 or later.

```
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python scripts/collect_oecd.py      # optional: re-collect the data (about 4 minutes)
python run_all.py                   # full analysis (about 30-60 minutes on a laptop CPU)
python build_manuscript.py          # tables, figure captions and online appendix (Word files in submission/)
```

The first run downloads the sentence encoder `sentence-transformers/all-mpnet-base-v2`
(about 420 MB). Every analytic setting (seeds, thresholds, bootstrap draws) is in
`config/config.yaml`; the configuration and the topic label file are hashed into
`results/summary.json` on every run.

Simulation study (results already in `simulations/results/`):

```
python simulations/power_study.py           # Studies 1-2: calibration and power
python simulations/power_study.py --roles   # Study 3: role-level power
python simulations/recovery_study.py        # Study 4: recovery on synthetic corpora
```

Tests: `pytest -q` (unit tests and a full run on a synthetic file; every synthetic row is
marked SYNTHETIC).

## Code

| File | Purpose |
|---|---|
| `scripts/collect_oecd.py` | Data collection from the public Navigator pages |
| `run_all.py` | Whole analysis; writes `results/` |
| `src/govtopics/load.py` | Column mapping, exclusions, language filter, place-name masking |
| `src/govtopics/mask.py` | Masking of place names, nationalities, languages and currencies |
| `src/govtopics/instruments.py` | Hood and Margetts tool coding (`config/instrument_type_map.csv`, `config/instrument_map.csv`) |
| `src/govtopics/topics.py` | BERTopic, seed stability, coherence, LDA robustness |
| `src/govtopics/compare.py` | Profiles, smoothed cluster bootstrap, permutation tests, clustering and stability, role shares, theme gaps with BH-FDR, instrument tests |
| `src/govtopics/coverage.py` | Coverage audit of 13 Indian AI instruments (`config/india_inventory.csv`) |
| `src/govtopics/figures.py` | Figures 1-4 (300 dpi) |
| `src/govtopics/validation.py`, `scripts/score_validation.py` | Human validation tasks and scoring (sheets in `results/validation/`) |
| `build_manuscript.py` | Builds the tables, figure captions and online appendix from `results/` |

## Outputs

| Article | File |
|---|---|
| Table 2 (topics, labels, roles) | `results/tables/Table2_topics.csv`, `config/topic_labels.csv` |
| Table 3 (India similarity) | `results/tables/Table3_india_similarity.csv` |
| Table 4 (theme differences) | `results/tables/Table4_india_gaps.csv` |
| Table 5 (instruments) | `results/tables/Table5_instrument_tests.csv` |
| Supplementary Tables S1-S18 | `results/tables/TableS*.csv` |
| Figures 1-4 | `results/figures/` |
| All reported numbers | `results/summary.json` |

## Licence

Code: MIT licence (see `LICENSE`). Data: collected from OECD.AI and subject to the OECD's
terms of use; cite OECD.AI as above when reusing it.
