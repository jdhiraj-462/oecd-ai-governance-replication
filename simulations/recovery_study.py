#!/usr/bin/env python3
"""Text-level recovery simulation (Online Appendix C, Study 3).

Builds five SYNTHETIC corpora (tests/make_fixture.py, different seeds) with ten
planted themes and runs the paper's BERTopic specification on each. Because the
OECD.AI data and the sentence-embedding model are not needed for this check, it
uses the offline TF-IDF/SVD embedding backend; it therefore tests the clustering,
outlier and c-TF-IDF stages, not the SBERT encoder.

Reports the adjusted Rand index between planted themes and recovered topics, and
whether the gap test recovers the planted direction of India's lowest-weighted theme.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tests"))
from govtopics import compare, load, topics  # noqa: E402
import make_fixture  # noqa: E402

OUT = ROOT / "simulations" / "results"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in range(5):
        tmp = OUT / f"_fixture_{seed}.csv"
        make_fixture.main(str(tmp), n=700, seed=seed)
        cfg = yaml.safe_load(open(ROOT / "config/config.yaml"))
        cfg["paths"]["raw"] = str(tmp); cfg["sample"]["english_only"] = False
        cfg["bertopic"]["hdbscan"]["min_cluster_size"] = 15
        df = load.build_corpus(cfg, [])
        df = df[~df["subnational"]].reset_index(drop=True)
        planted = df["title"].str.extract(r"SYNTHETIC (\w+) initiative")[0].values
        emb = topics.embed(df["text"].tolist(), cfg, "tfidf-svd")
        _, t, _ = topics.fit_bertopic(df["text"].tolist(), emb, cfg, 42)
        ari = adjusted_rand_score(planted, t)
        # planted regulation share is lowest for India by construction of make_fixture
        cfg["gaps"]["benchmarks"] = ["European Union"]; cfg["similarity"]["bootstrap_reps"] = 300
        g = compare.india_gaps(df["jurisdiction"].values, t, cfg)
        reg_topic = pd.Series(t[planted == "regulation"]).mode().iloc[0]
        rr = g[g.topic == reg_topic].iloc[0]
        rows.append({"seed": seed, "n_docs": len(df), "n_topics": len(set(t) - {-1}), "ari": ari,
                     "regulation_gap_vs_eu": rr["gap"], "regulation_gap_flagged": bool(rr["sig_fdr"] and rr["gap"] > 0)})
        tmp.unlink()
    res = pd.DataFrame(rows); res.to_csv(OUT / "recovery.csv", index=False)
    summ = {"n_corpora": len(res), "ari_mean": float(res.ari.mean()), "ari_min": float(res.ari.min()),
            "topics_recovered": res.n_topics.tolist(), "planted_gap_flagged": int(res.regulation_gap_flagged.sum())}
    (OUT / "recovery_summary.json").write_text(json.dumps(summ, indent=2))
    print(res.to_string(index=False)); print(summ)


if __name__ == "__main__":
    main()
