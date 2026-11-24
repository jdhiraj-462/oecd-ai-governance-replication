"""End-to-end test on SYNTHETIC data. Run from the replication folder:  pytest -q"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from govtopics import compare, validation  # noqa: E402


def test_alpha_perfect_and_chance():
    assert validation.krippendorff_alpha_nominal(list("aabbc"), list("aabbc")) == 1.0
    a = validation.krippendorff_alpha_nominal(list("abab"), list("baba"))
    assert a < 0


def test_bh_fdr_monotone():
    sig, adj = compare.bh_fdr(np.array([0.001, 0.02, 0.04, 0.5]), 0.1)
    assert sig[:3].all() and not sig[3]


def test_end_to_end(tmp_path):
    fx = tmp_path / "fixture.csv"
    subprocess.run([sys.executable, str(ROOT / "tests/make_fixture.py"), str(fx)], check=True, cwd=ROOT)
    cfg = yaml.safe_load(open(ROOT / "config/config.yaml"))
    cfg["paths"] = {"raw": str(fx), "processed": str(tmp_path / "processed"), "results": str(tmp_path / "results"),
                    "topic_labels": str(tmp_path / "topic_labels.csv")}
    cfg["sample"]["english_only"] = False
    cfg["bertopic"]["seeds"] = [42, 57]
    cfg["bertopic"]["hdbscan"]["min_cluster_size"] = 15
    cfg["bertopic"]["seeds"] = [42, 57]
    cp = tmp_path / "cfg.yaml"; cp.write_text(yaml.safe_dump(cfg))
    subprocess.run([sys.executable, "run_all.py", "--config", str(cp), "--embedding-backend", "tfidf-svd", "--fast"],
                   check=True, cwd=ROOT)
    s = json.loads((tmp_path / "results/summary.json").read_text())
    assert s["corpus"]["n_docs"] > 500
    assert s["bertopic"]["n_topics"] >= 3
    assert {p["comparator"] for p in s["india_pairs"]} >= {"European Union", "United States", "EU27_POOLED", "NON_OECD_POOLED"}
    assert s["bootstrap"]["type"].startswith("cluster")
    assert s["corpus"]["n_subnational_flagged"] > 0
    assert any(r["variant"] == "subnational initiatives included" for r in s["sensitivity"]["rows"])
    assert "coverage" in s and s["coverage"]["n_inventory"] == 13
    assert {r["role"] for r in s["role_shares"]} == {"development", "control", "promotion"}
    for f in ("Figure1_similarity_heatmap.tif", "Figure2_dendrogram.jpg", "Figure3_india_gaps.tif"):
        assert (tmp_path / "results/figures" / f).exists()
