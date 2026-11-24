#!/usr/bin/env python3
"""Monte Carlo study of the gap test used in the paper (Online Appendix C).

Study 1 (power and false discovery). A benchmark topic profile is drawn from a
flat Dirichlet over K = 15 topics. India's profile equals the benchmark except on
three topics, where India's share is lower by `delta` (mass redistributed to the
other topics). Initiatives are drawn independently. The gap test of
compare.india_gaps (bootstrap CI and p-value, Benjamini-Hochberg at q = 0.10) is
applied. (An earlier version of the test used a plain percentile bootstrap; its
results are kept in results/naive_*.csv because they show why the smoothed bootstrap
was adopted: with 25 Indian initiatives against 150 benchmark initiatives it flagged
at least one false gap in 87 percent of null replications.) Reported: power (share of true gaps flagged), false discovery proportion,
and the probability of at least one false flag. delta = 0 is the global null.

Study 2 (interval coverage under clustering). Initiatives come in programmes of
four that share a topic tendency (Dirichlet around the unit's profile with total
concentration 5, i.e. strong within-programme similarity). We compare 95 percent interval coverage of the true gap for the
initiative-level bootstrap and the cluster bootstrap used in the paper.

    python simulations/power_study.py            # full study (about 10-20 min on 4 cores)
    python simulations/power_study.py --quick    # smoke test
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from govtopics import compare  # noqa: E402

K, N_GAP = 15, 3
OUT = ROOT / "simulations" / "results"


def cfg_for(B: int) -> dict:
    return {"sample": {"india_label": "India", "eu_label": "Bench", "us_label": "US"},
            "groups": {"oecd_members": [], "eu27_members": []},
            "gaps": {"benchmarks": ["Bench"], "min_benchmark_share": 0.02, "fdr_q": 0.10},
            "similarity": {"bootstrap_reps": B, "ci": 0.95, "smoothing": 0.5}}


def profiles(rng, delta):
    pb = rng.dirichlet(np.ones(K))
    cand = np.where(pb >= max(0.05, delta + 0.01))[0]
    gap_t = rng.choice(cand, min(N_GAP, len(cand)), replace=False) if delta > 0 and len(cand) else np.array([], int)
    pi = pb.copy()
    removed = 0.0
    for g in gap_t:
        new = pb[g] - delta; removed += pb[g] - new; pi[g] = new
    rest = np.setdiff1d(np.arange(K), gap_t)
    pi[rest] += removed * pb[rest] / pb[rest].sum()
    return pb, pi, set(gap_t.tolist())


def one_power(args):
    n_in, n_b, delta, rep, B = args
    rng = np.random.default_rng(10_000 * rep + int(1000 * delta) + n_in + 7 * n_b)
    pb, pi, true = profiles(rng, delta)
    t = np.concatenate([rng.choice(K, n_in, p=pi), rng.choice(K, n_b, p=pb)])
    j = np.array(["India"] * n_in + ["Bench"] * n_b)
    res = compare.india_gaps(j, t, cfg_for(B), seed=rep)
    flagged = set(res.loc[res.sig_fdr & (res.gap > 0), "topic"].astype(int))
    tp = len(flagged & true); fp = len(flagged - true)
    return {"n_india": n_in, "n_bench": n_b, "delta": delta, "rep": rep,
            "power": tp / len(true) if true else np.nan, "fdp": fp / len(flagged) if flagged else 0.0,
            "any_false": float(fp > 0)}


def draw_clustered(rng, p, n, size=4, conc=5.0):
    out, grp, g = [], [], 0
    while len(out) < n:
        q = rng.dirichlet(conc * p + 1e-3)  # programme-level tendency around p (total concentration = conc)
        m = min(size, n - len(out))
        out += list(rng.choice(K, m, p=q)); grp += [g] * m; g += 1
    return np.array(out), np.array(grp)


def one_coverage(args):
    rep, B, n_in, n_b = args
    rng = np.random.default_rng(99_000 + rep)
    pb, pi, true = profiles(rng, 0.08)
    ti, gi = draw_clustered(rng, pi, n_in); tb, gb = draw_clustered(rng, pb, n_b)
    t = np.concatenate([ti, tb]); j = np.array(["India"] * n_in + ["Bench"] * n_b)
    groups = np.concatenate([["I%d" % x for x in gi], ["B%d" % x for x in gb]])
    true_gap = pb - pi
    out = {"rep": rep}
    for name, g in (("initiative", None), ("cluster", groups)):
        res = compare.india_gaps(j, t, cfg_for(B), groups=g, seed=rep)
        res = res.set_index("topic")
        ids = sorted(set(t))
        cov = [(res.loc[k, "lo"] <= true_gap[k] <= res.loc[k, "hi"]) for k in ids if k in res.index]
        width = [(res.loc[k, "hi"] - res.loc[k, "lo"]) for k in ids if k in res.index]
        out[f"coverage_{name}"] = float(np.mean(cov)); out[f"width_{name}"] = float(np.mean(width))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--roles", action="store_true", help="run only the role-level study (Study 4)")
    a = ap.parse_args()
    if a.roles:
        return
    reps, B = (20, 200) if a.quick else (100, 400)
    OUT.mkdir(parents=True, exist_ok=True)
    grid = [(n_in, n_b, d, r, B) for n_in in (25, 50, 100) for n_b in (50, 150)
            for d in (0.0, 0.03, 0.05, 0.08, 0.12) for r in range(reps)]
    with Pool(a.procs) as pool:
        p = pd.DataFrame(pool.map(one_power, grid, chunksize=10))
        c = pd.DataFrame(pool.map(one_coverage, [(r, B, 60, 120) for r in range(reps)], chunksize=5))
    p.to_csv(OUT / "power_raw.csv", index=False); c.to_csv(OUT / "coverage_raw.csv", index=False)
    tab = p.groupby(["n_india", "n_bench", "delta"]).agg(power=("power", "mean"), fdr=("fdp", "mean"),
                                                         p_any_false=("any_false", "mean")).reset_index()
    tab.to_csv(OUT / "power_summary.csv", index=False)
    cov = c[["coverage_initiative", "coverage_cluster", "width_initiative", "width_cluster"]].mean().to_dict()
    summ = {"reps_per_cell": reps, "bootstrap_reps": B, "K": K, "n_true_gaps": N_GAP,
            "power": tab.to_dict(orient="records"), "coverage": cov,
            "null_fwer_max": float(tab[tab.delta == 0]["p_any_false"].max()),
            "fdr_max_nonnull": float(tab[tab.delta > 0]["fdr"].max())}
    (OUT / "power_summary.json").write_text(json.dumps(summ, indent=2))
    print(tab.to_string(index=False)); print(cov)




def one_role_power(args):
    """Study 4: the same gaps, tested at the level of state roles (15 topics -> 3 roles of 5).
    All three true gaps fall in the 'control' role, so the role-level gap is about 3 * delta."""
    n_in, n_b, delta, rep, B = args
    rng = np.random.default_rng(55_000 + rep + int(1000 * delta) + n_in)
    pb = rng.dirichlet(np.ones(K))
    ctrl = np.arange(5, 10)
    cand = ctrl[pb[ctrl] >= delta + 0.01]
    gap_t = rng.choice(cand, min(N_GAP, len(cand)), replace=False) if delta > 0 and len(cand) else np.array([], int)
    pi = pb.copy(); removed = 0.0
    for gt in gap_t:
        pi[gt] -= delta; removed += delta
    rest = np.setdiff1d(np.arange(K), ctrl)
    pi[rest] += removed * pb[rest] / pb[rest].sum()
    t = np.concatenate([rng.choice(K, n_in, p=pi), rng.choice(K, n_b, p=pb)])
    j = np.array(["India"] * n_in + ["Bench"] * n_b)
    roles = {k: ("development" if k < 5 else "control" if k < 10 else "promotion") for k in range(K)}
    rs = compare.role_shares(j, t, roles, cfg_for(B), seed=rep)
    r = rs[rs.role == "control"].iloc[0]
    return {"n_india": n_in, "delta_per_topic": delta, "true_role_gap": float(pb[ctrl].sum() - pi[ctrl].sum()),
            "detected": float(r["lo"] > 0), "false_detect": float(delta == 0 and (r["lo"] > 0 or r["hi"] < 0))}


def role_study(reps=100, B=300, procs=2):
    grid = [(n, 150, d, r, B) for n in (25, 50, 100) for d in (0.0, 0.03, 0.05) for r in range(reps)]
    with Pool(procs) as pool:
        df = pd.DataFrame(pool.map(one_role_power, grid, chunksize=10))
    tab = df.groupby(["n_india", "delta_per_topic"]).agg(power=("detected", "mean"), false_rate=("false_detect", "mean"),
                                                         mean_true_role_gap=("true_role_gap", "mean")).reset_index()
    tab.to_csv(OUT / "role_power_summary.csv", index=False)
    return tab


if __name__ == "__main__":
    if "--roles" in sys.argv:
        print(role_study().to_string(index=False))
    else:
        main()
