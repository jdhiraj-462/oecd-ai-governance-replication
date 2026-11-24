"""Country topic profiles, similarity, clustering, benchmarks and India comparisons.

A *unit* is either one jurisdiction or a pooled group (for example EU-27 =
EU institutions plus the 27 member states). A *benchmark* for the gap analysis
is a unit or a mean over jurisdictions (GLOBAL_MEAN, NON_OECD_MEAN), where each
jurisdiction counts once regardless of how many initiatives it reports.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import jensenshannon, squareform
from sklearn.metrics import silhouette_score


# ---------------------------------------------------------------- units
def unit_members(name: str, cfg: dict, all_j) -> list[str]:
    S = cfg["sample"]
    if name == "EU27_POOLED":
        return [S["eu_label"]] + [m for m in cfg["groups"]["eu27_members"] if m in set(all_j)]
    if name == "NON_OECD_POOLED":
        return [j for j in set(all_j) if j not in set(cfg["groups"]["oecd_members"]) and j != S["eu_label"]
                and j != S["india_label"]]
    return [name]


def unit_label(name: str) -> str:
    return {"EU27_POOLED": "EU-27 pooled (EU institutions and member states)",
            "NON_OECD_POOLED": "non-OECD jurisdictions pooled", "GLOBAL_MEAN": "mean of all other jurisdictions",
            "NON_OECD_MEAN": "mean of non-OECD jurisdictions"}.get(name, name)


# ---------------------------------------------------------------- profiles
def counts_from_assignments(juris: pd.Series, topics: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"j": np.asarray(juris), "t": topics})
    df = df[df["t"] != -1]
    return pd.crosstab(df["j"], df["t"])


def counts_from_theta(juris: pd.Series, theta: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(theta, index=np.asarray(juris)).groupby(level=0).sum()


def normalise(counts: pd.DataFrame, smoothing: float) -> pd.DataFrame:
    c = counts.astype(float) + smoothing
    return c.div(c.sum(axis=1), axis=0)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def similarity_matrices(profiles: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    P = profiles.values
    n = len(P)
    cos = np.array([[cosine(P[i], P[j]) for j in range(n)] for i in range(n)])
    jsd = np.array([[jensenshannon(P[i], P[j], base=2) for j in range(n)] for i in range(n)])
    idx = profiles.index
    return pd.DataFrame(cos, idx, idx), pd.DataFrame(jsd, idx, idx)


def _profile(topic_vec: np.ndarray, n_topics: int, smoothing: float) -> np.ndarray:
    c = np.bincount(topic_vec, minlength=n_topics).astype(float) + smoothing
    return c / c.sum()


def _draw_shares(v: np.ndarray, K: int, rng, prior: float = 0.5) -> np.ndarray:
    """One posterior draw of a unit's topic shares given its (resampled) initiatives:
    Dirichlet(counts + prior), Jeffreys prior 0.5. Keeps uncertainty for empty cells,
    where a plain percentile bootstrap would report zero variance (see Appendix C)."""
    return rng.dirichlet(np.bincount(v, minlength=K) + prior)


def _post_mean(v: np.ndarray, K: int, prior: float = 0.5) -> np.ndarray:
    c = np.bincount(v, minlength=K) + prior
    return c / c.sum()


def _recode(topics: np.ndarray) -> tuple[np.ndarray, int, dict]:
    ids = {v: i for i, v in enumerate(sorted(set(topics)))}
    return np.array([ids[v] for v in topics]), len(ids), ids


def _resample(idx: np.ndarray, groups: np.ndarray | None, rng) -> np.ndarray:
    """Bootstrap indices; if groups are given, resample whole groups (cluster bootstrap)."""
    if groups is None:
        return rng.choice(idx, len(idx))
    g = groups[idx]
    ug = np.unique(g)
    pick = rng.choice(ug, len(ug))
    return np.concatenate([idx[g == u] for u in pick])


def reference_percentile(jsd: pd.DataFrame, a: str, b: str) -> float | None:
    """Share of all off-diagonal jurisdiction pairs that are closer (smaller JSD) than a-b."""
    if a not in jsd.index or b not in jsd.index:
        return None
    iu = np.triu_indices(len(jsd), 1)
    allv = jsd.values[iu]
    return float((allv < jsd.loc[a, b]).mean())


# ---------------------------------------------------------------- India vs comparators
def india_comparisons(juris, topics, cfg, groups=None, seed=2026) -> tuple[pd.DataFrame, dict]:
    """Cosine and JSD between India and each comparator unit, with bootstrap intervals
    (cluster bootstrap if `groups` is given) and a permutation test."""
    rng = np.random.default_rng(seed)
    S, Sim = cfg["sample"], cfg["similarity"]
    keep = topics != -1
    j = np.asarray(juris)[keep]
    t, K, _ = _recode(topics[keep])
    g = None if groups is None else np.asarray(groups)[keep]
    sm = Sim["smoothing"]
    lo_q, hi_q = (1 - Sim["ci"]) / 2, 1 - (1 - Sim["ci"]) / 2
    i_idx = np.where(j == S["india_label"])[0]
    rows, draws = [], {}
    for comp in cfg["comparators"]:
        mem = unit_members(comp, cfg, j)
        c_idx = np.where(np.isin(j, mem))[0]
        row = {"comparator": comp, "label": unit_label(comp), "n_india": len(i_idx), "n_comp": len(c_idx)}
        if len(i_idx) < 5 or len(c_idx) < 5:
            rows.append(row); continue
        cs, js = [], []
        for _ in range(Sim["bootstrap_reps"]):
            pa = _draw_shares(t[_resample(i_idx, g, rng)], K, rng)
            pb = _draw_shares(t[_resample(c_idx, g, rng)], K, rng)
            cs.append(cosine(pa, pb)); js.append(jensenshannon(pa, pb, base=2))
        pa, pb = _profile(t[i_idx], K, sm), _profile(t[c_idx], K, sm)
        obs = jensenshannon(pa, pb, base=2)
        pooled = np.concatenate([t[i_idx], t[c_idx]]); na = len(i_idx); hits = 0
        for _ in range(Sim["permutation_reps"]):
            p = rng.permutation(pooled)
            if jensenshannon(_profile(p[:na], K, sm), _profile(p[na:], K, sm), base=2) >= obs:
                hits += 1
        # Point estimates are posterior medians from the same smoothed-bootstrap draws as the
        # interval. With few Indian initiatives the plug-in profile is less noisy than any
        # draw, so plug-in values can fall outside the interval; they are kept as descriptives.
        row.update({"cosine": float(np.median(cs)), "jsd_plugin": float(obs), "cosine_plugin": float(cosine(pa, pb)),
                    "cosine_lo": float(np.quantile(cs, lo_q)),
                    "cosine_hi": float(np.quantile(cs, hi_q)), "jsd": float(np.median(js)),
                    "jsd_lo": float(np.quantile(js, lo_q)), "jsd_hi": float(np.quantile(js, hi_q)),
                    "perm_p_jsd": (hits + 1) / (Sim["permutation_reps"] + 1)})
        draws[comp] = np.array(cs)
        rows.append(row)
    diff = {}
    if S["eu_label"] in draws and S["us_label"] in draws:
        d = draws[S["eu_label"]] - draws[S["us_label"]]
        diff = {"cos_diff_eu_minus_us": float(d.mean()), "lo": float(np.quantile(d, lo_q)),
                "hi": float(np.quantile(d, hi_q)), "share_draws_eu_closer": float((d > 0).mean())}
    return pd.DataFrame(rows), diff


# ---------------------------------------------------------------- clustering
def cluster_from_profiles(prof: pd.DataFrame, k: int, method: str) -> np.ndarray:
    _, jsd = similarity_matrices(prof)
    D = jsd.values.copy(); np.fill_diagonal(D, 0)
    Z = linkage(squareform(D, checks=False), method=method)
    return fcluster(Z, k, criterion="maxclust")


def cluster_countries(profiles: pd.DataFrame, jsd: pd.DataFrame, cfg):
    D = jsd.loc[profiles.index, profiles.index].values.copy()
    np.fill_diagonal(D, 0)
    Z = linkage(squareform(D, checks=False), method=cfg["clustering"]["method"])
    rows = []
    for k in cfg["clustering"]["k_range"]:
        if k >= len(profiles):
            break
        lab = fcluster(Z, k, criterion="maxclust")
        if len(set(lab)) < 2:
            continue
        rows.append({"k": k, "silhouette": silhouette_score(D, lab, metric="precomputed")})
    sil = pd.DataFrame(rows)
    best_k = int(sil.loc[sil["silhouette"].idxmax(), "k"]) if len(sil) else 1
    member = pd.DataFrame({"jurisdiction": profiles.index, "cluster": fcluster(Z, best_k, criterion="maxclust")})
    return member, sil, Z


def cluster_stability(juris, topics, units: list[str], members: pd.DataFrame, cfg, focal: str,
                      reps: int = 200, seed=2026) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bootstrap initiatives within each jurisdiction, rebuild profiles, re-cluster at the
    chosen k. Returns (a) how often each jurisdiction is clustered with the focal unit and
    (b) Hennig's (2007) cluster-wise mean Jaccard stability for each original cluster."""
    rng = np.random.default_rng(seed)
    keep = topics != -1
    j = np.asarray(juris)[keep]
    t, K, _ = _recode(topics[keep])
    sm = cfg["similarity"]["smoothing"]
    k = members["cluster"].nunique()
    orig = dict(zip(members["jurisdiction"], members["cluster"]))
    idx_by = {u: np.where(j == u)[0] for u in units}
    co = pd.Series(0.0, index=units)
    jac = {c: [] for c in sorted(set(orig.values()))}
    for _ in range(reps):
        prof = pd.DataFrame([_profile(t[rng.choice(idx_by[u], len(idx_by[u]))], K, sm) for u in units], index=units)
        lab = dict(zip(units, cluster_from_profiles(prof, k, cfg["clustering"]["method"])))
        if focal in lab:
            co += pd.Series({u: float(lab[u] == lab[focal]) for u in units})
        for c in jac:
            A = {u for u, v in orig.items() if v == c}
            best = max(len(A & {u for u, v in lab.items() if v == b}) / len(A | {u for u, v in lab.items() if v == b})
                       for b in set(lab.values()))
            jac[c].append(best)
    co = (co / reps).drop(focal, errors="ignore").sort_values(ascending=False)
    cojd = co.rename("share_of_bootstraps_with_" + focal).reset_index().rename(columns={"index": "jurisdiction"})
    jtab = pd.DataFrame([{"cluster": c, "size": sum(1 for v in orig.values() if v == c),
                          "mean_jaccard": float(np.mean(v))} for c, v in jac.items()])
    return cojd, jtab


# ---------------------------------------------------------------- benchmarks and gaps
def _bench_groups(bench: str, j: np.ndarray, cfg) -> tuple[list[np.ndarray], str]:
    """Return a list of index arrays. One array = pooled unit; several = mean over jurisdictions."""
    S = cfg["sample"]
    others = [c for c in pd.unique(j) if c != S["india_label"]]
    if bench == "GLOBAL_MEAN":
        return [np.where(j == c)[0] for c in others], "mean"
    if bench == "NON_OECD_MEAN":
        sel = [c for c in others if c not in set(cfg["groups"]["oecd_members"]) and c != S["eu_label"]
               and c not in set(cfg["groups"]["eu27_members"])]
        return [np.where(j == c)[0] for c in sel], "mean"
    mem = unit_members(bench, cfg, j)
    return [np.where(np.isin(j, mem))[0]], "pooled"


def bh_fdr(p: np.ndarray, q: float):
    p = np.asarray(p, dtype=float); n = len(p)
    order = np.argsort(p); ranked = p[order] * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n); out[order] = np.minimum(adj, 1)
    return out <= q, out


def india_gaps(juris, topics, cfg, groups=None, seed=2026) -> pd.DataFrame:
    """gap = benchmark share - India share (positive: India gives the theme less weight).

    Shares are posterior means under a Jeffreys Dirichlet prior. Uncertainty comes from a
    smoothed bootstrap: each replicate resamples initiatives (whole organisations when
    `groups` is given) and then draws shares from Dirichlet(counts + 0.5). For mean
    benchmarks each jurisdiction counts once. Two-sided p from the replicate distribution;
    Benjamini-Hochberg within each benchmark."""
    rng = np.random.default_rng(seed)
    keep = topics != -1
    j = np.asarray(juris)[keep]
    t, K, ids = _recode(topics[keep]); inv = {v: k for k, v in ids.items()}
    g = None if groups is None else np.asarray(groups)[keep]
    S, G, Sim = cfg["sample"], cfg["gaps"], cfg["similarity"]
    i_idx = np.where(j == S["india_label"])[0]
    lo_q, hi_q = (1 - Sim["ci"]) / 2, 1 - (1 - Sim["ci"]) / 2
    out = []
    for bench in G["benchmarks"]:
        grp, _ = _bench_groups(bench, j, cfg)
        grp = [x for x in grp if len(x) > 0]
        if len(i_idx) == 0 or not grp:
            continue
        bshare = np.mean([_post_mean(t[x], K) for x in grp], axis=0)
        ishare = _post_mean(t[i_idx], K)
        boot = np.zeros((Sim["bootstrap_reps"], K))
        for b in range(Sim["bootstrap_reps"]):
            bb = np.mean([_draw_shares(t[_resample(x, g, rng)], K, rng) for x in grp], axis=0)
            boot[b] = bb - _draw_shares(t[_resample(i_idx, g, rng)], K, rng)
        for k in range(K):
            p = 2 * min((boot[:, k] <= 0).mean(), (boot[:, k] >= 0).mean())
            out.append({"benchmark": bench, "benchmark_label": unit_label(bench), "topic": int(inv[k]),
                        "india_share": float(ishare[k]), "bench_share": float(bshare[k]),
                        "gap": float(bshare[k] - ishare[k]),
                        "lo": float(np.quantile(boot[:, k], lo_q)), "hi": float(np.quantile(boot[:, k], hi_q)),
                        "p_boot": max(p, 1 / Sim["bootstrap_reps"]), "n_bench_units": len(grp)})
    res = pd.DataFrame(out)
    if res.empty:
        return res
    res["eligible"] = (res["bench_share"] >= G["min_benchmark_share"]) | (res["india_share"] >= G["min_benchmark_share"])
    res["sig_fdr"] = False; res["p_fdr"] = np.nan
    for bench, idx in res[res["eligible"]].groupby("benchmark").groups.items():
        sig, padj = bh_fdr(res.loc[idx, "p_boot"].values, G["fdr_q"])
        res.loc[idx, "sig_fdr"] = sig; res.loc[idx, "p_fdr"] = padj
    res["direction"] = np.where(res["gap"] > 0, "India gives less weight", "India gives more weight")
    return res


def role_shares(juris, topics, roles: dict[int, str], cfg, groups=None, seed=2026) -> pd.DataFrame:
    """Expectation E1: share of initiatives in each state role (development, control,
    promotion) for India and each benchmark; difference benchmark - India with CI."""
    rng = np.random.default_rng(seed)
    keep = topics != -1
    j = np.asarray(juris)[keep]
    r = np.array([roles.get(int(x), "other") for x in topics[keep]])
    g = None if groups is None else np.asarray(groups)[keep]
    Sim, S = cfg["similarity"], cfg["sample"]
    lo_q, hi_q = (1 - Sim["ci"]) / 2, 1 - (1 - Sim["ci"]) / 2
    i_idx = np.where(j == S["india_label"])[0]
    out = []
    for bench in cfg["gaps"]["benchmarks"]:
        grp, _ = _bench_groups(bench, j, cfg)
        grp = [x for x in grp if len(x) > 0]
        if len(i_idx) == 0 or not grp:
            continue
        R_ = ["development", "control", "promotion", "other"]
        rc = np.array([R_.index(x) if x in R_ else 3 for x in r])
        obs_b = np.mean([_post_mean(rc[x], 4) for x in grp], axis=0); obs_i = _post_mean(rc[i_idx], 4)
        bt = np.array([np.mean([_draw_shares(rc[_resample(x, g, rng)], 4, rng) for x in grp], axis=0)
                       - _draw_shares(rc[_resample(i_idx, g, rng)], 4, rng) for _ in range(Sim["bootstrap_reps"])])
        for k, role in enumerate(R_[:3]):
            out.append({"benchmark": bench, "benchmark_label": unit_label(bench), "role": role,
                        "india_share": float(obs_i[k]), "bench_share": float(obs_b[k]),
                        "diff_bench_minus_india": float(obs_b[k] - obs_i[k]),
                        "lo": float(np.quantile(bt[:, k], lo_q)), "hi": float(np.quantile(bt[:, k], hi_q))})
    return pd.DataFrame(out)


# ---------------------------------------------------------------- instruments
def instrument_contrast(df: pd.DataFrame, cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    """NATO-tool mix and binding share for India vs comparators. Binding tests use only
    initiatives whose binding status is recorded (missing is never treated as non-binding)."""
    from scipy.stats import chi2_contingency, fisher_exact
    S = cfg["sample"]
    all_j = df["jurisdiction"].unique()
    rows, share_rows = [], []
    india = df[df.jurisdiction == S["india_label"]]
    share_rows.append(india["nato_tool"].value_counts(normalize=True).rename(S["india_label"]))
    for comp in cfg["comparators"]:
        mem = unit_members(comp, cfg, all_j)
        c = df[df.jurisdiction.isin(mem)]
        if c.empty or india.empty:
            continue
        share_rows.append(c["nato_tool"].value_counts(normalize=True).rename(comp))
        tab = pd.DataFrame({"india": india["nato_tool"].value_counts(), "comp": c["nato_tool"].value_counts()}).fillna(0).T
        tab = tab.loc[:, tab.sum() > 0]
        row = {"comparator": comp, "label": unit_label(comp)}
        if tab.shape[1] > 1:
            chi2, p, dof, _ = chi2_contingency(tab)
            n = tab.values.sum()
            row.update({"chi2": chi2, "dof": dof, "p": p, "cramers_v": float(np.sqrt(chi2 / (n * (min(tab.shape) - 1))))})
        bi = india["binding_status"].value_counts(); bc = c["binding_status"].value_counts()
        a = [bi.get("binding", 0), bi.get("non-binding", 0)]; b = [bc.get("binding", 0), bc.get("non-binding", 0)]
        row.update({"binding_share_india": a[0] / sum(a) if sum(a) else np.nan,
                    "binding_share_comp": b[0] / sum(b) if sum(b) else np.nan,
                    "binding_missing_india": float((india["binding_status"] == "missing").mean()),
                    "binding_missing_comp": float((c["binding_status"] == "missing").mean()),
                    "fisher_p_binding": fisher_exact([a, b])[1] if sum(a) and sum(b) else np.nan,
                    "authority_share_india": float((india["nato_tool"] == "authority").mean()),
                    "authority_share_comp": float((c["nato_tool"] == "authority").mean())})
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(share_rows).fillna(0)


def field_missingness(df: pd.DataFrame, focal: list[str]) -> pd.DataFrame:
    """Share of initiatives with missing structured fields, by jurisdiction (stewardship evidence)."""
    def miss(s):
        return s.isna() | s.astype(str).str.strip().isin(["", "nan", "None", "<NA>"])
    out = df.assign(m_binding=df["binding_status"].eq("missing"), m_status=miss(df["status"]),
                    m_year=df["start_year"].isna(), m_instrument=miss(df["instrument"]),
                    m_org=miss(df["organisation"]))
    g = out.groupby("jurisdiction")[["m_binding", "m_status", "m_year", "m_instrument", "m_org"]].mean()
    g.loc["ALL"] = out[["m_binding", "m_status", "m_year", "m_instrument", "m_org"]].mean()
    first = [x for x in focal + ["ALL"] if x in g.index]
    return pd.concat([g.loc[first], g.drop(index=first)])
