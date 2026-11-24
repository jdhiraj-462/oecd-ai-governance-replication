#!/usr/bin/env python3
"""Build the submission files for the Journal of Information Policy.

    python build_manuscript.py            # after run_all.py has produced results/summary.json

Inputs : manuscript/manuscript.md (text with [[TOKENS]]), results/summary.json,
         results/validation/validation_scores.json (after human coding), config/data_access.yaml
Outputs: submission/Manuscript_anonymous.docx, submission/Tables.docx,
         submission/Figure_captions_and_alt_text.docx, submission/figures/*.tif

Placeholders: [[TOKEN]] is replaced by a computed value. If the value does not
exist yet, the text appears highlighted as RESULT PENDING. [[AUTHOR: ...]] notes
are always highlighted and must be resolved by the authors before submission.
The script prints how many placeholders remain.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).parent
MS = ROOT / "manuscript"
OUT = ROOT / "submission"
RES = ROOT / "results"


# ------------------------------------------------------------------ helpers
def f2(x, d=2):
    return "–" if x is None or pd.isna(x) else f"{x:.{d}f}"


def pct(x, d=0):
    return "–" if x is None or pd.isna(x) else f"{100 * x:.{d}f} percent"


def pfmt(p):
    if p is None or pd.isna(p):
        return "–"
    return "p < .001" if p < 0.001 else f"p = {p:.3f}".replace("0.", ".")


def listing(items):
    items = [str(x) for x in items]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    sep = "; " if any("," in x for x in items) else ", "
    if len(items) == 2 and sep == ", ":
        return f"{items[0]} and {items[1]}"
    return sep.join(items[:-1]) + f"{sep}and " + items[-1]


def lc(x: str) -> str:
    """Lower-case the first letter unless it starts an acronym (AI, R&D, SME)."""
    return x if (len(x) > 1 and (x[1].isupper() or not x[1].isalpha())) else x[:1].lower() + x[1:]


def clean_label(s):
    return re.sub(r"^T\d+:\s*", "", str(s))


# ------------------------------------------------------------------ tokens
def ok(x):
    return x is not None and not (isinstance(x, float) and np.isnan(x))


def _n(x, d):
    v = f"{x:.{d}f}"
    return f"{x:.3f}" if v in ("-0.00", "0.00") and x != 0 else v


def ci(r, a="lo", b="hi", d=2):
    return f"95% CI {_n(r[a], d)} to {_n(r[b], d)}"


def sim_tokens() -> dict[str, str]:
    """Simulation results exist independently of the OECD data and are always inserted."""
    t: dict[str, str] = {}
    SR = ROOT / "simulations/results"
    try:
        naive = pd.read_csv(SR / "naive_power_summary.csv")
        t["SIM_NAIVE_FWER"] = pct(naive.query("n_india == 25 and n_bench == 150 and delta == 0")["p_any_false"].iloc[0])
        p = pd.read_csv(SR / "power_summary.csv")
        t["SIM_FWER_MAX"] = pct(p[p.delta == 0]["p_any_false"].max())
        t["SIM_POW_100"] = pct(p.query("n_india == 100 and n_bench == 150 and delta == 0.12")["power"].iloc[0])
        t["SIM_POW_50"] = pct(p.query("n_india == 50 and n_bench == 150 and delta == 0.12")["power"].iloc[0])
        cov = json.loads((SR / "power_summary.json").read_text())["coverage"]
        t["SIM_COV_CLUSTER"] = pct(cov["coverage_cluster"]); t["SIM_COV_INIT"] = pct(cov["coverage_initiative"])
        rec = json.loads((SR / "recovery_summary.json").read_text())
        t["SIM_ARI"] = f"{rec['ari_mean']:.2f}"; t["SIM_GAP_FOUND"] = str(rec["planted_gap_flagged"])
    except FileNotFoundError:
        pass
    return t


def tokens() -> dict[str, str]:
    t: dict[str, str] = {}
    acc = yaml.safe_load(open(ROOT / "config/data_access.yaml"))
    for k, v in acc.items():
        if v not in (None, "", "TBD"):
            t[k.upper()] = str(v)
    t.update(sim_tokens())
    cfgp = yaml.safe_load(open(ROOT / "config/config.yaml"))
    t["IN_N_PLANNED"] = str(cfgp["validation"]["instrument_sample_size"])
    sp = RES / "summary.json"
    if not sp.exists():
        return t
    s = json.loads(sp.read_text())
    if s["run"].get("embedding_backend") != "sbert" or s["run"].get("fast_mode"):
        print("WARNING: summary.json comes from a test or fast run. Numbers are NOT inserted.")
        return t
    IN, EU, US = "India", "European Union", "United States"
    c, b, cl = s["corpus"], s["bertopic"], s["clusters"]
    t.update({
        "N_DOCS": f"{c['n_docs']:,}", "N_JUR": str(c["n_jurisdictions"]), "N_INDIA": str(c["n_india"]),
        "N_EU": str(c["n_eu"]), "N_US": str(c["n_us"]), "N_EU27": str(c["n_eu27_pooled"]),
        "N_NON_OECD": str(c["n_non_oecd_jurisdictions"]), "N_SUBNATIONAL": str(c["n_subnational_flagged"]),
        "N_SUBNATIONAL_INDIA": str(c["n_subnational_india"]),
        "YEAR_MIN": f"{c['year_min']:.0f}" if c.get("year_min") else "–", "YEAR_MAX": f"{c['year_max']:.0f}" if c.get("year_max") else "–",
        "N_TOPICS": str(b["n_topics"]), "OUTLIER_SHARE": pct(b["outlier_share_before_reduction"]),
        "CV": f2(b["c_v"]), "NPMI": f2(b["c_npmi"]), "ARI_MEAN": f2(b["seed_ari_mean"]), "ARI_MIN": f2(b["seed_ari_min"]),
        "K_RANGE": f"{b['seed_k_range'][0]} to {b['seed_k_range'][1]}",
        "COS_EU_PLUGIN": f2(next((p.get("cosine_plugin") for p in s["india_pairs"] if p.get("comparator") == EU), None)),
        "COS_US_PLUGIN": f2(next((p.get("cosine_plugin") for p in s["india_pairs"] if p.get("comparator") == US), None)),
        "SEED_EU": f"{b['seed_cos_eu'][0]:.2f} to {b['seed_cos_eu'][2]:.2f}",
        "SEED_US": f"{b['seed_cos_us'][0]:.2f} to {b['seed_cos_us'][2]:.2f}",
        "SEED_SAME": f"{b['seed_same_ordering']} of {b['n_seeds']}",
        "N_CLUSTERED": str(cl["n_countries_clustered"]), "BEST_K": str(cl["best_k"]), "SILHOUETTE": f2(cl["best_silhouette"]),
        "JACCARD_ALL": f2(cl["mean_jaccard_all"]),
        "INDIA_PEERS": listing(sorted(cl["india_cluster_members"])) or "no other jurisdiction",
        "STABLE_PEERS": listing(cl["india_stable_peers"]) or "no jurisdiction",
        "STABLE_PEERS_PHRASE": (f"identifies {listing(cl['india_stable_peers'])} as {'a stable peer' if len(cl['india_stable_peers']) == 1 else 'stable peers'}"
                                if cl["india_stable_peers"] else "identifies no jurisdiction that shares India's cluster in at least 70 percent of replicates"),
        "INDIA_WITH_EU": "shares a cluster with" if cl["india_cluster"] == cl["eu_cluster"] else "sits in a different cluster from",
        "INDIA_WITH_US": "shares a cluster with" if cl["india_cluster"] == cl["us_cluster"] else "sits in a different cluster from",
        "INDIA_CLUSTER_SENT": {(True, True): "India shares a cluster with both the EU institutions and the United States.",
                               (True, False): "India shares a cluster with the EU institutions but not with the United States.",
                               (False, True): "India shares a cluster with the United States but not with the EU institutions.",
                               (False, False): "India sits in a different cluster from both the EU institutions and the United States."}[
                                   (cl["india_cluster"] == cl["eu_cluster"], cl["india_cluster"] == cl["us_cluster"])],
        "BOOT_CLUSTERS": f"{s['bootstrap']['n_clusters']:,}" if s["bootstrap"].get("n_clusters") else "no organisation field; initiative-level",
    })
    # clusters described for the global section
    cm = cl["cluster_members"]
    t["CLUSTER_TEXT"] = " ".join(f"Cluster {k} contains {listing(v)}." for k, v in cm.items() if len(v) <= 12) or ""
    if cl["best_silhouette"] < 0.25:
        small = [v for v in cm.values() if len(v) <= 3]
        t["CLUSTER_TEXT"] = (f"A silhouette below 0.25 indicates weak structure: the selected solution sets apart only {listing(['the ' + x if x in ('United States', 'European Union', 'United Kingdom', 'Netherlands') else x for x in sum(small, [])])}, "
                             f"and the remaining {max(len(v) for v in cm.values())} jurisdictions form one group. National AI portfolios, measured this way, "
                             "vary more continuously than in distinct models.")
        n_other = len(cl["india_cluster_members"])
        t["INDIA_PEERS"] = f"{n_other} other jurisdictions"
        t["STABLE_PEERS_PHRASE"] = (f"keeps {len(cl['india_stable_peers'])} of them in India's cluster in at least 70 percent of replicates, "
                                    "which reflects the size of the group rather than close kinship")
        t["STABLE_PEERS"] = None
    if s.get("india_nearest"):
        t["INDIA_NEAREST"] = listing(f"{'the EU institutions' if r['jurisdiction'] == EU else r['jurisdiction']} ({r['cosine']:.2f})" for r in s["india_nearest"][:3])
    rp = s.get("reference_percentile", {})
    if EU in rp: t["REF_EU"] = pct(rp[EU])
    if US in rp: t["REF_US"] = pct(rp[US])
    L = s["length"]["median_chars"]
    t.update({"LEN_IN": f"{L.get(IN, float('nan')):.0f}", "LEN_EU": f"{L.get(EU, float('nan')):.0f}",
              "LEN_US": f"{L.get(US, float('nan')):.0f}",
              "LEN_TEST": f"H = {s['length']['kruskal_H']:.1f}, {pfmt(s['length']['kruskal_p'])}" if s["length"]["kruskal_H"] else "not computable"})
    ms = s["missingness"]
    t["MISS_BIND_ALL"] = pct(ms["ALL"]["m_binding"]); t["MISS_STATUS_ALL"] = pct(ms["ALL"]["m_status"])
    mb, mst = ms["ALL"]["m_binding"], ms["ALL"]["m_status"]
    t["STEWARD_TEXT"] = (f"The binding field is missing for {pct(mb)} of initiatives (Table S13), which limits any test of whether countries govern AI by rule or by guidance. The status field is almost always recorded but rarely updated when a measure ends or is revoked."
                         if mb >= 0.10 and mst < 0.10 else
                         f"The binding field is missing for {pct(mb)} of initiatives and the status field for {pct(mst)} (Table S13), which limits any test of whether countries govern AI by rule or by guidance, and whether measures remain in force."
                         if max(mb, mst) >= 0.10 else
                         f"The binding and status fields are recorded for most initiatives (missing for {pct(mb)} and {pct(mst)}; Table S13), but they record legal form, not enforcement, and say nothing about scale.")
    t["MISS_BIND_IN"] = pct(ms.get(IN, {}).get("m_binding")) if IN in ms else "–"
    cov = dict(s["coverage"])
    ac = RES / "coverage_analyst_check.csv"
    if ac.exists() and not cov["basis"].startswith("confirmed"):
        a = pd.read_csv(ac)
        cov.update({"n_covered": int(a["analyst_covered"].sum()), "share_covered": float(a["analyst_covered"].mean()),
                    "basis": "automatic search, with each candidate match read and checked",
                    "missing": a.loc[~a["analyst_covered"], "instrument"].tolist()})
    tm = ROOT / "config/instrument_type_map.csv"
    if tm.exists():
        t["N_TYPES"] = str(len(pd.read_csv(tm)))
    t.update({"COV_N": str(cov["n_inventory"]), "COV_COVERED": str(cov["n_covered"]), "COV_BASIS": cov["basis"],
              "COV_MISSING": listing(cov["missing"]) or "none"})
    share = cov["share_covered"]
    t["COV_TEXT"] = ("Most major instruments are recorded, so the comparison describes India's actual portfolio reasonably well."
                     if share >= 0.75 else
                     "A substantial part of India's governance activity, particularly sectoral frameworks and recent rules, is therefore missing from the shared record. "
                     "Where India appears to give less weight to a theme, part of the difference may be a reporting gap rather than a policy gap, "
                     "and the comparison should be read as describing India's internationally visible portfolio.")
    # similarity
    pairs = {p["comparator"]: p for p in s["india_pairs"] if "cosine" in p}
    for key, comp in (("EU", EU), ("EU27", "EU27_POOLED"), ("US", US), ("NONOECD", "NON_OECD_POOLED")):
        p = pairs.get(comp)
        if p:
            t[f"COS_{key}"] = f"{p['cosine']:.2f} ({ci(p, 'cosine_lo', 'cosine_hi')})"
    d = s.get("india_eu_minus_us") or {}
    if d:
        t["DIFF"] = f"{d['cos_diff_eu_minus_us']:+.2f} (95% CI {d['lo']:+.2f} to {d['hi']:+.2f})"
        if d["lo"] > 0:
            t["CLOSER_TO"] = "closer to the EU institutions than to the United States"
        elif d["hi"] < 0:
            t["CLOSER_TO"] = "closer to the United States than to the EU institutions"
        else:
            t["CLOSER_TO"] = "not reliably closer to the EU institutions or to the United States"
    # global themes
    gt = s.get("global_themes", [])
    if gt:
        wide = [lc(clean_label(x["label"])) for x in gt[:3]]
        narrow = [lc(clean_label(x["label"])) for x in gt[-2:]]
        t["GLOBAL_THEMES_TEXT"] = (f"The most widespread themes, present in the portfolios of the largest share of jurisdictions, are {listing(wide)} "
                                   f"(present in {pct(gt[0]['share_of_jurisdictions'])}, {pct(gt[1]['share_of_jurisdictions'])}, and "
                                   f"{pct(gt[2]['share_of_jurisdictions'])} of jurisdictions). The least widespread are {listing(narrow)}.")
    rg = s.get("global_role_shares", {})
    t["ROLE_DEV_ALL"] = pct(rg.get("development", 0)); t["ROLE_CTRL_ALL"] = pct(rg.get("control", 0))
    t["ROLE_PROM_ALL"] = pct(rg.get("promotion", 0))
    # E1
    rs = [r for r in s["role_shares"] if r["role"] == "control"]
    byb = {r["benchmark"]: r for r in rs}
    e1 = []
    for bkey, name in (("EU27_POOLED", "the pooled EU-27"), (EU, "the EU institutions"), (US, "the United States"),
                       ("NON_OECD_MEAN", "the non-OECD mean")):
        r = byb.get(bkey)
        if r:
            e1.append(f"{pct(r['bench_share'])} for {name} (difference {r['diff_bench_minus_india']:+.2f}, {ci(r)})")
    ri = byb.get("EU27_POOLED")
    if ri:
        if ri["lo"] > 0:
            verdict = "E1 is supported: the interval for the EU-27 comparison lies above zero."
        elif ri["hi"] <= 0:
            verdict = "E1 is contradicted: the interval for the EU-27 comparison lies at or below zero."
        elif ri["diff_bench_minus_india"] < 0:
            verdict = (f"The estimate runs against E1: India's control share is higher than the pooled EU-27's, although the interval includes zero, "
                       "so E1 is neither supported nor formally contradicted. The EU-level comparison points the other way, which shows how much the "
                       "result depends on whether member states' programs are counted.")
        else:
            verdict = "The estimate is in the expected direction, but the interval for the EU-27 comparison includes zero, so E1 is not supported at conventional precision."
        t["E1_TEXT"] = (f"Control-role topics account for {pct(ri['india_share'])} of India's initiatives, against "
                        f"{listing(e1)}. {verdict}")
        if ri["lo"] > 0:
            t["ABSTRACT_E1"] = "India gives the control functions of the state less weight than the pooled EU-27."
        elif ri["diff_bench_minus_india"] < 0:
            t["ABSTRACT_E1"] = (f"Contrary to expectation, control-role themes take a larger share of India's portfolio than of the pooled EU-27's "
                                f"({pct(ri['india_share'])} against {pct(ri['bench_share'])}), although the interval includes zero.")
        else:
            t["ABSTRACT_E1"] = "India's weight on control functions is not reliably lower than the pooled EU-27's."
        t["E1_SHORT"] = ("E1 is not supported: India's control share is, if anything, higher than the pooled EU-27's."
                         if (ri["lo"] <= 0 and ri["diff_bench_minus_india"] < 0) else verdict)
    # exploratory role differences whose interval excludes zero
    rn = {"development": "development", "promotion": "promotion", "control": "control"}
    sig = [r for r in s["role_shares"] if r["role"] in rn and (r["lo"] > 0 or r["hi"] < 0)]
    bname = {"EU27_POOLED": "the pooled EU-27", EU: "the EU institutions", US: "the United States",
             "GLOBAL_MEAN": "the mean of all other jurisdictions", "NON_OECD_MEAN": "the non-OECD mean"}
    if sig:
        t["ROLE_EXPLORE"] = ("Outside the pre-specified test, two role differences have intervals that exclude zero: " if len(sig) == 2 else
                             f"Outside the pre-specified test, {len(sig)} role differences have intervals that exclude zero: ") + listing(
            f"India gives {'less' if r['diff_bench_minus_india'] > 0 else 'more'} weight to {r['role']} than {bname.get(r['benchmark'], r['benchmark'])} "
            f"({pct(r['india_share'])} against {pct(r['bench_share'])}; difference {r['diff_bench_minus_india']:+.2f}, {ci(r)})" for r in sig) +             ". These comparisons were not pre-specified and are not corrected for multiplicity."
        prom = [r for r in sig if r["role"] == "promotion" and r["diff_bench_minus_india"] > 0]
        dev = [r for r in sig if r["role"] == "development" and r["diff_bench_minus_india"] < 0]
        bits = []
        if dev: bits.append(f"weighted toward development ({pct(dev[0]['india_share'])} of initiatives)")
        if prom: bits.append(f"light on the promotion of adoption ({pct(prom[0]['india_share'])}, against {pct(prom[0]['bench_share'])} for {bname.get(prom[0]['benchmark'])})")
        t["ROLE_SHORT"] = ("India's portfolio is " + " and ".join(bits) + ".") if bits else ""
    else:
        t["ROLE_EXPLORE"] = "No other role difference has an interval that excludes zero."
        t["ROLE_SHORT"] = ""
    # gaps
    g = s.get("gaps", {})
    names = {EU: "the EU institutions", "EU27_POOLED": "the pooled EU-27", US: "the United States",
             "GLOBAL_MEAN": "the mean of all other jurisdictions", "NON_OECD_MEAN": "the non-OECD mean"}

    def gl(rows):
        return listing(f"{lc(clean_label(r['label']))} ({r['gap']:+.2f}, {ci(r)})" for r in rows[:4])
    parts = []
    for bk, nm in names.items():
        rows = g.get(bk, [])
        n = s.get("n_tests", {}).get(bk, 0)
        parts.append(f"Relative to {nm}, {len(rows)} of {n} tested themes {'shows' if len(rows) == 1 else 'show'} a significant shortfall"
                     + (f": {gl(rows)}." if rows else "."))
    t["GAP_TEXT"] = " ".join(parts) + (" Given the power of the test (see Data and Methods), themes not listed may still differ by up to about ten percentage points.")
    t["GAPS_GLOBAL"] = gl(g.get("GLOBAL_MEAN", [])) or "no theme"
    ov = s.get("overweights", {})
    t["OVERWEIGHTS"] = listing(sorted({lc(clean_label(r["label"])) for rows in ov.values() for r in rows})) or "none of the themes"
    # E2
    tests = {r["comparator"]: r for r in s["instruments"]["tests"]}
    e2 = []
    for comp, nm in (("EU27_POOLED", "the pooled EU-27"), (EU, "the EU institutions"), (US, "the United States"),
                     ("NON_OECD_POOLED", "non-OECD jurisdictions")):
        r = tests.get(comp)
        if r and ok(r.get("authority_share_comp")):
            e2.append(f"{nm} ({pct(r['authority_share_comp'])}; tool mix {('χ²(' + str(int(r['dof'])) + ') = ' + format(r['chi2'], '.1f') + ', ' + pfmt(r['p']) + ', V = ' + format(r['cramers_v'], '.2f')) if ok(r.get('chi2')) else 'not testable'})")
    r27 = tests.get("EU27_POOLED") or tests.get(EU)
    if r27:
        a_in, a_c = r27["authority_share_india"], r27["authority_share_comp"]
        b_in, b_c = r27["binding_share_india"], r27["binding_share_comp"]
        lower = a_in < a_c and (not ok(b_in) or not ok(b_c) or b_in < b_c)
        fp = r27.get("fisher_p_binding"); small = abs(a_in - a_c) < 0.05 and (not ok(b_in) or not ok(b_c) or abs(b_in - b_c) < 0.05)
        if not lower:
            E2V, E2S = "E2 is contradicted: India's shares are not lower than the EU-27's on both measures.", "contradicted"
        elif small or (ok(fp) and fp > 0.05 and ok(r27.get("p")) and r27["p"] > 0.05):
            E2V = (f"E2 is not contradicted, since both shares are lower for India, but the differences are small "
                   f"({(a_c - a_in) * 100:.0f} and {((b_c - b_in) * 100) if ok(b_in) and ok(b_c) else float('nan'):.0f} percentage points) and neither test distinguishes India from the EU-27. "
                   "Both portfolios use authority tools and binding instruments rarely, so E2 receives no meaningful support.")
            E2S = "not contradicted but without meaningful support"
        else:
            E2V, E2S = "E2 is supported on both measures.", "supported"
        t["E2_TEXT"] = (f"Authority tools account for {pct(a_in)} of India's initiatives, against {listing(e2)}. "
                        f"Among initiatives with recorded binding status, {pct(b_in) if ok(b_in) else 'an unknown share'} of India's are binding, against "
                        f"{pct(b_c) if ok(b_c) else 'an unknown share'} of the EU-27's (Fisher's exact test, {pfmt(r27.get('fisher_p_binding'))}; "
                        f"binding status missing for {pct(r27['binding_missing_india'])} of India's and {pct(r27['binding_missing_comp'])} of the EU-27's initiatives). "
                        + E2V)
        sh = s["instruments"]["shares"]; si, sc = sh.get(IN, {}), sh.get("EU27_POOLED", {})
        tools = [k for k in ("nodality", "authority", "treasure", "organization") if k in si and k in sc]
        top_diff = max(tools, key=lambda k: si[k] - sc[k]) if tools else None
        low_diff = min(tools, key=lambda k: si[k] - sc[k]) if tools else None
        t["E2_TEXT"] += (f" The larger difference lies elsewhere in the tool mix: India relies more on {top_diff} tools ({pct(si[top_diff])} against {pct(sc[top_diff])}) "
                         f"and less on {low_diff} ({pct(si[low_diff])} against {pct(sc[low_diff])}), although the overall tool-mix test against the EU-27 is {pfmt(r27.get('p'))}."
                         if top_diff else "")
        t["E2_SHORT"] = E2S
        t["INSTRUMENT_SUMMARY"] = (f"uses authority tools and binding instruments about as rarely as the pooled EU-27 ({pct(a_in)} against {pct(a_c)} of initiatives use authority tools)"
                                   + (f" and relies more on {top_diff} tools ({pct(si[top_diff])} against {pct(sc[top_diff])})" if top_diff else ""))
    # verdict / implication / conclusion text
    gl_rows = g.get("GLOBAL_MEAN", []) + g.get("NON_OECD_MEAN", [])
    ctrl_gaps = sorted({lc(clean_label(r["label"])) for r in gl_rows if r.get("role") == "control"})
    t["VERDICT_TEXT"] = (f"{t.get('E1_SHORT', '')} On instruments, E2 is {t.get('E2_SHORT', 'untested')}. "
                         f"{t.get('ROLE_SHORT', '')} "
                         f"On proximity, India's profile is {t.get('CLOSER_TO', 'not comparable')}" + (f", and its most reliable peers are {t['STABLE_PEERS']}. " if t.get("STABLE_PEERS") else
                         f"; the jurisdictions closest to it are {t['INDIA_NEAREST']}, and the clustering shows no distinct group to which India belongs. ")
                         + (f"Against the benchmarks in which each jurisdiction counts once, the control-role themes on which India gives significantly less weight are {listing(ctrl_gaps)}."
                            if ctrl_gaps else "Against the benchmarks in which each jurisdiction counts once, no control-role theme shows a significant shortfall for India, which given the power of the test does not rule out differences of moderate size."))
    t["IMPLICATION_TEXT"] = (
        (f"Where India's shortfalls concern control-role themes ({listing(ctrl_gaps)}), the issue is institutional capacity more than legislation. "
         "India's guidelines favor existing regulators over a new AI law, a model that depends on regulators having technical staff, information-gathering powers, and incident data in their sectors.")
        if ctrl_gaps else
        ("The evidence does not single out a control-role theme on which India falls short of jurisdictions in general. "
         "The policy question is then less which themes to add than how to make existing commitments enforceable and visible, which depends on regulators having technical staff, information-gathering powers, and incident data in their sectors."))
    pa = [r for r in g.get("EU27_POOLED", []) + g.get("GLOBAL_MEAN", []) if "public administration" in r["label"].lower()]
    t["IMPLICATION_TEXT"] = (
        (f"{t.get('ROLE_SHORT', '')} The one theme-level shortfall that survives correction is the use of AI in public administration and citizen services, "
         f"the most widespread theme in the database, which accounts for {pct(pa[0]['india_share'])} of India's recorded initiatives against {pct(pa[0]['bench_share'])} of the pooled EU-27's. "
         "For a country whose digital public infrastructure is a model for others, this is the clearest case in which the database may record a reporting gap rather than a policy gap: "
         "AI deployments by Indian ministries and states appear seldom to be submitted as initiatives. Either way, the remedy runs through information: "
         "systematic reporting of public-sector AI use, of the kind the Netherlands and the United Kingdom publish in public algorithm registers, would make India's state use of AI visible and open to scrutiny. "
         if pa else t["IMPLICATION_TEXT"] + " ")
        + "On the control side, the evidence does not single out a theme on which India falls short, so the policy question is less which themes to add than how to make existing commitments "
          "enforceable, which depends on regulators having technical staff, information-gathering powers, and incident data in their sectors.")
    t["CONCLUSION_TEXT"] = (f"{t.get('E1_SHORT', '')} {t.get('ROLE_SHORT', '')} In its choice of instruments, India {t.get('INSTRUMENT_SUMMARY', '')}. "
                            f"The database records {t['COV_COVERED']} of {t['COV_N']} major Indian instruments.")
    # sensitivity, periods, LDA
    sv = s.get("sensitivity")
    if sv:
        t["SENS_SAME"] = f"{sv['n_same_ordering']} of {sv['n_variants']} variants"
        t["SENS_DETAIL"] = listing(f"{r['variant']} (EU {r['eu']:.2f}, US {r['us']:.2f})" for r in sv["rows"][1:]
                                   if ok(r.get("eu")) and ok(r.get("us")))
    pr = s.get("periods", [])
    t["PERIODS"] = listing(f"{r['period']}, {r['eu']:.2f} with the EU institutions and {r['us']:.2f} with the United States ({r['n_india']} Indian initiatives)"
                           if ok(r.get("eu")) and ok(r.get("us")) else f"{r['period']}, too few initiatives to compare ({r['n_india']} Indian)" for r in pr)
    valid = [r for r in pr if ok(r.get("eu")) and ok(r.get("us"))]
    if len(valid) >= 2:
        de = valid[-1]["eu"] - valid[0]["eu"]; du = valid[-1]["us"] - valid[0]["us"]
        t["PERIOD_TEXT"] = (f"Between the earliest and latest comparable periods, India's similarity changed by {de:+.2f} with the EU institutions and {du:+.2f} with the United States; "
                            "period estimates rest on fewer initiatives and indicate direction only.")
    else:
        t["PERIOD_TEXT"] = "Too few periods have enough initiatives to describe change over time."
    r = s.get("lda_robustness")
    if r:
        t.update({"LDA_K": str(r["k"]), "LDA_CV": f2(r["c_v"]), "LDA_RHO": f2(r["spearman_similarity_bertopic_vs_lda"]),
                  "LDA_IN_EU": f2(r.get("india_eu_cosine_lda")), "LDA_IN_US": f2(r.get("india_us_cosine_lda"))})
        if ok(r.get("india_eu_cosine_lda")) and ok(r.get("india_us_cosine_lda")) and d:
            same = (r["india_eu_cosine_lda"] > r["india_us_cosine_lda"]) == (d["cos_diff_eu_minus_us"] > 0)
            t["LDA_TEXT"] = ("The ordering of India's similarity to the EU institutions and the United States is the same under both models."
                             if same else "The ordering differs between the two models, so the comparison of India's proximity to the EU institutions and the United States should be treated as model-dependent.")
    # validation
    vp = RES / "validation/validation_scores.json"
    if vp.exists():
        v = json.loads(vp.read_text())
        for pre, key in (("WI", "word_intrusion"), ("LF", "label_fit"), ("IN", "instrument_coding")):
            if key in v:
                x = v[key]
                t[f"{pre}_N"] = str(x["n_items"]); t[f"{pre}_ACC"] = pct((x["accuracy_coder_a"] + x["accuracy_coder_b"]) / 2)
                t[f"{pre}_ALPHA"] = f2(x["krippendorff_alpha"])
    if s.get("roles_provisional") or not s.get("labels_from_authors"):
        print("WARNING: topic labels or roles are not yet coded by the authors; E1 and theme labels are provisional.")
        t.pop("E1_TEXT", None); t.pop("ABSTRACT_E1", None); t.pop("E1_SHORT", None); t.pop("VERDICT_TEXT", None)
    # role robustness to alternative coding
    ra = {(r["benchmark"], r["role"]): r for r in s.get("role_shares_alt", [])}
    if ra:
        c27 = ra.get(("EU27_POOLED", "control")); p27 = ra.get(("EU27_POOLED", "promotion"))
        if c27 and p27:
            t["ROLE_ALT_TEXT"] = (f"Under the alternative coding of {s.get('n_topics_recoded_alt', 7)} borderline topics, the pre-specified control comparison with the pooled EU-27 "
                                  f"becomes {pct(c27['india_share'])} against {pct(c27['bench_share'])} (difference {c27['diff_bench_minus_india']:+.2f}, {ci(c27)}), "
                                  f"and the promotion comparison {pct(p27['india_share'])} against {pct(p27['bench_share'])} (difference {p27['diff_bench_minus_india']:+.2f}, {ci(p27)}).")
    dt = s.get("dominated_topics", {})
    drow = next((r for r in (s.get("sensitivity") or {}).get("rows", []) if str(r.get("variant", "")).startswith("topics dominated")), None)
    if dt and drow:
        prim = s["sensitivity"]["rows"][0]
        t["DOM_TEXT"] = (f"The variant that drops the {['zero','one','two','three','four','five','six'][len(dt)] if len(dt) < 7 else len(dt)} topics in which one jurisdiction contributes more than a quarter of initiatives "
                         f"({listing(lc(v['label']) + ', ' + pct(v['share']) + ' from ' + ('the ' if v['top_jurisdiction'] in ('United States',) else '') + v['top_jurisdiction'] for v in dt.values())}) "
                         f"leaves India's similarity to the EU institutions almost unchanged ({prim['eu']:.2f} to {drow['eu']:.2f}) but raises its similarity to the United States from {prim['us']:.2f} to {drow['us']:.2f}. "
                         "Part of India's distance from the United States therefore reflects how the United States reports its own federal programs.")
    else:
        t["DOM_TEXT"] = ""
    cp_ = ROOT / "data/processed/corpus.csv"
    if cp_.exists():
        _c = pd.read_csv(cp_)
        t["EXACT_SHARE"] = pct(_c["nato_rule"].fillna("").str.startswith("exact:").mean())
    t["INTRO_STATEMENT"] = (
        f"Where does India's AI governance sit between the European Union and the United States? Across {t.get('N_DOCS', '')} initiatives in the OECD.AI Policy Navigator, "
        f"India's portfolio is not reliably closer to either. It is weighted toward building capacity and light on promoting adoption, especially AI in public administration, "
        f"and it uses binding rules as rarely as the EU-27. The database records only {t.get('COV_COVERED', '')} of {t.get('COV_N', '')} major Indian AI instruments, "
        "so international comparison understates India's regulatory activity. Better reporting of sectoral and public-sector AI measures is a low-cost policy lever.")
    return t


def fill(md: str, tok: dict[str, str]) -> tuple[str, int]:
    pending = 0

    def rep(m):
        nonlocal pending
        key = m.group(1).strip()
        if key.startswith("AUTHOR:"):
            pending += 1
            return f"[AUTHOR NOTE: {key[7:].strip()}]{{custom-style=\"Placeholder\"}}"
        if key in tok:
            return tok[key]
        pending += 1
        return f"[RESULT PENDING: {key}]{{custom-style=\"Placeholder\"}}"
    return re.sub(r"\[\[(.+?)\]\]", rep, md, flags=re.S), pending


# ------------------------------------------------------------------ docx styling
def make_reference_docx(path: Path):
    from docx import Document
    from docx.enum.text import WD_LINE_SPACING
    from docx.oxml.ns import qn
    from docx.shared import Pt
    subprocess.run(["pandoc", "-o", str(path), "--print-default-data-file", "reference.docx"], check=True)
    d = Document(path)
    for st in d.styles:
        try:
            f = st.font
        except AttributeError:
            continue
        f.name = "Times New Roman"; f.color.rgb = None
        rpr = st.element.get_or_add_rPr()
        rf = rpr.find(qn("w:rFonts"))
        if rf is not None:
            for a in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
                rf.attrib.pop(qn(a), None)
            for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
                rf.set(qn(a), "Times New Roman")
    for name in ("Normal", "Body Text", "First Paragraph", "Block Text"):
        if name in [s.name for s in d.styles]:
            st = {x.name: x for x in d.styles}[name]
            st.font.size = Pt(12)
            pf = st.paragraph_format
            pf.line_spacing_rule = WD_LINE_SPACING.DOUBLE; pf.space_before = Pt(0); pf.space_after = Pt(0)
            if name in ("Body Text",):
                pf.first_line_indent = Pt(36)
    byid = {x.style_id: x for x in d.styles}
    for name in ("Heading 1", "Heading 2", "Heading 3", "Title"):
        st = byid[name.replace(" ", "")]
        st.font.size = Pt(12); st.font.bold = True; st.font.italic = name == "Heading 3"
        st.paragraph_format.space_before = Pt(12); st.paragraph_format.space_after = Pt(0)
        st.paragraph_format.line_spacing_rule = WD_LINE_SPACING.DOUBLE
    byid["Title"].paragraph_format.alignment = 1
    for name in ("Footnote Text",):
        if name in [s.name for s in d.styles]:
            fs = {x.name: x for x in d.styles}[name]
            fs.font.size = Pt(10)
            fs.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
            fs.paragraph_format.space_after = Pt(6)
    from docx.enum.style import WD_STYLE_TYPE
    comp = {x.name: x for x in d.styles}.get("Compact")
    if comp is not None:  # table cells and tight lists: single spaced, 10 pt
        comp.font.size = Pt(10)
        comp.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
        comp.paragraph_format.space_before = Pt(0); comp.paragraph_format.space_after = Pt(0)
        comp.paragraph_format.first_line_indent = Pt(0)
    ph = d.styles.add_style("Placeholder", WD_STYLE_TYPE.CHARACTER)
    ph.font.highlight_color = 7  # yellow
    ph.font.bold = True
    d.core_properties.author = ""; d.core_properties.last_modified_by = ""
    d.save(path)


def footnotes_to_endnotes(docx: Path):
    """Convert pandoc footnotes into Word endnotes (JIP final checklist asks for endnotes),
    with Arabic numerals, placed at the end of the main section (before the bibliography)."""
    tmp = Path(tempfile.mkdtemp())
    with zipfile.ZipFile(docx) as z:
        z.extractall(tmp)
    w = tmp / "word"
    fn = w / "footnotes.xml"
    if fn.exists():
        x = fn.read_text(encoding="utf-8")
        x = x.replace("w:footnotes", "w:endnotes").replace("<w:footnote ", "<w:endnote ") \
             .replace("</w:footnote>", "</w:endnote>").replace("w:footnoteRef", "w:endnoteRef")
        (w / "endnotes.xml").write_text(x, encoding="utf-8"); fn.unlink()
        doc = (w / "document.xml").read_text(encoding="utf-8").replace("w:footnoteReference", "w:endnoteReference")
        (w / "document.xml").write_text(doc, encoding="utf-8")
        rels = (w / "_rels/document.xml.rels").read_text(encoding="utf-8")
        rels = rels.replace("relationships/footnotes", "relationships/endnotes").replace("footnotes.xml", "endnotes.xml")
        (w / "_rels/document.xml.rels").write_text(rels, encoding="utf-8")
        ct = (tmp / "[Content_Types].xml").read_text(encoding="utf-8")
        ct = ct.replace("/word/footnotes.xml", "/word/endnotes.xml").replace("footnotes+xml", "endnotes+xml")
        (tmp / "[Content_Types].xml").write_text(ct, encoding="utf-8")
    # Blank identifying document properties (JIP anonymity rule)
    core = tmp / "docProps/core.xml"
    if core.exists():
        c = core.read_text(encoding="utf-8")
        c = re.sub(r"<dc:creator>.*?</dc:creator>", "<dc:creator></dc:creator>", c, flags=re.S)
        c = re.sub(r"<cp:lastModifiedBy>.*?</cp:lastModifiedBy>", "<cp:lastModifiedBy></cp:lastModifiedBy>", c, flags=re.S)
        core.write_text(c, encoding="utf-8")
    docx.unlink()
    with zipfile.ZipFile(docx, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(tmp.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(tmp).as_posix())
    shutil.rmtree(tmp)


SETTINGS = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:settings xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            '<w:zoom w:percent="100"/><w:defaultTabStop w:val="720"/>'
            '<w:characterSpacingControl w:val="doNotCompress"/>{endnotes}'
            '<w:compat><w:compatSetting w:name="compatibilityMode" w:uri="http://schemas.microsoft.com/office/word" w:val="15"/></w:compat>'
            '<w:themeFontLang w:val="en-US"/></w:settings>')
ENDNOTE_PR = ('<w:endnotePr><w:pos w:val="sectEnd"/><w:numFmt w:val="decimal"/>'
              '<w:endnote w:id="-1"/><w:endnote w:id="0"/></w:endnotePr>')


PPR_ORDER = ["pStyle", "keepNext", "keepLines", "pageBreakBefore", "framePr", "widowControl", "numPr",
             "suppressLineNumbers", "pBdr", "shd", "tabs", "suppressAutoHyphens", "kinsoku", "wordWrap",
             "overflowPunct", "topLinePunct", "autoSpaceDE", "autoSpaceDN", "bidi", "adjustRightInd",
             "snapToGrid", "spacing", "ind", "contextualSpacing", "mirrorIndents", "suppressOverlap", "jc",
             "textDirection", "textAlignment", "textboxTightWrap", "outlineLvl", "divId", "cnfStyle", "rPr",
             "sectPr", "pPrChange"]
TCPR_ORDER = ["cnfStyle", "tcW", "gridSpan", "hMerge", "vMerge", "tcBorders", "shd", "noWrap", "tcMar",
              "textDirection", "tcFitText", "vAlign", "hideMark"]
TBLPR_ORDER = ["tblStyle", "tblpPr", "tblOverlap", "bidiVisual", "tblStyleRowBandSize", "tblStyleColBandSize", "tblW",
               "jc", "tblCellSpacing", "tblInd", "tblBorders", "shd", "tblLayout", "tblCellMar", "tblLook",
               "tblCaption", "tblDescription"]
STYLE_ORDER = ["name", "aliases", "basedOn", "next", "link", "autoRedefine", "uiPriority", "semiHidden",
               "unhideWhenUsed", "qFormat", "locked", "personal", "personalCompose", "personalReply", "rsid",
               "pPr", "rPr", "tblPr", "trPr", "tcPr", "tblStylePr"]


def _reorder(xml: str) -> str:
    from lxml import etree
    W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    root = etree.fromstring(xml.encode("utf-8"))
    for tag, order in (("pPr", PPR_ORDER), ("tcPr", TCPR_ORDER), ("style", STYLE_ORDER), ("tblPr", TBLPR_ORDER)):
        for el in root.iter(W + tag):
            kids = list(el)
            kids.sort(key=lambda k: order.index(etree.QName(k).localname) if etree.QName(k).localname in order else 99)
            for k in kids:
                el.remove(k); el.append(k)
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True).decode("utf-8")


def finalize_docx(docx: Path):
    """Write schema-ordered settings (pandoc's defaults fail OOXML validation), blank the
    author fields, and drop folder entries from the zip."""
    tmp = Path(tempfile.mkdtemp())
    with zipfile.ZipFile(docx) as z:
        z.extractall(tmp)
    has_end = (tmp / "word/endnotes.xml").exists()
    if has_end:
        from lxml import etree
        W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
        dp = tmp / "word/document.xml"
        root = etree.fromstring(dp.read_bytes())
        for sp in root.iter(W + "sectPr"):
            for old in sp.findall(W + "endnotePr"):
                sp.remove(old)
            pr = etree.SubElement(sp, W + "endnotePr")
            etree.SubElement(pr, W + "pos").set(W + "val", "sectEnd")
            etree.SubElement(pr, W + "numFmt").set(W + "val", "decimal")
            sp.remove(pr)
            idx = 0
            for i, k in enumerate(sp):
                if etree.QName(k).localname in ("headerReference", "footerReference", "footnotePr"):
                    idx = i + 1
            sp.insert(idx, pr)
        dp.write_bytes(etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True))
    for part in ("word/styles.xml", "word/document.xml"):
        st = tmp / part
        st.write_text(_reorder(st.read_text(encoding="utf-8")), encoding="utf-8")
    nb = tmp / "word/numbering.xml"
    if nb.exists():  # nsid must be 8 hex digits
        nb.write_text(re.sub(r'(<w:nsid w:val=")([0-9A-Fa-f]{1,7})(")', lambda m: m.group(1) + m.group(2).rjust(8, "0") + m.group(3),
                             nb.read_text(encoding="utf-8")), encoding="utf-8")
    (tmp / "word/settings.xml").write_text(SETTINGS.format(endnotes=ENDNOTE_PR if has_end else ""), encoding="utf-8")
    core = tmp / "docProps/core.xml"
    if core.exists():
        c = core.read_text(encoding="utf-8")
        c = re.sub(r"<dc:creator>.*?</dc:creator>", "<dc:creator></dc:creator>", c, flags=re.S)
        c = re.sub(r"<cp:lastModifiedBy>.*?</cp:lastModifiedBy>", "<cp:lastModifiedBy></cp:lastModifiedBy>", c, flags=re.S)
        core.write_text(c, encoding="utf-8")
    docx.unlink()
    with zipfile.ZipFile(docx, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(tmp.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(tmp).as_posix())
    shutil.rmtree(tmp)


def pandoc(md_text: str, out: Path, ref: Path):
    src = Path(tempfile.mkdtemp()) / "src.md"   # never next to the output: it could overwrite a source file
    src.write_text(md_text, encoding="utf-8")
    subprocess.run(["pandoc", str(src), "-f", "markdown+bracketed_spans+raw_attribute", "-o", str(out),
                    "--reference-doc", str(ref)], check=True)
    src.unlink()


# ------------------------------------------------------------------ tables & captions
def tables_docx(ref: Path, labels: dict[int, str]):
    from docx import Document
    from docx.shared import Pt
    d = Document(ref)
    for p in list(d.paragraphs):
        p._element.getparent().remove(p._element)
    T = RES / "tables"
    specs = [
        ("Table 1", "Initiatives in the analysed corpus, by jurisdiction",
         "Counts after the exclusions described in the Data section (subnational entries excluded). The 20 jurisdictions with the most initiatives are shown, plus India, the European Union institutions, and the United States if not already listed. The full list is in supplementary Table S1.",
         "TableS1_initiatives_by_jurisdiction.csv", None),
        ("Table 2", "Topics estimated by BERTopic, with the authors' labels and state-role coding",
         "Labels and roles were assigned by two authors independently after reading the ten highest-weighted terms and twenty representative initiatives per topic, and reconciled before any country comparison. Roles follow Papyshev and Yarime: development, control, promotion, or other. Share is the proportion of all initiatives assigned to the topic.",
         "Table2_topics.csv", ["topic", "label", "role", "top_terms", "n_docs", "share"]),
        ("Table 3", "Similarity of India's topic profile to the comparator units",
         "Cosine similarity (higher means more alike) and Jensen-Shannon distance (base 2; lower means more alike). Intervals come from a cluster bootstrap that resamples responsible organisations and draws shares from a Dirichlet distribution with a Jeffreys prior in each replicate (2,000 replicates). The permutation p-value tests whether India's and the comparator's initiatives could come from one topic distribution. EU27_POOLED: EU institutions and the 27 member states. NON_OECD_POOLED: all non-OECD jurisdictions except India.",
         "Table3_india_similarity.csv", ["label", "n_india", "n_comp", "cosine", "cosine_lo", "cosine_hi", "jsd", "jsd_lo", "jsd_hi", "perm_p_jsd"]),
        ("Table 4", "Theme-level differences between each benchmark and India",
         "Gap is the benchmark's share of initiatives on a theme minus India's share (posterior means under a Jeffreys prior); positive values mean India gives the theme less weight. Intervals from the smoothed cluster bootstrap. Themes on which neither side places at least two percent of its initiatives are not tested. Significance: Benjamini-Hochberg at a false discovery rate of 0.10 within each benchmark. Shown: all significant differences and all differences of at least three percentage points; the full table is supplementary file Table4_india_gaps.csv. GLOBAL_MEAN and NON_OECD_MEAN count each jurisdiction once.",
         "Table4_india_gaps.csv", ["benchmark_label", "label", "role", "india_share", "bench_share", "gap", "lo", "hi", "p_fdr", "sig_fdr"]),
        ("Table 5", "Policy instrument mix and binding status: India compared with each comparator",
         "Instruments were coded to Hood and Margetts' four tools by the rules in supplementary file instrument_map.csv. Chi-square tests compare the distribution across tools (Cramér's V as effect size). Binding shares are computed only over initiatives whose binding status is recorded; missing shares are reported. Fisher's exact test compares binding shares.",
         "Table5_instrument_tests.csv", ["label", "authority_share_india", "authority_share_comp", "chi2", "dof", "p", "cramers_v",
                                         "binding_share_india", "binding_share_comp", "binding_missing_india", "binding_missing_comp", "fisher_p_binding"]),
    ]
    first = True
    for num, title, note, fname, cols in specs:
        fp = T / fname
        if not first:
            d.add_page_break()
        first = False
        p = d.add_paragraph(); r = p.add_run(f"{num}. {title}"); r.bold = True
        if not fp.exists():
            d.add_paragraph("[RESULT PENDING: run run_all.py on the OECD.AI data]")
            d.add_paragraph(f"Note: {note}")
            continue
        df = pd.read_csv(fp)
        if num == "Table 1":
            focal = df[df.jurisdiction.isin(["India", "European Union", "United States"])]
            df = pd.concat([df.head(20), focal]).drop_duplicates("jurisdiction")
        if num == "Table 4":
            df = df[df["eligible"] & (df["sig_fdr"] | (df["gap"].abs() >= 0.03))].copy()
            df["label"] = df["label"].map(clean_label)
            df = df.sort_values(["benchmark", "gap"], ascending=[True, False])
        if cols:
            df = df[[c for c in cols if c in df.columns]]
        if "label" in df.columns:
            df["label"] = df["label"].map(clean_label)
        tbl = d.add_table(rows=1, cols=len(df.columns)); tbl.style = "Table"
        for i, c in enumerate(df.columns):
            tbl.rows[0].cells[i].text = str(c).replace("_", " ")
        for _, row in df.iterrows():
            cells = tbl.add_row().cells
            for i, v in enumerate(row):
                cells[i].text = (f"{v:.3f}" if isinstance(v, float) else str(v))
        for row in tbl.rows:
            for cell in row.cells:
                for pp in cell.paragraphs:
                    pp.paragraph_format.line_spacing = 1.0
                    for rr in pp.runs:
                        rr.font.size = Pt(9)
        d.add_paragraph(f"Note: {note}")
        d.add_paragraph("Source: Authors' analysis of the OECD.AI Policy Navigator.")
    d.save(OUT / "Tables.docx")


def captions_docx(ref: Path):
    from docx import Document
    d = Document(ref)
    for p in list(d.paragraphs):
        p._element.getparent().remove(p._element)
    caps = yaml.safe_load(open(MS / "figure_captions.yaml"))
    tok = tokens()
    for fig in caps:
        p = d.add_paragraph(); p.add_run(f"{fig['number']}. {fig['caption']}").bold = True
        d.add_paragraph(f"File: {fig['file']}.tif (also supplied as .jpg)")
        d.add_paragraph(f"<alt text> {fig['alt']}")
        ext, _ = fill(fig["extended"], tok)
        d.add_paragraph(f"<extended description> {re.sub(r'\{custom-style=.*?\}', '', ext)}")
        d.add_paragraph("")
    d.save(OUT / "Figure_captions_and_alt_text.docx")


# ------------------------------------------------------------------ online appendix
def md_table(df: pd.DataFrame) -> str:
    """Pipe table whose separator dash counts are proportional to content width, so pandoc
    gives each column a sensible share of the text width (equal short dashes squeeze them)."""
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            if np.isnan(v):
                return "–"
            return f"{int(v)}" if float(v).is_integer() and abs(v) >= 1 else f"{v:.3f}"
        return str(v)
    cols = list(df.columns)
    head = [str(c).replace("_", " ") for c in cols]
    rows = [[fmt(v).replace("|", "/") for v in r] for _, r in df.iterrows()]
    w = [max([len(head[i])] + [min(len(r[i]), 60) for r in rows]) for i in range(len(cols))]
    total = sum(w)
    dashes = [max(3, round(100 * x / total)) for x in w]
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("-" * d for d in dashes) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def _labels_with_examples() -> pd.DataFrame:
    lab = pd.read_csv(ROOT / "config/topic_labels.csv")[["topic", "label", "role", "alt_role", "n_docs"]]
    dt, cp = RES / "tables/doc_topics.csv", ROOT / "data/processed/corpus.csv"
    if dt.exists() and cp.exists():
        d = pd.read_csv(dt).merge(pd.read_csv(cp)[["doc_id", "title"]], on="doc_id", how="left")
        ex = {k: "; ".join(g.sample(min(3, len(g)), random_state=2026)["title"].astype(str).str.slice(0, 70))
              for k, g in d.groupby("topic") if k != -1}
        lab["three sampled initiatives"] = lab["topic"].map(ex)
    return lab.rename(columns={"alt_role": "alternative role", "n_docs": "initiatives"})


def appendix_md() -> str:
    cfg = yaml.safe_load(open(ROOT / "config/config.yaml"))
    b, l, sm = cfg["bertopic"], cfg["lda"], cfg["similarity"]
    hp = pd.DataFrame([
        ("Sentence encoder", b["embedding_model"]), ("UMAP", ", ".join(f"{k}={v}" for k, v in b["umap"].items())),
        ("HDBSCAN", ", ".join(f"{k}={v}" for k, v in b["hdbscan"].items())),
        ("c-TF-IDF vectorizer", f"ngram {cfg['text']['ngram_range']}, min_df={b['vectorizer']['min_df']}, max_df={b['vectorizer']['max_df']}, reduce_frequent_words=True"),
        ("Outliers", "reassigned by c-TF-IDF similarity (sensitivity: excluded)"),
        ("Seeds", ", ".join(map(str, b["seeds"])) + f" (primary {b['primary_seed']})"),
        ("LDA", f"K grid {l['k_grid']}, passes={l['passes']}, iterations={l['iterations']}, alpha={l['alpha']}, eta={l['eta']}, seeds {l['seeds']}"),
        ("Bootstrap", f"{sm['bootstrap_reps']} replicates; cluster = {cfg.get('bootstrap_cluster_field')}; Dirichlet(counts + 0.5) draw per replicate"),
        ("Permutation test", f"{sm['permutation_reps']} permutations"),
        ("Gap test", f"BH-FDR q = {cfg['gaps']['fdr_q']}; minimum share {cfg['gaps']['min_benchmark_share']}"),
        ("Clustering", f"{cfg['clustering']['method']} linkage on Jensen-Shannon distance; k chosen by silhouette over {cfg['clustering']['k_range']}; 200 bootstrap re-clusterings"),
        ("Sample", f"min text {cfg['sample']['min_chars']} characters; min {cfg['sample']['min_initiatives_per_country']} initiatives for clustering; English only"),
    ], columns=["Component", "Setting"])
    imap = pd.read_csv(ROOT / "config/instrument_map.csv")[["pattern", "nato_tool", "rationale"]]
    SR = ROOT / "simulations/results"
    parts = ["---\ntitle: \"Online Appendix\"\n---\n",
             "**Mapping Global AI Governance: A Topic-Modeling Analysis of National AI Policies Using the OECD.AI Policy Database**\n",
             "# Appendix A. Model and inference settings, topic labels, and role coding\n", "All settings are read from config/config.yaml, which is hashed into every run's output. Before topic modeling, place names, nationalities, languages, and currency codes are replaced by a neutral token (spaCy en_core_web_sm entities GPE, NORP, LOC, and LANGUAGE, plus an exact list of country names and demonyms; src/govtopics/mask.py).\n", md_table(hp),
             "\n**Topic labels and state roles.** The alternative role is used only in the robustness check of Table S17b. Three initiative titles per topic are sampled at random (seed 2026) so that readers can judge each label.\n",
             md_table(_labels_with_examples()),
             "\n# Appendix B. Instrument codebook\n",
             "The Navigator assigns each initiative one type from a fixed list. Each type is mapped to one of Hood and Margetts' tools by the table below (config/instrument_type_map.csv). For the three generic types (other initiatives, project, programme), the keyword rules that follow are applied to the title. \n",
             md_table(pd.read_csv(ROOT / "config/instrument_type_map.csv").rename(columns={"initiative_type": "initiative type", "nato_tool": "tool"})),
             "\n**Keyword rules for generic types.**\n",
             md_table(imap)]
    parts.append("\n# Appendix C. Simulation study\n")
    try:
        naive = pd.read_csv(SR / "naive_power_summary.csv"); pw = pd.read_csv(SR / "power_summary.csv")
        cov = json.loads((SR / "power_summary.json").read_text())
        rec = pd.read_csv(SR / "recovery.csv")
        role = pd.read_csv(SR / "role_power_summary.csv") if (SR / "role_power_summary.csv").exists() else None
        parts += [
            "**Design.** Each replication draws a benchmark topic profile over fifteen topics from a flat Dirichlet distribution. India's profile equals the benchmark except on three topics, where India's share is lower by delta, with the removed mass spread proportionally over the other topics. Initiatives are drawn independently from each profile. The gap test of the main text is applied (bootstrap intervals, Benjamini-Hochberg at q = 0.10). Each condition has "
            f"{cov['reps_per_cell']} replications with {cov['bootstrap_reps']} bootstrap replicates. Power is the share of true gaps flagged; FDR is the mean false discovery proportion; P(any false) is the probability of at least one false flag. Code: simulations/power_study.py.\n",
            "**Study 1a. Plain percentile bootstrap (rejected).** Under the null (delta = 0), with 25 Indian and 150 benchmark initiatives, the procedure flagged at least one false gap in "
            f"{pct(naive.query('n_india == 25 and n_bench == 150 and delta == 0')['p_any_false'].iloc[0])} of replications. The cause is that a theme with no Indian initiatives has zero bootstrap variance on the Indian side.\n",
            md_table(naive.rename(columns={"fdr": "FDR", "p_any_false": "P(any false)"})),
            "\n**Study 1b. Smoothed bootstrap (adopted).** Each replicate resamples initiatives and draws shares from Dirichlet(counts + 0.5).\n",
            md_table(pw.rename(columns={"fdr": "FDR", "p_any_false": "P(any false)"})),
            "\n**Study 2. Interval coverage under clustering.** Initiatives come in programmes of four that share a topic tendency (Dirichlet around the unit's profile with total concentration 5). 60 Indian and 120 benchmark initiatives; true gap delta = 0.08 on three topics. Mean coverage of nominal 95 percent intervals across all topics: initiative-level bootstrap "
            f"{pct(cov['coverage']['coverage_initiative'])} (mean width {cov['coverage']['width_initiative']:.3f}); cluster bootstrap {pct(cov['coverage']['coverage_cluster'])} (mean width {cov['coverage']['width_cluster']:.3f}). Both smoothed procedures are conservative; the cluster version is the more conservative and is used in the main analysis.\n",
        ]
        if role is not None:
            parts += ["**Study 3. Role-level test (E1).** The fifteen topics are grouped into three roles of five; all three true gaps fall in the control role, so the role-level gap is roughly three times delta. Benchmark of 150 initiatives, 100 replications per condition. Detection means the 95 percent interval for the control-role difference excludes zero.\n",
                      md_table(role), "\n"]
        parts += ["**Study 4. Text-level recovery.** Five synthetic corpora of 700 initiatives with ten planted themes (tests/make_fixture.py; every row is labelled SYNTHETIC). The BERTopic specification of the main text is run with an offline TF-IDF/SVD embedding in place of the sentence encoder, so the check covers the clustering, outlier, and c-TF-IDF stages only. Synthetic texts are far cleaner than policy summaries; this confirms the pipeline works as specified and says nothing about accuracy on real data. Code: simulations/recovery_study.py.\n",
                  md_table(rec)]
    except FileNotFoundError:
        parts.append("Run simulations/power_study.py, simulations/power_study.py --roles, and simulations/recovery_study.py.")
    parts.append("\n# Appendix D. Inventory of major Indian AI instruments (coverage audit)\n")
    parts.append("Each instrument was checked against the source listed on 21 November 2026. Matching terms (config/india_inventory.csv) are used to search India's database entries; each candidate match is then read to decide whether it is the instrument in question. The audit result is in Table S18 of the replication files.\n")
    inv = pd.read_csv(ROOT / "config/india_inventory.csv")[["instrument", "year", "issuer", "type", "binding", "source_url"]]
    parts.append(md_table(inv))
    ap = RES / "tables/TableS18_india_coverage_audit.csv"
    if ap.exists() and (RES / "summary.json").exists():
        sm_ = json.loads((RES / "summary.json").read_text())
        if sm_["run"].get("embedding_backend") == "sbert" and not sm_["run"].get("fast_mode"):
            aud = pd.read_csv(ap)[["instrument", "auto_match", "best_title", "best_title_similarity"]]
            ac = RES / "coverage_analyst_check.csv"
            if ac.exists():
                aud = aud.merge(pd.read_csv(ac)[["instrument", "analyst_check"]].rename(columns={"analyst_check": "result of reading the candidate"}), on="instrument", how="left")
            parts += ["\n**Audit result.**\n", md_table(aud)]
    parts.append("\n# Appendix E. Supplementary tables\n\nTables S1 to S18 are supplied as CSV files in results/tables/ of the replication package: S1 initiatives by jurisdiction; S2 seed stability; S2b India's similarity by seed; S3 country topic profiles; S4 cosine matrix; S5 Jensen-Shannon matrix; S6 cluster membership; S7 silhouette; S8 instrument shares; S9 LDA grid; S10 LDA topics; S11 sensitivity; S12 periods; S13 field missingness; S14 co-clustering with India; S15 cluster-wise Jaccard stability; S16 initiative-level bootstrap; S17 role shares; S17b role shares under the alternative coding of borderline topics; S18 India coverage audit.\n")
    return "\n".join(parts)


# ------------------------------------------------------------------ main
def main():
    OUT.mkdir(exist_ok=True)
    ref = OUT / "_reference.docx"
    make_reference_docx(ref)
    tok = tokens()
    pending = 0
    if (MS / "manuscript.md").exists():  # manuscript text is not part of the public repository
        md, pending = fill((MS / "manuscript.md").read_text(encoding="utf-8"), tok)
        body = md.split("# Introduction", 1)[-1].split("# Notes", 1)[0]
        body = re.sub(r"\^\[(?:[^\[\]]|\[[^\]]*\])*\]", "", body)
        body = re.sub(r"\[(Table|Figure) \d+ about here\]", "", body)
        tok["BODY_WORDS"] = f"{len(re.findall(r'[A-Za-z0-9][^\s]*', re.sub(r'[#*]', ' ', body))):,}"
        pandoc(md, OUT / "Manuscript_anonymous.docx", ref)
        footnotes_to_endnotes(OUT / "Manuscript_anonymous.docx")
    if (MS / "cover_page.md").exists():  # not part of the public replication copy
        cover, _ = fill((MS / "cover_page.md").read_text(encoding="utf-8"), tok)
        pandoc(cover, OUT / "Title_page.docx", ref)
    pandoc(appendix_md(), OUT / "Online_Appendix.docx", ref)
    if (MS / "cover_letter.md").exists():
        letter, _ = fill((MS / "cover_letter.md").read_text(encoding="utf-8"), tok)
        pandoc(letter, OUT / "Cover_letter.docx", ref)
    labels = {}
    tables_docx(ref, labels)
    captions_docx(ref)
    figs = OUT / "figures"; figs.mkdir(exist_ok=True)
    for f in (RES / "figures").glob("Figure*.tif"):
        shutil.copy(f, figs / f.name)
    rv = ROOT / "review/Review_report.md"
    if rv.exists():
        pandoc(rv.read_text(encoding="utf-8"), ROOT / "review/Review_report.docx", ref)
        finalize_docx(ROOT / "review/Review_report.docx")
    ref.unlink()
    for f in OUT.glob("*.docx"):
        finalize_docx(f)
    print(f"Built submission/ . Placeholders still open in manuscript: {pending}")


if __name__ == "__main__":
    main()
