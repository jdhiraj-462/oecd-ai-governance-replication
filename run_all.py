#!/usr/bin/env python3
"""Run the full analysis for 'Mapping Global AI Governance'.

    python run_all.py              # full analysis (downloads the SBERT model once)
    python run_all.py --fast       # fewer bootstrap/permutation draws, for a first look
    pytest -q                      # tests on SYNTHETIC data

Every number in the manuscript is written to results/summary.json and inserted
into the manuscript by build_manuscript.py. Nothing is typed by hand.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import kruskal, spearmanr

sys.path.insert(0, str(Path(__file__).parent / "src"))
from govtopics import compare, coverage, figures, instruments, load, topics, validation  # noqa: E402

ROLE_KEYWORDS = {  # used ONLY if the authors have not coded roles; result is flagged provisional
    "control": r"regulat|ethic|risk|right|oversight|law|legal|privacy|safety|standard|trust|accountab|transparen|liabil|audit|certif",
    "development": r"research|compute|skill|education|talent|data|infrastructure|universit|scien|train|phd|supercomput",
    "promotion": r"adoption|sme|business|industr|startup|enterpris|sector|health|agricultur|public service|manufactur|invest",
}


def pair_cos(jv, tv, cfg, comps=("European Union", "United States", "EU27_POOLED"), smoothing=None):
    S = cfg["sample"]
    sm = cfg["similarity"]["smoothing"] if smoothing is None else smoothing
    keep = tv != -1
    jv, tv = np.asarray(jv)[keep], np.asarray(tv)[keep]
    if len(tv) == 0:
        return {}
    tv, K, _ = compare._recode(tv)
    out = {"n_india": int((jv == S["india_label"]).sum())}
    a = tv[jv == S["india_label"]]
    for c in comps:
        b = tv[np.isin(jv, compare.unit_members(c, cfg, jv))]
        key = {"European Union": "eu", "United States": "us", "EU27_POOLED": "eu27"}.get(c, c)
        out[key] = np.nan if len(a) < 5 or len(b) < 5 else compare.cosine(compare._profile(a, K, sm), compare._profile(b, K, sm))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--embedding-backend", default="sbert", choices=["sbert", "tfidf-svd"])
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--skip-lda", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    cfg = yaml.safe_load(open(args.config))
    if args.fast:
        cfg["similarity"]["bootstrap_reps"] = 200
        cfg["similarity"]["permutation_reps"] = 300
        cfg["lda"]["k_grid"] = cfg["lda"]["k_grid"][:3]
        cfg["lda"]["seeds"] = cfg["lda"]["seeds"][:1]
        cfg["lda"]["passes"] = 5
    R = Path(cfg["paths"]["results"]); P = Path(cfg["paths"]["processed"])
    T, F, M, V = R / "tables", R / "figures", R / "models", R / "validation"
    for d in (P, T, F, M, V):
        d.mkdir(parents=True, exist_ok=True)
    S = cfg["sample"]; IN, EU, US = S["india_label"], S["eu_label"], S["us_label"]
    summary: dict = {"run": {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "python": platform.python_version(),
                             "embedding_backend": args.embedding_backend, "fast_mode": args.fast,
                             "config": args.config,
                             "config_sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest()}}

    # 1. Corpus -------------------------------------------------------------------
    log: list[str] = []
    df_all = load.build_corpus(cfg, log)
    summary["run"]["raw_sha256"] = hashlib.sha256(Path(cfg["paths"]["raw"]).read_bytes()).hexdigest()
    df_all = instruments.classify(df_all, instruments.load_map("config/instrument_map.csv"),
                                  instruments.load_type_map("config/instrument_type_map.csv"))
    df_all["doc_id"] = range(len(df_all))
    main_mask = ~df_all["subnational"] if S["exclude_subnational"] else np.ones(len(df_all), bool)
    df = df_all[main_mask].reset_index(drop=True)
    log.append(f"main_sample_docs={len(df)} (subnational excluded={S['exclude_subnational']})")
    (R / "sample_log.txt").write_text("\n".join(log) + "\n")
    df_all.to_csv(P / "corpus_all.csv", index=False)

    per_j = df["jurisdiction"].value_counts()
    summary["corpus"] = {
        "n_docs": int(len(df)), "n_jurisdictions": int(df["jurisdiction"].nunique()),
        "n_india": int(per_j.get(IN, 0)), "n_eu": int(per_j.get(EU, 0)), "n_us": int(per_j.get(US, 0)),
        "n_eu27_pooled": int(df["jurisdiction"].isin(compare.unit_members("EU27_POOLED", cfg, df["jurisdiction"])).sum()),
        "n_non_oecd_jurisdictions": int(sum(1 for j in per_j.index if j not in set(cfg["groups"]["oecd_members"])
                                            and j not in set(cfg["groups"]["eu27_members"]) and j not in (EU, IN))),
        "n_subnational_flagged": int(df_all["subnational"].sum()),
        "n_subnational_india": int((df_all["subnational"] & (df_all["jurisdiction"] == IN)).sum()),
        "median_per_jurisdiction": float(per_j.median()),
        "year_min": float(df["start_year"].min()) if df["start_year"].notna().any() else None,
        "year_max": float(df["start_year"].max()) if df["start_year"].notna().any() else None,
        "share_unclassified_instrument": float((df["nato_tool"] == "unclassified").mean()),
        "share_status_known": float(df["is_active"].notna().mean()),
        "log": log,
    }
    per_j.rename_axis("jurisdiction").reset_index(name="n_initiatives").to_csv(T / "TableS1_initiatives_by_jurisdiction.csv", index=False)

    # Text-length diagnostic (R2.2)
    foc = df[df["jurisdiction"].isin([IN, EU, US])]
    groups_len = [g["text_chars"].values for _, g in foc.groupby("jurisdiction") if len(g) > 1]
    kw = kruskal(*groups_len) if len(groups_len) >= 2 else None
    summary["length"] = {"median_chars": foc.groupby("jurisdiction")["text_chars"].median().to_dict(),
                         "median_all": float(df["text_chars"].median()),
                         "kruskal_H": float(kw.statistic) if kw else None, "kruskal_p": float(kw.pvalue) if kw else None}
    miss = compare.field_missingness(df, [IN, EU, US])
    miss.to_csv(T / "TableS13_field_missingness.csv")
    summary["missingness"] = miss.loc[[x for x in (IN, EU, US, "ALL") if x in miss.index]].round(3).to_dict(orient="index")

    # Cluster-bootstrap groups (R5.1): responsible organisation; missing values become singletons.
    cf = cfg.get("bootstrap_cluster_field")
    if cf in df and df[cf].notna().any() and (df[cf].fillna("").astype(str).str.strip() != "").any():
        org = df[cf].fillna("").astype(str).str.strip().str.lower()
        groups = np.where(org.isin(["", "nan", "none", "<na>"]), "doc_" + df["doc_id"].astype(str), df["jurisdiction"] + "|" + org)
        summary["bootstrap"] = {"type": "cluster (responsible organisation)", "n_clusters": int(len(set(groups)))}
    else:
        groups = None
        summary["bootstrap"] = {"type": "initiative-level (no organisation field)", "n_clusters": None}

    # 2. Embeddings + BERTopic -----------------------------------------------------
    texts_all = df_all["text"].tolist()
    ep = M / f"emb_{args.embedding_backend}_{hashlib.sha1(''.join(texts_all).encode()).hexdigest()[:10]}.npy"
    emb_all = np.load(ep) if ep.exists() else topics.embed(texts_all, cfg, args.embedding_backend)
    np.save(ep, emb_all)
    emb = emb_all[np.asarray(main_mask)]
    texts = df["text"].tolist()
    seed = cfg["bertopic"]["primary_seed"]
    model, tops, n_out_raw = topics.fit_bertopic(texts, emb, cfg, seed)
    df["topic"] = tops
    ttab = topics.topic_table(model, tops)

    label_file = Path(cfg["paths"].get("topic_labels", "config/topic_labels.csv"))
    user_labels, user_roles = {}, {}
    if label_file.exists():
        lab = pd.read_csv(label_file)
        for r in lab.itertuples():
            if isinstance(getattr(r, "label", None), str) and r.label.strip():
                user_labels[int(r.topic)] = r.label.strip()
            if isinstance(getattr(r, "role", None), str) and r.role.strip().lower() in ("development", "control", "promotion", "other"):
                user_roles[int(r.topic)] = r.role.strip().lower()
        summary["labels_file_sha256"] = hashlib.sha256(label_file.read_bytes()).hexdigest()
    else:
        ttab[["topic", "n_docs", "top_terms"]].assign(label="", role="").to_csv(label_file, index=False)
    labels = {int(r.topic): user_labels.get(int(r.topic), f"T{r.topic}: " + ", ".join(r.top_terms.split(", ")[:3]))
              for r in ttab.itertuples()}
    import re as _re
    roles = {}
    for r in ttab.itertuples():
        if int(r.topic) in user_roles:
            roles[int(r.topic)] = user_roles[int(r.topic)]; continue
        sc = {k: len(_re.findall(v, r.top_terms.lower())) for k, v in ROLE_KEYWORDS.items()}
        roles[int(r.topic)] = max(sc, key=sc.get) if max(sc.values()) > 0 else "other"
    ttab["label"] = ttab["topic"].map(labels); ttab["role"] = ttab["topic"].map(roles)
    ttab.to_csv(T / "Table2_topics.csv", index=False)
    summary["labels_from_authors"] = len(user_labels) >= len(set(tops) - {-1}) > 0
    summary["roles_from_authors"] = len(user_roles) >= len(set(tops) - {-1}) > 0

    tokens = topics.tokenize(texts, cfg)
    bwords = topics.bertopic_words_as_tokens(model, tops)
    stab_rows, seed_pairs = [], []
    assign = {seed: tops}
    for s_ in cfg["bertopic"]["seeds"]:
        if s_ != seed:
            assign[s_] = topics.fit_bertopic(texts, emb, cfg, s_)[1]
        seed_pairs.append({"seed": s_, "n_topics": len(set(assign[s_]) - {-1}), **pair_cos(df["jurisdiction"].values, assign[s_], cfg)})
    from itertools import combinations
    from sklearn.metrics import adjusted_rand_score
    for a, b in combinations(assign, 2):
        stab_rows.append({"seed_a": a, "seed_b": b, "ari": adjusted_rand_score(assign[a], assign[b])})
    stab = pd.DataFrame(stab_rows); stab.to_csv(T / "TableS2_bertopic_seed_stability.csv", index=False)
    seedtab = pd.DataFrame(seed_pairs); seedtab.to_csv(T / "TableS2b_india_similarity_by_seed.csv", index=False)
    seedtab["closer"] = np.sign(seedtab["eu"] - seedtab["us"])
    summary["bertopic"] = {
        "n_topics": int(len(set(tops) - {-1})), "outlier_share_before_reduction": n_out_raw / len(tops),
        "c_v": topics.coherence(bwords, tokens, "c_v"), "c_npmi": topics.coherence(bwords, tokens, "c_npmi"),
        "seed_ari_mean": float(stab["ari"].mean()) if len(stab) else None,
        "seed_ari_min": float(stab["ari"].min()) if len(stab) else None,
        "seed_k_range": [int(seedtab["n_topics"].min()), int(seedtab["n_topics"].max())],
        "seed_cos_eu": [float(seedtab["eu"].min()), float(seedtab["eu"].median()), float(seedtab["eu"].max())],
        "seed_cos_us": [float(seedtab["us"].min()), float(seedtab["us"].median()), float(seedtab["us"].max())],
        "seed_same_ordering": int((seedtab["closer"] == seedtab["closer"].iloc[0]).sum()), "n_seeds": int(len(seedtab)),
    }
    model.save(str(M / "bertopic_primary"), serialization="pickle")
    df[["doc_id", "id", "jurisdiction", "topic"]].to_csv(T / "doc_topics.csv", index=False)
    df.to_csv(P / "corpus.csv", index=False)

    # Global theme structure (R4.4): which themes are most widespread across jurisdictions
    ct = compare.counts_from_assignments(df["jurisdiction"], tops)
    breadth = (ct > 0).mean(axis=0).sort_values(ascending=False)
    summary["global_themes"] = [{"label": labels[int(k)], "share_of_jurisdictions": float(v),
                                 "share_of_initiatives": float((tops == k).mean()), "role": roles[int(k)]}
                                for k, v in breadth.items()]
    role_global = pd.Series([roles.get(int(x), "other") for x in tops[tops != -1]]).value_counts(normalize=True)
    summary["global_role_shares"] = role_global.round(4).to_dict()

    # 3. Profiles, similarity, clusters ------------------------------------------------
    big = ct.index[ct.sum(axis=1) >= S["min_initiatives_per_country"]].tolist()
    for must in (IN, EU, US):
        if must in ct.index and must not in big:
            big.append(must)
    prof = compare.normalise(ct.loc[big], cfg["similarity"]["smoothing"])
    prof.to_csv(T / "TableS3_country_topic_profiles.csv")
    cos, jsd = compare.similarity_matrices(prof)
    cos.to_csv(T / "TableS4_cosine_similarity.csv"); jsd.to_csv(T / "TableS5_jsd.csv")
    members, sil, Z = compare.cluster_countries(prof, jsd, cfg)
    members.to_csv(T / "TableS6_clusters.csv", index=False); sil.to_csv(T / "TableS7_silhouette.csv", index=False)
    order = figures.dendro(Z, list(prof.index), F)
    figures.similarity_heatmap(cos, order, F)
    cl = dict(zip(members["jurisdiction"], members["cluster"]))
    co, jac = compare.cluster_stability(df["jurisdiction"].values, tops, list(prof.index), members, cfg, IN,
                                        reps=50 if args.fast else 200)
    co.to_csv(T / "TableS14_coclustering_with_india.csv", index=False); jac.to_csv(T / "TableS15_cluster_jaccard.csv", index=False)
    summary["clusters"] = {
        "n_countries_clustered": int(len(prof)), "best_k": int(members["cluster"].nunique()),
        "best_silhouette": float(sil["silhouette"].max()) if len(sil) else None,
        "india_cluster": cl.get(IN), "eu_cluster": cl.get(EU), "us_cluster": cl.get(US),
        "india_cluster_members": [k for k, v in cl.items() if v == cl.get(IN) and k != IN],
        "sizes": members["cluster"].value_counts().sort_index().to_dict(),
        "india_stable_peers": co[co.iloc[:, 1] >= 0.7]["jurisdiction"].tolist() if IN in prof.index else [],
        "india_cluster_jaccard": float(jac.loc[jac.cluster == cl.get(IN), "mean_jaccard"].iloc[0]) if cl.get(IN) else None,
        "mean_jaccard_all": float(jac["mean_jaccard"].mean()),
        "focal_units_below_threshold": [u for u in (IN, EU, US) if per_j.get(u, 0) < S["min_initiatives_per_country"]],
        "cluster_members": {int(c): sorted(g["jurisdiction"].tolist()) for c, g in members.groupby("cluster")},
    }
    if IN in cos.index:
        nn = cos.loc[IN].drop(IN).sort_values(ascending=False)
        summary["india_nearest"] = [{"jurisdiction": k, "cosine": float(v)} for k, v in nn.head(5).items()]
        summary["reference_percentile"] = {u: compare.reference_percentile(jsd, IN, u) for u in (EU, US) if u in jsd.index}

    # 4. India vs comparators ---------------------------------------------------------
    comp, diff = compare.india_comparisons(df["jurisdiction"], tops, cfg, groups=groups)
    comp.to_csv(T / "Table3_india_similarity.csv", index=False)
    comp_iid, _ = compare.india_comparisons(df["jurisdiction"], tops, cfg, groups=None, seed=7)
    comp_iid.to_csv(T / "TableS16_india_similarity_initiative_bootstrap.csv", index=False)
    summary["india_pairs"] = comp.to_dict(orient="records"); summary["india_eu_minus_us"] = diff

    gaps = compare.india_gaps(df["jurisdiction"], tops, cfg, groups=groups)
    if not gaps.empty:
        gaps["label"] = gaps["topic"].map(labels); gaps["role"] = gaps["topic"].map(roles)
        gaps.to_csv(T / "Table4_india_gaps.csv", index=False)
        figures.gap_plot(gaps, labels, F)
        cols = ["label", "role", "gap", "lo", "hi", "india_share", "bench_share"]
        summary["gaps"] = {b: gaps[(gaps.benchmark == b) & gaps.sig_fdr & (gaps.gap > 0)].sort_values("gap", ascending=False)[cols]
                           .to_dict(orient="records") for b in cfg["gaps"]["benchmarks"]}
        summary["overweights"] = {b: gaps[(gaps.benchmark == b) & gaps.sig_fdr & (gaps.gap < 0)].sort_values("gap")[cols]
                                  .to_dict(orient="records") for b in cfg["gaps"]["benchmarks"]}
        summary["n_tests"] = {b: int(gaps[(gaps.benchmark == b) & gaps.eligible].shape[0]) for b in cfg["gaps"]["benchmarks"]}
    rs = compare.role_shares(df["jurisdiction"], tops, roles, cfg, groups=groups)
    rs.to_csv(T / "TableS17_role_shares.csv", index=False)
    summary["role_shares"] = rs.to_dict(orient="records")
    # Robustness of role results to the coding of borderline topics (alt_role column, fixed with the labels)
    if label_file.exists() and "alt_role" in pd.read_csv(label_file).columns:
        lab2 = pd.read_csv(label_file)
        alt_roles = {int(r.topic): (r.alt_role if isinstance(r.alt_role, str) and r.alt_role.strip() else roles[int(r.topic)])
                     for r in lab2.itertuples() if int(r.topic) in roles}
        rs_alt = compare.role_shares(df["jurisdiction"], tops, alt_roles, cfg, groups=groups)
        rs_alt.to_csv(T / "TableS17b_role_shares_alternative_coding.csv", index=False)
        summary["role_shares_alt"] = rs_alt.to_dict(orient="records")
        summary["n_topics_recoded_alt"] = int(sum(alt_roles[k] != roles[k] for k in roles))
    summary["roles_provisional"] = not summary["roles_from_authors"]

    inst_tests, inst_shares = compare.instrument_contrast(df, cfg)
    inst_tests.to_csv(T / "Table5_instrument_tests.csv", index=False); inst_shares.to_csv(T / "TableS8_instrument_shares.csv")
    figures.instrument_plot(inst_shares, F)
    summary["instruments"] = {"tests": inst_tests.to_dict(orient="records"), "shares": inst_shares.round(4).to_dict(orient="index")}

    # 5. Sensitivity -----------------------------------------------------------------
    J = df["jurisdiction"].values
    sens = [{"variant": "primary specification", **pair_cos(J, tops, cfg)}]
    for mcs in (8, 20):
        c2 = copy.deepcopy(cfg); c2["bertopic"]["hdbscan"]["min_cluster_size"] = mcs
        sens.append({"variant": f"HDBSCAN minimum cluster size {mcs}", **pair_cos(J, topics.fit_bertopic(texts, emb, c2, seed)[1], cfg)})
    c3 = copy.deepcopy(cfg); c3["bertopic"]["reduce_outliers"] = False
    sens.append({"variant": "outliers excluded, not reassigned", **pair_cos(J, topics.fit_bertopic(texts, emb, c3, seed)[1], cfg)})
    rec = (df["start_year"] >= 2021).values
    sens.append({"variant": "initiatives starting 2021 or later", **pair_cos(J[rec], tops[rec], cfg)})
    act = (df["is_active"] == True).values  # noqa: E712
    if act.sum() > 20 and df["is_active"].notna().mean() > 0.5:
        sens.append({"variant": "active initiatives only", **pair_cos(J[act], tops[act], cfg)})
    sens.append({"variant": "no smoothing of profiles", **pair_cos(J, tops, cfg, smoothing=1e-9)})
    if df_all["subnational"].any():
        sub_idx = np.where(df_all["subnational"].values)[0]
        t_sub, _ = model.transform([texts_all[i] for i in sub_idx], embeddings=emb_all[sub_idx])
        J2 = np.concatenate([J, df_all["jurisdiction"].values[sub_idx]]); T2 = np.concatenate([tops, np.array(t_sub)])
        sens.append({"variant": "subnational initiatives included", **pair_cos(J2, T2, cfg)})
    if "text_raw" in df_all and (df_all["text_raw"] != df_all["text"]).any():
        raw_all = df_all["text_raw"].tolist()
        ep_raw = M / f"emb_{args.embedding_backend}_{hashlib.sha1(''.join(raw_all).encode()).hexdigest()[:10]}.npy"
        emb_raw = np.load(ep_raw) if ep_raw.exists() else topics.embed(raw_all, cfg, args.embedding_backend)
        np.save(ep_raw, emb_raw)
        raw_main = df["text_raw"].tolist()
        t_raw = topics.fit_bertopic(raw_main, emb_raw[np.asarray(main_mask)], cfg, seed)[1]
        sens.append({"variant": "no masking of place names", **pair_cos(J, t_raw, cfg)})
    # Topics dominated by one jurisdiction encode that jurisdiction's programs as much as a theme
    ct = pd.crosstab(df["topic"], df["jurisdiction"])
    conc = (ct.max(axis=1) / ct.sum(axis=1))
    dom = [int(k) for k in conc[conc > 0.25].index if k != -1]
    summary["dominated_topics"] = {int(k): {"label": labels.get(int(k)), "top_jurisdiction": ct.loc[k].idxmax(), "share": round(float(conc[k]), 3)} for k in dom}
    if dom:
        keep_d = ~np.isin(tops, dom)
        sens.append({"variant": "topics dominated by one jurisdiction removed", **pair_cos(J[keep_d], tops[keep_d], cfg)})
    sens = pd.DataFrame(sens)
    sens["closer_to"] = np.where(sens["eu"] > sens["us"], "European Union", np.where(sens["us"] > sens["eu"], "United States", "undetermined"))
    sens.to_csv(T / "TableS11_sensitivity.csv", index=False)
    summary["sensitivity"] = {"rows": sens.to_dict(orient="records"), "n_variants": int(len(sens) - 1),
                              "n_same_ordering": int((sens["closer_to"].iloc[1:] == sens["closer_to"].iloc[0]).sum())}

    per = []
    for lo, hi, lab_ in [(None, 2020, "2020 and earlier"), (2021, 2023, "2021 to 2023"), (2024, None, "2024 and later")]:
        m = df["start_year"].notna()
        if lo is not None: m &= df["start_year"] >= lo
        if hi is not None: m &= df["start_year"] <= hi
        per.append({"period": lab_, **pair_cos(J[m.values], tops[m.values], cfg)})
    per = pd.DataFrame(per); per.to_csv(T / "TableS12_periods.csv", index=False)
    summary["periods"] = per.to_dict(orient="records")

    # 6. LDA robustness -------------------------------------------------------------------
    if not args.skip_lda:
        grid = topics.lda_grid(tokens, cfg); grid.to_csv(T / "TableS9_lda_grid.csv", index=False)
        k_best = int(grid.groupby("k")["c_v"].mean().idxmax())
        lda, theta = topics.fit_lda(tokens, k_best, cfg, cfg["lda"]["primary_seed"])
        lprof = compare.normalise(compare.counts_from_theta(df["jurisdiction"], theta).loc[prof.index], cfg["similarity"]["smoothing"])
        lcos, _ = compare.similarity_matrices(lprof)
        iu = np.triu_indices(len(prof), 1)
        rho, p_rho = spearmanr(cos.values[iu], lcos.loc[cos.index, cos.columns].values[iu])
        pd.DataFrame({"topic": range(k_best), "top_terms": [", ".join(w for w, _ in lda.show_topic(j, 10)) for j in range(k_best)]}) \
            .to_csv(T / "TableS10_lda_topics.csv", index=False)
        rob = {"k": k_best, "c_v": float(grid[grid.k == k_best]["c_v"].mean()),
               "spearman_similarity_bertopic_vs_lda": float(rho), "spearman_p": float(p_rho)}
        for key, u in (("india_eu_cosine_lda", EU), ("india_us_cosine_lda", US)):
            if IN in lcos.index and u in lcos.index:
                rob[key] = float(lcos.loc[IN, u])
        summary["lda_robustness"] = rob

    # 7. Coverage audit (R3.1) ------------------------------------------------------------
    summary["coverage"] = coverage.audit(df_all, "config/india_inventory.csv", IN, T / "TableS18_india_coverage_audit.csv")

    # 8. Validation sheets ----------------------------------------------------------------
    rng = np.random.default_rng(cfg["validation"]["seed"])
    tt = {int(r.topic): r.top_terms.split(", ") for r in ttab.itertuples() if r.topic != -1}
    wi_s, wi_k = validation.word_intrusion(tt, cfg["validation"]["intrusion_tasks_per_topic"], rng)
    lf_s, lf_k = validation.label_fit(df, {k: v for k, v in labels.items() if k != -1}, cfg["validation"]["sample_size"], rng)
    in_s, in_k = validation.instrument_sheet(df, cfg["validation"]["instrument_sample_size"], rng)
    for who in ("coder_A", "coder_B"):
        for name, sh in (("word_intrusion", wi_s), ("label_fit", lf_s), ("instrument", in_s)):
            p_ = V / f"{name}_{who}.csv"
            if not p_.exists():  # never overwrite coders' work
                sh.to_csv(p_, index=False)
    wi_k.to_csv(V / "KEY_word_intrusion.csv", index=False); lf_k.to_csv(V / "KEY_label_fit.csv", index=False)
    in_k.to_csv(V / "KEY_instrument.csv", index=False)
    summary["validation_items"] = {"word_intrusion": int(len(wi_s)), "label_fit": int(len(lf_s)), "instrument": int(len(in_s))}

    summary["run"]["seconds"] = round(time.time() - t0, 1)
    (R / "summary.json").write_text(json.dumps(summary, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(f"Done in {summary['run']['seconds']} s. Results in {R}/")


if __name__ == "__main__":
    main()
