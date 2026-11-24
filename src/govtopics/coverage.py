"""Coverage audit: how much of India's documented AI governance appears in the database?

config/india_inventory.csv lists major Indian AI instruments, each checked against a
primary or authoritative source. For every item we search India's corpus entries for
its match terms and rank candidates by title similarity. The output has a 'confirmed'
column that the authors fill by reading the candidate entries (yes / no). Coverage is
reported from confirmed matches when available, otherwise from automatic matches,
and the summary records which.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd


def audit(df: pd.DataFrame, inventory_path: str, india_label: str, out_path: Path) -> dict:
    inv = pd.read_csv(inventory_path)
    india = df[df["jurisdiction"] == india_label]
    rows = []
    for _, it in inv.iterrows():
        rx = re.compile(it["match_terms"], re.I)
        cands = []
        for _, d in india.iterrows():
            blob = f"{d['title']} {d['text']}"
            if rx.search(blob):
                score = SequenceMatcher(None, str(it["instrument"]).lower(), str(d["title"]).lower()).ratio()
                cands.append((score, d["doc_id"], str(d["title"])[:120]))
        cands.sort(reverse=True)
        best = cands[0] if cands else (0.0, None, "")
        rows.append({"instrument": it["instrument"], "year": it["year"], "type": it["type"],
                     "binding": it["binding"], "auto_match": bool(cands), "n_candidates": len(cands),
                     "best_doc_id": best[1], "best_title": best[2], "best_title_similarity": round(best[0], 3),
                     "other_candidate_ids": ";".join(str(c[1]) for c in cands[1:6]), "source_url": it["source_url"]})
    res = pd.DataFrame(rows)
    if out_path.exists():  # keep the authors' confirmations across re-runs
        prev = pd.read_csv(out_path)
        if "confirmed" in prev.columns:
            res = res.merge(prev[["instrument", "confirmed"]], on="instrument", how="left")
    if "confirmed" not in res.columns:
        res["confirmed"] = ""
    res.to_csv(out_path, index=False)
    conf = res["confirmed"].fillna("").astype(str).str.strip().str.lower()
    manual = conf.isin(["yes", "no"]).all() and len(res) > 0
    covered = conf.eq("yes") if manual else res["auto_match"]
    return {"n_inventory": int(len(res)), "n_covered": int(covered.sum()), "share_covered": float(covered.mean()),
            "basis": "confirmed by authors" if manual else "automatic matching (not yet confirmed)",
            "missing": res.loc[~covered, "instrument"].tolist()}
