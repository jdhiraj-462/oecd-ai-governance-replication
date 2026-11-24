"""Map OECD.AI initiative types onto Hood and Margetts' NATO tools of government.

The mapping in config/instrument_map.csv is an author-defined codebook. It is
applied to the 'instrument' field first and, if that is empty, to the title.
Every assignment records which rule fired so it can be audited.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd


def load_map(path: str) -> pd.DataFrame:
    m = pd.read_csv(path)
    m["regex"] = m["pattern"].map(lambda p: re.compile(p, re.I))
    return m


def load_type_map(path: str) -> dict[str, str]:
    """Exact codebook for the Navigator's controlled list of initiative types."""
    m = pd.read_csv(path)
    return {str(a).strip().lower(): b for a, b in zip(m["initiative_type"], m["nato_tool"])}


def classify(df: pd.DataFrame, rules: pd.DataFrame, type_map: dict[str, str] | None = None) -> pd.DataFrame:
    tools, rule_hit, source = [], [], []
    type_map = type_map or {}
    for inst, title in zip(df["instrument"].fillna(""), df["title"].fillna("")):
        hit = None
        exact = type_map.get(str(inst).strip().lower())
        if exact and exact != "fallback":
            tools.append(exact); rule_hit.append("exact:" + str(inst)); source.append("instrument"); continue
        # generic types ("Project", "Other ... initiatives") fall back to the title
        fields = (("title", str(title)),) if exact == "fallback" else (("instrument", str(inst)), ("title", str(title)))
        for field_name, text in fields:
            for _, r in rules.iterrows():
                if text and r["regex"].search(text):
                    hit = (r["nato_tool"], r["pattern"], field_name)
                    break
            if hit:
                break
        hit = hit or ("unclassified", "", "")
        tools.append(hit[0]); rule_hit.append(hit[1]); source.append(hit[2])
    out = df.copy()
    out["nato_tool"] = tools
    out["nato_rule"] = rule_hit
    out["nato_source"] = source
    b = out["binding"].fillna("").astype(str).str.strip().str.lower()
    nonb = b.str.contains(r"non|not\b|^no$|^no\b|false|^0$|voluntary")
    yes = b.str.contains(r"^yes|binding|true|^1$|mandatory") & ~nonb
    out["binding_status"] = np.where(yes, "binding", np.where(nonb, "non-binding", "missing"))
    out["is_binding"] = out["binding_status"] == "binding"
    return out
