"""Human validation of topics (after Ying, Montgomery and Stewart 2022).

Two tasks are exported as coding sheets for two independent coders:
  1. Word intrusion: five top terms of a topic plus one intruder term that is
     high in another topic and low in this one. Coders mark the intruder.
  2. Label fit: an initiative's text and four candidate topic labels (the
     assigned label plus three random others). Coders pick the best fit.
Answer keys are written to a separate file the coders never see.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def word_intrusion(topic_terms: dict[int, list[str]], reps: int, rng) -> tuple[pd.DataFrame, pd.DataFrame]:
    sheet, key = [], []
    tids = list(topic_terms)
    for t in tids:
        own = topic_terms[t]
        for r in range(reps):
            others = [o for o in tids if o != t]
            if not others:
                break
            donor = rng.choice(others)
            pool = [w for w in topic_terms[donor][:5] if w not in own]
            if not pool or len(own) < 5:
                continue
            intruder = rng.choice(pool)
            words = list(rng.choice(own[:8], 5, replace=False)) + [intruder]
            rng.shuffle(words)
            item = f"WI-{t}-{r}"
            sheet.append({"item": item, **{f"w{i+1}": w for i, w in enumerate(words)}, "coder_answer": ""})
            key.append({"item": item, "topic": t, "intruder": intruder})
    return pd.DataFrame(sheet), pd.DataFrame(key)


def label_fit(docs: pd.DataFrame, labels: dict[int, str], n: int, rng) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = docs[docs["topic"] != -1]
    d = d.sample(min(n, len(d)), random_state=int(rng.integers(1e9)))
    sheet, key = [], []
    tids = list(labels)
    for _, row in d.iterrows():
        others = [t for t in tids if t != row["topic"]]
        cands = [row["topic"]] + list(rng.choice(others, min(3, len(others)), replace=False))
        rng.shuffle(cands)
        item = f"LF-{row['doc_id']}"
        sheet.append({"item": item, "text": row["text"][:1500],
                      **{f"option_{chr(65+i)}": labels[c] for i, c in enumerate(cands)}, "coder_answer": ""})
        key.append({"item": item, "true_topic": row["topic"],
                    "true_option": chr(65 + cands.index(row["topic"]))})
    return pd.DataFrame(sheet), pd.DataFrame(key)


def krippendorff_alpha_nominal(a: list, b: list) -> float:
    """Krippendorff's alpha, nominal data, two coders, no missing values."""
    vals = sorted(set(a) | set(b))
    ix = {v: i for i, v in enumerate(vals)}
    o = np.zeros((len(vals), len(vals)))
    for x, y in zip(a, b):
        o[ix[x], ix[y]] += 1; o[ix[y], ix[x]] += 1
    n_c = o.sum(axis=0); n = n_c.sum()
    if n <= 1:
        return float("nan")
    d_o = (n - np.trace(o)) / n
    d_e = (n * n - (n_c ** 2).sum()) / (n * (n - 1))
    return float(1 - d_o / d_e) if d_e > 0 else float("nan")


def score(sheet_a: pd.DataFrame, sheet_b: pd.DataFrame, key: pd.DataFrame, truth_col: str) -> dict:
    from sklearn.metrics import cohen_kappa_score
    m = key.merge(sheet_a[["item", "coder_answer"]].rename(columns={"coder_answer": "a"}), on="item") \
           .merge(sheet_b[["item", "coder_answer"]].rename(columns={"coder_answer": "b"}), on="item")
    m = m.dropna(subset=["a", "b"])
    m = m[(m["a"].fillna("").astype(str) != "") & (m["b"].fillna("").astype(str) != "")]
    a, b, t = m["a"].fillna("").astype(str).str.strip(), m["b"].fillna("").astype(str).str.strip(), m[truth_col].fillna("").astype(str)
    return {"n_items": int(len(m)), "accuracy_coder_a": float((a == t).mean()),
            "accuracy_coder_b": float((b == t).mean()), "cohen_kappa": float(cohen_kappa_score(a, b)),
            "krippendorff_alpha": krippendorff_alpha_nominal(list(a), list(b))}


def instrument_sheet(df: pd.DataFrame, n: int, rng) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reliability check of the rule-based NATO coding: coders assign each sampled initiative
    to authority / treasure / nodality / organization without seeing the automatic code."""
    d = df.sample(min(n, len(df)), random_state=int(rng.integers(1e9)))
    sheet = pd.DataFrame({"item": [f"IN-{x}" for x in d["doc_id"]], "title": d["title"].fillna("").astype(str).str[:200],
                          "instrument_type_in_database": d["instrument"].fillna("").astype(str),
                          "text": d["text"].str[:1200], "coder_answer": ""})
    key = pd.DataFrame({"item": sheet["item"], "auto_code": d["nato_tool"].values})
    return sheet, key
