"""Load an OECD.AI Policy Navigator export and build the analysis corpus.

The export format has changed over time (the Policy Navigator replaced the older
EC/OECD database of national AI policies), so columns are resolved through the
aliases in config/config.yaml rather than hard-coded.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

INTERNATIONAL_ORGS = {
    "oecd", "unesco", "united nations", "un", "g7", "g20", "council of europe",
    "gpai", "global partnership on ai", "world bank", "itu", "african union",
    "asean", "nato", "wto", "imf", "inter-american development bank", "idb",
    "european council of europe", "international", "multilateral",
}

# Spelling variants seen in OECD.AI exports and in user-assembled files.
JURISDICTION_ALIASES = {
    "united states of america": "United States", "usa": "United States", "us": "United States",
    "u.s.": "United States", "united states": "United States",
    "eu": "European Union", "european commission": "European Union",
    "european union": "European Union", "european parliament": "European Union",
    "uk": "United Kingdom", "great britain": "United Kingdom", "united kingdom": "United Kingdom",
    "korea": "Korea", "republic of korea": "Korea", "south korea": "Korea", "korea, republic of": "Korea",
    "turkey": "Türkiye", "turkiye": "Türkiye", "türkiye": "Türkiye",
    "viet nam": "Viet Nam", "vietnam": "Viet Nam",
    "russian federation": "Russia", "czech republic": "Czechia", "czechia": "Czechia",
    "slovak republic": "Slovakia", "india": "India",
}

SUBNATIONAL_RE = re.compile(r"\(\s*subnational\s*:", re.I)


def read_any(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No raw data at {path}. Download the OECD.AI Policy Navigator export "
            "(see README, section 'Getting the data') and save it there.")
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".json":
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return pd.json_normalize(data if isinstance(data, list) else data.get("data", data))
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode {path}")


def resolve_columns(df: pd.DataFrame, aliases: dict[str, list[str]]) -> dict[str, str | None]:
    lower = {c.lower().strip(): c for c in df.columns}
    out: dict[str, str | None] = {}
    for field, names in aliases.items():
        out[field] = next((lower[n.lower()] for n in names if n.lower() in lower), None)
    missing = [f for f in ("title", "description", "jurisdiction") if out[f] is None]
    if missing:
        raise KeyError(
            f"Required fields {missing} not found. Columns present: {list(df.columns)}. "
            "Add the right header names to config/config.yaml under 'columns'.")
    return out


def normalise_jurisdiction(value: str) -> str:
    v = re.sub(r"\s+", " ", str(value)).strip()
    return JURISDICTION_ALIASES.get(v.lower(), v)


_EXTRA_INTL = None


def _extra_intl() -> set[str]:
    """Names scraped by scripts/collect_oecd.py from the Navigator's international-organisation pages."""
    global _EXTRA_INTL
    if _EXTRA_INTL is None:
        f = Path(__file__).resolve().parents[2] / "data" / "raw" / "international_orgs.txt"
        names = {x.strip().lower() for x in f.read_text(encoding="utf-8").splitlines()} if f.exists() else set()
        # never treat the EU or a recognised country as an international organisation
        names -= {"european union", "eu"} | {k for k in JURISDICTION_ALIASES} | {v.lower() for v in JURISDICTION_ALIASES.values()}
        _EXTRA_INTL = {n for n in names if n}
    return _EXTRA_INTL


def is_international(j: str) -> bool:
    return j.lower() in INTERNATIONAL_ORGS or j.lower() in _extra_intl()


def clean_text(s: str) -> str:
    s = "" if pd.isna(s) else str(s)
    s = re.sub(r"<[^>]+>", " ", s)               # stray HTML
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)  # URLs
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def detect_language(text: str) -> str:
    try:
        from langdetect import DetectorFactory, detect
        DetectorFactory.seed = 0
        return detect(text)
    except Exception:  # langdetect missing or text too short
        return "unknown"


def build_corpus(cfg: dict, log: list[str]) -> pd.DataFrame:
    raw = read_any(cfg["paths"]["raw"])
    log.append(f"raw_rows={len(raw)}")
    cols = resolve_columns(raw, cfg["columns"])
    log.append("column_map=" + json.dumps(cols))

    df = pd.DataFrame({k: raw[v] if v else pd.NA for k, v in cols.items()})

    # Multi-country rows ("France; Germany") are split so each jurisdiction is counted once.
    df["jurisdiction"] = df["jurisdiction"].fillna("").astype(str).str.split(r"\s*[;|]\s*")
    df = df.explode("jurisdiction", ignore_index=True)
    df["jurisdiction"] = df["jurisdiction"].map(normalise_jurisdiction)

    s = cfg["sample"]
    # Subnational entries are flagged, not dropped, so a sensitivity run can include them
    # under their national unit. The main analysis excludes them (config: exclude_subnational).
    sub = df["title"].fillna("").astype(str).str.contains(SUBNATIONAL_RE) | \
        df["jurisdiction"].fillna("").astype(str).str.contains(SUBNATIONAL_RE)
    df["subnational"] = sub.values
    df["jurisdiction"] = df["jurisdiction"].fillna("").astype(str).str.replace(r"\s*\(\s*subnational\s*:[^)]*\)", "",
                                                                    regex=True, flags=re.I).str.strip()
    df["jurisdiction"] = df["jurisdiction"].map(normalise_jurisdiction)
    log.append(f"flagged_subnational={int(sub.sum())}")
    n1 = len(df)
    if s["exclude_international_orgs"]:
        df = df[~df["jurisdiction"].map(is_international)]
        log.append(f"dropped_international={n1 - len(df)}")

    parts = [df["title"].map(clean_text), df["description"].map(clean_text),
             df["objectives"].map(clean_text)]
    df["text"] = (parts[0] + ". " + parts[1] + " " + parts[2]).str.strip(" .")
    n2 = len(df)
    df = df[df["text"].str.len() >= s["min_chars"]]
    log.append(f"dropped_short={n2 - len(df)}")

    # Duplicate entries (same jurisdiction, same text) are removed.
    df["text_hash"] = df["text"].str.lower().map(lambda t: hashlib.sha1(t.encode()).hexdigest())
    n3 = len(df)
    df = df.drop_duplicates(["jurisdiction", "text_hash"])
    log.append(f"dropped_duplicates={n3 - len(df)}")

    df["lang"] = df["text"].map(detect_language)
    if s["english_only"]:
        n4 = len(df)
        df = df[df["lang"].isin(["en", "unknown"])]
        log.append(f"dropped_non_english={n4 - len(df)}")

    df["text_raw"] = df["text"]
    if cfg["text"].get("mask_geography", False):
        from . import mask
        masked, info = mask.mask_texts(df["text"].tolist(), extra_names=df["jurisdiction"].unique())
        df["text"] = masked
        log.append(f"masked_entities={info['entities_masked']} masked_list_terms={info['list_matches_masked']}")

    st = df["status"].fillna("").astype(str).str.lower()
    df["is_active"] = np.where(st.str.contains("inactive|ended|closed|revoked|repealed|completed|expired|discontinued"),
                               False, np.where(st.str.contains("active|ongoing|in force|in progress|implement"), True, np.nan))
    df["text_chars"] = df["text"].str.len()
    if df["id"].isna().all():
        df["id"] = df["text_hash"].str[:12]
    df["doc_id"] = range(len(df))
    df["start_year"] = pd.to_numeric(
        df["start_year"].fillna("").astype(str).str.extract(r"(\d{4})")[0], errors="coerce")
    log.append(f"final_docs={len(df)} jurisdictions={df['jurisdiction'].nunique()}")
    return df.reset_index(drop=True)
