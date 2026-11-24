#!/usr/bin/env python3
"""Collect the OECD.AI Policy Navigator (no official export or API; checked 23 Nov 2026).

Each public list page (20 initiatives) embeds the full initiative records as Angular
state JSON (<script id="ng-state">). This script reads that JSON for every list page,
keeps the raw pages for provenance, and writes one row per initiative.

    python scripts/collect_oecd.py            # full collection (about 4 minutes)
    python scripts/collect_oecd.py --check    # first page only

Output: data/raw/oecd_ai_policy_navigator.csv, data/raw/html/list_*.html,
        data/raw/international_orgs.txt, data/raw/collection_log.json
Staff names and e-mail addresses in the records are not kept.
"""
from __future__ import annotations

import argparse
import html as htmlmod
import json
import re
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

BASE = "https://oecd.ai"
LIST_URL = BASE + "/en/dashboards/policy-initiatives?orderBy=startYearDesc&page={page}"
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
HTMLDIR = RAW / "html"
HEADERS = {"User-Agent": "Mozilla/5.0 (academic research replication)"}
DELAY = 1.5
STATE_RE = re.compile(r'<script id="ng-state" type="application/json">(.*?)</script>', re.S)


def strip_html(x) -> str:
    if not x:
        return ""
    x = re.sub(r"<[^>]+>", " ", str(x))
    return re.sub(r"\s+", " ", htmlmod.unescape(x)).strip()


def names(lst) -> str:
    return "; ".join(d.get("name", "") for d in (lst or []) if isinstance(d, dict))


def page_state(html: str) -> dict:
    m = STATE_RE.search(html)
    return json.loads(m.group(1)) if m else {}


def records_from(state: dict):
    """Return (records, paging, intl_orgs) from one list page's state."""
    recs, paging, intl = [], {}, []
    for v in state.values():
        b = v.get("b") if isinstance(v, dict) else None
        if isinstance(b, dict) and "lastPage" in b:
            paging = {k: b.get(k) for k in ("currentPage", "lastPage", "total", "perPage")}
            recs = [b[k] for k in b if k.isdigit()] or list(b.get("data") or [])
        elif isinstance(b, list) and b and isinstance(b[0], dict) and "yearFounded" in b[0]:
            intl = [o.get("name") for o in b if o.get("name")]
    return recs, paging, intl


def row(r: dict) -> dict:
    c = r.get("gaiinCountry") or {}
    io = r.get("intergovernmentalOrganisation") or {}
    it = r.get("initiativeType") or {}
    country = c.get("name") or ""
    return {
        "slug": r.get("slug"), "oecd_id": r.get("id"),
        "url": f"{BASE}/en/dashboards/policy-initiatives/{r.get('slug')}",
        "Name": r.get("englishName") or r.get("originalName") or "",
        "Original name": r.get("originalName") or "",
        "Summary": strip_html(r.get("description")),
        "Initiative overview": strip_html(r.get("overview")) or strip_html(r.get("description")),
        "Country": country or (io.get("name") or ""),
        "Country category": c.get("category") or ("International organisation" if io else ""),
        "International organisation": io.get("name") or "",
        "Category": r.get("category") or "",
        "Initiative type": it.get("name") or r.get("otherInitiativeType") or "",
        "Binding": r.get("extentBinding") or "",
        "Status": r.get("status") or "",
        "Start Year": r.get("startYear"), "End Year": r.get("endYear"),
        "Target Sectors": names(r.get("targetSectors")),
        "OECD AI Principles": names(r.get("principles")),
        "AI Tags": names(r.get("tags")),
        "Responsible organisation": strip_html(r.get("responsibleOrganisation")),
        "Budget": strip_html(r.get("budget")), "Budget available": r.get("budgetAvailable"),
        "Is evaluated": r.get("isEvaluated"),
        "Website": r.get("website") or "",
        "Added on": r.get("createdAt"), "Updated on": r.get("updatedAt"),
        "Editorial status": r.get("editorialStatus") or "",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    HTMLDIR.mkdir(parents=True, exist_ok=True)
    s = requests.Session(); s.headers.update(HEADERS)
    rows, intl, page, last, total = [], set(), 1, None, None
    while True:
        fp = HTMLDIR / f"list_{page:04d}.html"
        if not fp.exists():
            for attempt in range(4):
                try:
                    r = s.get(LIST_URL.format(page=page), timeout=60); r.raise_for_status(); break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(5 * (attempt + 1))
            fp.write_text(r.text, encoding="utf-8"); time.sleep(DELAY)
        recs, paging, io = records_from(page_state(fp.read_text(encoding="utf-8")))
        intl.update(io)
        last, total = paging.get("lastPage", last), paging.get("total", total)
        rows += [row(x) for x in recs]
        print(f"page {page}/{last}: {len(recs)} records (total so far {len(rows)})", flush=True)
        if a.check:
            print(json.dumps({k: (str(v)[:150]) for k, v in rows[0].items()}, indent=1, ensure_ascii=False)); return
        if not recs or (last and page >= last):
            break
        page += 1
    df = pd.DataFrame(rows).drop_duplicates("slug")
    out = RAW / "oecd_ai_policy_navigator.csv"
    df.to_csv(out, index=False)
    (RAW / "international_orgs.txt").write_text("\n".join(sorted(intl)), encoding="utf-8")
    log = {"date": date.today().isoformat(), "list_url": LIST_URL, "navigator_total": total,
           "pages": page, "n_rows": len(df),
           "binding_counts": df["Binding"].replace("", "missing").value_counts().to_dict(),
           "empty_overview": int((df["Initiative overview"] == "").sum()),
           "csv_sha256": __import__("hashlib").sha256(out.read_bytes()).hexdigest()}
    (RAW / "collection_log.json").write_text(json.dumps(log, indent=2))
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
