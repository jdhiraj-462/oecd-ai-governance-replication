"""Build a SYNTHETIC test file shaped like an OECD.AI export.

!!! These rows are invented. They exist only so the pipeline can be tested
!!! end to end without the real data. Never report anything computed from them.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

THEMES = {
    "compute": "public supercomputing compute capacity GPU cluster cloud infrastructure data centre access for researchers and startups",
    "skills": "skills training university curriculum scholarships doctoral fellowships talent attraction workforce reskilling",
    "regulation": "risk based regulation high risk systems conformity assessment obligations providers fundamental rights enforcement penalties",
    "ethics": "ethical principles trustworthy fairness transparency accountability human oversight voluntary guidelines code of conduct",
    "health": "health care diagnostics hospitals medical imaging patient data clinical decision support",
    "agri": "agriculture farmers crop yield weather advisory rural extension services soil monitoring",
    "publicsector": "public sector administration e-government citizen services automation procurement of systems by ministries",
    "research": "research centres excellence basic research grants collaboration universities industry partnerships",
    "security": "national security defence cyber security critical infrastructure resilience threat detection",
    "data": "data sharing open government data data spaces personal data protection privacy consent",
}
TYPES = {"compute": "Grants and funding", "skills": "Education and training", "regulation": "Regulation",
         "ethics": "Principles/guidelines/frameworks for trustworthy AI", "health": "Public sector pilot",
         "agri": "Grants and funding", "publicsector": "Public sector adoption", "research": "Research centre",
         "security": "National strategy", "data": "Data access and sharing regulation"}
JURIS = ["India", "European Union", "United States", "France", "Germany", "Japan", "Korea", "Canada",
         "Brazil", "Kenya", "Singapore", "United Kingdom", "Australia", "Chile", "Egypt", "Nigeria",
         "Indonesia", "Spain", "Italy", "Norway"]
ORGS = [f"Ministry {c}" for c in "ABCDEFGH"]
FILLER = "The measure will be implemented over several years with ministries agencies and stakeholders coordinating delivery and monitoring progress."


def main(out: str, n: int = 700, seed: int = 0):
    rng = np.random.default_rng(seed)
    keys = list(THEMES)
    # Each jurisdiction gets its own theme mix so there is structure to recover.
    mix = {j: rng.dirichlet(np.full(len(keys), 0.6)) for j in JURIS}
    mix["India"] = rng.dirichlet([3, 3, 0.2, 1.5, 3, 3, 2, 1, 0.5, 1])
    mix["European Union"] = rng.dirichlet([1, 1, 4, 3, 0.5, 0.3, 1, 1.5, 0.5, 3])
    mix["United States"] = rng.dirichlet([3, 1, 0.5, 1, 1, 0.3, 1.5, 3, 3, 0.7])
    rows = []
    for i in range(n):
        j = rng.choice(JURIS, p=np.r_[0.08, 0.08, 0.08, np.full(len(JURIS) - 3, 0.76 / (len(JURIS) - 3))])
        k = rng.choice(keys, p=mix[j])
        words = THEMES[k].split()
        body = " ".join(rng.choice(words, 25)) + ". " + FILLER
        sub = j in ("India", "United States") and rng.random() < 0.08
        rows.append({"Name": f"SYNTHETIC {k} initiative {i}" + (" (Subnational: State X)" if sub else ""),
                     "Initiative overview": body, "Organisation": str(rng.choice(ORGS)),
                     "Country": j, "Initiative type": TYPES[k],
                     "Binding": "Yes" if k in ("regulation", "data") and rng.random() < 0.7 else "No",
                     "Start Year": int(rng.integers(2017, 2026)),
                     "Status": "Active" if rng.random() < 0.8 else "Inactive"})
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {n} SYNTHETIC rows to {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tests/fixture_SYNTHETIC.csv")
