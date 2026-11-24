#!/usr/bin/env python3
"""Score the two human validation tasks once both coders have filled their sheets.

    python scripts/score_validation.py

Reads results/validation/{word_intrusion,label_fit}_coder_{A,B}.csv (column
'coder_answer' filled in) and the KEY files, writes validation_scores.json.
Word intrusion: coders type the intruder word. Label fit: coders type A, B, C or D.
Instrument coding: coders type authority, treasure, nodality or organization.
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from govtopics.validation import score  # noqa: E402

V = ROOT / "results/validation"
out = {
    "word_intrusion": score(pd.read_csv(V / "word_intrusion_coder_A.csv"), pd.read_csv(V / "word_intrusion_coder_B.csv"),
                            pd.read_csv(V / "KEY_word_intrusion.csv"), "intruder"),
    "label_fit": score(pd.read_csv(V / "label_fit_coder_A.csv"), pd.read_csv(V / "label_fit_coder_B.csv"),
                       pd.read_csv(V / "KEY_label_fit.csv"), "true_option"),
}
if (V / "instrument_coder_A.csv").exists() and (V / "instrument_coder_B.csv").exists():
    out["instrument_coding"] = score(pd.read_csv(V / "instrument_coder_A.csv"), pd.read_csv(V / "instrument_coder_B.csv"),
                                     pd.read_csv(V / "KEY_instrument.csv"), "auto_code")
(V / "validation_scores.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
