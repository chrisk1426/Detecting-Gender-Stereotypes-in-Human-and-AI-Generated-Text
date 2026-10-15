"""Load StereoSet (public dev set), keep gender items, and reshape to one labeled sentence per row.

gold_label: 0 = anti-stereotype, 1 = stereotype, 2 = unrelated (dropped).
Intersentence items: the context sentence is joined to the candidate continuation.
Intrasentence items: the candidate already has the blank filled in.
Each context becomes one pair, with pair_id shared by its stereotype and anti-stereotype sentence.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"


def load_stereoset_gender():
    rows = []
    for part in ("intersentence", "intrasentence"):
        df = pd.read_parquet(RAW / f"stereoset_{part}_validation.parquet")
        for _, r in df[df["bias_type"] == "gender"].iterrows():
            pair_id = f"ss_{part[:5]}_{r['id']}"
            for sent, gold in zip(r["sentences"]["sentence"], r["sentences"]["gold_label"]):
                if gold == 2:
                    continue
                text = f"{r['context']} {sent}" if part == "intersentence" else sent
                rows.append({"pair_id": pair_id, "sentence": text.strip(),
                             "label": int(gold), "source": f"stereoset_{part[:5]}"})
    return pd.DataFrame(rows)
