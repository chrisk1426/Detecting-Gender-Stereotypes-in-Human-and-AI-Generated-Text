"""Load CrowS-Pairs, keep the gender pairs, and reshape to one labeled sentence per row.

Label: sent_more (the more stereotype-consistent sentence) -> 1, sent_less -> 0.
Each pair shares a pair_id so a train/test split can keep both halves together.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def load_crows_gender(path=ROOT / "data" / "raw" / "crows_pairs_anonymized.csv"):
    raw = pd.read_csv(path).rename(columns={"Unnamed: 0": "orig_idx"})
    gender = raw[raw["bias_type"] == "gender"]

    rows = []
    for _, r in gender.iterrows():
        pair_id = f"crows_{r['orig_idx']}"
        rows.append({"pair_id": pair_id, "sentence": r["sent_more"], "label": 1, "source": "crows"})
        rows.append({"pair_id": pair_id, "sentence": r["sent_less"], "label": 0, "source": "crows"})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = load_crows_gender()
    print(f"{len(df)} sentences from {df['pair_id'].nunique()} pairs (merge.py writes the combined CSV)")
    print(df["label"].value_counts().to_dict())
    print(df.head(4).to_string())
