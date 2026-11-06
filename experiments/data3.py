"""Combined dataset for the 3-class judge and the gender filter, with the same 5 grouped folds as before.

Rows:
  labeled     merged_gender.csv (1,518), label 1 stereotype / 0 anti-stereotype, folds from folds.csv
  old_neutral neutral_generated_clean.csv: the first neutral set after the LLM check, label 2, in its pair's fold
  v2          neutral_v2.csv (gen_neutral_v2.py): kinds listed there; new groups get folds by a seeded shuffle

Columns: sentence, label (0 anti, 1 stereo, 2 neutral), gender (1 refers to a gender, 0 not), kind, group, fold, pair_id.
Test sentences (ood_test.csv) are never in here.
"""
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EXT = ROOT / "data" / "external"


def load():
    folds = pd.read_csv(ROOT / "data" / "processed" / "folds.csv")
    lab = pd.read_csv(ROOT / "data" / "processed" / "merged_gender.csv").merge(folds, on="pair_id")
    lab = lab.assign(gender=1, kind="labeled_" + lab["source"])
    old = pd.read_csv(EXT / "neutral_generated_clean.csv").merge(folds, on="pair_id")
    old = old.assign(label=2, gender=1, kind="old_neutral")
    v2 = pd.read_csv(EXT / "neutral_v2.csv").rename(columns={"label3": "label"})
    groups = v2["group"].unique()
    rng = np.random.default_rng(72)
    fold_of = dict(zip(groups, rng.permutation(np.arange(len(groups)) % 5)))
    v2 = v2.assign(fold=v2["group"].map(fold_of), pair_id=v2["group"])
    cols = ["sentence", "label", "gender", "kind", "group", "fold", "pair_id"]
    df = pd.concat([lab[cols], old[cols], v2[cols]], ignore_index=True)
    df = df[~df["sentence"].duplicated(keep="first")].reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = load()
    print(df.pivot_table(index="kind", columns="label", values="sentence", aggfunc="size", fill_value=0))
    print(df.pivot_table(index="fold", columns="label", values="sentence", aggfunc="size"))
