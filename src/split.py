"""Assign every sentence to one of 5 cross-validation folds, keeping each pair together.

Pairs that share a sentence (the 5 StereoSet duplicates) are grouped too, so the same text never lands in
both train and test. Writes data/processed/folds.csv: pair_id, group, fold.
"""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
N_FOLDS = 5
SEED = 72


def pair_groups(df):
    """Union-find over pair_ids linked by a shared sentence."""
    parent = {p: p for p in df["pair_id"].unique()}

    def find(p):
        while parent[p] != p:
            parent[p] = parent[parent[p]]
            p = parent[p]
        return p

    for pids in df.groupby("sentence")["pair_id"].unique():
        for other in pids[1:]:
            parent[find(other)] = find(pids[0])
    return {p: find(p) for p in parent}


def make_folds(df):
    groups = df["pair_id"].map(pair_groups(df))
    folds = pd.Series(-1, index=df.index)
    splitter = GroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for k, (_, test_idx) in enumerate(splitter.split(df, df["label"], groups)):
        folds.iloc[test_idx] = k
    return pd.DataFrame({"pair_id": df["pair_id"], "group": groups, "fold": folds}).drop_duplicates("pair_id")


if __name__ == "__main__":
    df = pd.read_csv(ROOT / "data" / "processed" / "merged_gender.csv")
    folds = make_folds(df)
    out = ROOT / "data" / "processed" / "folds.csv"
    folds.to_csv(out, index=False)

    check = df.merge(folds, on="pair_id")
    assert check.groupby("sentence")["fold"].nunique().max() == 1, "a sentence appears in two folds"
    print(f"{folds['pair_id'].nunique()} pairs in {folds['group'].nunique()} groups -> {out.name}")
    print(check.pivot_table(index="fold", columns="label", values="sentence", aggfunc="size"))
