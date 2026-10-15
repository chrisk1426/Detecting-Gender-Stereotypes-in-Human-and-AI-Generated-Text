"""Merge CrowS-Pairs and StereoSet gender data into one table and sanity-check it."""
from pathlib import Path
import pandas as pd
from load_crows import load_crows_gender
from load_stereoset import load_stereoset_gender

ROOT = Path(__file__).resolve().parent.parent


def merge():
    merged = pd.concat([load_crows_gender(), load_stereoset_gender()], ignore_index=True)

    sizes = merged.groupby("pair_id")["label"].agg(["size", "nunique"])
    assert (sizes["size"] == 2).all() and (sizes["nunique"] == 2).all(), "every pair needs one 0 and one 1"
    assert merged["sentence"].str.strip().str.len().gt(0).all(), "empty sentence found"
    return merged


if __name__ == "__main__":
    df = merge()
    out = ROOT / "data" / "processed" / "merged_gender.csv"
    df.to_csv(out, index=False)
    print(f"{len(df)} sentences, {df['pair_id'].nunique()} pairs -> {out.name}\n")
    print(df.groupby("source")["label"].agg(sentences="size", stereotype="sum"))
    print("\noverall label balance:", df["label"].value_counts().to_dict())
    print("duplicate sentences (same text, any label):", df["sentence"].duplicated().sum())
    print("\nsamples:")
    for src in df["source"].unique():
        print(df[df["source"] == src].head(2)[["pair_id", "label", "sentence"]].to_string(index=False, max_colwidth=110))
