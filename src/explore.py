"""Explore the merged gender dataset: class balance, sentence length, and word frequencies by label.

Descriptive only. Do not use these results to choose features for the classifiers.
"""
import re
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"

FEMALE = {"she", "her", "hers", "herself", "woman", "women", "girl", "girls", "mother", "mom", "wife",
          "sister", "daughter", "grandmother", "lady", "ladies", "schoolgirl", "female"}
MALE = {"he", "him", "his", "himself", "man", "men", "boy", "boys", "father", "dad", "husband",
        "brother", "son", "grandfather", "gentleman", "gentlemen", "schoolboy", "male"}


def tokens(s):
    return re.findall(r"[a-z]+(?:'[a-z]+)?", s.lower())


def main():
    df = pd.read_csv(ROOT / "data" / "processed" / "merged_gender.csv")
    df["toks"] = df["sentence"].map(tokens)
    df["n_tokens"] = df["toks"].map(len)
    df["n_chars"] = df["sentence"].str.len()

    print("== Class balance ==")
    print(df["label"].value_counts().to_dict())
    print(df.groupby("source")["label"].agg(sentences="size", stereotype="sum"), "\n")

    print("== Sentence length (tokens) by label ==")
    print(df.groupby("label")["n_tokens"].describe().round(2))
    print("\nby source and label (mean tokens):")
    print(df.pivot_table(index="source", columns="label", values="n_tokens", aggfunc="mean").round(2))
    wide = df.pivot(index="pair_id", columns="label", values="n_tokens")
    diff = wide[1] - wide[0]
    print(f"\nWithin pairs, stereotype minus anti-stereotype length: mean {diff.mean():.2f} tokens; "
          f"longer in {(diff > 0).mean():.1%}, shorter in {(diff < 0).mean():.1%}, equal in {(diff == 0).mean():.1%}")
    by_src = diff.groupby(df.drop_duplicates("pair_id").set_index("pair_id")["source"]).mean().round(2)
    print("mean within-pair difference by source:", by_src.to_dict(), "\n")

    print("== Word frequencies by label (sentences containing the word) ==")
    df_pos = Counter(w for t in df.loc[df.label == 1, "toks"] for w in set(t))
    df_neg = Counter(w for t in df.loc[df.label == 0, "toks"] for w in set(t))
    n_pos, n_neg = (df.label == 1).sum(), (df.label == 0).sum()
    vocab = [w for w in set(df_pos) | set(df_neg) if df_pos[w] + df_neg[w] >= 8]
    table = pd.DataFrame({"word": vocab,
                          "stereo_count": [df_pos[w] for w in vocab],
                          "anti_count": [df_neg[w] for w in vocab]})
    table["log_odds"] = np.log((table.stereo_count + 1) / (n_pos + 2)) - np.log((table.anti_count + 1) / (n_neg + 2))
    table = table.sort_values("log_odds", ascending=False)
    table.to_csv(OUT / "word_log_odds.csv", index=False)
    print("Most stereotype-skewed (min 8 occurrences):")
    print(table.head(15).round(2).to_string(index=False))
    print("\nMost anti-stereotype-skewed:")
    print(table.tail(15).iloc[::-1].round(2).to_string(index=False))

    print("\n== Gendered words by label (share of sentences) ==")
    df["has_female"] = df["toks"].map(lambda t: bool(FEMALE & set(t)))
    df["has_male"] = df["toks"].map(lambda t: bool(MALE & set(t)))
    print(df.groupby("label")[["has_female", "has_male"]].mean().round(3))
    print("(sentences with neither:", f"{(~df.has_female & ~df.has_male).mean():.1%})")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    bins = np.arange(0, df["n_tokens"].max() + 3, 2)
    for lab, name in ((1, "stereotype"), (0, "anti-stereotype")):
        axes[0].hist(df.loc[df.label == lab, "n_tokens"], bins=bins, alpha=0.6, label=name)
    axes[0].set(xlabel="tokens per sentence", ylabel="sentences", title="Length by label")
    axes[0].legend()
    axes[1].hist(diff, bins=np.arange(-12.5, 13.5, 1), color="gray")
    axes[1].set(xlabel="stereotype minus anti-stereotype (tokens)", ylabel="pairs", title="Within-pair length difference")
    fig.tight_layout()
    fig.savefig(OUT / "length_by_label.png", dpi=150)
    print("\nsaved outputs/word_log_odds.csv and outputs/length_by_label.png")


if __name__ == "__main__":
    main()
