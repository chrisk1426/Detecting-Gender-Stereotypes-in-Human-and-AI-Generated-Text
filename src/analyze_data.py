"""Dataset analysis behind DATANOTES.md: folds, pair structure, length, gender words, vocabulary,
discriminative words (with a pair-flip permutation null), label quality, and domain coverage.

Needs data/processed/merged_gender.csv and folds.csv (run merge.py and split.py first), plus
data/raw/crows_pairs_anonymized.csv and the StereoSet parquet files (run download_stereoset.py) for the
label-quality section. Prints everything to stdout and writes outputs/word_zscores.csv.
Descriptive only: do not use these results to choose features. Fixed seed, so output is reproducible.
"""
import ast
import collections
import difflib
import math
import re
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
SOURCES = ["crows", "stereoset_inter", "stereoset_intra"]
rng = np.random.default_rng(72)

# Hand-built lexicons. Possessives and plurals are normalised before matching. Names are not covered.
FEMALE = set("she her hers herself woman women girl girls mother mommy mom wife sister daughter grandmother grandma "
             "lady ladies schoolgirl female gal girlfriend aunt niece queen actress waitress mum nana maid bride wives "
             "mothers sisters daughters grandmothers moms".split())
MALE = set("he him his himself man men boy boys father dad daddy husband brother son grandfather grandpa gentleman "
           "gentlemen schoolboy male guy guys boyfriend uncle nephew king actor waiter bridegroom husbands fathers "
           "brothers sons grandfathers dads males".split())
OCCUPATIONS = set("doctor nurse engineer teacher professor lawyer secretary programmer scientist pilot chef cook manager "
                  "boss ceo cashier driver mechanic plumber surgeon dentist firefighter police officer soldier janitor "
                  "librarian receptionist accountant banker farmer carpenter artist writer author student waitress waiter "
                  "athlete coach principal babysitter housekeeper tailor".split())


def tokens(s):
    return re.findall(r"[a-z]+(?:'[a-z]+)?", s.lower())


def header(title):
    print(f"\n##### {title}")


def normalise(tok):
    tok = re.sub(r"'s$", "", tok.lower())
    return tok[:-1] if tok.endswith("s") and tok[:-1] in FEMALE | MALE else tok


def gender_class(sentence):
    toks = {normalise(t) for t in re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", sentence)}
    f, m = bool(FEMALE & toks), bool(MALE & toks)
    return "both" if f and m else "F" if f else "M" if m else "none"


def has_name_like_token(sentence):
    words = re.findall(r"[A-Za-z']+", sentence)[1:]
    return any(w[0].isupper() and w not in ("I", "I'm", "I'll", "I've", "I'd") for w in words) or bool(
        re.match(r"[A-Z][a-z]+ (was|is|had|has|went|wanted|did|got|couldn't|didn't)", sentence))


def chi2_2x2(a, b, c, d):
    n = a + b + c + d
    chi = sum((o - r * k / n) ** 2 / (r * k / n) for o, r, k in
              [(a, a + b, a + c), (b, a + b, b + d), (c, c + d, a + c), (d, c + d, b + d)])
    return chi, math.erfc(math.sqrt(chi / 2))


def sign_flip(diffs, n_perm=20000):
    obs = diffs.mean()
    signs = rng.choice([-1, 1], size=(n_perm, len(diffs)))
    return obs, (np.abs((signs * diffs).mean(1)) >= abs(obs) - 1e-12).mean()


def folds_section(df, folds):
    header("Folds")
    m = df.merge(folds, on="pair_id")
    print(m.pivot_table(index="fold", columns="label", values="sentence", aggfunc="size"))
    print(m.pivot_table(index="fold", columns="source", values="sentence", aggfunc="size"))
    print(f"groups: {folds.group.nunique()}, pairs: {folds.pair_id.nunique()}, "
          f"groups with more than one pair: {(folds.groupby('group').size() > 1).sum()}")


def pair_structure_section(df):
    header("Pair edit structure (token-level difference between the two sentences of a pair)")
    rows = []
    for pid, g in df.groupby("pair_id"):
        a, b = g[g.label == 1].iloc[0], g[g.label == 0].iloc[0]
        sm = difflib.SequenceMatcher(None, a.toks, b.toks, autojunk=False)
        n_diff, swaps = 0, []
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op != "equal":
                n_diff += max(i2 - i1, j2 - j1)
                if op == "replace" and i2 - i1 == 1 and j2 - j1 == 1:
                    swaps.append((a.toks[i1], b.toks[j1]))
        rows.append(dict(source=a.source, n_diff=n_diff, swaps=swaps))
    pairs = pd.DataFrame(rows)
    for s in SOURCES:
        x = pairs[pairs.source == s]
        print(f"{s}: pairs={len(x)} median={x.n_diff.median()} mean={x.n_diff.mean():.2f} "
              f"==1: {(x.n_diff == 1).mean():.1%} <=2: {(x.n_diff <= 2).mean():.1%} <=3: {(x.n_diff <= 3).mean():.1%}")
    for s in ("crows", "stereoset_intra"):
        swaps = collections.Counter(sw for l in pairs[pairs.source == s].swaps for sw in l)
        print(f"{s} top (stereotype word, anti word) swaps:", swaps.most_common(12))


def length_section(df):
    header("Sentence length")
    print(df.groupby("source").n_tokens.describe().round(2))
    print(df.groupby(["source", "label"]).n_tokens.mean().round(2).unstack())
    print("mean characters by source:", df.groupby("source").n_chars.mean().round(1).to_dict(),
          "| overall:", round(df.n_chars.mean(), 1))
    wide = df.pivot(index="pair_id", columns="label", values="n_tokens")
    diffs = (wide[1] - wide[0]).values
    src = df.drop_duplicates("pair_id").set_index("pair_id").source.reindex(wide.index).values
    print("paired diff (stereotype - anti), all pairs: mean=%.3f p=%.3f, ties %.1f%%"
          % (*sign_flip(diffs), (diffs == 0).mean() * 100))
    for s in SOURCES:
        x = diffs[src == s]
        print(f"  {s}: mean=%.3f p=%.3f ties=%.1f%% longer=%.1f%% shorter=%.1f%%"
              % (*sign_flip(x), (x == 0).mean() * 100, (x > 0).mean() * 100, (x < 0).mean() * 100))


def gender_section(df):
    header("Gender words")
    df["gender_class"] = df.sentence.map(gender_class)
    print(pd.crosstab([df.source, df.label], df.gender_class, normalize="index").round(3))
    wide = df.pivot(index="pair_id", columns="label", values="gender_class")
    wide["source"] = df.drop_duplicates("pair_id").set_index("pair_id").source
    for s in SOURCES:
        x = wide[wide.source == s]
        fm, mf = int(((x[1] == "F") & (x[0] == "M")).sum()), int(((x[1] == "M") & (x[0] == "F")).sum())
        same = int((x[1] == x[0]).sum())
        n, k = fm + mf, min(fm, mf)
        p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else float("nan")
        print(f"{s}: stereotype=F & anti=M {fm}; stereotype=M & anti=F {mf}; same class {same}; "
              f"other {len(x) - fm - mf - same}; exact sign test p={p:.3f}")
    crows = df[df.source == "crows"]
    for name, cls in (("female-only", "F"), ("male-only", "M")):
        a = int(((crows.label == 1) & (crows.gender_class == cls)).sum())
        c = int(((crows.label == 0) & (crows.gender_class == cls)).sum())
        chi, p = chi2_2x2(a, 262 - a, c, 262 - c)
        print(f"crows {name}: stereotype {a}/262, anti {c}/262, chi2={chi:.2f}, p={p:.3f}")
    none = df[df.gender_class == "none"]
    print(f"sentences with no lexicon word: {len(none)} {none.source.value_counts().to_dict()}; "
          f"with a name-like capitalised token: {int(none.sentence.map(has_name_like_token).sum())}")


def vocabulary_section(df):
    header("Vocabulary")
    for s in SOURCES + ["ALL"]:
        x = df if s == "ALL" else df[df.source == s]
        counts = collections.Counter(t for l in x.toks for t in l)
        n_tok, n_types, hapax = sum(counts.values()), len(counts), sum(1 for v in counts.values() if v == 1)
        print(f"{s}: tokens={n_tok} types={n_types} type/token={n_types / n_tok:.3f} "
              f"types used once={hapax} ({hapax / n_types:.1%})")
    v_crows = {t for l in df[df.source == "crows"].toks for t in l}
    v_ss = {t for l in df[df.source != "crows"].toks for t in l}
    print("crows vs stereoset vocabulary: jaccard=%.3f; share of crows types in stereoset=%.3f; "
          "share of stereoset types in crows=%.3f"
          % (len(v_crows & v_ss) / len(v_crows | v_ss), len(v_crows & v_ss) / len(v_crows), len(v_crows & v_ss) / len(v_ss)))
    print("top 20 tokens:", collections.Counter(t for l in df.toks for t in l).most_common(20))


def discriminative_words_section(df, n_perm=500, min_count=5, prior_mass=500.0):
    header("Discriminative words: log-odds with informative Dirichlet prior, pair-flip permutation null")
    counts = collections.Counter(t for l in df.toks for t in l)
    vocab = [w for w, v in counts.items() if v >= min_count]
    index = {w: i for i, w in enumerate(vocab)}
    X = np.zeros((len(df), len(vocab)))
    for r, toks in enumerate(df.toks):
        for t in toks:
            if t in index:
                X[r, index[t]] += 1
    prior = prior_mass * X.sum(0) / X.sum()

    def zscores(labels):
        y1, y0 = X[labels == 1].sum(0), X[labels == 0].sum(0)
        n1, n0 = y1.sum(), y0.sum()
        delta = np.log((y1 + prior) / (n1 + prior_mass - y1 - prior)) - np.log((y0 + prior) / (n0 + prior_mass - y0 - prior))
        return delta / np.sqrt(1 / (y1 + prior) + 1 / (y0 + prior)), y1, y0

    labels = df.label.values
    z, y1, y0 = zscores(labels)
    table = pd.DataFrame(dict(word=vocab, stereotype=y1.astype(int), anti=y0.astype(int), z=z)).sort_values("z", ascending=False)
    table.to_csv(ROOT / "outputs" / "word_zscores.csv", index=False)
    print(f"words with at least {min_count} occurrences: {len(vocab)}")
    print("toward stereotype:\n", table.head(15).round(2).to_string(index=False))
    print("toward anti-stereotype:\n", table.tail(15).iloc[::-1].round(2).to_string(index=False))

    pair_ids = np.unique(df.pair_id.values)
    position = {p: i for i, p in enumerate(pair_ids)}
    row_pair = np.array([position[p] for p in df.pair_id.values])
    null_max = []
    for _ in range(n_perm):
        flip = rng.integers(0, 2, len(pair_ids))
        null_max.append(np.abs(zscores(np.where(flip[row_pair] == 1, 1 - labels, labels))[0]).max())
    null_max = np.array(null_max)
    p95 = np.percentile(null_max, 95)
    print("observed max|z|=%.2f; permutation max|z| 95th=%.2f, 99th=%.2f" % (np.abs(z).max(), p95, np.percentile(null_max, 99)))
    print("words above the 95th-percentile threshold:", table[np.abs(table.z) > p95].round(2).to_dict("records"))


def label_quality_section():
    header("Label quality: CrowS-Pairs validators")
    crows = pd.read_csv(RAW / "crows_pairs_anonymized.csv")
    crows = crows[crows.bias_type == "gender"].copy()
    crows["n_gender"] = crows.annotations.map(lambda s: sum("gender" in a for a in ast.literal_eval(s)))
    print(crows.n_gender.value_counts().sort_index().to_dict(),
          "| >=4/5: %.1f%% | 5/5: %.1f%% | <=2/5: %.1f%%"
          % ((crows.n_gender >= 4).mean() * 100, (crows.n_gender == 5).mean() * 100, (crows.n_gender <= 2).mean() * 100))
    print("writers:", crows.anon_writer.nunique(), "| most prolific writer's share: %.1f%%"
          % (crows.anon_writer.value_counts().iloc[0] / len(crows) * 100))
    print("mean validators tagging gender, by build direction:", crows.groupby("stereo_antistereo").n_gender.mean().round(2).to_dict())

    header("Label quality: StereoSet annotators, and target terms")
    items, targets = [], []
    for part in ("intersentence", "intrasentence"):
        path = RAW / f"stereoset_{part}_validation.parquet"
        if not path.exists():
            print(f"missing {path.name}: run src/download_stereoset.py"); return
        ss = pd.read_parquet(path)
        for _, r in ss[ss.bias_type == "gender"].iterrows():
            targets.append((part, r.target))
            for gold, lab in zip(r.sentences["gold_label"], r.sentences["labels"]):
                votes = list(lab["label"])
                items.append(dict(part=part, gold=int(gold), agree=sum(v == gold for v in votes) / len(votes),
                                  unanimous=len(set(votes)) == 1, has_related=3 in votes))
    items = pd.DataFrame(items)
    print("candidates:", len(items), "| candidates where some annotator answered 'related' (3):", int(items.has_related.sum()))
    kept = items[items.gold != 2]
    print(kept.groupby(["part", "gold"]).agg(candidates=("agree", "size"), mean_agree=("agree", "mean"),
                                             unanimous=("unanimous", "mean")).round(3))
    print("overall (gold 0/1): mean agreement %.3f, unanimous %.1f%%, >=4/5 agree %.1f%%"
          % (kept.agree.mean(), kept.unanimous.mean() * 100, (kept.agree >= 0.8).mean() * 100))
    t = pd.DataFrame(targets, columns=["part", "target"])
    print("unique targets:", t.target.nunique(), "of", len(t), "contexts:", t.target.value_counts().to_dict())


def coverage_section(df, folds):
    header("Domain coverage, duplicates, chance bound")
    df["has_occupation"] = df.toks.map(lambda t: bool(OCCUPATIONS & set(t)))
    print("sentences with an occupation word: %.3f" % df.has_occupation.mean(),
          df.groupby("source").has_occupation.mean().round(3).to_dict())
    print(collections.Counter(t for l in df.toks for t in l if t in OCCUPATIONS).most_common(15))
    print("duplicate rows:", df.sentence.duplicated(keep=False).sum(), "| extra copies:", df.sentence.duplicated().sum())
    print("accuracy needed for one-sided p<.05 vs 0.5, n=1518: %.3f" % (0.5 + 1.645 * math.sqrt(0.25 / 1518)),
          "| 95%% CI half-width at 0.60: %.3f" % (1.96 * math.sqrt(0.6 * 0.4 / 1518)))
    path = RAW / "stereoset_intersentence_validation.parquet"
    if path.exists():
        rows = []
        for part in ("intersentence", "intrasentence"):
            ss = pd.read_parquet(RAW / f"stereoset_{part}_validation.parquet")
            ss = ss[ss.bias_type == "gender"]
            rows += [(f"ss_{part[:5]}_{i}", t) for i, t in zip(ss.id, ss.target)]
        per_fold = pd.DataFrame(rows, columns=["pair_id", "target"]).merge(folds, on="pair_id")
        counts = per_fold.pivot_table(index="fold", columns="target", values="pair_id", aggfunc="size", fill_value=0)
        print("fewest contexts for any target in any fold:", int(counts.values.min()))


def main():
    df = pd.read_csv(ROOT / "data" / "processed" / "merged_gender.csv")
    folds = pd.read_csv(ROOT / "data" / "processed" / "folds.csv")
    df["toks"] = df.sentence.map(tokens)
    df["n_tokens"] = df.toks.map(len)
    df["n_chars"] = df.sentence.str.len()
    folds_section(df, folds)
    pair_structure_section(df)
    length_section(df)
    gender_section(df)
    vocabulary_section(df)
    discriminative_words_section(df)
    label_quality_section()
    coverage_section(df, folds)


if __name__ == "__main__":
    main()
