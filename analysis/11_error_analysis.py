"""Y2-Y7 of docs/MetricsAnalysisPlan.md: where do the models fail? All from saved predictions plus data files (CPU only).

  Y2 hard sentences: wrong for how many of the 4 models           -> error_hard_sentences.csv (sentences wrong for all 4)
  Y3 error overlap between models, and an "oracle" upper bound    -> error_overlap.csv
  Y4 errors vs sentence properties (length, gender words, negation, occupation word, source, label)  -> error_by_property.csv
  Y5 errors vs label quality (CrowS validators, StereoSet annotator agreement)                        -> error_by_label_quality.csv
  Y6 words over-represented in Qwen's errors (log-odds, with a permutation threshold)                 -> error_words.csv
  Y7 accuracy by StereoSet target term and by CrowS writer                                             -> error_by_target_writer.csv
  chart: error_by_group.png (Y4/Y5/Y7 subgroup accuracy with 95% intervals)
Intervals are Wilson intervals that treat sentences as independent; the two sentences of a pair are not, so read them as
a little too narrow. Differences between subgroups are descriptive, not tests.
"""
from pathlib import Path
import ast, re, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from analyze_data import FEMALE, MALE, OCCUPATIONS, gender_class, tokens  # same lexicons as DATANOTES.md

OUT, RAW = ROOT / "outputs" / "metrics", ROOT / "data" / "raw"
df = pd.read_csv(OUT / "predictions_all.csv")
MODELS = ["tfidf_lr", "mpnet_rf", "qwen14b", "ensemble"]
NAME = {"tfidf_lr": "TF-IDF + LR", "mpnet_rf": "mpnet + RF", "qwen14b": "Qwen 14B", "ensemble": "Ensemble"}
err = pd.DataFrame({m: 1 - df[f"{m}_correct"] for m in MODELS})
rng = np.random.default_rng(72)


def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    c, h = p + z * z / (2 * n), z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


# ---- Y2: hard sentences -------------------------------------------------------------------------------------------------
n_wrong = err.sum(axis=1)
print("Y2  number of models (of 4) that get a sentence wrong:", n_wrong.value_counts().sort_index().to_dict())
allw = df[n_wrong == 4]
print(f"    wrong for all 4: {len(allw)} sentences ({len(allw) / len(df):.1%}); by source {allw.source.value_counts().to_dict()}; "
      f"by label {allw.label.map({1: 'stereotype', 0: 'anti'}).value_counts().to_dict()}")
qw = df[err.qwen14b == 1]
print(f"    Qwen's 273 errors: wrong for all 4 models = {int((n_wrong[qw.index] == 4).sum())}, shared with 1 or 2 other models = "
      f"{int((n_wrong[qw.index].isin([2, 3])).sum())}, Qwen alone = {int((n_wrong[qw.index] == 1).sum())}")
pair_of = df.groupby("pair_id").sentence.apply(list)
allw.assign(partner_sentence=[[t for t in pair_of[p] if t != s][0] for p, s in zip(allw.pair_id, allw.sentence)])[
    ["pair_id", "source", "label", "sentence", "partner_sentence", "qwen14b_prob", "ensemble_prob"]].to_csv(
    OUT / "error_hard_sentences.csv", index=False)

# ---- Y3: overlap and oracle -----------------------------------------------------------------------------------------------
rows = []
for a in MODELS:
    for b in MODELS:
        if a < b:
            both = int(((err[a] == 1) & (err[b] == 1)).sum())
            rows.append({"measure": "shared errors", "first": NAME[a], "second": NAME[b], "errors_first": int(err[a].sum()),
                         "errors_second": int(err[b].sum()), "shared": both,
                         "share_of_smaller_error_set": round(both / min(err[a].sum(), err[b].sum()), 3),
                         "jaccard": round(both / int(((err[a] == 1) | (err[b] == 1)).sum()), 3)})
for combo in (("qwen14b", "ensemble"), ("qwen14b", "mpnet_rf"), ("qwen14b", "mpnet_rf", "tfidf_lr"), tuple(MODELS)):
    rows.append({"measure": "oracle accuracy (right if any of these is right)", "first": " + ".join(NAME[m] for m in combo),
                 "oracle_accuracy": round(1 - err[list(combo)].min(axis=1).mean(), 3)})
ov = pd.DataFrame(rows)
ov.to_csv(OUT / "error_overlap.csv", index=False)
print("\nY3  error overlap and oracle:\n", ov.fillna("").to_string(index=False))

# ---- sentence properties and label quality ------------------------------------------------------------------------------------
neg = pd.read_csv(ROOT / "experiments" / "feat" / "negation.csv")
assert len(neg) == len(df)
df["negation"] = neg["neg"].to_numpy().astype(bool)
df["negation_label"] = np.where(df.negation, "has negation", "no negation")
df["n_tokens"] = df.sentence.str.split().str.len()
df["length"] = pd.cut(df.n_tokens, [0, 8, 12, 17, 1000], labels=["<=8 words", "9-12", "13-17", "18+"])
df["gender_words"] = df.sentence.map(gender_class).map({"F": "female words only", "M": "male words only", "both": "both", "none": "no gender word"})
df["occupation"] = df.sentence.map(lambda s: bool(OCCUPATIONS & set(tokens(s)))).map({True: "occupation word", False: "no occupation word"})
df["label_name"] = df.label.map({1: "label: stereotype", 0: "label: anti"})

crows = pd.read_csv(RAW / "crows_pairs_anonymized.csv").rename(columns={"Unnamed: 0": "orig_idx"})
crows = crows[crows.bias_type == "gender"].copy()
crows["n_gender"] = crows.annotations.map(lambda s: sum("gender" in a for a in ast.literal_eval(s)))
quality, target, writer = {}, {}, {}
for r in crows.itertuples():
    quality[f"crows_{r.orig_idx}"] = "CrowS: 5/5 validators say gender" if r.n_gender == 5 else \
        "CrowS: 4/5" if r.n_gender == 4 else "CrowS: 3 or fewer"
    writer[f"crows_{r.orig_idx}"] = r.anon_writer
for part in ("intersentence", "intrasentence"):
    ss = pd.read_parquet(RAW / f"stereoset_{part}_validation.parquet")
    for _, r in ss[ss.bias_type == "gender"].iterrows():
        weakest = min(np.mean(np.array(l["label"]) == g) for g, l in zip(r.sentences["gold_label"], r.sentences["labels"]) if g != 2)
        quality[f"ss_{part[:5]}_{r['id']}"] = "StereoSet: all annotators agree" if weakest == 1 else \
            "StereoSet: agreement 0.8" if weakest >= 0.8 else "StereoSet: agreement 0.6 or less"
        target[f"ss_{part[:5]}_{r['id']}"] = r.target
df["label_quality"] = df.pair_id.map(quality)
assert df.label_quality.notna().all(), "unmatched pair ids"
df["target"], df["writer"] = df.pair_id.map(target), df.pair_id.map(writer)

# ---- Y4 / Y5 / Y7 tables ----------------------------------------------------------------------------------------------------------
def group_table(col, family, order=None, data=df, min_n=1):
    rows = []
    for val, g in data.groupby(col, observed=True):
        if len(g) < min_n:
            continue
        for m in ("qwen14b", "ensemble"):
            k = int(g[f"{m}_correct"].sum())
            lo, hi = wilson(k, len(g))
            rows.append({"family": family, "group": str(val), "model": NAME[m], "n": len(g), "accuracy": round(k / len(g), 3),
                         "ci_low": round(lo, 3), "ci_high": round(hi, 3)})
    return rows

rows = []
for col, fam in (("source", "source"), ("label_name", "dataset label"), ("length", "length"), ("gender_words", "gender words"),
                 ("negation_label", "negation"), ("occupation", "occupation")):
    rows += group_table(col, fam)
prop = pd.DataFrame(rows)
prop.to_csv(OUT / "error_by_property.csv", index=False)
print("\nY4  accuracy by sentence property (Qwen / ensemble, n):")
print(prop.pivot_table(index=["family", "group"], columns="model", values="accuracy").join(
    prop[prop.model == "Qwen 14B"].set_index(["family", "group"]).n).round(3).to_string())

qual = pd.DataFrame(group_table("label_quality", "label quality"))
qual.to_csv(OUT / "error_by_label_quality.csv", index=False)
print("\nY5  accuracy by label quality:")
print(qual.pivot_table(index=["group"], columns="model", values="accuracy").join(
    qual[qual.model == "Qwen 14B"].set_index("group").n).round(3).to_string())
# Is the pair-level label-quality signal real? Compare mean quality-group accuracy only within each source.
print("    by source and quality (Qwen):")
print(df.groupby(["source", "label_quality"]).qwen14b_correct.agg(["mean", "size"]).round(3).to_string())

tw = pd.DataFrame(group_table("target", "StereoSet target term", data=df[df.target.notna()]))
wr = df[df.writer.notna()].groupby("writer").filter(lambda g: len(g) >= 20)
tw = pd.concat([tw, pd.DataFrame(group_table("writer", "CrowS writer (20+ sentences)", data=wr))])
tw.to_csv(OUT / "error_by_target_writer.csv", index=False)
print("\nY7  StereoSet target terms, Qwen accuracy:")
t = tw[(tw.family == "StereoSet target term") & (tw.model == "Qwen 14B")].sort_values("accuracy")
print(t[["group", "n", "accuracy"]].to_string(index=False))
w = tw[(tw.family.str.startswith("CrowS writer")) & (tw.model == "Qwen 14B")]
print(f"    CrowS writers with 20+ sentences: {len(w)}; Qwen accuracy range {w.accuracy.min():.2f}-{w.accuracy.max():.2f}; "
      f"({df.writer.nunique()} writers in all)")

# ---- Key contrasts, with a pair-level bootstrap (resample whole pairs) ----------------------------------------------------------
pairs = df.pair_id.unique()
rows_of = {p: np.flatnonzero((df.pair_id == p).to_numpy()) for p in pairs}
draws = [np.concatenate([rows_of[p] for p in rng.choice(pairs, len(pairs))]) for _ in range(2000)]


def contrast(name, mask_a, mask_b, m):
    ok = df[f"{m}_correct"].to_numpy()
    d = [ok[i][mask_a[i]].mean() - ok[i][mask_b[i]].mean() for i in draws]
    est = ok[mask_a].mean() - ok[mask_b].mean()
    print(f"    {name} [{NAME[m]}]: {est:+.3f}  95% interval {np.percentile(d, 2.5):+.3f} to {np.percentile(d, 97.5):+.3f}")
    return {"contrast": name, "model": NAME[m], "difference": round(est, 3), "ci_low": round(np.percentile(d, 2.5), 3),
            "ci_high": round(np.percentile(d, 97.5), 3)}

print("\nKey contrasts (difference in accuracy, whole pairs resampled):")
gw, ql = df.gender_words.to_numpy(), df.label_quality.to_numpy()
ck = []
for m in ("qwen14b", "ensemble"):
    ck.append(contrast("female-only minus male-only sentences", gw == "female words only", gw == "male words only", m))
    ck.append(contrast("StereoSet all-agree minus agreement 0.6 or less", ql == "StereoSet: all annotators agree", ql == "StereoSet: agreement 0.6 or less", m))
    ck.append(contrast("CrowS 5/5 validators minus 3 or fewer", ql == "CrowS: 5/5 validators say gender", ql == "CrowS: 3 or fewer", m))
    ck.append(contrast("negation minus no negation", df.negation.to_numpy(), ~df.negation.to_numpy(), m))
pd.DataFrame(ck).to_csv(OUT / "error_key_contrasts.csv", index=False)

# ---- Y6: words in Qwen's errors ---------------------------------------------------------------------------------------------------
docs = [set(tokens(s)) for s in df.sentence]
vocab = pd.Series([w for d in docs for w in d]).value_counts()
vocab = vocab[vocab >= 10].index
X = np.array([[w in d for w in vocab] for d in docs], dtype=float)
a0 = 0.5  # prior pseudo-count per word (informative Dirichlet, Monroe et al. 2008)


def zscores(wrong):
    ca, cb = X[wrong == 1].sum(0) + a0, X[wrong == 0].sum(0) + a0
    na, nb = ca.sum(), cb.sum()
    d = np.log(ca / (na - ca)) - np.log(cb / (nb - cb))
    return d / np.sqrt(1 / ca + 1 / cb)

wrong = err.qwen14b.to_numpy()
z = zscores(wrong)
null = [np.abs(zscores(rng.permutation(wrong))).max() for _ in range(300)]
thr = np.percentile(null, 95)
tab = pd.DataFrame({"word": vocab, "z": z.round(2), "in_errors": X[wrong == 1].sum(0).astype(int),
                    "in_correct": X[wrong == 0].sum(0).astype(int)}).sort_values("z")
tab["above_permutation_threshold"] = tab.z.abs() > thr
tab.to_csv(OUT / "error_words.csv", index=False)
print(f"\nY6  words (in 10+ sentences): max|z| observed {np.abs(z).max():.2f}, permutation 95th percentile {thr:.2f}")
print("    more in Qwen's errors:", tab.tail(8)[["word", "z", "in_errors", "in_correct"]].values.tolist())
print("    more in correct answers:", tab.head(8)[["word", "z", "in_errors", "in_correct"]].values.tolist())
print("    words clearing the threshold:", int(tab.above_permutation_threshold.sum()))

# ---- Chart -----------------------------------------------------------------------------------------------------------------------------------
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e8e7e3"
COLOR = {"Qwen 14B": "#1baf7a", "Ensemble": "#eda100"}
sel = pd.concat([prop[prop.family.isin(["source", "length", "gender words", "negation"])], qual])
groups = list(dict.fromkeys(zip(sel.family, sel.group)))
fig, ax = plt.subplots(figsize=(8.6, 0.33 * len(groups) + 1.9), facecolor=SURFACE)
fig.subplots_adjust(left=0.36, right=0.97, top=1 - 1.15 / (0.33 * len(groups) + 1.9), bottom=0.09)
ax.set_facecolor(SURFACE)
for i, (fam, grp) in enumerate(groups):
    for m, off in (("Qwen 14B", -0.12), ("Ensemble", 0.12)):
        r = sel[(sel.family == fam) & (sel.group == grp) & (sel.model == m)].iloc[0]
        ax.plot([r.ci_low, r.ci_high], [i + off] * 2, color=COLOR[m], lw=2, solid_capstyle="round")
        ax.plot(r.accuracy, i + off, "o", color=COLOR[m], ms=5, label=m if i == 0 else None)
    ax.text(0.595, i, f"n={int(r.n)}", va="center", fontsize=7.5, color=INK2)
for m in COLOR:
    ax.axvline(df[f"{'qwen14b' if m == 'Qwen 14B' else 'ensemble'}_correct"].mean(), color=COLOR[m], lw=0.8, ls=":")
ax.set_yticks(range(len(groups)), [g if f in ("source", "label quality") or g.startswith(("CrowS", "StereoSet")) else f"{f}: {g}" for f, g in groups], fontsize=8.5, color=INK2)
ax.invert_yaxis()
ax.set_xlim(0.58, 1.0)
ax.set_xlabel("Accuracy (95% interval; dotted line = overall)", color=INK2)
ax.grid(axis="x", color=GRID)
for s_ in ("top", "right"):
    ax.spines[s_].set_visible(False)
ax.spines["left"].set_color(GRID); ax.spines["bottom"].set_color(GRID)
ax.legend(loc="upper right", frameon=False, fontsize=8.5)
fig.text(0.01, 0.985, "Accuracy tracks label quality far more than sentence features", ha="left", va="top", fontsize=12.5, fontweight="bold", color=INK)
fig.text(0.01, 0.945, "Length and negation barely matter; female-only sentences score about 6 points above male-only.", ha="left", va="top", fontsize=9, color=INK2)
fig.savefig(OUT / "error_by_group.png", dpi=160)
