"""Step C scoring of docs/MetricsAnalysisPlan.md: unblind the audit and compare the human labels with the dataset and models.

Run only after BOTH sheets in outputs/audit/ are fully labeled. Reads the private answer key.

Single-sentence sheet (C1 random sample, C3 targeted weak-label set, scored separately):
  - human vs the dataset's label (only rows where the human said stereotypical or anti-stereotypical)
  - how often the human said neutral / unsure, and what the dataset said for those
  - human vs Qwen and vs the ensemble; Qwen's own accuracy on the same rows (is the sample representative of 0.82?)
Pair sheet (C2): human pick vs the dataset's "more stereotypical" sentence vs each model's higher-probability sentence.

Writes (these contain answers, so they are not in private/):
  outputs/audit/audit_results_single.csv   one row per sentence, with a blank `error_category` column to fill in
  outputs/audit/audit_results_pairs.csv
"""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parent.parent
AUD = ROOT / "outputs" / "audit"
MODELS = ["tfidf_lr", "mpnet_rf", "qwen14b", "ensemble"]
LABEL = {"stereotypical": 1, "anti-stereotypical": 0}


def wilson(k, n, z=1.96):
    """95% interval for a share k/n; wide when n is small, which is the point."""
    if n == 0:
        return np.nan, np.nan
    p, d = k / n, 1 + z * z / n
    c, h = p + z * z / (2 * n), z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def share(k, n):
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k / n:.2f}  (95% interval {lo:.2f}-{hi:.2f})" if n else "n/a"


# ---- Single sentences ---------------------------------------------------------------------------------------------
sheet = pd.read_csv(AUD / "audit_single.csv", dtype=str, keep_default_na=False)
key = pd.read_csv(AUD / "private" / "audit_key_single.csv")
m = sheet.merge(key, on="audit_id", suffixes=("_sheet", ""))
assert len(m) == len(sheet) == 40, "single sheet and key do not match"
assert (m.sentence_sheet == m.sentence).all(), "a sentence in the sheet was edited"
m["label_text"] = m.your_label.str.strip().str.lower()
bad = set(m.label_text) - set(LABEL) - {"neutral", "unsure"}
assert not bad, f"unexpected labels: {bad}"
m["human"] = m.label_text.map(LABEL)  # NaN for neutral / unsure
m["human_agrees"] = np.where(m.human.isna(), np.nan, (m.human == m.true_label).astype(float))
m["error_category"] = ""

print("=" * 78, "\nSINGLE SENTENCES\n" + "=" * 78)
for part, title in (("C1", "C1: random sample"), ("C3", "C3: targeted weak-label set")):
    s = m[m.part == part]
    b = s[s.human.notna()]
    print(f"\n--- {title} ({len(s)} sentences) ---")
    print("your labels:", s.label_text.value_counts().to_dict())
    print("human vs dataset label (stereotypical/anti only):", share(int(b.human_agrees.sum()), len(b)))
    if len(b) > 1 and b.human.nunique() > 1:
        print(f"  Cohen's kappa vs dataset: {cohen_kappa_score(b.human.astype(int), b.true_label):.2f}")
    print("what the dataset said for your neutral/unsure rows:",
          s[s.human.isna()].true_label.value_counts().rename({1: "stereotype", 0: "anti"}).to_dict())
    for mod in ("qwen14b", "ensemble"):
        print(f"{mod}: accuracy on these {len(s)} sentences {share(int((s[mod + '_pred'] == s.true_label).sum()), len(s))}"
              f" | agrees with you {share(int((b.human == b[mod + '_pred']).sum()), len(b))}")
    print("your label x dataset label:")
    print(pd.crosstab(s.label_text, s.true_label.map({1: "dataset: stereotype", 0: "dataset: anti"})).to_string())

c1 = m[(m.part == "C1") & m.human.notna()]
print("\n--- C1 only: your agreement with the dataset by source ---")
for src, g in c1.groupby("source"):
    print(f"  {src}: {share(int(g.human_agrees.sum()), len(g))}")
print("--- C1 only: by your confidence ---")
for conf, g in c1.groupby("confidence_1to3"):
    print(f"  confidence {conf}: {share(int(g.human_agrees.sum()), len(g))}")

# Where you and the dataset disagree, is Qwen with you or with the dataset?
dis = m[m.human_agrees == 0]
print(f"\n--- the {len(dis)} sentences where you disagree with the dataset (C1 + C3) ---")
print(f"Qwen sides with the dataset on {int((dis.qwen14b_pred == dis.true_label).sum())}, with you on {int((dis.qwen14b_pred == dis.human).sum())}")
print(f"ensemble sides with the dataset on {int((dis.ensemble_pred == dis.true_label).sum())}, with you on {int((dis.ensemble_pred == dis.human).sum())}")

cols = ["audit_id", "part", "selection_reason", "source", "pair_id", "sentence", "your_label", "confidence_1to3", "reason",
        "true_label", "qwen14b_prob", "qwen14b_pred", "ensemble_pred", "human_agrees", "error_category"]
m[cols].to_csv(AUD / "audit_results_single.csv", index=False)

# ---- Pairs ---------------------------------------------------------------------------------------------------------
psheet = pd.read_csv(AUD / "audit_pairs.csv", dtype=str, keep_default_na=False)
pkey = pd.read_csv(AUD / "private" / "audit_key_pairs.csv")
p = psheet.merge(pkey, on="audit_id")
assert len(p) == len(psheet) == 10, "pair sheet and key do not match"
pick_col = "your_pick_first_second_equal_unsure"
p["pick_text"] = p[pick_col].str.strip().str.lower()
assert not (set(p.pick_text) - {"first", "second", "equal", "unsure"}), "unexpected pick"
p["human"] = p.pick_text.map({"first": 1, "second": 2})
p["human_agrees"] = np.where(p.human.isna(), np.nan, (p.human == p.true_more_stereotypical).astype(float))
for mod in MODELS:
    p[f"{mod}_pick"] = np.where(p[f"{mod}_prob_1"] > p[f"{mod}_prob_2"], 1, 2)
p["error_category"] = ""

print("\n" + "=" * 78, "\nPAIRS (which sentence is more stereotypical?)\n" + "=" * 78)
b = p[p.human.notna()]
print("your picks:", p.pick_text.value_counts().to_dict())
print("you vs dataset:", share(int(b.human_agrees.sum()), len(b)))
for mod in MODELS:
    print(f"{mod:9s}: picks the dataset's sentence in {share(int((p[mod + '_pick'] == p.true_more_stereotypical).sum()), len(p))}"
          f" | agrees with you {share(int((b[mod + '_pick'] == b.human).sum()), len(b))}")
print("by source (you vs dataset):", {s: share(int(g.human_agrees.sum()), int(g.human.notna().sum())) for s, g in p.groupby("source")})

pcols = ["audit_id", "pair_id", "source", "sentence_1", "sentence_2", pick_col, "confidence_1to3", "reason",
         "true_more_stereotypical", "qwen14b_pick", "ensemble_pick", "human_agrees", "error_category"]
p[pcols].to_csv(AUD / "audit_results_pairs.csv", index=False)
print(f"\nwrote {AUD / 'audit_results_single.csv'} and {AUD / 'audit_results_pairs.csv'}")
