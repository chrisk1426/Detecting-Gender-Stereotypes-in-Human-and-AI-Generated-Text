"""Section D of docs/MetricsAnalysisPlan.md: draw a sample of Qwen 14B's real mistakes to read and categorize.

The blind audit holds too few errors (only 2 for Qwen), so this reviews known mistakes directly (not blind).
Sample (seed 72): 10 errors per source, 5 where the dataset says stereotype (Qwen missed it) and 5 where the dataset says
anti (Qwen called it a stereotype); pairs already used in the audit are skipped.

Writes outputs/audit/error_review.csv with context for each error and blank `category` / `note` columns to fill in:
  partner sentence + Qwen's score for it, whether Qwen got the partner wrong too (it swapped the pair) or right
  (it gave both sentences the same label), label quality (CrowS validators / StereoSet annotator agreement),
  negation flag, sentence length.
"""
from pathlib import Path
import ast
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs" / "audit" / "error_review.csv"
rng = np.random.default_rng(72)

df = pd.read_csv(ROOT / "outputs" / "metrics" / "predictions_all.csv")
neg = pd.read_csv(ROOT / "experiments" / "feat" / "negation.csv")
assert len(neg) == len(df)
df["has_negation"] = neg["neg"].to_numpy()
df["n_tokens"] = df.sentence.str.split().str.len()

# Label quality per pair (same definitions as src/analyze_data.py)
crows = pd.read_csv(RAW / "crows_pairs_anonymized.csv").rename(columns={"Unnamed: 0": "orig_idx"})
crows = crows[crows.bias_type == "gender"]
quality = {f"crows_{r.orig_idx}": f"{sum('gender' in a for a in ast.literal_eval(r.annotations))}/5 validators said gender"
           for r in crows.itertuples()}
for part in ("intersentence", "intrasentence"):
    ss = pd.read_parquet(RAW / f"stereoset_{part}_validation.parquet")
    for _, r in ss[ss.bias_type == "gender"].iterrows():
        weakest = min(np.mean(np.array(lab["label"]) == gold)
                      for gold, lab in zip(r.sentences["gold_label"], r.sentences["labels"]) if gold != 2)
        quality[f"ss_{part[:5]}_{r['id']}"] = f"annotator agreement {weakest:.1f}"

audit_pairs = set(pd.read_csv(ROOT / "outputs" / "audit" / "audit_results_single.csv").pair_id) | \
              set(pd.read_csv(ROOT / "outputs" / "audit" / "audit_results_pairs.csv").pair_id)
errors = df[(df.qwen14b_correct == 0) & ~df.pair_id.isin(audit_pairs)]
print(f"Qwen errors overall: {int((df.qwen14b_correct == 0).sum())}; available after skipping audit pairs: {len(errors)}")

rows = []
for source in ["crows", "stereoset_inter", "stereoset_intra"]:
    for lab in (1, 0):
        pool = errors[(errors.source == source) & (errors.label == lab)]
        take = pool.loc[rng.permutation(pool.index)[:5]]
        rows.append(take)
sample = pd.concat(rows)
sample = sample.loc[rng.permutation(sample.index)]  # shuffle so sources and labels are mixed

out = []
for i, (idx, r) in enumerate(sample.iterrows(), 1):
    partner = df[(df.pair_id == r.pair_id) & (df.index != idx)].iloc[0]
    out.append({"review_id": f"E{i:02d}", "source": r.source, "pair_id": r.pair_id, "sentence": r.sentence,
                "dataset_label": "stereotype" if r.label == 1 else "anti", "qwen_prob": round(r.qwen14b_prob, 3),
                "qwen_says": "stereotype" if r.qwen14b_pred == 1 else "anti",
                "ensemble_says": "stereotype" if r.ensemble_pred == 1 else "anti",
                "partner_sentence": partner.sentence, "partner_qwen_prob": round(partner.qwen14b_prob, 3),
                "pair_outcome": "swapped: both sentences wrong" if partner.qwen14b_correct == 0
                                else "same label given to both sentences",
                "label_quality": quality.get(r.pair_id, ""), "has_negation": int(r.has_negation), "n_tokens": int(r.n_tokens),
                "category": "", "note": ""})
out = pd.DataFrame(out)
if OUT.exists() and pd.read_csv(OUT)["category"].notna().any():
    raise SystemExit(f"{OUT} already has categories filled in; not overwriting. Move it aside to regenerate the sample.")
out.to_csv(OUT, index=False)

# Counts only (no sentences) so the review can be done one at a time
print(f"wrote {OUT} ({len(out)} rows)")
print("by source and dataset label:", out.groupby(["source", "dataset_label"]).size().to_dict())
print("pair outcome:", out.pair_outcome.value_counts().to_dict())
print("with negation:", int(out.has_negation.sum()), "| mean tokens:", round(out.n_tokens.mean(), 1),
      "(all sentences:", round(df.n_tokens.mean(), 1), ")")
print("Qwen overconfident (prob <0.05 or >0.95):", int(((out.qwen_prob < 0.05) | (out.qwen_prob > 0.95)).sum()), "of", len(out))
