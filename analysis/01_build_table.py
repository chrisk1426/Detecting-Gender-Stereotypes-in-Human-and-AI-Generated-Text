"""Step A of docs/MetricsAnalysisPlan.md: one table with every model's out-of-fold P(stereotype) per sentence.

Reads only saved predictions (no GPU). Writes outputs/metrics/predictions_all.csv with one row per sentence:
  sentence info   pair_id, sentence, label, source, fold
  <model>_prob    P(stereotype) from a model that never trained on that sentence's fold
  <model>_pred    1 if prob > 0.5 else 0
  <model>_correct 1 if pred == label
The stacked ensemble mirrors experiments/ensemble.py, which prints its score but does not save its predictions.
"""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
FEAT = ROOT / "experiments" / "feat"
OUT = ROOT / "outputs" / "metrics"

df = pd.read_csv(ROOT / "data" / "processed" / "merged_gender.csv")
df = df.merge(pd.read_csv(ROOT / "data" / "processed" / "folds.csv")[["pair_id", "fold"]], on="pair_id", how="left")
y, f = df["label"].to_numpy(), df["fold"].to_numpy()
assert len(df) == 1518 and not np.isnan(f).any(), "merged_gender.csv / folds.csv do not line up"


def check_rows(other, name):
    """Prediction files are row-aligned with merged_gender.csv; prove it instead of assuming it."""
    assert len(other) == len(df), f"{name}: {len(other)} rows, expected {len(df)}"
    for col in ("pair_id", "sentence", "label", "fold"):
        if col in other and not (other[col].to_numpy() == df[col].to_numpy()).all():
            raise SystemExit(f"{name}: column '{col}' does not match merged_gender.csv row by row")


def feat_prob(name):
    d = pd.read_csv(FEAT / name)
    check_rows(d[["fold"]], name)
    return d["prob"].to_numpy()


# Baselines from earlier work (already carry sentence/label/fold, so they can be checked directly)
base = pd.read_csv(ROOT / "outputs" / "predictions.csv")
check_rows(base, "outputs/predictions.csv")
tfidf = pd.read_csv(ROOT / "tf-idf" / "results" / "predictions.csv")
check_rows(tfidf, "tf-idf/results/predictions.csv")

# Same nine models the README's stacked ensemble uses (names and files copied from experiments/ensemble.py)
ENSEMBLE_INPUTS = {"features + LR": "oof_feature_pipeline_7B+14B.csv", "features + RF": "oof_rf.csv",
                   "LoRA 7B 8ep (3 seeds)": "ft_lora7b_ep8_3seeds.csv",
                   "LoRA 7B 16ep": "ft_Qwen2.5-7B-Instruct_lora_ep16_s72.csv",
                   "LoRA 14B 6ep": "ft_Qwen2.5-14B-Instruct_lora_4bit_ep6_s72.csv",
                   "LoRA 14B 12ep": "ft_Qwen2.5-14B-Instruct_lora_4bit_ep12_s72.csv",
                   "LoRA 7B 4bit + GEST": "ft_Qwen2.5-7B-Instruct_lora_4bit_gest_pairs_ep4_s72.csv",
                   "LoRA 14B + GEST": "ft_Qwen2.5-14B-Instruct_lora_4bit_gest_pairs_ep4_s72.csv",
                   "LoRA 7B + GEST": "ft_Qwen2.5-7B-Instruct_lora_gest_pairs_ep4_s72.csv"}
P = {n: feat_prob(p) for n, p in ENSEMBLE_INPUTS.items()}
logit = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / np.clip(1 - p, 1e-4, 1))
Z = np.column_stack([logit(p) for p in P.values()])
stacked = np.zeros(len(y))
for k in range(5):  # fit on the other 4 folds' out-of-fold probabilities, score the held-out fold
    m = LogisticRegression(C=1.0).fit(Z[f != k], y[f != k])
    stacked[f == k] = m.predict_proba(Z[f == k])[:, 1]

# Models shown in the table, in the order of the story: chance -> bag of words -> embeddings -> Qwen -> ensemble
MODELS = {"majority": base["majority_prob"].to_numpy(),
          "tfidf_lr": tfidf["tfidf_lr_prob"].to_numpy(),
          "mpnet_rf": base["mpnet_rf_prob"].to_numpy(),
          "qwen14b": P["LoRA 14B 12ep"],
          "ensemble": stacked}

out = df.copy()
for name, p in MODELS.items():
    out[f"{name}_prob"] = p
    out[f"{name}_pred"] = (p > 0.5).astype(int)
    out[f"{name}_correct"] = (out[f"{name}_pred"] == out["label"]).astype(int)

OUT.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT / "predictions_all.csv", index=False)

# Sanity check against the README. README numbers are the mean of the five per-fold accuracies, so compute that.
README = {"tfidf_lr": 0.567, "mpnet_rf": 0.604, "qwen14b": 0.820, "ensemble": 0.838}
print(f"wrote {OUT / 'predictions_all.csv'}  ({len(out)} rows)\n")
print(f"{'model':10s} {'mean-of-folds':>14s} {'pooled':>8s} {'README':>8s}")
bad = []
for name in MODELS:
    folds_acc = np.mean([out.loc[out.fold == k, f"{name}_correct"].mean() for k in range(5)])
    pooled = out[f"{name}_correct"].mean()
    ref = README.get(name)
    print(f"{name:10s} {folds_acc:14.3f} {pooled:8.3f} {'' if ref is None else f'{ref:8.3f}'}")
    if ref is not None and abs(folds_acc - ref) > 0.01:
        bad.append(name)
print("\nCHECK FAILED for: " + ", ".join(bad) if bad else "\nCheck passed: every model is within 0.01 of the README.")
