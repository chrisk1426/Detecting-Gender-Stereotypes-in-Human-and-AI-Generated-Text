"""Post-training: combine out-of-fold P(stereotype) from the feature pipeline and the fine-tuned models.

Every input probability is out-of-fold (made by a model that never trained on that sentence's fold).
  average  - mean of the probabilities, nothing fitted
  stacked  - logistic regression on the logits, fitted on 4 folds' out-of-fold probs, scored on the 5th
  + calibrated threshold - decision threshold chosen on the 4 training folds instead of 0.5
"""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).parent
df = pd.read_csv(HERE.parent / "data" / "processed" / "merged_gender.csv")
df = df.merge(pd.read_csv(HERE.parent / "data" / "processed" / "folds.csv"), on="pair_id")
y, f = df["label"].to_numpy(), df["fold"].to_numpy()

sources = {"features + LR": "oof_feature_pipeline_7B+14B.csv", "features + RF": "oof_rf.csv",
           "LoRA 7B 8ep (3 seeds)": "ft_lora7b_ep8_3seeds.csv",
           "LoRA 7B 16ep": "ft_Qwen2.5-7B-Instruct_lora_ep16_s72.csv",
           "LoRA 14B 6ep": "ft_Qwen2.5-14B-Instruct_lora_4bit_ep6_s72.csv",
           "LoRA 14B 12ep": "ft_Qwen2.5-14B-Instruct_lora_4bit_ep12_s72.csv",
           "LoRA 7B 4bit + GEST": "ft_Qwen2.5-7B-Instruct_lora_4bit_gest_pairs_ep4_s72.csv",
           "LoRA 14B + GEST": "ft_Qwen2.5-14B-Instruct_lora_4bit_gest_pairs_ep4_s72.csv",
           "LoRA 7B + GEST": "ft_Qwen2.5-7B-Instruct_lora_gest_pairs_ep4_s72.csv"}
P = {n: pd.read_csv(HERE / "feat" / p)["prob"].to_numpy() for n, p in sources.items() if (HERE / "feat" / p).exists()}
logit = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / np.clip(1 - p, 1e-4, 1))


def acc(prob, thr=None):
    out = []
    for k in range(5):
        te, tr = f == k, f != k
        t = 0.5 if thr is None else thr(prob[tr], y[tr])
        out.append(((prob[te] > t) == y[te]).mean())
    return np.mean(out), np.std(out)


def best_threshold(p, yy):
    grid = np.quantile(p, np.linspace(0.3, 0.7, 41))
    return grid[np.argmax([((p > t) == yy).mean() for t in grid])]


def pairwise(prob):
    pr = df.assign(p=prob).pivot_table(index="pair_id", columns="label", values="p")
    return (pr[1] > pr[0]).mean()


rows = []
for n, p in P.items():
    rows.append((n, *acc(p), pairwise(p)))
import itertools
for a, b in [] if "--pairs" not in __import__("sys").argv else itertools.combinations(P, 2):
    avg2 = (P[a] + P[b]) / 2
    rows.append((f"avg {a} + {b}", *acc(avg2), pairwise(avg2)))
if len(P) > 1:
    avg = np.mean(list(P.values()), axis=0)
    rows.append(("average of all", *acc(avg), pairwise(avg)))
    Z = np.column_stack([logit(p) for p in P.values()])
    stacked = np.zeros(len(y))
    for k in range(5):
        m = LogisticRegression(C=1.0).fit(Z[f != k], y[f != k])
        stacked[f == k] = m.predict_proba(Z[f == k])[:, 1]
    rows.append(("stacked (LR on logits)", *acc(stacked), pairwise(stacked)))
    rows.append(("stacked + tuned threshold", *acc(stacked, best_threshold), pairwise(stacked)))
for n, a, s, pw in rows:
    print(f"{n:28s} acc {a:.3f} (sd {s:.3f})   pairwise {pw:.3f}")
