"""Tier 1 extra metrics X1-X5 of docs/MetricsAnalysisPlan.md, from outputs/metrics/predictions_all.csv.

  X1  ci.csv               95% bootstrap intervals for accuracy, macro F1 and AUC (resampling whole PAIRS)
  X2  model_comparison.csv paired bootstrap of accuracy differences between models, plus McNemar's exact test
  X3  calibration.csv, calibration.png   Brier score, expected calibration error (10 equal-width bins), reliability diagram
  X4  accuracy_by_confidence.csv, .png   accuracy within confidence bins (confidence = max(P, 1-P))
  X5  mcc_kappa.csv        Matthews correlation coefficient and Cohen's kappa against the true labels
Accuracy here is pooled over all 1,518 sentences (summary.csv uses the mean of the 5 folds; they differ by <0.003).
Resamples: 5,000, seed 72.
"""
from pathlib import Path
from math import comb
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score, f1_score, matthews_corrcoef, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "metrics"
df = pd.read_csv(OUT / "predictions_all.csv")
MODELS = ["majority", "tfidf_lr", "mpnet_rf", "qwen14b", "ensemble"]
NAME = {"majority": "Majority guess", "tfidf_lr": "TF-IDF + LR", "mpnet_rf": "mpnet + random forest",
        "qwen14b": "Qwen 14B (LoRA)", "ensemble": "Ensemble (9 models)"}
COLOR = {"majority": "#9b9a94", "tfidf_lr": "#2a78d6", "mpnet_rf": "#eb6834", "qwen14b": "#1baf7a", "ensemble": "#eda100"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e8e7e3"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1, "axes.axisbelow": True,
                     "font.size": 10, "legend.frameon": False})

pairs = df.pair_id.unique()
idx = {p: np.flatnonzero((df.pair_id == p).to_numpy()) for p in pairs}
rng = np.random.default_rng(72)
B = 5000
boot = [np.concatenate([idx[p] for p in rng.choice(pairs, len(pairs))]) for _ in range(B)]  # same resamples for every model
y = df.label.to_numpy()


def acc(m, i=slice(None)): return df[f"{m}_correct"].to_numpy()[i].mean()
def f1(m, i=slice(None)): return f1_score(y[i], df[f"{m}_pred"].to_numpy()[i], average="macro")
def auc(m, i=slice(None)): return roc_auc_score(y[i], df[f"{m}_prob"].to_numpy()[i])


# ---- X1: bootstrap intervals ---------------------------------------------------------------------------------------
rows = []
for m in MODELS:
    for metric, fn in (("accuracy", acc), ("macro_F1", f1), ("AUC", auc)):
        if metric == "AUC" and m == "majority":
            continue
        v = np.array([fn(m, i) for i in boot])
        rows.append({"model": NAME[m], "metric": metric, "estimate": round(fn(m), 3),
                     "ci_low": round(np.percentile(v, 2.5), 3), "ci_high": round(np.percentile(v, 97.5), 3)})
ci = pd.DataFrame(rows)
ci.to_csv(OUT / "ci.csv", index=False)
print("X1  95% intervals (resampling pairs):\n", ci.pivot(index="model", columns="metric", values=["estimate", "ci_low", "ci_high"]).round(3).to_string())

# ---- X2: paired comparisons ----------------------------------------------------------------------------------------
def mcnemar_exact(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)  # Python ints: no overflow

rows = []
for a, b_ in (("ensemble", "qwen14b"), ("qwen14b", "mpnet_rf"), ("qwen14b", "tfidf_lr"), ("mpnet_rf", "tfidf_lr")):
    d = np.array([acc(a, i) - acc(b_, i) for i in boot])
    ca, cb = df[f"{a}_correct"].to_numpy(), df[f"{b_}_correct"].to_numpy()
    only_a, only_b = int(((ca == 1) & (cb == 0)).sum()), int(((ca == 0) & (cb == 1)).sum())
    rows.append({"comparison": f"{NAME[a]} minus {NAME[b_]}", "accuracy_diff": round(acc(a) - acc(b_), 4),
                 "ci_low": round(np.percentile(d, 2.5), 4), "ci_high": round(np.percentile(d, 97.5), 4),
                 "share_of_resamples_diff_le_0": round((d <= 0).mean(), 4),
                 "only_first_right": only_a, "only_second_right": only_b, "mcnemar_exact_p": round(mcnemar_exact(only_a, only_b), 4)})
cmp_ = pd.DataFrame(rows)
cmp_.to_csv(OUT / "model_comparison.csv", index=False)
print("\nX2  paired comparisons:\n", cmp_.to_string(index=False))

# ---- X3: calibration -------------------------------------------------------------------------------------------------
def ece_bins(m, nb=10):
    p, edges = df[f"{m}_prob"].to_numpy(), np.linspace(0, 1, nb + 1)
    b = np.minimum(np.digitize(p, edges[1:-1]), nb - 1)
    return [(k, int((b == k).sum()), p[b == k].mean(), y[b == k].mean()) for k in range(nb) if (b == k).any()]

rows, fig, axs = [], plt.figure(figsize=(9.5, 4.6)), None
ax = fig.add_axes([0.07, 0.14, 0.40, 0.66])
ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls="--")
for m in MODELS[1:]:
    bins = ece_bins(m)
    n = sum(b[1] for b in bins)
    ece = sum(b[1] / n * abs(b[2] - b[3]) for b in bins)
    rows.append({"model": NAME[m], "brier": round(np.mean((df[f"{m}_prob"] - y) ** 2), 4), "ece": round(ece, 4)})
    ax.plot([b[2] for b in bins], [b[3] for b in bins], marker="o", ms=4, lw=1.6, color=COLOR[m], label=NAME[m])
ax.set(xlabel="Model's P(stereotype)", ylabel="Share actually stereotype", xlim=(0, 1), ylim=(0, 1))
ax.legend(loc="upper left", fontsize=8.5)
ax2 = fig.add_axes([0.56, 0.14, 0.42, 0.66])
for m, off in (("qwen14b", -0.2), ("ensemble", 0.2)):
    hist, _ = np.histogram(df[f"{m}_prob"], bins=10, range=(0, 1))
    ax2.bar(np.arange(10) + off, hist, width=0.4, color=COLOR[m], label=NAME[m])
ax2.set(xlabel="Model's P(stereotype), 10 bins", ylabel="Sentences", xticks=range(10), xticklabels=[f"{k/10:.1f}" for k in range(10)])
ax2.legend(fontsize=8.5)
fig.text(0.01, 0.98, "Qwen is far more extreme than it is reliable", ha="left", va="top", fontsize=13, fontweight="bold", color=INK)
fig.text(0.01, 0.925, "Left: points on the dashed line mean the stated probability is honest. Right: how many sentences land in each probability bin.",
         ha="left", va="top", fontsize=9.5, color=INK2)
fig.savefig(OUT / "calibration.png", dpi=160)
cal = pd.DataFrame(rows)
cal.to_csv(OUT / "calibration.csv", index=False)
print("\nX3  calibration (lower is better):\n", cal.to_string(index=False))

# ---- X4: accuracy by confidence ----------------------------------------------------------------------------------------
edges = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0001]
labels = ["50-60", "60-70", "70-80", "80-90", "90-95", "95-99", "99-100"]
rows = []
for m in MODELS[1:]:
    conf = np.maximum(df[f"{m}_prob"], 1 - df[f"{m}_prob"])
    b = pd.cut(conf, edges, labels=labels, right=False)
    for lab in labels:
        s = df[b == lab]
        rows.append({"model": NAME[m], "confidence_%": lab, "n": len(s),
                     "accuracy": round(s[f"{m}_correct"].mean(), 3) if len(s) else np.nan})
acb = pd.DataFrame(rows)
acb.to_csv(OUT / "accuracy_by_confidence.csv", index=False)
fig, ax = plt.subplots(figsize=(8.5, 4.8))
fig.subplots_adjust(left=0.08, right=0.99, bottom=0.14, top=0.78)
for m, off in (("qwen14b", -0.2), ("ensemble", 0.2)):  # the two weaker models fill only the lowest bins, so they stay in the CSV only
    s_ = acb[acb.model == NAME[m]]
    ax.bar(np.arange(7) + off, s_.accuracy.fillna(0), width=0.4, color=COLOR[m], label=NAME[m])
    for k, (n, a) in enumerate(zip(s_.n, s_.accuracy)):
        ax.text(k + off, (0 if np.isnan(a) else a) + 0.015, f"n={n}", ha="center", fontsize=7.5, color=INK2)
ax.set(xticks=range(7), xticklabels=labels, xlabel="Model confidence, max(P, 1 - P), in %", ylabel="Accuracy", ylim=(0, 1.1))
ax.legend(ncol=2, fontsize=8.5, loc="upper left")
fig.text(0.01, 0.98, "Most of Qwen's mistakes are made at 99%+ confidence", ha="left", va="top", fontsize=13, fontweight="bold", color=INK)
fig.text(0.01, 0.925, "Accuracy within each confidence band (n = sentences in the band). A reliable model climbs from left to right.",
         ha="left", va="top", fontsize=9.5, color=INK2)
fig.savefig(OUT / "accuracy_by_confidence.png", dpi=160)
print("\nX4  accuracy by confidence (n in brackets):")
print(acb.assign(v=acb.accuracy.astype(str) + " (" + acb.n.astype(str) + ")").pivot(index="confidence_%", columns="model", values="v").loc[labels].to_string())

# ---- X5: MCC and kappa --------------------------------------------------------------------------------------------------
mk = pd.DataFrame([{"model": NAME[m], "mcc": round(matthews_corrcoef(y, df[f"{m}_pred"]), 3),
                    "cohen_kappa": round(cohen_kappa_score(y, df[f"{m}_pred"]), 3)} for m in MODELS])
mk.to_csv(OUT / "mcc_kappa.csv", index=False)
print("\nX5  chance-corrected agreement with the true labels:\n", mk.to_string(index=False))
