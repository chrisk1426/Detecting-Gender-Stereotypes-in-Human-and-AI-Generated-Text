"""Step B of docs/MetricsAnalysisPlan.md: core metrics and charts (B1-B8) from outputs/metrics/predictions_all.csv.

Tables written to outputs/metrics/ (every chart also has its numbers in a table):
  summary.csv      accuracy (mean of 5 folds), fold sd, macro F1, AUC, pairwise accuracy        (B1, B8)
  per_label.csv    precision / recall / F1 for each label                                        (B4)
  by_source.csv    accuracy per data source                                                      (B6)
  by_fold.csv      accuracy per fold                                                             (B7)
Charts written to outputs/metrics/:
  accuracy_by_model.png (B2)  confusion_matrices.png (B3)  roc.png (B5)  by_source.png (B6)  by_fold.png (B7)
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "metrics"
df = pd.read_csv(OUT / "predictions_all.csv")
y = df["label"].to_numpy()

MODELS = ["majority", "tfidf_lr", "mpnet_rf", "qwen14b", "ensemble"]  # chance -> bag of words -> embeddings -> Qwen -> ensemble
NAME = {"majority": "Majority guess", "tfidf_lr": "TF-IDF + LR", "mpnet_rf": "mpnet + random forest",
        "qwen14b": "Qwen 14B (LoRA)", "ensemble": "Ensemble (9 models)"}
SOURCE = {"crows": "CrowS-Pairs", "stereoset_inter": "StereoSet inter", "stereoset_intra": "StereoSet intra"}
# Fixed colour per model (palette slots 1-4, validated for adjacent pairs); the baseline is neutral grey.
COLOR = {"majority": "#9b9a94", "tfidf_lr": "#2a78d6", "mpnet_rf": "#eb6834", "qwen14b": "#1baf7a", "ensemble": "#eda100"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e8e7e3"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1, "axes.axisbelow": True,
                     "font.size": 10, "legend.frameon": False})


def title(fig, main, sub):
    fig.text(0.01, 0.98, main, ha="left", va="top", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.01, 0.925, sub, ha="left", va="top", fontsize=9.5, color=INK2)


def fold_acc(m):
    return np.array([df.loc[df.fold == k, f"{m}_correct"].mean() for k in range(5)])


def pairwise(m):
    """Share of pairs where the stereotype sentence gets a higher P(stereotype) than its anti-stereotype partner."""
    pr = df.pivot_table(index="pair_id", columns="label", values=f"{m}_prob")
    return (pr[1] > pr[0]).mean()


# ---- B1, B8: one row per model --------------------------------------------------------------------------
summary = pd.DataFrame([dict(model=m, accuracy=fold_acc(m).mean(), fold_sd=fold_acc(m).std(),
                             macro_f1=f1_score(y, df[f"{m}_pred"], average="macro", zero_division=0),
                             auc=roc_auc_score(y, df[f"{m}_prob"]),
                             pairwise_acc=np.nan if m == "majority" else pairwise(m)) for m in MODELS])
summary.to_csv(OUT / "summary.csv", index=False)

# ---- B4: precision / recall / F1 per label --------------------------------------------------------------
rows = []
for m in MODELS:
    p, r, f1, s = precision_recall_fscore_support(y, df[f"{m}_pred"], labels=[0, 1], zero_division=0)
    for i, lab in enumerate(["anti-stereotype (0)", "stereotype (1)"]):
        rows.append(dict(model=m, label=lab, precision=p[i], recall=r[i], f1=f1[i], support=int(s[i])))
per_label = pd.DataFrame(rows)
per_label.to_csv(OUT / "per_label.csv", index=False)

# ---- B6, B7: accuracy by source and by fold (tables first, charts below) ---------------------------------
by_source = df.groupby("source")[[f"{m}_correct" for m in MODELS]].mean().loc[list(SOURCE)]
by_source.columns = MODELS
by_source.insert(0, "n_sentences", df.groupby("source").size().loc[list(SOURCE)])
by_source.to_csv(OUT / "by_source.csv")
by_fold = pd.DataFrame({m: fold_acc(m) for m in MODELS}).rename_axis("fold")
by_fold.to_csv(OUT / "by_fold.csv")

# ---- B2: accuracy per model -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 3.9))
yp = np.arange(len(MODELS))[::-1]
acc, sd = summary["accuracy"].to_numpy(), summary["fold_sd"].to_numpy()
ax.barh(yp, acc, height=0.5, color=[COLOR[m] for m in MODELS])
ax.errorbar(acc, yp, xerr=sd, fmt="none", ecolor=INK2, elinewidth=1, capsize=3)
for a, s, yy in zip(acc, sd, yp):
    ax.text(a + s + 0.015, yy, f"{a:.3f}", va="center", fontsize=10, color=INK)
ax.set_yticks(yp, [NAME[m] for m in MODELS])
ax.set_xlim(0, 1)
ax.set_xlabel("Accuracy (0.5 = coin flip)")
ax.grid(axis="y", visible=False)
ax.tick_params(axis="y", length=0)
title(fig, "Accuracy on sentences the model never trained on",
      "Mean of 5 folds; whisker = one fold standard deviation. 1,518 sentences.")
fig.tight_layout(rect=(0, 0, 1, 0.88))
fig.savefig(OUT / "accuracy_by_model.png", dpi=200)
plt.close(fig)

# ---- B3: confusion matrices -----------------------------------------------------------------------------
cmap = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#184f95"])
show = ["tfidf_lr", "mpnet_rf", "qwen14b", "ensemble"]
fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.9))
for ax, m in zip(axes, show):
    cm = np.zeros((2, 2), int)
    for t, p in zip(y, df[f"{m}_pred"]):
        cm[t, p] += 1
    share = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(share, cmap=cmap, vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n{share[i, j]:.0%}", ha="center", va="center", fontsize=11,
                    color="white" if share[i, j] > 0.5 else INK)
    ax.set_xticks([0, 1], ["anti", "stereotype"])
    ax.set_yticks([0, 1], ["anti", "stereotype"], rotation=90, va="center")
    ax.set_xlabel("Model said")
    ax.set_title(NAME[m], fontsize=10.5, color=INK, pad=8)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
axes[0].set_ylabel("True label")
title(fig, "Where each model's mistakes go",
      "Rows = true label, columns = model's answer. Each cell: sentence count and share of that true label.")
fig.subplots_adjust(left=0.05, right=0.99, bottom=0.17, top=0.78, wspace=0.28)  # tight_layout clips the label on equal-aspect panels
fig.savefig(OUT / "confusion_matrices.png", dpi=200)
plt.close(fig)

# ---- B5: ROC curves -------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.6, 5.6))
ax.plot([0, 1], [0, 1], color="#9b9a94", linewidth=1)
for m in show:
    fpr, tpr, _ = roc_curve(y, df[f"{m}_prob"])
    ax.plot(fpr, tpr, color=COLOR[m], linewidth=2, solid_capstyle="round",
            label=f"{NAME[m]}  (AUC {summary.set_index('model').loc[m, 'auc']:.3f})")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.01)
ax.set_aspect("equal")
ax.set_xlabel("False positive rate (anti sentences called stereotype)")
ax.set_ylabel("True positive rate (stereotypes found)")
ax.legend(loc="lower right", fontsize=9)
title(fig, "ROC curves", "Grey diagonal = coin flip (AUC 0.5). Closer to the top-left corner is better.")
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.savefig(OUT / "roc.png", dpi=200)
plt.close(fig)

# ---- B6: accuracy by source -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 4.3))
w, step = 0.17, 0.19
x = np.arange(len(SOURCE))
for i, m in enumerate(show):
    pos = x + (i - 1.5) * step
    vals = by_source[m].to_numpy()
    ax.bar(pos, vals, width=w, color=COLOR[m], label=NAME[m])
    if m == "ensemble":
        for px, v in zip(pos, vals):
            ax.text(px, v + 0.012, f"{v:.2f}", ha="center", fontsize=9, color=INK)
ax.axhline(0.5, color="#9b9a94", linewidth=1)
ax.text(2.48, 0.505, "coin flip", ha="right", va="bottom", fontsize=8.5, color=INK2)
ax.set_xticks(x, [f"{SOURCE[s]}\n(n={by_source.loc[s, 'n_sentences']})" for s in SOURCE])
ax.set_ylim(0, 1)
ax.set_ylabel("Accuracy")
ax.grid(axis="x", visible=False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=4, fontsize=9)
title(fig, "Accuracy by data source", "Pooled over all folds. Labels show the ensemble.")
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.savefig(OUT / "by_source.png", dpi=200)
plt.close(fig)

# ---- B7: accuracy by fold -------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 4.3))
for i, m in enumerate(show):
    xs = np.arange(5) + (i - 1.5) * 0.12
    ax.plot(xs, by_fold[m], "o", color=COLOR[m], markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5,
            label=NAME[m])
ax.axhline(0.5, color="#9b9a94", linewidth=1)
ax.text(4.45, 0.505, "coin flip", ha="right", va="bottom", fontsize=8.5, color=INK2)
ax.set_xticks(range(5), [f"fold {k}" for k in range(5)])
ax.set_ylim(0.4, 0.95)
ax.set_ylabel("Accuracy")
ax.grid(axis="x", visible=False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=9)
title(fig, "Accuracy in each of the 5 folds", "Each dot is one model on one held-out fold. Spread shows how stable the result is.")
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.savefig(OUT / "by_fold.png", dpi=200)
plt.close(fig)

# ---- Print tables and check against the README -----------------------------------------------------------
pd.set_option("display.width", 200)
print(summary.round(3).to_string(index=False), "\n")
print(per_label.round(3).to_string(index=False), "\n")
print(by_source.round(3).to_string(), "\n")
print(by_fold.round(3).to_string(), "\n")

README = {("pairwise", "ensemble"): 0.891, ("pairwise", "mpnet_rf"): 0.675, ("sd", "ensemble"): 0.020,
          ("source", "crows"): 0.573, ("source", "stereoset_inter"): 0.597, ("source", "stereoset_intra"): 0.643}
got = {("pairwise", "ensemble"): summary.set_index("model").loc["ensemble", "pairwise_acc"],
       ("pairwise", "mpnet_rf"): summary.set_index("model").loc["mpnet_rf", "pairwise_acc"],
       ("sd", "ensemble"): summary.set_index("model").loc["ensemble", "fold_sd"],
       **{("source", s): by_source.loc[s, "mpnet_rf"] for s in SOURCE}}
bad = [f"{k}: got {got[k]:.3f}, README {v:.3f}" for k, v in README.items() if abs(got[k] - v) > 0.01]
print("CHECK FAILED:\n  " + "\n  ".join(bad) if bad else "Check passed: pairwise accuracy, fold sd and accuracy by source match the README within 0.01.")
