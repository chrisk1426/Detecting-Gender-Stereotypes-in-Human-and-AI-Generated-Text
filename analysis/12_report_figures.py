"""Report figures for the Results section (Figures 1-8). Reads saved outputs only; writes PNGs to outputs/report_figures/.

Fig 4 reads outputs/audit/private/audit_key_single.csv (git-ignored; regenerate with 03_make_audit_sheet.py).
Hand-entered numbers: the Fig 4 labeler counts come from 13_score_second_labeler.py and 14_audit_vs_three_class.py,
and the Fig 7 and 8 percentages from outputs/shap/*/summary.txt.
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
M = REPO / "outputs" / "metrics"
A = REPO / "outputs" / "audit"
OUT = REPO / "outputs" / "report_figures"; OUT.mkdir(exist_ok=True)

BG, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, AQUA, GREY, LGREY = "#2a78d6", "#eb6834", "#1baf7a", "#9b9a93", "#d6d5cf"
plt.rcParams.update({"font.size": 10, "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
                     "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.spines.left": False, "axes.grid": False, "legend.frameon": False})


def head(fig, title, sub, top=None, extra=0.0):
    import textwrap
    W, H = fig.get_figwidth(), fig.get_figheight()
    tl = textwrap.wrap(title, int(W * 9.2)); sl = textwrap.wrap(sub, int(W * 12.5))
    y = 0.12
    fig.text(0.012, 1 - y / H, "\n".join(tl), fontsize=13, weight="bold", va="top", linespacing=1.25)
    y += 0.25 * len(tl) + 0.06
    fig.text(0.012, 1 - y / H, "\n".join(sl), fontsize=9.5, color=INK2, va="top", linespacing=1.3)
    y += 0.19 * len(sl) + 0.30 + extra
    fig.subplots_adjust(top=1 - y / H)


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def hgrid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=0.8); ax.set_axisbelow(True)


# ---------- Fig 1: accuracy by model with 95% CI ----------
ci = pd.read_csv(M / "ci.csv"); ci = ci[ci.metric == "accuracy"]
order = ["Majority guess", "TF-IDF + LR", "mpnet + random forest", "Qwen 14B (LoRA)", "Ensemble (9 models)"]
lab = ["Majority\nguess", "TF-IDF +\nlogistic regression", "mpnet +\nrandom forest", "Qwen 14B\n(LoRA)", "Stacked ensemble\n(9 models)"]
r = ci.set_index("model").loc[order]
fig, ax = plt.subplots(figsize=(8, 5))
cols = [GREY, GREY, GREY, BLUE, BLUE]
x = np.arange(5)
ax.bar(x, r.estimate, width=0.5, color=cols, zorder=3)
ax.errorbar(x, r.estimate, yerr=[r.estimate - r.ci_low, r.ci_high - r.estimate], fmt="none", ecolor=INK, elinewidth=1.2, capsize=4, zorder=4)
for xi, (e, hi) in enumerate(zip(r.estimate, r.ci_high)):
    ax.text(xi, hi + 0.02, f"{e:.3f}", ha="center", fontsize=10, weight="bold")
ax.axhline(0.5, color=INK2, ls="--", lw=1, zorder=2, label="Chance (0.50)")
ax.legend(loc="upper left")
ax.set_xticks(x); ax.set_xticklabels(lab); ax.set_ylim(0, 1.0); ax.set_ylabel("Accuracy (out-of-fold, 1,518 sentences)")
hgrid(ax)
head(fig, "Accuracy by model",
     "Fine-tuning a 14B language model lifts accuracy from about 0.60 to 0.82. Bars show out-of-fold accuracy with 95% confidence intervals (resampling whole pairs, 5,000 times). Grey = baselines, blue = larger models.")
save(fig, "fig1_accuracy_by_model.png")

# ---------- Fig 2: pair outcomes ----------
po = pd.read_csv(M / "pair_outcomes.csv")
names = {"TF-IDF + LR": "TF-IDF + logistic regression", "mpnet + random forest": "mpnet + random forest",
         "Qwen 14B (LoRA)": "Qwen 14B (LoRA)", "Ensemble (9 models)": "Stacked ensemble (9 models)"}
fig, ax = plt.subplots(figsize=(8.5, 4.2))
rows = po.iloc[::-1].reset_index(drop=True)
for i, rw in rows.iterrows():
    parts = [rw.both_right, rw.same_label_to_both, rw.swapped]; left = 0
    for p, c, tc in zip(parts, [BLUE, ORANGE, AQUA], ["white", INK, INK]):
        w = p / rw.pairs
        ax.barh(i, w - 0.004, left=left, color=c, height=0.5, zorder=3)
        if w >= 0.08: ax.text(left + w / 2, i, f"{w*100:.0f}%", ha="center", va="center", color=tc, fontsize=10, weight="bold")
        else: ax.text(left + w + 0.008, i, f"{w*100:.0f}%", ha="left", va="center", fontsize=9.5, color=INK, weight="bold")
        left += w
ax.set_yticks(range(len(rows))); ax.set_yticklabels([names[m] for m in rows.model])
ax.set_xlim(0, 1.08); ax.set_xticks([0, .25, .5, .75, 1]); ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
ax.set_xlabel("Share of the 759 pairs")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=BLUE, label="Both sentences right"), Patch(color=ORANGE, label="Same label to both sentences"),
                   Patch(color=AQUA, label="Labels swapped")], loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
head(fig, "Pair outcomes",
     "Most mistakes come from giving both sentences of a pair the same label. Each pair has one stereotype and one anti-stereotype sentence; bars show what each model did with it (out-of-fold).")
save(fig, "fig2_pair_outcomes.png")

# ---------- Fig 3: accuracy by confidence band (Qwen and ensemble) ----------
ab = pd.read_csv(M / "accuracy_by_confidence.csv")
bands = ["50-60", "60-70", "70-80", "80-90", "90-95", "95-99", "99-100"]
mid = [0.55, 0.65, 0.75, 0.85, 0.925, 0.97, 0.995]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
for ax, m, ttl in zip(axes, ["Qwen 14B (LoRA)", "Ensemble (9 models)"], ["Qwen 14B (LoRA)", "Stacked ensemble (9 models)"]):
    d = ab[ab.model == m].set_index("confidence_%").reindex(bands)
    x = np.arange(len(bands))
    ax.bar(x, d.accuracy, width=0.55, color=BLUE, zorder=3)
    ax.scatter(x, mid, marker="_", s=420, color=INK, linewidths=1.6, zorder=5, label="Accuracy if perfectly calibrated")
    for xi, (a, n) in enumerate(zip(d.accuracy, d.n)):
        if n and not np.isnan(a): ax.text(xi, 0.03, f"{a*100:.0f}%", ha="center", color="white", fontsize=8, weight="bold", zorder=6) if a > 0.15 else None
    ax.set_xticks(x); ax.set_xticklabels([f"{b.replace('-', '–')}%\nn={int(n):,}" for b, n in zip(bands, d.n)], fontsize=8.5)
    ax.set_title(ttl, fontsize=11, weight="bold", loc="left", color=INK); ax.set_ylim(0, 1.05); hgrid(ax)
    ax.set_xlabel("Model's stated confidence in its chosen label")
axes[0].set_ylabel("Share of sentences it got right"); axes[0].legend(loc="upper left", fontsize=8.5)
head(fig, "Accuracy by confidence",
     "Qwen is 99%+ confident on most sentences but right only 86% of the time there. Accuracy within each confidence band (out-of-fold); dashes mark the accuracy a perfectly calibrated model would have. 169 of Qwen's 273 errors fall in the 99–100% band.", extra=0.15)
save(fig, "fig3_accuracy_by_confidence.png")

# ---------- Fig 4: audit labels by labeler ----------
key = pd.read_csv(REPO / "outputs" / "audit" / "private" / "audit_key_single.csv")
ds = key.true_label.value_counts()
rows = [("Dataset label", [ds.get(1, 0), ds.get(0, 0), 0, 0]),
        ("Human 1 (blind)", [17, 13, 10, 0]),
        ("Human 2 (blind)", [11, 2, 23, 4]),
        ("Three-class model", [20, 20, 0, 0])]
cats = [("Stereotypical", BLUE, "white"), ("Anti-stereotypical", ORANGE, INK), ("Neutral", GREY, INK), ("Unsure", LGREY, INK)]
fig, ax = plt.subplots(figsize=(8.5, 4.0))
for i, (nm, vals) in enumerate(rows[::-1]):
    left = 0
    for v, (cn, c, tc) in zip(vals, cats):
        if v:
            ax.barh(i, v - 0.25, left=left, color=c, height=0.5, zorder=3)
            ax.text(left + v / 2, i, str(int(v)), ha="center", va="center", color=tc, fontsize=10, weight="bold")
        left += v
ax.set_yticks(range(4)); ax.set_yticklabels([r[0] for r in rows[::-1]]); ax.set_xlim(0, 40); ax.set_xlabel("Number of the 40 audit sentences")
ax.legend(handles=[Patch(color=c, label=n) for n, c, _ in cats], loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4)
head(fig, "Audit labels by labeler",
     "Human 2 labeled more than half the audit sentences neutral; the dataset and the model labeled none. Each row shows how a labeler (or the model) labeled the same 40 single sentences. The dataset has only stereotype and anti-stereotype labels.")
save(fig, "fig4_audit_labels.png")

# ---------- Fig 5: categories of reviewed Qwen errors ----------
er = pd.read_csv(A / "error_review.csv").category.value_counts()
nice = {"real model miss": "Real model miss", "possibly neutral / not really about gender": "Possibly neutral, or not about gender",
        "contested stereotype": "Contested stereotype", "questionable label": "Questionable label",
        "role vs statement": "Role vs. statement", "bad data": "Bad data"}
er = er.sort_values()
fig, ax = plt.subplots(figsize=(8.5, 4.2))
cols = [BLUE if k == "real model miss" else ORANGE for k in er.index]
ax.barh(range(len(er)), er.values, color=cols, height=0.5, zorder=3)
for i, v in enumerate(er.values): ax.text(v + 0.2, i, str(v), va="center", weight="bold")
ax.set_yticks(range(len(er))); ax.set_yticklabels([nice[k] for k in er.index]); ax.set_xlim(0, 11)
ax.set_xlabel("Number of the 30 reviewed Qwen errors"); hgrid(ax, "x")
ax.legend(handles=[Patch(color=BLUE, label="The model's mistake"), Patch(color=ORANGE, label="A label or data problem")], loc="lower right")
head(fig, "Categories of reviewed errors",
     "Only 9 of 30 reviewed Qwen errors are clear model misses. Each error was read and categorized by hand (10 per source); the review is not blind, because we already knew they were errors.")
save(fig, "fig5_error_categories.png")

# ---------- Fig 6: old vs new neutral pipeline on hand-written test ----------
t = pd.read_csv(REPO / "experiments" / "feat" / "ood_test_summary.csv")
tot = t[t.category == "ALL"].iloc[0]; t = t[t.category != "ALL"].sort_values("new")
nm = {"animals": "Animals", "gender_anti_stereotype": "Anti-stereotypes about gender", "gender_fact": "Gendered facts",
      "gender_neutral_claim": "Gendered neutral claims", "gender_stereotype": "Stereotypes about gender",
      "no_gender_group": "Groups not defined by gender", "no_gender_person": "Individuals, no gender mentioned",
      "no_people": "Sentences with no people", "non_gender_stereotype": "Non-gender stereotypes"}
fig, ax = plt.subplots(figsize=(9, 5.2))
y = np.arange(len(t)); h = 0.34
ax.barh(y + h / 2, t.old, height=h - 0.03, color=GREY, zorder=3, label="Earlier pipeline (neutral detector + two-class judges)")
ax.barh(y - h / 2, t.new, height=h - 0.03, color=BLUE, zorder=3, label="New pipeline (gender filter + three-class Qwen)")
for yi, (o, n_) in enumerate(zip(t.old, t.new)):
    ax.text(o + 0.01, yi + h / 2, f"{o:.2f}", va="center", fontsize=8.5); ax.text(n_ + 0.01, yi - h / 2, f"{n_:.2f}", va="center", fontsize=8.5, weight="bold")
ax.set_yticks(y); ax.set_yticklabels([f"{nm[c]} (n={int(n)})" for c, n in zip(t.category, t.n)], fontsize=9)
ax.set_xlim(0, 1.12); ax.set_xlabel("Accuracy on hand-written test sentences (3 classes: stereotype, anti-stereotype, neutral)"); hgrid(ax, "x")
ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.13), ncol=1, fontsize=9)
head(fig, "Earlier vs. new neutral pipeline",
     f"The new three-class pipeline reaches {tot.new:.2f} on the test set, up from {tot.old:.2f}. Accuracy by sentence type on 92 hand-written sentences that nothing trains on (written by Claude, not yet reviewed).")
save(fig, "fig6_neutral_pipeline.png")

# ---------- Fig 7: SHAP attribution share ----------
rows = [("Qwen (first adapter),\ntraining data", [24, 4, 43, 23, 6]), ("Qwen (adapter trained with GEST),\ntraining data", [24, 4, 41, 23, 8]),
        ("Qwen on held-out\nGEST pairs", [17, 0, 36, 30, 17]), ("Neutral detector", [19, 2, 42, 26, 11])]
cats = [("Attribute words", BLUE, "white"), ("Gender words", ORANGE, INK), ("First names", AQUA, INK), ("Function words", GREY, INK), ("Punctuation", LGREY, INK)]
order_idx = [2, 0, 1, 3, 4]   # attribute, gender, names, function, punct
fig, ax = plt.subplots(figsize=(9, 4.6))
for i, (nm_, v) in enumerate(rows[::-1]):
    left = 0
    for k, (cn, c, tc) in zip(order_idx, cats):
        w = v[k]
        if w:
            ax.barh(i, w - 0.3, left=left, color=c, height=0.5, zorder=3)
            if w >= 5: ax.text(left + w / 2, i, f"{w}%", ha="center", va="center", color=tc, fontsize=9.5, weight="bold")
        left += w
ax.set_yticks(range(4)); ax.set_yticklabels([r[0] for r in rows[::-1]], fontsize=9); ax.set_xlim(0, 100)
ax.set_xlabel("Share of total SHAP attribution (absolute value)")
ax.legend(handles=[Patch(color=c, label=n) for n, c, _ in cats], loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=5, fontsize=8.5)
head(fig, "SHAP attribution by word type",
     "Attribute words carry more of the attribution than gender words in every model. Bars show where the SHAP attribution goes, by word type.")
save(fig, "fig7_shap_attribution.png")

# ---------- Fig 8: SHAP sign flips under gender swap ----------
rows = [("Qwen (first adapter), CrowS gender swaps", 49, BLUE), ("Qwen (adapter trained with GEST), CrowS swaps", 50, BLUE),
        ("Qwen on held-out GEST swaps", 40, BLUE), ("Neutral detector, CrowS swaps", 4, GREY)]
fig, ax = plt.subplots(figsize=(9, 3.9))
for i, (nm_, v, c) in enumerate(rows[::-1]):
    ax.barh(i, v, color=c, height=0.5, zorder=3); ax.text(v + 1, i, f"{v}%", va="center", weight="bold")
ax.set_yticks(range(4)); ax.set_yticklabels([r[0] for r in rows[::-1]], fontsize=9); ax.set_xlim(0, 65); hgrid(ax, "x")
ax.set_xlabel("Share of shared words whose SHAP value changes sign when only the gender is swapped")
head(fig, "SHAP sign flips under gender swap",
     "Qwen reads attribute words together with gender; the neutral detector does not. In sentence pairs that differ only by gender, 40–50% of Qwen's word effects flip direction, against 4% for the neutral detector.")
save(fig, "fig8_shap_sign_flips.png")
print("ok", sorted(p.name for p in OUT.glob("*.png")))
