"""Step E3 of docs/MetricsAnalysisPlan.md: three-class results (stereotype / anti-stereotype / neutral) for the pipeline.

Pipeline (experiments/predict.py --neutral): if the neutral detector says P(stereotype-related) < 0.5 the sentence is called
"neutral"; otherwise Qwen decides stereotype vs anti. Qwen cannot be run on CPU, so Qwen's answers exist only for the 1,518
labeled sentences (saved out-of-fold predictions). Neutral test sentences that pass the detector therefore go in a
"passed to Qwen (not run)" column; nothing is invented for them.

Rows (true class) and where the detector scores come from:
  stereotype / anti    759 + 759 labeled sentences, out-of-fold detector (same recipe and folds as neutral_clf.py)
  neutral, generated   734 Qwen-7B-written neutral sentences, out-of-fold detector (these are the detector's own negatives)
  neutral, hand-written  60 sentences by Chris, saved detector (never trained on them)
  neutral, StereoSet unrelated  497 gender sentences with an unrelated second sentence, saved detector (never trained on them)
Also: the audit sentences Julia labeled "neutral", and whether the detector discards more of Qwen's errors than its correct answers.
Runs on CPU (~/cosc72-stereotype-judge/.venv-neutral). Writes outputs/metrics/three_class.csv and three_class_confusion.png.
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
EXT = ROOT / "data" / "external"
folds = pd.read_csv(ROOT / "data" / "processed" / "folds.csv")
pred = pd.read_csv(ROOT / "outputs" / "metrics" / "predictions_all.csv")
oof = pd.read_csv(ROOT / "outputs" / "metrics" / "neutral_oof.csv")  # from 08_neutral_oof.py
assert (oof.pair_id.to_numpy() == oof.pair_id.to_numpy()).all()
lab = pred.merge(oof[["pair_id", "sentence", "p_stereo_related"]], on=["pair_id", "sentence"])
assert len(lab) == len(pred) == 1518

gen = pd.read_csv(EXT / "neutral_generated.csv").merge(folds, on="pair_id")
hand = pd.read_csv(EXT / "neutral_sentences.csv").sentence
unrel = pd.read_csv(EXT / "stereoset_unrelated_gender.csv").sentence
enc = SentenceTransformer("Snowflake/snowflake-arctic-embed-l-v2.0")
embed = lambda s: enc.encode(list(s), batch_size=32, normalize_embeddings=True)

# Out-of-fold scores for the generated neutral sentences (same fold recipe as 08), saved detector for the held-out neutral sets.
lab_full = pd.read_csv(ROOT / "data" / "processed" / "merged_gender.csv").merge(folds, on="pair_id")
E_lab, E_gen, E_hand, E_unrel = embed(lab_full.sentence), embed(gen.sentence), embed(hand), embed(unrel)
X, y = np.vstack([E_lab, E_gen]), np.r_[np.ones(len(E_lab)), np.zeros(len(E_gen))]
f = np.r_[lab_full.fold, gen.fold]
p_gen = np.zeros(len(gen))
for k in range(5):
    m = LogisticRegression(C=1.0, max_iter=2000).fit(X[f != k], y[f != k])
    p_gen[(gen.fold == k).to_numpy()] = m.predict_proba(E_gen[(gen.fold == k).to_numpy()])[:, 1]
final = LogisticRegression(C=1.0, max_iter=2000).fit(X, y)
p_hand, p_unrel = final.predict_proba(E_hand)[:, 1], final.predict_proba(E_unrel)[:, 1]

# ---- Confusion table -------------------------------------------------------------------------------------------------
COLS = ["called neutral", "called stereotype", "called anti", "passed to Qwen (not run)"]
rows = []
for name, label in (("true: stereotype", 1), ("true: anti-stereotype", 0)):
    s = lab[lab.label == label]
    neutral = s.p_stereo_related < 0.5
    passed = s[~neutral]
    rows.append([name, int(neutral.sum()), int((passed.qwen14b_pred == 1).sum()), int((passed.qwen14b_pred == 0).sum()), 0])
for name, p in (("true: neutral (generated, out-of-fold)", p_gen), ("true: neutral (hand-written)", p_hand),
                ("true: neutral (StereoSet unrelated)", p_unrel)):
    rows.append([name, int((p < 0.5).sum()), 0, 0, int((p >= 0.5).sum())])
cm = pd.DataFrame(rows, columns=["row"] + COLS).set_index("row")
cm["total"] = cm.sum(axis=1)
cm.to_csv(ROOT / "outputs" / "metrics" / "three_class.csv")
print("Three-class table (counts):\n", cm.to_string())

# ---- What the detector does to the stereotype-vs-anti job -------------------------------------------------------------------
neutral = lab.p_stereo_related < 0.5
for m in ("qwen14b", "ensemble"):
    ok = lab[f"{m}_correct"] == 1
    print(f"\n{m}: accuracy alone {ok.mean():.3f}; through the pipeline (neutral call counts as wrong) {(ok & ~neutral).mean():.3f}; "
          f"on sentences that pass the detector {ok[~neutral].mean():.3f} (n={int((~neutral).sum())})")
    print(f"  detector calls neutral: {int(neutral.sum())} labeled sentences; of those {m} was right on {int((ok & neutral).sum())} "
          f"and wrong on {int((~ok & neutral).sum())} (overall error rate {(~ok).mean():.3f}, on the neutral-called {(~ok[neutral]).mean():.3f})")

# ---- Audit sentences Julia labeled neutral ----------------------------------------------------------------------------------
aud = pd.read_csv(ROOT / "outputs" / "audit" / "audit_results_single.csv").merge(
    oof[["sentence", "p_stereo_related"]], on="sentence", how="left")
for lbl in ("neutral", "unsure"):
    a = aud[aud.your_label.str.strip().str.lower() == lbl]
    print(f"\naudit sentences Julia labeled '{lbl}': n={len(a)}, detector called neutral {int((a.p_stereo_related < 0.5).sum())}, "
          f"scores {sorted(a.p_stereo_related.round(2))}")
a = aud[aud.your_label.str.strip().str.lower().isin(["stereotypical", "anti-stereotypical"])]
print(f"audit sentences Julia labeled stereotypical/anti: n={len(a)}, detector called neutral {int((a.p_stereo_related < 0.5).sum())}")

# ---- Chart --------------------------------------------------------------------------------------------------------------------
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("c", [SURFACE, "#2a78d6"])
share = cm[COLS].div(cm["total"], axis=0)
fig = plt.figure(figsize=(9.6, 4.9), facecolor=SURFACE)
ax = fig.add_axes([0.34, 0.10, 0.64, 0.60], facecolor=SURFACE)
ax.imshow(share.to_numpy(), cmap=cmap, vmin=0, vmax=1, aspect="auto")
for i in range(len(cm)):
    for j, c in enumerate(COLS):
        v = int(cm.iloc[i][c])
        ax.text(j, i, f"{v}\n{share.iloc[i, j]:.0%}", ha="center", va="center", fontsize=9,
                color="white" if share.iloc[i, j] > 0.55 else INK)
ax.set_xticks(range(4), [c.replace(" (", "\n(") for c in COLS], fontsize=9, color=INK2)
ax.set_yticks(range(len(cm)), [f"{r.split(': ')[1]}  (n={int(t)})" for r, t in zip(cm.index, cm.total)], fontsize=9, color=INK2)
ax.xaxis.tick_top()
for s_ in ax.spines.values():
    s_.set_visible(False)
ax.set_ylabel("True class", color=INK2)
fig.text(0.01, 0.98, "Neutral check: keeps stereotypes in, misses most 'unrelated' text", ha="left", va="top",
         fontsize=12.5, fontweight="bold", color=INK)
fig.text(0.01, 0.915, "Pipeline: neutral check first, then Qwen. Counts and row shares. Qwen was not run on the neutral sentences,\n"
         "so those that pass the check are not split into stereotype / anti.", ha="left", va="top", fontsize=8.8, color=INK2)
fig.savefig(ROOT / "outputs" / "metrics" / "three_class_confusion.png", dpi=160)
