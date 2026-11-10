"""Step E2 of docs/MetricsAnalysisPlan.md: out-of-fold scores from Chris's neutral detector for every labeled sentence.

Same recipe as experiments/neutral_clf.py (Snowflake-L embeddings, logistic regression C=1, labeled sentences = 1,
generated neutral sentences = 0, the project's 5 grouped folds), but it saves each labeled sentence's score from the
model that did NOT train on its pair. Sanity check: CV accuracy should match the README (0.924).
Runs on CPU (~/cosc72-stereotype-judge/.venv-neutral). Writes outputs/metrics/neutral_oof.csv (pair_id, sentence, label,
fold, p_stereo_related) and outputs/audit/error_review_neutral.csv (the 30 reviewed errors with these scores).
"""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
folds = pd.read_csv(ROOT / "data" / "processed" / "folds.csv")
lab = pd.read_csv(ROOT / "data" / "processed" / "merged_gender.csv").merge(folds, on="pair_id")
gen = pd.read_csv(ROOT / "data" / "external" / "neutral_generated.csv").merge(folds, on="pair_id")
enc = SentenceTransformer("Snowflake/snowflake-arctic-embed-l-v2.0")
embed = lambda s: enc.encode(list(s), batch_size=32, normalize_embeddings=True)
E_lab, E_gen = embed(lab.sentence), embed(gen.sentence)

X, y = np.vstack([E_lab, E_gen]), np.r_[np.ones(len(lab)), np.zeros(len(gen))]
f, L = np.r_[lab.fold, gen.fold], len(lab)
p = np.zeros(len(y))
for k in range(5):
    p[f == k] = LogisticRegression(C=1.0, max_iter=2000).fit(X[f != k], y[f != k]).predict_proba(X[f == k])[:, 1]
acc = np.mean([((p[f == k] > 0.5) == y[f == k]).mean() for k in range(5)])
print(f"CV accuracy {acc:.3f} (README says 0.924); labeled kept as stereotype-related {(p[:L] > .5).mean():.1%}; "
      f"generated neutral called neutral {(p[L:] < .5).mean():.1%}")

oof = lab[["pair_id", "sentence", "label", "fold"]].assign(p_stereo_related=p[:L].round(4))
oof.to_csv(ROOT / "outputs" / "metrics" / "neutral_oof.csv", index=False)

er = pd.read_csv(ROOT / "outputs" / "audit" / "error_review.csv").merge(oof[["pair_id", "sentence", "p_stereo_related"]], on=["pair_id", "sentence"], how="left")
assert er.p_stereo_related.notna().all() and len(er) == 30
er["detector_says"] = np.where(er.p_stereo_related < 0.5, "neutral", "stereotype-related")
er["my_label_neutral"] = er.category.eq("possibly neutral / not really about gender")
er.drop(columns=["partner_sentence"]).to_csv(ROOT / "outputs" / "audit" / "error_review_neutral.csv", index=False)
print("\nout-of-fold detector on the 30 reviewed errors, by my category:")
print(er.groupby("category").agg(n=("review_id", "size"), called_neutral=("detector_says", lambda s: (s == "neutral").sum()),
                                 mean_p=("p_stereo_related", "mean")).round(2).to_string())
a, b = er[er.my_label_neutral], er[~er.my_label_neutral]
print(f"\nmy 'possibly neutral' (n={len(a)}): called neutral {(a.detector_says == 'neutral').sum()}, mean P {a.p_stereo_related.mean():.2f}")
print(f"all other errors   (n={len(b)}): called neutral {(b.detector_says == 'neutral').sum()}, mean P {b.p_stereo_related.mean():.2f}")
