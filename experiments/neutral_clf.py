"""Neutral detector: is a sentence neutral, or does it express a (pro- or anti-) gender stereotype?

Positives (stereotype-related): our 1,518 labeled sentences. Negatives (neutral): one generated neutral sentence per
pair (gen_neutral.py). Snowflake-L embeddings + logistic regression, evaluated on the same 5 grouped folds (a neutral
sentence is in its pair's fold). Also scored on two neutral sets it never trains on: my 60 hand-written sentences and
StereoSet's "unrelated" gender sentences. Control: LLM paraphrases of stereotype sentences (gen_neutral.py
--paraphrase), scored by the model whose test fold holds their pair, must still be flagged as stereotype-related.

  python experiments/neutral_clf.py      # prints the evaluation, saves the final detector to models/neutral_lr.npz
"""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).parent
ROOT = HERE.parent
df = pd.read_csv(ROOT / "data" / "processed" / "merged_gender.csv").merge(
    pd.read_csv(ROOT / "data" / "processed" / "folds.csv"), on="pair_id")
gen = pd.read_csv(ROOT / "data" / "external" / "neutral_generated.csv").merge(
    pd.read_csv(ROOT / "data" / "processed" / "folds.csv"), on="pair_id")
hand = pd.read_csv(ROOT / "data" / "external" / "neutral_sentences.csv")["sentence"]
unrel = pd.read_csv(ROOT / "data" / "external" / "stereoset_unrelated_gender.csv")["sentence"]
para = pd.read_csv(ROOT / "data" / "external" / "stereo_paraphrased.csv").merge(
    pd.read_csv(ROOT / "data" / "processed" / "folds.csv"), on="pair_id")

enc = SentenceTransformer("Snowflake/snowflake-arctic-embed-l-v2.0")
embed = lambda s: enc.encode(list(s), batch_size=32, normalize_embeddings=True)
E_lab = np.load(HERE / "emb" / "snowflake-arctic-l-v2.0.npy")
E_gen, E_hand, E_unrel, E_para = embed(gen["sentence"]), embed(hand), embed(unrel), embed(para["sentence"])

X = np.vstack([E_lab, E_gen])
y = np.r_[np.ones(len(E_lab)), np.zeros(len(E_gen))]  # 1 = stereotype-related, 0 = neutral
f = np.r_[df["fold"], gen["fold"]]
L = len(E_lab)
p, p_para = np.zeros(len(y)), np.zeros(len(para))
for k in range(5):
    m = LogisticRegression(C=1.0, max_iter=2000).fit(X[f != k], y[f != k])
    p[f == k] = m.predict_proba(X[f == k])[:, 1]
    p_para[para["fold"] == k] = m.predict_proba(E_para[para["fold"] == k])[:, 1]
acc = [((p[f == k] > 0.5) == y[f == k]).mean() for k in range(5)]
print(f"CV accuracy {np.mean(acc):.3f} (sd {np.std(acc):.3f})")
print(f"  labeled sentences kept as stereotype-related: {(p[:L] > 0.5).mean():.1%}")
print(f"  generated neutral called neutral:             {(p[L:] < 0.5).mean():.1%}")
print(f"  LLM-paraphrased stereotypes kept as stereotype-related (control): {(p_para > 0.5).mean():.1%}")

m = LogisticRegression(C=1.0, max_iter=2000).fit(X, y)
for name, E in [("hand-written neutral (60)", E_hand), ("StereoSet unrelated (497)", E_unrel)]:
    print(f"  {name} called neutral: {(m.predict_proba(E)[:, 1] < 0.5).mean():.1%}")
pd.DataFrame({"sentence": hand, "p_stereo": m.predict_proba(E_hand)[:, 1]}).to_csv(HERE / "feat" / "neutral_hand_scores.csv", index=False)
np.savez(ROOT / "models" / "neutral_lr.npz", coef=m.coef_[0], intercept=m.intercept_[0])  # used by predict.py --neutral
print("saved models/neutral_lr.npz")
