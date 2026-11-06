"""Gender filter: does a sentence refer to gender at all? First step of the 3-class pipeline. GPU for embedding.

Replaces the first neutral detector (neutral_clf.py), which learned "general claim = stereotype-related" because its
neutral examples were all people doing everyday things. Here:
  positive (gender-related): labeled stereotype/anti sentences, gendered neutral sentences (old cleaned set and
                             "Men like apples."-style claims), gendered stereotype claims and their swaps
  negative (not gender):     no-people facts, animals, first person, non-gender groups, non-gender stereotypes
Whether a gender-related sentence is a stereotype is left to the 3-class Qwen judge.

Snowflake-L embeddings; logistic regression vs XGBoost on the same 5 grouped folds (data3.py). Also scored on the
hand-written test set (ood_test.csv), which nothing trains on. Saves the logistic regression to models/gender_filter.npz.

  python experiments/gender_filter.py
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import data3

df = data3.load()
test = pd.read_csv(ROOT / "data" / "external" / "ood_test.csv")
enc = SentenceTransformer("Snowflake/snowflake-arctic-embed-l-v2.0", device="cuda")
embed = lambda s: enc.encode(list(s), batch_size=64, normalize_embeddings=True)
X, Xt = embed(df["sentence"]), embed(test["sentence"])
y, f, yt = df["gender"].to_numpy(), df["fold"].to_numpy(), test["gender"].to_numpy()


def lr():
    return LogisticRegression(C=1.0, max_iter=3000)


def xgb():
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1, subsample=0.8, colsample_bytree=0.5,
                         tree_method="hist")


rows, oof = [], {}
for name, make in [("logistic regression", lr), ("xgboost", xgb)]:
    p = np.zeros(len(y))
    for k in range(5):
        p[f == k] = make().fit(X[f != k], y[f != k]).predict_proba(X[f == k])[:, 1]
    oof[name] = p
    pt = make().fit(X, y).predict_proba(Xt)[:, 1]
    cv = [((p[f == k] > 0.5) == y[f == k]).mean() for k in range(5)]
    rows.append({"model": name, "cv_acc": np.mean(cv), "cv_sd": np.std(cv),
                 "labeled_kept": (p[df["kind"].str.startswith("labeled")] > 0.5).mean(),
                 "test_acc": ((pt > 0.5) == yt).mean(),
                 "test_gender_kept": (pt[yt == 1] > 0.5).mean(), "test_nongender_removed": (pt[yt == 0] < 0.5).mean()})
res = pd.DataFrame(rows)
print(res.round(3).to_string(index=False))

p = oof["logistic regression"]
print("\nCV accuracy by kind (logistic regression):")
print(df.assign(ok=(p > 0.5) == y).groupby("kind")["ok"].agg(["mean", "size"]).round(3).to_string())

m = lr().fit(X, y)
pt = m.predict_proba(Xt)[:, 1]
miss = test.assign(p_gender=pt)[(pt > 0.5) != yt]
print(f"\ntest-set misses ({len(miss)}):")
print(miss[["sentence", "gender", "p_gender"]].round(2).to_string(index=False))
out = HERE / "feat"
res.to_csv(out / "gender_filter_results.csv", index=False)
df.assign(p_gender_lr=oof["logistic regression"], p_gender_xgb=oof["xgboost"]).to_csv(
    out / "gender_filter_oof.csv", index=False)
np.savez(ROOT / "models" / "gender_filter.npz", coef=m.coef_[0], intercept=m.intercept_[0])
print("saved models/gender_filter.npz")
