"""3-class baselines on the 3-class judge's data and folds (3,774 sentences, 5 grouped folds).

Same recipes as the 2-class versions:
  majority      most frequent training-fold class
  tfidf_lr      TF-IDF (1-gram or 1+2-gram) + LR, C and n-grams tuned by grouped inner CV (tf-idf/train_tfidf.py)
  mpnet_lr      all-mpnet-base-v2 + standardised LR, C = 0.1 (src/train.py)
  mpnet_rf      all-mpnet-base-v2 + random forest, 300 trees, grid tuned by grouped inner CV (src/train.py)
  qwen          saved out-of-fold 3-class Qwen2.5-14B QLoRA predictions
  stack         LR on the three models' out-of-fold log-probs, fitted on the other 4 folds (not reported)
Also fits each baseline on all 3,774 and scores the 92-sentence hand-written test set.
The forest's leaf grid and depth cap (max_depth 15, min_samples_leaf 10/30/50) differ from the 2-class src/train.py.
CPU, ~10 min. Writes outputs/three_class/baselines_3class.csv (metrics) and baselines_3class_oof.csv (probabilities).

  python experiments/baselines_3class.py
"""
import sys, numpy as np, pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

REPO = Path(__file__).resolve().parent.parent; OUT = REPO / "outputs" / "three_class"
df = pd.read_csv(REPO / "experiments/feat/ft_Qwen2.5-14B-Instruct_lora_4bit_3class_ep6_s72.csv")
test = pd.read_csv(REPO / "experiments/feat/ood_test_predictions.csv")
text, y, folds, groups = df.sentence.to_numpy(), df.label.to_numpy(), df.fold.to_numpy(), df.group.to_numpy()

def tfidf():
    pipe = Pipeline([("tfidf", TfidfVectorizer(lowercase=True)), ("lr", LogisticRegression(max_iter=5000))])
    return GridSearchCV(pipe, {"tfidf__ngram_range": [(1, 1), (1, 2)], "lr__C": [0.1, 0.3, 1, 3, 10, 30]},
                        cv=GroupKFold(n_splits=4), scoring="accuracy", n_jobs=-1)
def mlr(): return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=0.1))
def mrf():
    return GridSearchCV(RandomForestClassifier(n_estimators=300, max_depth=15, random_state=72, n_jobs=-1),
                        {"max_features": ["sqrt", 0.1, 0.3], "min_samples_leaf": [10, 30, 50]},
                        cv=GroupKFold(n_splits=4), scoring="accuracy")

from sentence_transformers import SentenceTransformer
enc = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
emb = lambda s: enc.encode(list(s), batch_size=64, normalize_embeddings=True, show_progress_bar=False)
E, Et = emb(text), emb(test.sentence)
print("embedded", E.shape, flush=True)

models = {"tfidf_lr": (tfidf, text, test.sentence.to_numpy()), "mpnet_lr": (mlr, E, Et), "mpnet_rf": (mrf, E, Et)}
P, test_pred = {}, {}
for name, (make, X, Xt) in models.items():
    prob = np.zeros((len(y), 3))
    for k in range(5):
        tr = folds != k
        m = make()
        if isinstance(m, GridSearchCV): m.fit(X[tr], y[tr], groups=groups[tr])
        else: m.fit(X[tr], y[tr])
        prob[~tr] = m.predict_proba(X[~tr])
        if name == "mpnet_rf":
            d = [e.get_depth() for e in m.best_estimator_.estimators_]
            print(f"  rf fold {k}: {m.best_params_}  inner acc {m.best_score_:.3f}  depth mean {np.mean(d):.1f} max {max(d)}", flush=True)
    P[name] = prob
    m = make()
    if isinstance(m, GridSearchCV): m.fit(X, y, groups=groups)
    else: m.fit(X, y)
    test_pred[name] = m.predict(Xt)
    print("done", name, flush=True)
P["qwen"] = df[["p0", "p1", "p2"]].to_numpy()
test_pred["qwen"] = test.new_qwen_only.to_numpy()

# stacked ensemble: for each test fold, fit LR on the other folds' out-of-fold log-probs
Z = np.hstack([np.log(np.clip(P[n], 1e-6, 1)) for n in ["tfidf_lr", "mpnet_rf", "qwen"]])
st = np.zeros((len(y), 3))
for k in range(5):
    tr = folds != k
    st[~tr] = LogisticRegression(max_iter=5000).fit(Z[tr], y[tr]).predict_proba(Z[~tr])
P["stack"] = st

yt = test.label3.to_numpy()
maj_cv = np.zeros(len(y), int)
for k in range(5): maj_cv[folds == k] = np.bincount(y[folds != k]).argmax()
rows = [{"model": "majority", "pred": maj_cv, "test": np.full(len(yt), np.bincount(y).argmax())}]
rows += [{"model": n, "pred": P[n].argmax(1), "test": test_pred.get(n)} for n in ["tfidf_lr", "mpnet_lr", "mpnet_rf", "qwen", "stack"]]
lab = df.kind.str.startswith("labeled").to_numpy()
out = []
for r in rows:
    p = r["pred"]
    fa = [accuracy_score(y[folds == k], p[folds == k]) for k in range(5)]
    out.append({"model": r["model"], "cv_acc": accuracy_score(y, p), "cv_macro_f1": f1_score(y, p, average="macro"),
                "fold_sd": np.std(fa, ddof=1),
                "f1_anti": f1_score(y, p, labels=[0], average=None)[0], "f1_stereo": f1_score(y, p, labels=[1], average=None)[0],
                "f1_neutral": f1_score(y, p, labels=[2], average=None)[0],
                "labeled_acc_3class": accuracy_score(y[lab], p[lab]),
                "test_acc": accuracy_score(yt, r["test"]) if r["test"] is not None else np.nan,
                "test_macro_f1": f1_score(yt, r["test"], average="macro") if r["test"] is not None else np.nan})
res = pd.DataFrame(out).round(3)
OUT.mkdir(parents=True, exist_ok=True)
res.to_csv(OUT / "baselines_3class.csv", index=False)
pd.DataFrame({f"{n}_p{c}": P[n][:, c] for n in P for c in range(3)}).assign(sentence=text, label=y, fold=folds).to_csv(
    OUT / "baselines_3class_oof.csv", index=False)
print(res.to_string(index=False))
