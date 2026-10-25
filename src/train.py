"""Train and score all classifiers with 5-fold grouped cross-validation (folds from split.py).

Models: majority class, BERT mean-pool + LR, mpnet + LR, mpnet + random forest (tuned), fine-tuned BERT. Add Julia's TF-IDF + LR to models() to score it too.
Writes:
  outputs/metrics.csv           per-label precision / recall / F1 and accuracy, pooled over out-of-fold predictions
  outputs/metrics_by_fold.csv   accuracy and macro F1 per fold (for spread)
  outputs/predictions.csv       out-of-fold prediction + P(stereotype) per sentence per model (for error analysis)
  outputs/rf_best_params.csv    random-forest settings chosen in each outer fold

The random forest is tuned with nested CV: a grid search over grouped inner folds of each training split,
so the outer test fold is never used to pick settings.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from embed import load_embeddings
from finetune import BertFineTuned

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
LABELS = {1: "stereotype", 0: "anti-stereotype"}


def embed_lr():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=0.1))


def tuned_rf():
    grid = {"max_features": ["sqrt", 0.1, 0.3], "min_samples_leaf": [1, 3, 10]}
    return GridSearchCV(RandomForestClassifier(n_estimators=300, random_state=72, n_jobs=-1), grid,
                        cv=GroupKFold(n_splits=4), scoring="accuracy")


def models(df):
    text = df["sentence"].to_numpy()
    return {
        "majority": (DummyClassifier(strategy="most_frequent"), text),
        "bert_lr": (embed_lr(), load_embeddings("bert", df["sentence"])),
        "mpnet_lr": (embed_lr(), load_embeddings("mpnet", df["sentence"])),
        "mpnet_rf": (tuned_rf(), load_embeddings("mpnet", df["sentence"])),
        "bert_ft": (BertFineTuned(), text),
    }


def cross_validate(model, X, y, folds, groups):
    pred, prob = np.zeros(len(y), dtype=int), np.zeros(len(y))
    for k in np.unique(folds):
        test = folds == k
        fit_kw = {"groups": groups[~test]} if isinstance(model, GridSearchCV) else {}
        model.fit(X[~test], y[~test], **fit_kw)
        if isinstance(model, GridSearchCV):
            model.chosen_.append({"fold": k, **model.best_params_, "inner_accuracy": model.best_score_})
        pred[test] = model.predict(X[test])
        prob[test] = model.predict_proba(X[test])[:, list(model.classes_).index(1)]
    return pred, prob


def label_metrics(name, y, pred):
    p, r, f, n = precision_recall_fscore_support(y, pred, labels=[1, 0], zero_division=0)
    rows = [{"model": name, "label": LABELS[lab], "precision": p[i], "recall": r[i], "f1": f[i], "support": n[i]}
            for i, lab in enumerate([1, 0])]
    rows.append({"model": name, "label": "macro avg", "precision": p.mean(), "recall": r.mean(), "f1": f.mean(),
                 "support": n.sum(), "accuracy": accuracy_score(y, pred)})
    return rows


def main():
    df = pd.read_csv(ROOT / "data" / "processed" / "merged_gender.csv")
    df = df.merge(pd.read_csv(ROOT / "data" / "processed" / "folds.csv")[["pair_id", "group", "fold"]], on="pair_id")
    y, folds, groups = df["label"].to_numpy(), df["fold"].to_numpy(), df["group"].to_numpy()

    metrics, by_fold = [], []
    preds = df[["pair_id", "sentence", "label", "source", "fold"]].copy()
    all_models = models(df)
    all_models["mpnet_rf"][0].chosen_ = []
    for name, (model, X) in all_models.items():
        pred, prob = cross_validate(model, X, y, folds, groups)
        preds[f"{name}_pred"], preds[f"{name}_prob"] = pred, prob.round(4)
        metrics += label_metrics(name, y, pred)
        for k in np.unique(folds):
            m = folds == k
            by_fold.append({"model": name, "fold": k, "accuracy": accuracy_score(y[m], pred[m]),
                            "macro_f1": f1_score(y[m], pred[m], average="macro", zero_division=0)})

    metrics = pd.DataFrame(metrics)
    by_fold = pd.DataFrame(by_fold)
    metrics.to_csv(OUT / "metrics.csv", index=False)
    by_fold.to_csv(OUT / "metrics_by_fold.csv", index=False)
    preds.to_csv(OUT / "predictions.csv", index=False)
    rf_params = pd.DataFrame(all_models["mpnet_rf"][0].chosen_)
    rf_params.to_csv(OUT / "rf_best_params.csv", index=False)

    print("== Per-label metrics (pooled out-of-fold) ==")
    print(metrics.drop(columns="accuracy").round(3).to_string(index=False))
    print("\n== Accuracy across folds (mean ± sd) ==")
    print(by_fold.groupby("model")["accuracy"].agg(["mean", "std"]).round(3))
    print("\n== Random forest settings chosen per fold ==")
    print(rf_params.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
