"""TF-IDF + logistic regression stereotype classifier, scored with the shared 5-fold grouped split.

Models (names match the column prefixes in Chris's outputs/predictions.csv so results can be compared):
  tfidf_lr     unigram TF-IDF + logistic regression
  tfidf_lr_12  unigram + bigram TF-IDF + logistic regression

The vectorizer is part of the pipeline, so vocabulary and IDF weights are fitted on each training fold only.
Stop words are NOT removed: scikit-learn's English list contains he, she, her and his, which carry the signal.
The regularisation strength C (and which n-gram range, for tfidf_lr_12) is tuned by nested CV: a grid search over
grouped inner folds of each training split, so the outer test fold never picks settings.

Writes to tf-idf/results/ (same format as Chris's outputs, in a separate folder):
  metrics.csv            per-label precision / recall / F1 and accuracy, pooled over out-of-fold predictions
  metrics_by_fold.csv    accuracy and macro F1 per fold
  predictions.csv        out-of-fold prediction + P(stereotype) per sentence per model
  best_params.csv        settings chosen in each outer fold
  top_words.csv          highest-weighted words per label from the unigram model fitted on all data,
                         with how many of the 5 per-fold models also rank each word in their top 15
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "results"
LABELS = {1: "stereotype", 0: "anti-stereotype"}
C_GRID = [0.1, 0.3, 1, 3, 10, 30]
TOP_K = 15


def tuned_model(ngram_ranges):
    pipe = Pipeline([("tfidf", TfidfVectorizer(lowercase=True, stop_words=None)),
                     ("lr", LogisticRegression(max_iter=5000))])
    grid = {"tfidf__ngram_range": ngram_ranges, "lr__C": C_GRID}
    return GridSearchCV(pipe, grid, cv=GroupKFold(n_splits=4), scoring="accuracy")


def cross_validate(model, text, y, folds, groups):
    """Out-of-fold predictions. Returns pred, P(stereotype), chosen params per fold, and per-fold fitted models."""
    pred, prob = np.zeros(len(y), dtype=int), np.zeros(len(y))
    chosen, fitted = [], {}
    for k in np.unique(folds):
        test = folds == k
        model.fit(text[~test], y[~test], groups=groups[~test])
        chosen.append({"fold": k, "ngram_range": str(model.best_params_["tfidf__ngram_range"]),
                       "C": model.best_params_["lr__C"], "inner_accuracy": model.best_score_})
        fitted[k] = model.best_estimator_
        pred[test] = model.predict(text[test])
        prob[test] = model.predict_proba(text[test])[:, list(model.classes_).index(1)]
    return pred, prob, chosen, fitted


def label_metrics(name, y, pred):
    p, r, f, n = precision_recall_fscore_support(y, pred, labels=[1, 0], zero_division=0)
    rows = [{"model": name, "label": LABELS[lab], "precision": p[i], "recall": r[i], "f1": f[i], "support": n[i]}
            for i, lab in enumerate([1, 0])]
    rows.append({"model": name, "label": "macro avg", "precision": p.mean(), "recall": r.mean(), "f1": f.mean(),
                 "support": n.sum(), "accuracy": accuracy_score(y, pred)})
    return rows


def pairwise_accuracy(df, name):
    """Share of pairs where the stereotype sentence gets a strictly higher P(stereotype) than its partner."""
    wide = df.pivot(index="pair_id", columns="label", values=f"{name}_prob")
    return (wide[1] > wide[0]).mean(), (wide[1] == wide[0]).mean()


def word_weights(pipe):
    """Coefficient per vocabulary word; positive pushes toward 'stereotype'."""
    words = pipe.named_steps["tfidf"].get_feature_names_out()
    return pd.Series(pipe.named_steps["lr"].coef_[0], index=words)


def top_words(final_pipe, fold_pipes):
    weights = word_weights(final_pipe)
    per_fold = [word_weights(p) for p in fold_pipes.values()]
    rows = []
    for label, series in (("stereotype", weights.sort_values(ascending=False)),
                          ("anti-stereotype", weights.sort_values())):
        for rank, (word, w) in enumerate(series.head(30).items(), start=1):
            in_top = 0
            for fw in per_fold:
                top = fw.sort_values(ascending=(label == "anti-stereotype")).head(TOP_K).index
                in_top += word in set(top)
            rows.append({"label": label, "rank": rank, "word": word, "weight": round(w, 3),
                         "folds_in_top_15": in_top})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(exist_ok=True)
    df = pd.read_csv(ROOT / "data" / "processed" / "merged_gender.csv")
    df = df.merge(pd.read_csv(ROOT / "data" / "processed" / "folds.csv")[["pair_id", "group", "fold"]], on="pair_id")
    text, y = df["sentence"].to_numpy(), df["label"].to_numpy()
    folds, groups = df["fold"].to_numpy(), df["group"].to_numpy()

    specs = {"tfidf_lr": [(1, 1)], "tfidf_lr_12": [(1, 2)]}
    preds = df[["pair_id", "sentence", "label", "source", "fold"]].copy()
    metrics, by_fold, params, fold_models = [], [], [], {}
    for name, ngram_ranges in specs.items():
        pred, prob, chosen, fitted = cross_validate(tuned_model(ngram_ranges), text, y, folds, groups)
        preds[f"{name}_pred"], preds[f"{name}_prob"] = pred, prob.round(4)
        metrics += label_metrics(name, y, pred)
        params += [{"model": name, **c} for c in chosen]
        fold_models[name] = fitted
        for k in np.unique(folds):
            m = folds == k
            by_fold.append({"model": name, "fold": k, "accuracy": accuracy_score(y[m], pred[m]),
                            "macro_f1": f1_score(y[m], pred[m], average="macro", zero_division=0)})

    metrics, by_fold, params = pd.DataFrame(metrics), pd.DataFrame(by_fold), pd.DataFrame(params)
    metrics.to_csv(OUT / "metrics.csv", index=False)
    by_fold.to_csv(OUT / "metrics_by_fold.csv", index=False)
    preds.to_csv(OUT / "predictions.csv", index=False)
    params.to_csv(OUT / "best_params.csv", index=False)

    # Interpretation: unigram model with the most commonly chosen C, fitted on all data.
    best_c = params[params.model == "tfidf_lr"]["C"].mode().iloc[0]
    final = tuned_model([(1, 1)]).estimator.set_params(lr__C=best_c).fit(text, y)
    words = top_words(final, fold_models["tfidf_lr"])
    words.to_csv(OUT / "top_words.csv", index=False)

    print("== Per-label metrics (pooled out-of-fold) ==")
    print(metrics.drop(columns="accuracy").round(3).to_string(index=False))
    print("\n== Accuracy across folds (mean, sd) ==")
    print(by_fold.groupby("model")["accuracy"].agg(["mean", "std"]).round(3))
    print("\n== Accuracy by source ==")
    for name in specs:
        print(name, (preds[f"{name}_pred"] == preds["label"]).groupby(preds["source"]).mean().round(3).to_dict())
    print("\n== Pairwise accuracy (stereotype sentence scores higher than its partner; share of exact ties) ==")
    for name in specs:
        acc, ties = pairwise_accuracy(preds, name)
        print(f"{name}: {acc:.3f} (ties {ties:.3f})")
    print("\n== Settings chosen per fold ==")
    print(params.round(3).to_string(index=False))
    print(f"\n== Top words, unigram model fitted on all data (C={best_c}) ==")
    for label in ("stereotype", "anti-stereotype"):
        w = words[words.label == label].head(15)
        print(f"{label}: " + ", ".join(f"{r.word} ({r.weight:+.2f}, {r.folds_in_top_15}/5)" for r in w.itertuples()))


if __name__ == "__main__":
    main()
