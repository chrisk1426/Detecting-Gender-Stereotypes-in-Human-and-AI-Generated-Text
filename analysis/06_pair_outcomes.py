"""Analysis Y1 of docs/MetricsAnalysisPlan.md: for each pair, are both sentences right, one right, or both wrong?

Each pair has one stereotype and one anti sentence. If a model gets exactly one right, it gave both sentences the SAME label
(it could not tell them apart). If it gets both wrong, it swapped them. Writes outputs/metrics/pair_outcomes.csv.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "outputs" / "metrics" / "predictions_all.csv")
NAME = {"tfidf_lr": "TF-IDF + LR", "mpnet_rf": "mpnet + random forest", "qwen14b": "Qwen 14B (LoRA)", "ensemble": "Ensemble (9 models)"}

rows = []
for m, name in NAME.items():
    right = df.groupby("pair_id")[f"{m}_correct"].sum()  # 2 = both right, 1 = same label to both, 0 = swapped
    n = len(right)
    errors = int((df[f"{m}_correct"] == 0).sum())
    same, swapped = int((right == 1).sum()), int((right == 0).sum())
    over = df[df[f"{m}_correct"] == 0][f"{m}_prob"]
    rows.append({"model": name, "pairs": n, "both_right": int((right == 2).sum()), "same_label_to_both": same,
                 "swapped": swapped, "share_both_right": round((right == 2).mean(), 3),
                 "share_of_errors_from_same_label": round(same / errors, 3), "errors": errors,
                 "errors_with_prob_below_0.05_or_above_0.95": int(((over < 0.05) | (over > 0.95)).sum())})
out = pd.DataFrame(rows)
out.to_csv(ROOT / "outputs" / "metrics" / "pair_outcomes.csv", index=False)
print(out.to_string(index=False))
