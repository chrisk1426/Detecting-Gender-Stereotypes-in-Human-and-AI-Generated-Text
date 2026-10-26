"""Gender-swap scores for the neutral check. Run on a CUDA GPU.

For every sentence, scores it and its gender-swapped version (she<->he, woman<->man, ...) with the saved adapters
(averaged), and writes experiments/feat/neutral_swap.csv with columns set, sentence, label, p, p_swap.

Sets:
  neutral    data/external/neutral_sentences.csv: hand-written everyday sentences with a gender word, no stereotype
  unrelated  StereoSet gender items with gold_label 2 ("unrelated" continuations), which merge.py drops
  labeled    our 1,518 labeled sentences. The final adapters were trained on these, so this is optimistic.

  python experiments/neutral_eval.py models/qwen14b_lora models/qwen14b_lora_gest
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, torch

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from llm_features import swap_gender
from predict import load, score


def unrelated():
    cached = ROOT / "data" / "external" / "stereoset_unrelated_gender.csv"  # for machines without pyarrow
    if cached.exists():
        return pd.read_csv(cached)["sentence"].tolist()
    rows = []
    for part in ("intersentence", "intrasentence"):
        df = pd.read_parquet(ROOT / "data" / "raw" / f"stereoset_{part}_validation.parquet")
        for _, r in df[df["bias_type"] == "gender"].iterrows():
            for sent, gold in zip(r["sentences"]["sentence"], r["sentences"]["gold_label"]):
                if gold == 2:
                    rows.append(f"{r['context']} {sent}".strip() if part == "intersentence" else sent)
    pd.DataFrame({"sentence": rows}).to_csv(cached, index=False)
    return rows


if __name__ == "__main__":
    labeled = pd.read_csv(ROOT / "data" / "processed" / "merged_gender.csv")
    df = pd.concat([
        pd.DataFrame({"set": "neutral", "sentence": pd.read_csv(ROOT / "data" / "external" / "neutral_sentences.csv")["sentence"], "label": -1}),
        pd.DataFrame({"set": "unrelated", "sentence": unrelated(), "label": -1}),
        pd.DataFrame({"set": "labeled", "sentence": labeled["sentence"], "label": labeled["label"]}),
    ], ignore_index=True)
    df["swapped"] = [swap_gender(s) for s in df["sentence"]]
    has = df["swapped"].notna().to_numpy()
    p, p_swap = [], []
    for a in sys.argv[1:]:
        tok, model = load(a)
        p.append(score(tok, model, df["sentence"].to_numpy()))
        q = np.full(len(df), np.nan)
        q[has] = score(tok, model, df.loc[has, "swapped"].to_numpy())
        p_swap.append(q)
        del model; torch.cuda.empty_cache()
    df["p"], df["p_swap"] = np.mean(p, axis=0), np.mean(p_swap, axis=0)
    out = HERE / "feat" / "neutral_swap.csv"
    df.to_csv(out, index=False)
    print("wrote", out, df.groupby("set").size().to_dict())
