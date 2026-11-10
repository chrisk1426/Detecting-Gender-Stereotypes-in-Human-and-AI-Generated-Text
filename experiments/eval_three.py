"""Evaluate the new pipeline (gender filter -> 3-class Qwen) against the old one (neutral detector -> two 2-class Qwen
judges). GPU.

Part 1, cross-validation (no GPU needed): combines the out-of-fold gender filter and 3-class predictions.
Part 2, hand-written test set (ood_test.csv; sentences that also appear in the training data are dropped): runs the
final models of both pipelines on every test sentence. Old pipeline labels: neutral / stereotype / anti-stereotype; a
sentence the new filter removes counts as neutral (label 2), so both use the same 3 labels.

  python experiments/eval_three.py FEAT_3CLASS.csv ADAPTER_3CLASS [OLD_ADAPTER ...]
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, torch

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import data3

NAMES = {0: "anti", 1: "stereo", 2: "neutral"}
feat3, adapter3, old_adapters = sys.argv[1], sys.argv[2], sys.argv[3:]

# ---- Part 1: cross-validation ----
cv = pd.read_csv(feat3)
gf = pd.read_csv(HERE / "feat" / "gender_filter_oof.csv")[["sentence", "p_gender_lr"]]
cv = cv.merge(gf, on="sentence", how="left")
P = cv[["p0", "p1", "p2"]].to_numpy()
cv["qwen"] = P.argmax(1)
cv["pipe"] = np.where(cv["p_gender_lr"] < 0.5, 2, cv["qwen"])
lab = cv["kind"].str.startswith("labeled")
sa = lab & (cv["pipe"] != 2)
print("== cross-validation (out-of-fold) ==")
print(f"3-class accuracy, Qwen alone:            {(cv['qwen'] == cv['label']).mean():.3f}")
print(f"3-class accuracy, filter + Qwen:         {(cv['pipe'] == cv['label']).mean():.3f}")
print(f"labeled sentences called neutral:        {(cv.loc[lab, 'pipe'] == 2).mean():.3f}")
print(f"stereo vs anti on labeled (p1 vs p0):    {((cv.loc[lab, 'p1'] > cv.loc[lab, 'p0']) == cv.loc[lab, 'label']).mean():.3f}"
      "   (old 2-class: 0.833 ensemble)")
print(f"  same, only those not called neutral:   {(cv.loc[sa, 'pipe'] == cv.loc[sa, 'label']).mean():.3f}")
print("accuracy by kind (filter + Qwen):")
print(cv.assign(ok=cv["pipe"] == cv["label"]).groupby("kind")["ok"].agg(["mean", "size"]).round(3).to_string())


# ---- Part 2: test set ----
def load_clf(adapter, num_labels):
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
    import json
    base = json.loads((Path(adapter) / "adapter_config.json").read_text())["base_model_name_or_path"]
    tok = AutoTokenizer.from_pretrained(adapter)
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    m = AutoModelForSequenceClassification.from_pretrained(base, num_labels=num_labels, quantization_config=q,
                                                           dtype=torch.float16, device_map={"": 0})
    m.config.pad_token_id = tok.pad_token_id
    return tok, PeftModel.from_pretrained(m, adapter).eval()


@torch.no_grad()
def softmax(tok, model, texts):
    out = []
    for i in range(0, len(texts), 32):
        enc = tok(list(texts[i:i + 32]), padding=True, truncation=True, max_length=96, return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(torch.softmax(model(**enc).logits.float(), -1).cpu().numpy())
    return np.concatenate(out)


test = pd.read_csv(ROOT / "data" / "external" / "ood_test.csv")
seen = set(data3.load()["sentence"].str.lower().str.strip())
overlap = test["sentence"].str.lower().str.strip().isin(seen)
print(f"\n== hand-written test set: {len(test)} sentences, {overlap.sum()} dropped (also in training data) ==")
test = test[~overlap].reset_index(drop=True)
texts = test["sentence"].to_numpy()

from sentence_transformers import SentenceTransformer
E = SentenceTransformer("Snowflake/snowflake-arctic-embed-l-v2.0", device="cuda").encode(
    list(texts), normalize_embeddings=True)
sig = lambda w: 1 / (1 + np.exp(-(E @ w["coef"] + w["intercept"])))
test["p_gender"] = sig(np.load(ROOT / "models" / "gender_filter.npz"))
tok, m = load_clf(adapter3, 3)
P = softmax(tok, m, texts)
del m; torch.cuda.empty_cache()
test["new"] = np.where(test["p_gender"] < 0.5, 2, P.argmax(1))
test["new_qwen_only"] = P.argmax(1)

if old_adapters:
    test["p_related_old"] = sig(np.load(ROOT / "models" / "neutral_lr.npz"))
    ps = []
    for a in old_adapters:
        tok, m = load_clf(a, 2)
        ps.append(softmax(tok, m, texts)[:, 1])
        del m; torch.cuda.empty_cache()
    test["old"] = np.where(test["p_related_old"] < 0.5, 2, (np.mean(ps, 0) > 0.5).astype(int))

cols = [c for c in ["old", "new_qwen_only", "new"] if c in test]
summ = test.groupby("category")[cols].agg(lambda s: (s == test.loc[s.index, "label3"]).mean())
summ.loc["ALL"] = [(test[c] == test["label3"]).mean() for c in cols]
summ.insert(0, "n", test.groupby("category").size().reindex(summ.index).fillna(len(test)).astype(int))
print(summ.round(2).to_string())
for c in cols:
    print(f"\nconfusion, {c} (rows true, columns predicted):")
    print(pd.crosstab(test["label3"].map(NAMES), test[c].map(NAMES)).to_string())
print("\nnew pipeline misses:")
miss = test[test["new"] != test["label3"]]
print(miss.assign(true=miss["label3"].map(NAMES), pred=miss["new"].map(NAMES))[
    ["sentence", "category", "true", "pred", "p_gender"]].round(2).to_string(index=False))
test.to_csv(HERE / "feat" / "ood_test_predictions.csv", index=False)
summ.to_csv(HERE / "feat" / "ood_test_summary.csv")
