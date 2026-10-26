"""Score sentences with saved LoRA adapters (from finetune_cv.py --final). Run on a CUDA GPU.

With several adapters, P(stereotype) is their average (the demo ensemble: 0.833 in CV).

  python experiments/predict.py models/qwen14b_lora models/qwen14b_lora_gest -- "She is very emotional." "He is very emotional."
  python experiments/predict.py models/qwen14b_lora --train      # sanity check: accuracy on our training data
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parent.parent


def load(adapter):
    """Base model in 4-bit (as trained) plus the adapter. The base model name is read from the adapter config."""
    base = json.loads((Path(adapter) / "adapter_config.json").read_text())["base_model_name_or_path"]
    tok = AutoTokenizer.from_pretrained(adapter)
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForSequenceClassification.from_pretrained(base, num_labels=2, quantization_config=q,
                                                               dtype=torch.float16, device_map={"": 0})
    model.config.pad_token_id = tok.pad_token_id
    return tok, PeftModel.from_pretrained(model, adapter).eval()


@torch.no_grad()
def score(tok, model, texts):
    out = []
    for i in range(0, len(texts), 32):
        enc = tok(list(texts[i:i + 32]), padding=True, truncation=True, max_length=96, return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(torch.softmax(model(**enc).logits.float(), -1)[:, 1].cpu().numpy())
    return np.concatenate(out)


if __name__ == "__main__":
    args = sys.argv[1:]
    adapters = args[:args.index("--")] if "--" in args else [a for a in args if not a.startswith("--")]
    if "--train" in args:
        df = pd.read_csv(ROOT / "data" / "processed" / "merged_gender.csv")
        texts, y = df["sentence"].to_numpy(), df["label"].to_numpy()
    else:
        texts, y = np.array(args[args.index("--") + 1:]), None
    probs = []
    for a in adapters:
        tok, model = load(a)
        probs.append(score(tok, model, texts))
        del model; torch.cuda.empty_cache()
    p = np.mean(probs, axis=0)
    if y is not None:
        print(f"training-data accuracy {((p > 0.5) == y).mean():.3f} (not a test score: these sentences were trained on)")
    else:
        for t, v in zip(texts, p):
            print(f"{v:.2f}  {'stereotype' if v > 0.5 else 'anti-stereotype'}  {t}")
