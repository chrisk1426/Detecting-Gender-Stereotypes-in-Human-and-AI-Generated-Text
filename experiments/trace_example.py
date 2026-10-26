"""Trace example sentences through the final pipeline and dump every intermediate value as JSON (for the chart).

  python experiments/trace_example.py models/qwen14b_lora models/qwen14b_lora_gest -- "sentence" ...
"""
import json, sys
from pathlib import Path
import numpy as np, torch
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from predict import load

args = sys.argv[1:]
adapters, texts = args[:args.index("--")], args[args.index("--") + 1:]
out = [{"sentence": t} for t in texts]

w = np.load(ROOT / "models" / "neutral_lr.npz")
E = SentenceTransformer("Snowflake/snowflake-arctic-embed-l-v2.0").encode(texts, normalize_embeddings=True)
z = E @ w["coef"] + w["intercept"]
contrib = E * w["coef"]
for o, e, zz, c in zip(out, E, z, contrib):
    o["emb_dim"] = int(len(e))
    o["emb_first"] = [round(float(x), 3) for x in e[:6]]
    o["neutral_logit"] = round(float(zz), 3)
    o["p_related"] = round(float(1 / (1 + np.exp(-zz))), 3)

with torch.no_grad():
    for a in adapters:
        tok, model = load(a)
        for o in out:
            enc = tok(o["sentence"], return_tensors="pt").to("cuda")
            o["tokens"] = [tok.decode([i]) for i in enc["input_ids"][0]]
            with torch.autocast("cuda", dtype=torch.float16):
                lg = model(**enc).logits[0].float()
            o.setdefault("adapters", []).append({"name": Path(a).name, "logits": [round(float(x), 3) for x in lg],
                                                 "p_stereo": round(float(torch.softmax(lg, -1)[1]), 4)})
        del model; torch.cuda.empty_cache()
for o in out:
    o["p_avg"] = round(float(np.mean([a["p_stereo"] for a in o["adapters"]])), 4)
    o["label"] = "neutral" if o["p_related"] < 0.5 else "stereotype" if o["p_avg"] > 0.5 else "anti-stereotype"
print("JSON" + json.dumps(out))
