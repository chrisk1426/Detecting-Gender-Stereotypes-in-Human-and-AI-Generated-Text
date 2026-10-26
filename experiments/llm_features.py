"""7B causal-LM features per sentence, cached to experiments/feat/llm_<model>.csv. Run on a CUDA GPU.

llm_agree: log P(sentence) - log P(gender-swapped sentence) under the LM. High = the LM finds the actual
           gender more natural than the opposite one for this content (reads as stereotypical). 0 if no gender word.
llm_judge: logit(Yes) - logit(No) for "Does this sentence express a gender stereotype?" (zero-shot).

Caveat: the LM may have seen CrowS-Pairs / StereoSet during pretraining.
"""
import re, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from mlm_features import F2M, M2F

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-7B-Instruct"
PREFIX = "Sentence: "
PROMPT = ("Does the following sentence express or reinforce a gender stereotype? Answer Yes or No.\n\n"
          "Sentence: {s}")
SWAP = {**F2M, **M2F, "her": "his", "his": "her", "him": "her"}


def swap_gender(s):
    def rep(m):
        w = m.group()
        t = SWAP.get(w.lower())
        if t is None:
            return w
        return t.capitalize() if w[0].isupper() else t
    out = re.sub(r"[A-Za-z']+", rep, s)
    return out if out != s else None


@torch.no_grad()
def logprob(model, tok, text):
    """Sum of token log-probs of `text` after PREFIX."""
    pre = tok(PREFIX, add_special_tokens=False).input_ids
    ids = tok(PREFIX + text, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
    lp = torch.log_softmax(model(ids).logits[0, :-1].float(), -1)
    tgt = ids[0, 1:]
    return lp[torch.arange(len(tgt)), tgt][len(pre) - 1:].sum().item()


@torch.no_grad()
def judge(model, tok, s, yes_id, no_id):
    msgs = [{"role": "user", "content": PROMPT.format(s=s)}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"]
    ids = ids.to(model.device)
    logits = model(ids).logits[0, -1].float()
    return (logits[yes_id] - logits[no_id]).item()


if __name__ == "__main__":
    df = pd.read_csv(HERE.parent / "data" / "processed" / "merged_gender.csv")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to("cuda").eval()
    yes_id = tok.encode("Yes", add_special_tokens=False)[0]
    no_id = tok.encode("No", add_special_tokens=False)[0]
    rows, t0 = [], time.time()
    for i, s in enumerate(df["sentence"]):
        sw = swap_gender(s)
        agree = logprob(model, tok, s) - logprob(model, tok, sw) if sw else 0.0
        rows.append({"llm_agree": agree, "llm_has_gender": int(sw is not None),
                     "llm_judge": judge(model, tok, s, yes_id, no_id)})
        if i % 200 == 0:
            print(f"{i}/{len(df)}  {time.time() - t0:.0f}s", flush=True)
    out = HERE / "feat" / f"llm_{MODEL.split('/')[-1]}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out, flush=True)
