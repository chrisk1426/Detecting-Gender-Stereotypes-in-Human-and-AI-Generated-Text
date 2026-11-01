"""Generate one neutral sentence per labeled pair with an instruct LLM. Run on a CUDA GPU.

The neutral sentence keeps the pair's people and setting but states an ordinary fact or action, so a
neutral-vs-stereotype classifier has to learn the stereotype content rather than the topic.
Writes data/external/neutral_generated.csv (pair_id, sentence).

--paraphrase: instead, rewrite each stereotype sentence in the model's own words, keeping its meaning. This is a control
for the neutral detector: LLM-worded stereotypes must still be flagged, or the detector learned "LLM style = neutral".
Writes data/external/stereo_paraphrased.csv.

  python experiments/gen_neutral.py Qwen/Qwen2.5-7B-Instruct [--paraphrase]
"""
import re, sys
from pathlib import Path
import pandas as pd, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
MODEL = next((a for a in sys.argv[1:] if not a.startswith("--")), "Qwen/Qwen2.5-7B-Instruct")
PARA = "--paraphrase" in sys.argv
PROMPT = """Here are two sentences from a dataset about gender stereotypes:
1. {a}
2. {b}

Write ONE new sentence about the same kind of person or setting that is completely neutral: it describes an
ordinary action, event or fact. It must not say or imply anything about what people of a gender are like, good
at, or supposed to do, and must not describe anyone's personality, abilities or appearance. Keep a gender word
(she, he, woman, man, mother, father, sister, brother, ...). Keep it about the same length.
Answer with the sentence only."""
PARA_PROMPT = """Rewrite this sentence in your own words. Keep exactly the same meaning, including any opinion or
generalization it makes, and keep the gender words. Answer with the rewritten sentence only.

{a}"""
GENDER = re.compile(r"\b(she|he|her|him|his|hers|woman|women|man|men|girl|girls|boy|boys|mother|mothers|father|fathers|"
                    r"mom|mommy|dad|daddy|sister|brother|wife|husband|daughter|son|aunt|uncle|grandmother|grandfather|"
                    r"lady|ladies|gentleman|gentlemen|female|male|schoolgirl|schoolboy|gentlemen|bride|groom)\b", re.I)


@torch.no_grad()
def main():
    df = pd.read_csv(ROOT / "data" / "processed" / "merged_gender.csv")
    if PARA:  # one item per stereotype sentence
        df = df[df["label"] == 1]
    pairs = df.groupby("pair_id", sort=False)["sentence"].apply(list)
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to("cuda").eval()
    torch.manual_seed(0)
    rows = []
    items = list(pairs.items())
    for i in range(0, len(items), 32):
        chunk = items[i:i + 32]
        texts = [tok.apply_chat_template([{"role": "user", "content": PARA_PROMPT.format(a=s[0]) if PARA
                                           else PROMPT.format(a=s[0], b=s[1])}],
                                         add_generation_prompt=True, tokenize=False) for _, s in chunk]
        enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
        out = model.generate(**enc, max_new_tokens=60, do_sample=True, temperature=0.7, top_p=0.9,
                             pad_token_id=tok.eos_token_id)
        for (pid, _), o in zip(chunk, out[:, enc["input_ids"].shape[1]:]):
            s = tok.decode(o, skip_special_tokens=True).strip().strip('"').split("\n")[0].strip()
            rows.append({"pair_id": pid, "sentence": s})
        print(f"{i + len(chunk)}/{len(items)}", flush=True)
    out = pd.DataFrame(rows)
    keep = out["sentence"].str.contains(GENDER) & out["sentence"].str.split().str.len().between(4, 40)
    print(f"kept {keep.sum()} of {len(out)} (dropped: no gender word or odd length)")
    out[keep].to_csv(ROOT / "data" / "external" / ("stereo_paraphrased.csv" if PARA else "neutral_generated.csv"), index=False)


if __name__ == "__main__":
    main()
