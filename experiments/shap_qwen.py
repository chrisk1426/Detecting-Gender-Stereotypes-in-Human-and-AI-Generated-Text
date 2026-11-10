"""SHAP token attributions and gender x attribute interactions for the Qwen LoRA judge. Run on a CUDA GPU.

  python experiments/shap_qwen.py models/qwen14b_lora               # all 1,518 sentences
  python experiments/shap_qwen.py models/qwen14b_lora --n 300       # random sample of pairs (both sentences kept)
  python experiments/shap_qwen.py models/qwen14b_lora --data data/external/gest_pairs.csv --n 1500   # held-out text
  python experiments/shap_qwen.py --neutral                         # the neutral detector (see below)
  python experiments/shap_qwen.py --hf distilbert-base-uncased-finetuned-sst-2-english --n 6   # CPU smoke test

What is explained: the log-odds of P(stereotype) from the adapter. SHAP's Partition explainer masks groups of tokens
(masked text -> "...") and splits the change in log-odds among the tokens, so token values sum to f(x) - f(all masked).

Gender x attribute interaction (interactions.csv): for each sentence, G = its gendered words, a = one of its top-3
non-gender content words by |SHAP|. With f = log-odds and x\S = x with S masked,
    interaction = f(x) - f(x\\G) - f(x\\a) + f(x\\G\\a)
> 0: the attribute pushes toward "stereotype" more when the gender word is present (the model reads the pairing);
~ 0: gender and attribute act separately (additive).

--neutral explains the neutral detector instead (Snowflake-L embedding + logistic regression, models/neutral_lr.npz):
f = its logit of "stereotype-related" (vs neutral). Sentences: its training data (the 1,518 labeled sentences + the
734 generated neutral ones) and two neutral sets it never trained on (60 hand-written, 497 StereoSet "unrelated").
label: 1 stereotype, 0 anti-stereotype, -1 neutral. Output folder: outputs/shap/neutral_lr/.

Writes outputs/shap/<adapter>/: tokens.csv, interactions.csv, examples.html (with --shard i/k: tokens_iofk.csv, ...;
shap_analyze.py merges them). Then run experiments/shap_analyze.py.
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd, shap, torch
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shap_utils import GENDER, PUNCT, group_words, key

ROOT = Path(__file__).resolve().parent.parent
MASK = "..."


def make_scorer(args):
    """Returns (tokenizer, f) with f(list[str]) -> log-odds of the 'stereotype' class, computed as the logit
    difference (logit_1 - logit_0) so it does not saturate the way log(p / (1 - p)) does for confident predictions."""
    if args.neutral:  # neutral detector: logit of "stereotype-related" from LR on the Snowflake-L embedding
        from sentence_transformers import SentenceTransformer
        enc = SentenceTransformer("Snowflake/snowflake-arctic-embed-l-v2.0", device="cuda")
        w = np.load(ROOT / "models" / "neutral_lr.npz")
        f = lambda texts: enc.encode([str(t) for t in texts], batch_size=args.batch,
                                     normalize_embeddings=True) @ w["coef"] + w["intercept"]
        return enc.tokenizer, f
    if args.hf:  # small unquantized model for testing the pipeline without a GPU
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.hf)
        model = AutoModelForSequenceClassification.from_pretrained(args.hf).eval()
        dev, amp = "cpu", torch.autocast("cpu", enabled=False)
    else:
        from predict import load
        tok, model = load(args.adapter)
        dev, amp = "cuda", torch.autocast("cuda", dtype=torch.float16)

    @torch.no_grad()
    def f(texts):
        texts, out = [str(t) for t in texts], []
        for i in range(0, len(texts), args.batch):
            enc = tok(texts[i:i + args.batch], padding=True, truncation=True, max_length=96,
                      return_tensors="pt").to(dev)
            with amp:
                lg = model(**enc).logits.float()
            out.append((lg[:, 1] - lg[:, 0]).cpu().numpy())
        return np.concatenate(out)
    return tok, f


def masked(tokens, words, drop):
    """Rebuild the sentence from tokens with the words at indices `drop` replaced by '...'."""
    out = list(tokens)
    for w in drop:
        idx = words[w][1]
        first, last = tokens[idx[0]], tokens[idx[-1]]
        lead, trail = first[:len(first) - len(first.lstrip())], last[len(last.rstrip()):]
        for i in idx:
            out[i] = ""
        out[idx[0]] = lead + MASK + trail
    return "".join(out)


def neutral_data():
    """The neutral detector's training sentences plus the two held-out neutral sets (label -1 = neutral)."""
    ext = ROOT / "data" / "external"
    lab = pd.read_csv(ROOT / "data" / "processed" / "merged_gender.csv")
    gen = pd.read_csv(ext / "neutral_generated.csv").assign(label=-1, source="neutral_generated")
    held = [pd.read_csv(ext / f"{n}.csv")[["sentence"]].assign(label=-1, source=n)
              .pipe(lambda d: d.assign(pair_id=[f"{n}_{i}" for i in range(len(d))]))
            for n in ("neutral_sentences", "stereoset_unrelated_gender")]
    return pd.concat([lab, gen, *held], ignore_index=True)[["pair_id", "sentence", "label", "source"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("adapter", nargs="?", default="models/qwen14b_lora")
    ap.add_argument("--neutral", action="store_true", help="explain the neutral detector instead of an adapter")
    ap.add_argument("--hf", help="any HF sequence-classification model instead of the adapter (smoke test)")
    ap.add_argument("--n", type=int, default=0, help="sample this many sentences (by pair); 0 = all")
    ap.add_argument("--max-evals", type=int, default=400, help="SHAP model calls per sentence")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default="data/processed/merged_gender.csv",
                    help="CSV with pair_id, sentence, label (e.g. data/external/gest_pairs.csv for held-out text)")
    ap.add_argument("--batch", type=int, default=32, help="forward-pass batch size")
    ap.add_argument("--shard", default="0/1", help="i/k: run pairs i, i+k, ... (one GPU each; analyze merges shards)")
    args = ap.parse_args()

    df = neutral_data() if args.neutral else pd.read_csv(ROOT / args.data)
    if "source" not in df:
        df["source"] = Path(args.data).stem.removesuffix("_pairs")
    if args.n:
        pairs = df.pair_id.drop_duplicates().sample(frac=1, random_state=args.seed)
        df = df[df.pair_id.isin(pairs.iloc[:max(1, args.n // 2)])]
    i, k = map(int, args.shard.split("/"))
    df = df[df.pair_id.map({p: j % k for j, p in enumerate(df.pair_id.unique())}) == i]  # sid = row in full data
    tag = "neutral_lr" if args.neutral else Path(args.hf).name if args.hf else Path(args.adapter).name
    if Path(args.data).name != "merged_gender.csv" and not args.neutral:
        tag += "__" + Path(args.data).stem
    sfx = f"_{i}of{k}" if k > 1 else ""
    out_dir = ROOT / "outputs" / "shap" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    tok, f = make_scorer(args)
    explainer = shap.Explainer(f, shap.maskers.Text(tok, mask_token=MASK, collapse_mask_token=True))
    print(f"SHAP on {len(df)} sentences, max_evals={args.max_evals} ...", flush=True)
    sv = explainer(df.sentence.tolist(), max_evals=args.max_evals, batch_size=max(64, args.batch))
    fx = f(df.sentence.tolist())
    sids = df.index.to_numpy()

    tok_rows, jobs = [], []  # jobs: (sid, attr_rank, attr word, gender words, [4 texts])
    for s, (sid, r) in enumerate(df.iterrows()):
        tokens, vals = list(sv.data[s]), np.asarray(sv.values[s], float)
        for p, (t, v) in enumerate(zip(tokens, vals)):
            tok_rows.append((sid, r.pair_id, r.label, r.source, p, t, v, float(sv.base_values[s]), fx[s]))
        words = group_words(tokens)
        keys = [key(w) for w, _ in words]
        g = [i for i, k in enumerate(keys) if k in GENDER]
        attrs = [i for i, (w, _) in enumerate(words)
                 if keys[i] and keys[i] not in GENDER and keys[i] not in ENGLISH_STOP_WORDS and not PUNCT.match(w)]
        if not g or not attrs:
            continue
        wshap = {i: vals[words[i][1]].sum() for i in attrs}
        for rank, a in enumerate(sorted(attrs, key=lambda i: -abs(wshap[i]))[:3]):
            texts = [masked(tokens, words, d) for d in (g, [a], g + [a])]
            jobs.append((s, rank, keys[a], wshap[a], " ".join(keys[i] for i in g),
                         vals[[t for i in g for t in words[i][1]]].sum(), texts))

    print(f"interactions: {len(jobs)} (sentence, attribute) pairs ...", flush=True)
    flat = f([t for j in jobs for t in j[-1]]).reshape(-1, 3) if jobs else np.zeros((0, 3))
    inter = pd.DataFrame([dict(sid=sids[s], attr_rank=rk, attr=a, attr_shap=ash, gender_words=gw, gender_shap=gsh,
                               f_full=fx[s], f_no_gender=m[0], f_no_attr=m[1], f_no_both=m[2],
                               interaction=fx[s] - m[0] - m[1] + m[2],
                               masked_example=t[2])
                          for (s, rk, a, ash, gw, gsh, t), m in zip(jobs, flat)])
    if len(inter):
        inter = df[["pair_id", "label", "source", "sentence"]].rename_axis("sid").reset_index().merge(inter, on="sid")

    pd.DataFrame(tok_rows, columns=["sid", "pair_id", "label", "source", "pos", "token", "shap", "base", "fx"]) \
        .to_csv(out_dir / f"tokens{sfx}.csv", index=False)
    inter.to_csv(out_dir / f"interactions{sfx}.csv", index=False)
    src = df.source.to_numpy()
    pick = list(np.flatnonzero(src == "crows")[:10]) + list(np.flatnonzero(src != "crows")[:6])
    (out_dir / f"examples{sfx}.html").write_text(shap.plots.text(sv[pick], display=False))
    print(f"wrote {out_dir}/tokens{sfx}.csv, interactions{sfx}.csv, examples{sfx}.html")


if __name__ == "__main__":
    main()
