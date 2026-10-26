"""Fine-tune a sequence classifier with 5-fold grouped CV and write out-of-fold P(stereotype). Run on a CUDA GPU.

Per outer fold: hold out ~15% of the training groups as validation, train for up to EPOCHS epochs,
keep the epoch with the best validation accuracy, predict the test fold with it. The test fold never
picks anything. Averages over SEEDS seeds.

  python experiments/finetune_cv.py roberta-large
  python experiments/finetune_cv.py Qwen/Qwen2.5-7B-Instruct --lora --epochs 8 --seeds 72,73
Options:
  --epochs N        max epochs (best one picked on validation)
  --seeds a,b       seeds to average
  --pair            add a pairwise ranking loss: within each training pair, the stereotype sentence should
                    score higher than its partner. Prediction still sees one sentence at a time.
  --4bit            load the base model in 4-bit (QLoRA), for 14B on a 16 GB GPU
  --extra file.csv  extra training rows (sentence,label,pair_id), added to training folds only
  --one             train one unfinished fold, then exit (fresh process per fold frees GPU memory)
  --fold k          train only fold k (to run folds in parallel on several GPUs)

Writes experiments/feat/ft_<tag>.csv (one row per sentence: prob, fold) and prints accuracy.
"""
import gc, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.model_selection import GroupShuffleSplit
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parent.parent
MODEL = sys.argv[1]
LORA = "--lora" in sys.argv
PAIR = "--pair" in sys.argv
QUANT = "--4bit" in sys.argv
arg = lambda name, default: sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default
EPOCHS = int(arg("--epochs", 4 if LORA else 10))
LR = 1e-4 if LORA else 1e-5
BATCH = 8 if LORA else 16
SEEDS = [int(x) for x in arg("--seeds", "72" if LORA else "72,73,74").split(",")]
EXTRA = arg("--extra", None)
PAIR_WEIGHT = 1.0


def build(seed):
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if LORA:
        from peft import LoraConfig, get_peft_model
        if QUANT:
            from peft import prepare_model_for_kbit_training
            from transformers import BitsAndBytesConfig
            q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
            model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2, quantization_config=q,
                                                                       dtype=torch.float16, device_map={"": 0})
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2, dtype=torch.float16)
        model.config.pad_token_id = tok.pad_token_id
        model = get_peft_model(model, LoraConfig(task_type="SEQ_CLS", r=16, lora_alpha=32, lora_dropout=0.1,
                                                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
        for n, p in model.named_parameters():  # trainable params in fp32 for stable fp16 training
            if p.requires_grad:
                p.data = p.data.float()
    else:
        model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)
    return tok, (model if QUANT else model.to("cuda"))


@torch.no_grad()
def predict(tok, model, texts):
    model.eval()
    out = []
    for i in range(0, len(texts), 32):
        enc = tok(list(texts[i:i + 32]), padding=True, truncation=True, max_length=96, return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(**enc).logits.float()
        out.append(torch.softmax(logits, -1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def batches(tr, pairs, rng):
    """Shuffled training batches. With --pair, each batch holds whole pairs (both sentences)."""
    if not PAIR:
        order = rng.permutation(tr)
        return [order[i:i + BATCH] for i in range(0, len(order), BATCH)]
    by_pair = pd.Series(tr).groupby(pairs[tr]).apply(list)
    full = [v for v in by_pair if len(v) == 2]
    order = rng.permutation(len(full))
    per = BATCH // 2
    return [np.array([i for j in order[b:b + per] for i in full[j]]) for b in range(0, len(order), per)]


def pair_loss(logits, labels, idx, pairs):
    """softplus(-(margin of stereo sentence - margin of its anti partner)), over pairs inside the batch."""
    z = logits[:, 1] - logits[:, 0]
    pid = pairs[idx]
    terms = []
    for p in np.unique(pid):
        m = np.where(pid == p)[0]
        if len(m) == 2:
            s, a = (m[0], m[1]) if labels[m[0]] == 1 else (m[1], m[0])
            terms.append(torch.nn.functional.softplus(-(z[s] - z[a])))
    return torch.stack(terms).mean() if terms else z.sum() * 0


def train_fold(texts, y, groups, pairs, test_texts, seed, extra=None):
    tr, va = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed).split(texts, y, groups))
    if extra is not None:  # extra rows join training only; validation stays our data
        n = len(texts)
        texts = np.concatenate([texts, extra["sentence"].to_numpy()])
        y = np.concatenate([y, extra["label"].to_numpy()])
        pairs = np.concatenate([pairs, extra["pair_id"].to_numpy()])
        tr = np.concatenate([tr, np.arange(n, len(texts))])
    tok, model = build(seed)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01)
    steps = EPOCHS * len(batches(tr, pairs, np.random.default_rng(0)))
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler()
    rng = np.random.default_rng(seed)
    best_acc, best_state, best_ep = -1, None, 0
    for ep in range(1, EPOCHS + 1):
        model.train()
        for b in batches(tr, pairs, rng):
            enc = tok(list(texts[b]), padding=True, truncation=True, max_length=96, return_tensors="pt").to("cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(**enc, labels=torch.tensor(y[b], device="cuda"))
                loss = out.loss
                if PAIR:
                    loss = loss + PAIR_WEIGHT * pair_loss(out.logits.float(), y[b], b, pairs)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad()
        acc = ((predict(tok, model, texts[va]) > 0.5) == y[va]).mean()
        if acc > best_acc:
            best_acc, best_ep = acc, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                          if not LORA or "lora" in k or "score" in k}
    model.load_state_dict(best_state, strict=False)
    prob = predict(tok, model, test_texts)
    del model, opt, sched, scaler, best_state
    gc.collect(); torch.cuda.empty_cache()
    return prob, best_ep, best_acc


if __name__ == "__main__":
    df = pd.read_csv(ROOT / "data" / "processed" / "merged_gender.csv")
    df = df.merge(pd.read_csv(ROOT / "data" / "processed" / "folds.csv"), on="pair_id")
    X, y, f, g = df["sentence"].to_numpy(), df["label"].to_numpy(), df["fold"].to_numpy(), df["group"].to_numpy()
    pid = df["pair_id"].to_numpy()
    extra = pd.read_csv(EXTRA) if EXTRA else None
    prob = np.zeros(len(y))
    t0 = time.time()
    tag = (f"{MODEL.split('/')[-1]}{'_lora' if LORA else ''}{'_4bit' if QUANT else ''}{'_pair' if PAIR else ''}"
           f"{'_' + Path(EXTRA).stem if EXTRA else ''}_ep{EPOCHS}_s{'-'.join(map(str, SEEDS))}")
    folds = [int(arg("--fold", 0))] if "--fold" in sys.argv else range(5)
    for k in folds:
        tr, te = f != k, f == k
        part = ROOT / "experiments" / "feat" / f"ft_{tag}.fold{k}.npy"  # resume: skip finished folds
        if part.exists():
            prob[te] = np.load(part); print(f"fold {k}: loaded", flush=True); continue
        runs = [train_fold(X[tr], y[tr], g[tr], pid[tr], X[te], s, extra) for s in SEEDS]
        prob[te] = np.mean([r[0] for r in runs], axis=0)
        np.save(part, prob[te])
        if "--one" in sys.argv or "--fold" in sys.argv:
            print(f"fold {k}: acc {((prob[te] > 0.5) == y[te]).mean():.3f}  best epochs {[r[1] for r in runs]}", flush=True)
            sys.exit(0)
        print(f"fold {k}: acc {((prob[te] > 0.5) == y[te]).mean():.3f}  best epochs {[r[1] for r in runs]}  "
              f"val acc {[round(r[2], 3) for r in runs]}  {time.time() - t0:.0f}s", flush=True)
    accs = [((prob[f == k] > 0.5) == y[f == k]).mean() for k in range(5)]
    print(f"{MODEL}: accuracy {np.mean(accs):.3f} (sd {np.std(accs):.3f})", flush=True)
    out = ROOT / "experiments" / "feat" / f"ft_{tag}.csv"
    pd.DataFrame({"prob": prob, "fold": f}).to_csv(out, index=False)
    print("wrote", out)
