# Detecting Gender Stereotypes in Human- and AI-Generated Text

COSC 72 course project (Julia Allos, Chris Kang). We build a "stereotype judge": a classifier trained on
human-labeled gender-stereotype sentences that labels a sentence as stereotype, anti-stereotype, or neutral.
The proposal also planned to apply the judge to language-model-generated text to measure occupation-linked
gender bias. We dropped that stage and studied the judge itself instead (blind audit, error analysis, SHAP).
The report has the final scope; the proposal has the original plan.

## Status

Done: dataset merge and analysis (`DATANOTES.md`), grouped 5-fold split, a two-class development phase (baselines up
to a fine-tuned Qwen judge and a 9-model ensemble), the **final three-class pipeline** (gender filter + 3-class
Qwen2.5-14B judge) with three-class baselines, SHAP interpretability, the interactive demo, and the metrics, blind audit
and error analysis behind the report.
Not built (dropped from the proposal): the generation and aggregate scoring pipeline for language-model-generated text.

**Which numbers are current.** The final model is the three-class pipeline below; its results are in
[Results](#results-three-class). Everything under [Two-class development](#two-class-development-history) is the
earlier stereotype-vs-anti-stereotype task (balanced, so chance = 0.50). Those numbers are not comparable with the
three-class ones and the models there are not used for new text.

## Final pipeline (three classes)

```
sentence
  │
  ├─ 1. Gender filter: Snowflake-arctic-embed-l-v2.0 embedding (1,024 numbers, frozen) → logistic regression
  │       P(gender-related) < 0.5  →  "neutral" (shown as "not gender-related" in the demo), stop
  │
  └─ 2. 3-class judge: Qwen2.5-14B-Instruct (4-bit, frozen) + one LoRA adapter, 3-way head
          softmax over [anti-stereotype, stereotype, neutral]  →  highest probability is the label
```

Two models, both in `models/`: `gender_filter.npz` (CPU) and `qwen14b_lora_3class` (GPU, ~10 GB VRAM). There is no
ensemble. The filter adds nothing measurable to accuracy (the judge alone scores the same); it is kept because it is
cheap and keeps clearly non-gender text off the GPU.

Run it on new text with the demo server (`python demo/server.py`, see [Demo](#demo)); its `/api/classify` endpoint
returns the filter score, the three probabilities and the label. Step-by-step chart with real examples:
[interactive pipeline page](https://claude.ai/artifact/9LPNDXBa58tejhcr381ork) (source: `docs/pipeline.html`).

Training details:
- Judge: LoRA rank 16 on q/k/v/o, lr 1e-4, 4-bit base (fits a 16 GB GPU). In cross-validation the best of up to 6
  epochs is picked on a 15% validation slice of the training folds; the saved model is trained on all 3,774 sentences
  for 5 epochs (`finetune_cv.py --three`).
- Filter: logistic regression on frozen Snowflake-L embeddings (`gender_filter.py`). XGBoost was compared and scored
  the same (0.992 vs 0.992 grouped-CV accuracy on gender-related vs not).

## Results (three classes)

### How it is evaluated

Nested, grouped 5-fold cross-validation on all 3,774 sentences (`experiments/data3.py`). In each round:

| Set | Size | Used for |
|---|---|---|
| Test fold | ~755 sentences (749–771) | Scoring only; never used to train or choose anything |
| Validation | ~15% of the training groups (~450) | Choosing epochs (Qwen) or hyperparameters (baselines, by grouped inner CV) |
| Train | the rest (~2,570) | Fitting |

Both halves of a pair, and each generated stereotype with its gender swap, always share a fold. Over the 5 rounds every
sentence is scored exactly once by a model that never saw it, so the reported accuracy is **held-out (test)
accuracy**: correct predictions over all 3,774 (for Qwen: 3,366 / 3,774 = 0.892; the per-fold accuracies are
0.891, 0.883, 0.905, 0.897, 0.884, sd 0.009).

A second, separate check is a **hand-written test set** (`data/external/ood_test.csv`): 95 sentences in 9
categories, written by Claude, none in any training data (3 that overlapped were dropped, leaving 92).

### Main table

The classes are not balanced (1,713 neutral, 1,032 anti-stereotype, 1,029 stereotype), so always guessing "neutral"
scores 0.454, and **macro F1** (the unweighted mean of the three per-class F1s) is the fairer single number.
On the hand-written test set 69 of 92 sentences are neutral, so always guessing "neutral" already scores 0.750.

| Model | Held-out acc. | Macro F1 | Fold sd | F1 anti | F1 stereo | F1 neutral | Hand-written test acc. | Test macro F1 |
|---|---|---|---|---|---|---|---|---|
| Majority class (always neutral) | 0.454 | 0.208 | 0.021 | 0.000 | 0.000 | 0.624 | 0.750 | 0.286 |
| TF-IDF + logistic regression | 0.713 | 0.668 | 0.015 | 0.550 | 0.544 | 0.911 | 0.685 | 0.497 |
| all-mpnet-base-v2 + logistic regression | 0.704 | 0.656 | 0.014 | 0.532 | 0.531 | 0.906 | 0.804 | 0.637 |
| all-mpnet-base-v2 + random forest | 0.698 | 0.638 | 0.021 | 0.552 | 0.502 | 0.860 | 0.804 | 0.580 |
| **Qwen2.5-14B QLoRA, 3-class (final judge)** | **0.892** | **0.875** | 0.009 | 0.834 | 0.826 | 0.967 | **0.924** | **0.863** |
| Gender filter + Qwen (full pipeline) | 0.891 | 0.875 | 0.009 | 0.833 | 0.825 | 0.966 | 0.924 | 0.863 |

Sources: `outputs/three_class/baselines_3class.csv` (`experiments/baselines_3class.py`, CPU) and
`outputs/three_class/eval.log` (`experiments/eval_three.py`). Baselines are trained on exactly the same sentences and
folds as the judge. Settings: TF-IDF (1- or 1+2-grams) and C ∈ {0.1 … 30} tuned by grouped inner CV; mpnet + LR uses
standardised embeddings and C = 0.1; the random forest has 300 trees, `max_depth` 15, `min_samples_leaf` ∈ {10, 30, 50},
`max_features` ∈ {sqrt, 0.1, 0.3}, tuned by grouped inner CV (it chose leaf size 10 in every fold).

**What this shows**
- Every baseline learns neutral (F1 0.86–0.91) but not stereotype vs anti-stereotype (F1 ≈ 0.50–0.55). On the 1,513
  human-labeled sentences their three-class accuracy is 0.46–0.53, against 0.795 for the Qwen judge.
- The judge beats the best baseline by 17.9 points of held-out accuracy and 0.21 macro F1.
- On the hand-written test set, accuracy alone is misleading: TF-IDF + LR (0.685) is below always-neutral (0.750).
  Compare macro F1 instead (0.29 majority, 0.50–0.64 baselines, 0.86 judge).
- Stereotype vs anti-stereotype on the labeled sentences only: 0.822 for the 3-class judge, close to the best
  two-class result (0.833 for two averaged two-class judges). Adding the neutral class cost about a point there.
- The judge is usually near-certain, including when it is wrong.

### Hand-written test set in detail

Rows are the true class, columns the pipeline's label (`outputs/three_class/eval.log`).

| True ↓ / predicted → | anti | neutral | stereo | Correct |
|---|---|---|---|---|
| anti-stereotype | 7 | 0 | 2 | 7 / 9 |
| neutral | 2 | 64 | 3 | 64 / 69 |
| stereotype | 0 | 0 | 14 | 14 / 14 |
| **All** | | | | **85 / 92 = 0.924** |

| Category | n | Old pipeline | New pipeline |
|---|---|---|---|
| No people (facts) | 15 | 0.20 | 1.00 |
| Animals | 8 | 0.00 | 1.00 |
| People, no gender | 10 | 0.70 | 1.00 |
| Non-gender groups | 7 | 0.14 | 1.00 |
| Non-gender stereotypes | 7 | 0.00 | 0.86 |
| Gendered facts | 10 | 0.70 | 1.00 |
| Gendered neutral claims ("Men like apples.") | 12 | 0.00 | 0.67 |
| Gendered stereotypes | 14 | 1.00 | 1.00 |
| Gendered anti-stereotypes | 9 | 0.89 | 0.78 |
| **All** | **92** | **0.43** | **0.92** |

"Old pipeline" is the earlier neutral detector followed by two two-class judges (see
[Old neutral detector](#old-neutral-detector-replaced)), scored on the same three-class task. It failed on neutral
text (18 of 69 right): the detector passed most general claims to the judges, which can only answer stereotype or
anti-stereotype. It got 22 of 23 stereotype / anti-stereotype sentences right.

The 7 misses of the new pipeline: "Teenagers are irresponsible." (→ stereotype), "Men like apples." (→ anti),
"Women like apples.", "Girls go to school." (→ stereotype), "Men need sleep to stay healthy." (→ anti),
"Men are bad drivers.", "Men gossip all the time." (anti-stereotypes → stereotype). Gendered neutral claims are the
weakest category: only 138 such training sentences survived the generation check.

## Data

### Human-labeled data (stereotype vs anti-stereotype)

Two public datasets, gender items only, reshaped to one labeled sentence per row
(`1` = stereotype-consistent, `0` = anti-stereotype / less stereotypical).

| Source | Pairs | Sentences |
|---|---|---|
| CrowS-Pairs (gender) | 262 | 524 |
| StereoSet intersentence (gender) | 242 | 484 |
| StereoSet intrasentence (gender) | 255 | 510 |
| **Merged** | **759** | **1,518** |

How each source was handled:
- **CrowS-Pairs** (`data/raw/crows_pairs_anonymized.csv`): rows with `bias_type == "gender"`. `sent_more` -> 1,
  `sent_less` -> 0. We did not use the `stereo_antistereo` column as the label: it records how the writer built
  the pair, not which sentence is more stereotypical. (Checked by hand on a handful of rows only.)
- **StereoSet** (public dev/validation set from Hugging Face `McGill-NLP/stereoset`): `gold_label` 1 =
  stereotype, 0 = anti-stereotype, 2 = unrelated (dropped). Intersentence items join the context sentence to
  the candidate continuation. Intrasentence items use the candidate sentence with the blank already filled.
- Every context/pair gets a `pair_id`, so both sentences can be kept in the same fold when splitting.

The StereoSet "17,000 instances" figure is the full dataset. Most of it is a private test set; only the dev set
is public, which is why the gender portion is only about 500 contexts.

Output columns (`data/processed/merged_gender.csv`): `pair_id`, `sentence`, `label`, `source`
(`crows`, `stereoset_inter`, `stereoset_intra`).

### Generated data (the neutral class and extra stereotype pairs)

Neither dataset has a neutral class (StereoSet's "unrelated" options are nonsense continuations), and a judge trained
only on stereotype vs anti-stereotype forces every sentence into one of the two. So we added **2,261 sentences
written by Qwen**. The three-class training set is 3,774 sentences (`experiments/data3.py`): 1,513 labeled
(1,518 minus 5 duplicates) + 2,261 generated, about 60% generated.

| Added | Class | Sentences | Example | Script |
|---|---|---|---|---|
| Everyday sentence about the people / setting of each labeled pair | neutral | 394 | "She ordered a coffee and sat by the window." | `gen_neutral.py` (Qwen2.5-7B) |
| Facts, no people | neutral | 449 | "Lungs exchange oxygen and carbon dioxide." | `gen_neutral_v2.py` (Qwen2.5-14B) |
| First person, no gender | neutral | 264 | "I like pizza." | 〃 |
| Non-gender stereotypes | neutral | 198 | "Southerners love sweet tea and barbecue." | 〃 |
| Gendered neutral claims | neutral | 138 | "Fathers drive trucks for work." | 〃 |
| Animals | neutral | 136 | "Dogs are loyal." | 〃 |
| Gendered facts | neutral | 97 | "My sister works as an oceanographer." | 〃 |
| Non-gender groups | neutral | 37 | "Doctors work long hours." | 〃 |
| Gendered stereotype claims | stereotype | 274 | "Women are worse at fixing cars." | 〃 |
| Their gender swaps | anti-stereotype | 274 | "Men are worse at fixing cars." | 〃 |

How it was made and filtered:
- `gen_neutral.py`: for each labeled pair, Qwen2.5-7B writes one neutral sentence about the same kind of person and
  setting, so the model has to learn stereotype content rather than topic. Kept if it has a gender word and is 4–40
  words. Qwen2.5-14B later checked these and 190 of 734 were dropped as stereotype-related
  (`neutral_generated_dropped.csv`); 394 unique sentences remain.
- `gen_neutral_v2.py`: Qwen2.5-14B writes the kinds of sentence the first set lacked, across 30 topics. Sentences are
  deduplicated; gendered kinds must contain a gender word and the others must not. The same model then classifies each
  sentence with a separate prompt (no gender / gender stereotype / gendered but neutral / non-gender stereotype) and
  only sentences whose answer matches the intended kind are kept (`data/external/neutral_v2.csv`).
- Stereotype claims are generated by the same model as the neutral ones, and each comes with its gender swap, so
  writing style and sentence shape cannot separate the classes.
- Non-gender stereotypes are labeled neutral by design: the task is gender stereotypes only.

Limits of this data: no person has checked it. Qwen wrote and filtered it, and the judge is also Qwen, so some
agreement is built in. A blind human check of a random sample (two labelers, agreement reported) is the obvious next
step.

GEST (`kinit/gest` v1.1, Apache-2.0, 7,130 gender-swapped rows via `make_gest_pairs.py`) was used only in some
two-class models (e.g. `qwen14b_lora_gest`), never in the three-class pipeline.

### Licensing and attribution

CrowS-Pairs (Nangia et al., 2020) and StereoSet (Nadeem et al., 2021) are released under CC BY-SA 4.0, so the
files derived from them (`data/raw/`, `data/processed/`) carry the same license and need attribution. GEST
(Pikuliak et al., 2024; Apache-2.0) is in `data/external/`. The neutral and extra stereotype sentences in
`data/external/` were generated by Qwen2.5 (see Generated data above).

## Exploration findings (labeled data; descriptive only, not used to choose features)

Short version. The full analysis (sources, preprocessing, pair structure, label quality, word associations with
permutation tests, implications, draft Data section) is in [`DATANOTES.md`](DATANOTES.md).

- **Class balance:** the 1,518 labeled sentences are exactly 759 / 759 overall and within each source, so for the
  two-class task the majority baseline is 50%. The three-class training set is not balanced (45% neutral).
- **Sentence length:** a negligible label cue. Mean 12.08 tokens (stereotype) vs 12.27 (anti-stereotype); 71.5% of
  pairs have equal length. The stereotype sentence is about 0.6 tokens shorter in StereoSet intersentence
  (p = 0.005), the only source with a significant difference. Length differs far more by source
  (intrasentence ~8 tokens, intersentence ~16) but each source is 50/50.
- **Gendered words:** female words appear in 49% of stereotype vs 46% of anti-stereotype sentences; male words in
  53% vs 56%. The gender word alone barely predicts the label. (Word lists in `src/explore.py` are short, so treat
  these as approximate.)
- **Duplicates:** 5 sentences appear in more than one StereoSet context, always with the same label. Keep them
  in the same fold.
- **Label noise:** some StereoSet "stereotype" items are not really about gender (e.g. "My grandfather is old");
  CrowS annotators sometimes tagged gender pairs as another bias type. Useful for error analysis.

## Two-class development (history)

Before the neutral class existed, every model was trained and scored on the 1,518 labeled sentences only, as
stereotype vs anti-stereotype. Same 5 grouped folds, same held-out scoring. The task is balanced, so chance is 0.50.
**None of these numbers describe the final pipeline.** They show how we arrived at the Qwen judge.

### First baselines (`src/train.py`)

| Model | Accuracy | Macro F1 | Fold sd | Pairwise acc. |
|---|---|---|---|---|
| Majority class | 0.500 | 0.333 | 0.000 | – |
| BERT mean-pool + LR | 0.581 | 0.581 | 0.024 | 0.680 |
| all-mpnet-base-v2 + LR | 0.574 | 0.574 | 0.027 | 0.661 |
| all-mpnet-base-v2 + random forest (tuned, `min_samples_leaf` ∈ {1, 3, 10}) | 0.604 | 0.604 | 0.025 | 0.675 |
| Fine-tuned BERT (4 epochs, lr 2e-5) | 0.540 | 0.536 | 0.028 | 0.636 |
| TF-IDF + LR (`tf-idf/train_tfidf.py`) | 0.567 | 0.567 | | |

*Pairwise acc.* = share of pairs where the model gives the stereotype sentence a higher P(stereotype) than its
anti-stereotype partner.

Accuracy by source:

| Source | BERT + LR | mpnet + LR | mpnet + RF | Fine-tuned BERT |
|---|---|---|---|---|
| CrowS-Pairs | 0.550 | 0.525 | 0.573 | 0.510 |
| StereoSet inter | 0.620 | 0.576 | 0.597 | 0.560 |
| StereoSet intra | 0.576 | 0.624 | 0.643 | 0.553 |

Findings:
- All baselines were only 4–10 points above chance.
- Single-sentence classification is much harder than ranking within a pair (0.60 vs 0.68). The label is relative:
  "stereotype" means *more* stereotypical than its partner, not stereotypical on its own.
- The label depends on gender and attribute together ("she is nurturing" vs "he is nurturing"); neither word
  predicts it alone.
- CrowS-Pairs is hardest: most pairs differ only in the gender word, so the two sentences embed almost identically.
- Fine-tuning BERT did not help with ~1,200 training sentences.

Outputs: `outputs/metrics.csv`, `outputs/metrics_by_fold.csv`, `outputs/predictions.csv`,
`outputs/rf_best_params.csv`.

### Improved two-class classifiers (`experiments/`)

| Model | Accuracy | Script |
|---|---|---|
| Snowflake-arctic-embed-l-v2.0 + TF-IDF + gender × word, LR | 0.640 | `topk.py` |
| + roberta-large masked-LM gender feature | 0.653 | `mlm_features.py` |
| + Qwen2.5-7B/14B features (gender-swap likelihood, zero-shot judge), top-k, LR | 0.678 | `llm_features.py`, `topk.py` |
| Same features, random forest | 0.698 | `rf_tuned.py` |
| Qwen2.5-7B-Instruct LoRA, 8 epochs, 3 seeds averaged | 0.760 | `finetune_cv.py` |
| Qwen2.5-7B-Instruct LoRA + GEST training data | 0.781 | `finetune_cv.py --extra` |
| Qwen2.5-14B-Instruct QLoRA (4-bit) + GEST | 0.810 | `finetune_cv.py --4bit --extra` |
| Qwen2.5-14B-Instruct QLoRA (4-bit), 12 epochs (best single two-class model) | 0.820 | `finetune_cv.py --4bit` |
| Average of the two 14B runs | 0.833 | `ensemble.py` |
| Stacked ensemble of 9 models (LR on logits, fitted on training folds) | 0.838 (sd 0.020) | `ensemble.py` |

What helped: LLM fine-tuning (LoRA on q/k/v/o, r = 16, lr 1e-4) over frozen embeddings; a bigger model (14B > 7B);
more training data. What did not: stop-word removal, PCA (the signal is in low-variance directions), gender-swap
augmentation, MLP instead of LR/RF, a pairwise ranking loss, label-noise filtering, full fine-tuning of roberta-large
(collapsed to chance), other encoders (BERT-large, RoBERTa, mpnet all below Snowflake-L).

The 9-model ensemble was never used for new text and was not rebuilt for three classes. Many configurations were
tried, so the top numbers carry some selection bias (fold sd ≈ 0.02). All out-of-fold predictions are in
`experiments/feat/`, so `ensemble.py` reproduces this table without a GPU.

Two-class saved models (still in `models/`, used only as the "old pipeline" comparison and for SHAP):
`qwen14b_lora` (8 epochs) and `qwen14b_lora_gest` (+ GEST, 3 epochs), both trained on all 1,518 labeled sentences.
`python experiments/predict.py models/qwen14b_lora models/qwen14b_lora_gest --neutral -- "your sentence"` runs the
old pipeline.

### Old neutral detector (replaced)

The first fix for neutral text was a separate detector in front of the two-class judges: Snowflake-L + logistic
regression trained on labeled sentences vs the 734 `gen_neutral.py` sentences (`neutral_clf.py` →
`models/neutral_lr.npz`). It scored 0.924 in grouped CV on that data, but every neutral example it had seen was a
person doing an everyday thing, so it called almost any general claim stereotype-related ("Water is healthy." 0.81,
"Dogs are loyal." 0.93, "Men like apples." 0.93). On the hand-written test set that pipeline scored 0.43. It was
replaced by the gender filter and the 3-class judge. Also tried and dropped: calling a sentence neutral when its
gender-swapped version gets a similar score (`neutral_eval.py`); it called 54% of real labeled sentences neutral.

## Interpretability (SHAP)

Which words drive the Qwen judge, and whether it reads the *gender + attribute* pairing rather than either word alone.
These offline analyses use the **two-class** adapter `qwen14b_lora`; the demo's Explain button computes live SHAP for
the three-class judge.

Results: [interactive interpretability report](https://claude.ai/artifact/48u3FWFt3Mesq4qFqPKXi7) (source:
`docs/interpretability.html`; rebuild with `python experiments/shap_report.py`). Summary table:
`outputs/shap/comparison.md`.

- `shap_qwen.py` (GPU): SHAP Partition explainer on the adapter's log-odds of "stereotype" (masked tokens → "...").
  It also runs a masking interaction for each sentence: G = its gendered words, a = a top non-gender word,
  `f(x) − f(x\G) − f(x\a) + f(x\G\a)`. A value > 0 means the attribute pushes toward "stereotype" more when the gender
  word is present.
- `shap_analyze.py` (CPU) outputs word importance (mean |SHAP|), the share of attribution on gender vs. attribute
  words, the same attribute's SHAP in male vs. female sentences, CrowS gender swaps (do shared words keep their SHAP
  when only the gender changes?), and an interaction summary.
- Caveats: the adapters were trained on all 1,518 sentences, so this explains learned behavior, not held-out
  generalization. Masked inputs are off-distribution.

```bash
python experiments/shap_qwen.py models/qwen14b_lora --max-evals 800 --batch 64   # GPU, all 1,518: ~50 min on an RTX 8000
python experiments/shap_analyze.py outputs/shap/qwen14b_lora                      # CPU, tables + charts + summary.txt
python experiments/shap_report.py                                                  # CPU, rebuilds docs/interpretability.html
python experiments/shap_qwen.py models/qwen14b_lora --data data/external/gest_pairs.csv --n 1500 --max-evals 800  # held out: GEST was not in this adapter's training data
python experiments/shap_qwen.py --neutral --max-evals 800 --batch 64   # old neutral detector, ~45 min
```

## Demo

`demo/` is a web page where you type sentences and the three-class pipeline labels them (gender filter → 3-class 14B
judge), with each probability shown as a bar, a button that adds gender-swapped versions, and an **Explain** button
that computes live word-level SHAP values for the judge's predicted class (exact Shapley values up to 9 words, sampled
permutations beyond; ~0.1–0.4 s per short sentence).

```bash
python demo/server.py                  # on lisplab (GPU, ~10 GB VRAM): loads models once, serves localhost:8000
ssh -N -L 8000:localhost:8000 <you>@lisplab-1.thayer.dartmouth.edu   # on your laptop, then open http://localhost:8000
python demo/server.py --mock           # no GPU: fake scores, for editing the page
```

Options: `--gpu N`, `--port N`, `--host 0.0.0.0`, `--password-file FILE` (visitors must enter a shared password).

Shared link: Thayer's firewall blocks port 8000, so the lisplab copy (`/scratch2/f006j44/stereo-demo`) is served
through a Cloudflare named tunnel at https://demo.stereotypejudge.com, protected by `--password-file`. A watchdog
script restarts the server and tunnel if they stop; after a lisplab reboot, run `bash start-all.sh` there.

## Limitations

- About 60% of the three-class training data, including the whole neutral class, was generated and filtered by Qwen,
  and the judge is Qwen. No person has checked the generated data yet.
- The hand-written test set is small (92 sentences; one sentence moves accuracy by 1.1 points), 75% neutral, and was
  written by Claude, not reviewed by people.
- Stereotype vs anti-stereotype remains the hard part: 0.82 on the human-labeled sentences, often with too much
  confidence. Use the judge as a flag for human review, not as a measure of bias.
- Labels are noisy (see Exploration findings), and StereoSet's public gender data is small (~500 pairs).
- Qwen may have seen CrowS-Pairs and StereoSet in pretraining.

## Reproduce

Data and two-class baselines (CPU):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/download_stereoset.py   # fetches StereoSet into data/raw/
python src/merge.py                # writes data/processed/merged_gender.csv
python src/explore.py              # prints summaries, writes outputs/
python src/split.py                # writes data/processed/folds.csv
python src/analyze_data.py         # dataset analysis behind DATANOTES.md, writes outputs/word_zscores.csv
python src/embed.py                # caches BERT + mpnet embeddings in data/embeddings/
python src/train.py                # two-class baselines, writes outputs/
```

Three-class pipeline (fine-tuning needs a CUDA GPU; we used lisplab: RTX 5000 16 GB for 14B in 4-bit, 1–3 h per fold.
Extra packages: `peft bitsandbytes accelerate`):

```bash
python experiments/gen_neutral.py Qwen/Qwen2.5-7B-Instruct   # GPU, first neutral set
python experiments/gen_neutral_v2.py                         # GPU, ~15 min, rest of the generated data
python experiments/gender_filter.py                          # GPU for embeddings, ~1 min → models/gender_filter.npz
python experiments/finetune_cv.py Qwen/Qwen2.5-14B-Instruct --lora --4bit --three --epochs 6 --fold 0   # one fold per GPU
python experiments/finetune_cv.py Qwen/Qwen2.5-14B-Instruct --lora --4bit --three --epochs 5 --final models/qwen14b_lora_3class
python experiments/eval_three.py experiments/feat/ft_Qwen2.5-14B-Instruct_lora_4bit_3class_ep6_s72.csv \
    models/qwen14b_lora_3class models/qwen14b_lora models/qwen14b_lora_gest
python experiments/baselines_3class.py                       # CPU, ~10 min, three-class baselines table
```

Two-class development models:

```bash
python experiments/llm_features.py Qwen/Qwen2.5-7B-Instruct          # GPU, writes experiments/feat/llm_*.csv
python experiments/topk.py --llm=7B,14B                              # CPU, feature pipeline
python experiments/finetune_cv.py Qwen/Qwen2.5-14B-Instruct --lora --4bit --epochs 12 --fold 0   # one fold per GPU
python experiments/assemble_folds.py Qwen2.5-14B-Instruct_lora_4bit_ep12_s72   # join per-fold .npy
python experiments/ensemble.py                                       # CPU, two-class table from experiments/feat/*.csv
```

Scripts locate files relative to the repo root, so run them from anywhere inside it. The base model downloads from
Hugging Face on first use (~30 GB).

## Reproducing the report

The report's tables and figures come from the saved out-of-fold predictions in `experiments/feat/`, so they run on
a CPU; a GPU is only needed to retrain the Qwen models (commands above). The analysis scripts are numbered in
order, and each docstring names the step of `docs/MetricsAnalysisPlan.md` it implements:

```bash
python analysis/01_build_table.py             # every model's out-of-fold P(stereotype) per sentence
python analysis/02_metrics.py                 # core metrics and charts
python analysis/09_extra_metrics.py           # intervals, model comparisons, calibration, MCC/kappa
python analysis/06_pair_outcomes.py           # both right / same label to both / swapped
python analysis/08_neutral_oof.py             # neutral-detector scores for every labeled sentence
python analysis/10_three_class.py             # three-class results for the old neutral-detector pipeline
python analysis/05_error_review.py            # sample of Qwen's errors to read and categorize
python analysis/11_error_analysis.py          # where the models fail
python analysis/03_make_audit_sheet.py        # blind audit sheets and the private answer key
python analysis/04_score_audit.py             # unblind the audit (first labeler)
python analysis/13_score_second_labeler.py    # second labeler's sheets vs dataset, first labeler, Qwen
python analysis/14_audit_vs_three_class.py    # three-class pipeline vs the audit labels
python analysis/12_report_figures.py          # Figures 1-8 -> outputs/report_figures/
```

Most of these scripts (01–11) analyze the **two-class** models; 14 uses the three-class pipeline. The three-class
results tables come from `experiments/eval_three.py` and `experiments/baselines_3class.py` (see Results above).

The audit answer key (`outputs/audit/private/`) is deliberately not committed, so it stays hidden from labelers.
`03_make_audit_sheet.py` regenerates it (seed 72); steps 04, 13 and Figure 4 need it. The labeled sheets are in
`outputs/audit/` (`audit_results_*.csv` for the first labeler, `chris_labels_*.tsv` for the second). Tables are in
`outputs/metrics/`, and the manual error reviews are in `outputs/audit/`.
