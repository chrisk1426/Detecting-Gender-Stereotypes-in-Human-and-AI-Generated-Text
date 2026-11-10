# Detecting Gender Stereotypes in Human- and AI-Generated Text

COSC 72 course project (Julia Allos, Chris Kang). We build a "stereotype judge": a classifier trained on
human-labeled gender-stereotype sentences that labels a sentence as stereotype, anti-stereotype, or neutral.
The proposal also planned to apply the judge to language-model-generated text to measure occupation-linked
gender bias. We dropped that stage and studied the judge itself instead (blind audit, error analysis, SHAP).
The report has the final scope; the proposal has the original plan.

## Status

Done: dataset merge, exploration and data analysis (`DATANOTES.md`), grouped 5-fold split, baselines (majority,
TF-IDF + LR, embeddings + LR/RF), the fine-tuned Qwen judge and 9-model ensemble (0.838 accuracy, see
[Improved classifier](#improved-classifier-experiments)), the three-class pipeline with a gender filter, SHAP
interpretability, the interactive demo ([Demo](#demo)), and the metrics, blind audit and error analysis behind the
report ([Reproducing the report](#reproducing-the-report)).
Not built (dropped from the proposal): the generation pipeline and the aggregate scoring pipeline for
language-model-generated text.

## Classifiers (Chris)

Evaluation is 5-fold grouped cross-validation. Both halves of a pair stay in the same fold, and pairs that share a
sentence are grouped together (`src/split.py`, a stand-in until Julia's split is in). Metrics are pooled over
out-of-fold predictions.

| Model | Accuracy | Macro F1 | Fold sd | Pairwise acc. |
|---|---|---|---|---|
| Majority class | 0.500 | 0.333 | 0.000 | – |
| BERT mean-pool + LR | 0.581 | 0.581 | 0.024 | 0.680 |
| all-mpnet-base-v2 + LR | 0.574 | 0.574 | 0.027 | 0.661 |
| **all-mpnet-base-v2 + random forest (tuned)** | **0.604** | **0.604** | 0.025 | 0.675 |
| Fine-tuned BERT (4 epochs, lr 2e-5) | 0.540 | 0.536 | 0.028 | 0.636 |

*Pairwise acc.* = share of pairs where the model gives the stereotype sentence a higher P(stereotype) than its
anti-stereotype partner.

**Accuracy by source**

| Source | BERT + LR | mpnet + LR | mpnet + RF | Fine-tuned BERT |
|---|---|---|---|---|
| CrowS-Pairs | 0.550 | 0.525 | 0.573 | 0.510 |
| StereoSet inter | 0.620 | 0.576 | 0.597 | 0.560 |
| StereoSet intra | 0.576 | 0.624 | 0.643 | 0.553 |

**Findings**
- All models are only 4–10 points above chance. The mpnet random forest is best, but its lead over BERT + LR
  (2.3 points) is about one fold standard deviation.
- Single-sentence classification is much harder than ranking within a pair (0.60 vs 0.68). The label is relative:
  "stereotype" means *more* stereotypical than its partner, not stereotypical on its own.
- The label depends on gender and attribute together ("she is nurturing" vs "he is nurturing"). Neither word
  predicts the label alone, which suits a model that can combine features (random forest) better than a linear one.
- CrowS-Pairs is hardest. Most of its pairs differ only in the gender word, so the two sentences embed almost
  identically.
- Fine-tuning BERT did not help with ~1,200 training sentences and default settings. It over-predicts
  "stereotype" (recall 0.64 vs 0.45) and varies by about a point between runs on Apple GPUs.
- Embedding-based models are not interpretable: their features are embedding dimensions. Use Julia's TF-IDF + LR
  word weights to show which words matter.

**Setup:** LR uses standardised embeddings and C = 0.1 (untuned). The random forest (300 trees) is tuned by
nested CV: a grid over `max_features` ∈ {sqrt, 0.1, 0.3} and `min_samples_leaf` ∈ {1, 3, 10}, using grouped inner
folds of each training split, so the test fold never picks settings. Chosen settings vary by fold
(`outputs/rf_best_params.csv`).

**Outputs:** `outputs/metrics.csv` (per-label precision/recall/F1), `outputs/metrics_by_fold.csv`,
`outputs/predictions.csv` (out-of-fold predictions and probabilities for error analysis). `src/train.py` takes
about 10 minutes, most of it fine-tuning and the forest grid search.

## Improved classifier (`experiments/`)

Same data, same 5 grouped folds, single-sentence prediction. Every number is out-of-fold mean accuracy; nothing
was tuned on the test fold (epochs and top-k are picked on validation splits of the training folds).
Pipeline chart: [interactive version](https://claude.ai/artifact/9LPNDXBa58tejhcr381ork) (source: `docs/pipeline.html`, open it in a browser).

### Final pipeline (for new text)

```
sentence
  │
  ├─ 1. Neutral check: Snowflake-L embedding (1,024 numbers) → logistic regression
  │       P(stereotype-related) < 0.5  →  label "neutral", stop
  │
  ├─ 2. Two judges: Qwen2.5-14B (4-bit) + LoRA adapter, each outputs [anti, stereo] logits → softmax
  │       models/qwen14b_lora (8 epochs)   and   models/qwen14b_lora_gest (+ GEST, 3 epochs)
  │
  └─ 3. Average the two P(stereotype)  →  > 0.5 "stereotype", otherwise "anti-stereotype"
```

`python experiments/predict.py models/qwen14b_lora models/qwen14b_lora_gest --neutral -- "your sentence"` runs all
three steps (GPU). Worked example: "Women are too emotional to be leaders." → neutral check 0.994 → judges 100% and
99.99% → stereotype; "Men are too emotional to be leaders." → 0.03% → anti-stereotype; "She ordered a coffee and sat
by the window." → neutral check 0.114 → neutral. The chart above traces each step with these numbers
(`experiments/trace_example.py`).

Accuracy: step 2–3 scores 0.833 in cross-validation (stereotype vs anti-stereotype); the 9-model stacked ensemble
below (0.838) is the reported number but is not used for new text. The neutral check scores 0.924.

| Model | Accuracy | Script |
|---|---|---|
| Snowflake-arctic-embed-l-v2.0 + TF-IDF + gender × word, LR | 0.640 | `topk.py` |
| + roberta-large masked-LM gender feature | 0.653 | `mlm_features.py` |
| + Qwen2.5-7B/14B features (gender-swap likelihood, zero-shot judge), top-k, LR | 0.678 | `llm_features.py`, `topk.py` |
| Same features, random forest | 0.698 | `rf_tuned.py` |
| Qwen2.5-7B-Instruct LoRA, 8 epochs, 3 seeds averaged | 0.760 | `finetune_cv.py` |
| Qwen2.5-7B-Instruct LoRA + GEST training data | 0.781 | `finetune_cv.py --extra` |
| Qwen2.5-14B-Instruct QLoRA (4-bit) + GEST | 0.810 | `finetune_cv.py --4bit --extra` |
| **Qwen2.5-14B-Instruct QLoRA (4-bit), 12 epochs** (best single model) | **0.820** | `finetune_cv.py --4bit` |
| Average of the two 14B runs | 0.833 | `ensemble.py` |
| **Stacked ensemble of 9 models** (LR on logits, fitted on training folds) | **0.838** (sd 0.020) | `ensemble.py` |

Pairwise accuracy of the stacked ensemble is 0.891.

**What helped:** LLM fine-tuning (LoRA adapters on q/k/v/o, r = 16, lr 1e-4) over frozen embeddings; a bigger
model (14B > 7B); more training data. GEST (`kinit/gest` v1.1, Apache-2.0) is converted to gender-swapped pairs
(`make_gest_pairs.py` → `data/external/gest_pairs.csv`, 7,130 rows) and added to training folds only.

**What did not:** stop-word removal, PCA (the signal is in low-variance directions), gender-swap augmentation,
MLP instead of LR/RF, a pairwise ranking loss, label-noise filtering, full fine-tuning of roberta-large (collapsed).

**Caveats:** many configurations were tried, so the top numbers carry some selection bias (fold sd ≈ 0.02).
Qwen may have seen CrowS-Pairs / StereoSet in pretraining. Using an LLM to judge LLM-generated text is partly
circular.

**Running it:** fine-tuning needs a CUDA GPU (we used lisplab1: RTX 8000 48 GB for 7B, RTX 5000 16 GB for 14B
in 4-bit, about 1–3 h per fold). Extra packages: `peft bitsandbytes accelerate`. Example:

```bash
python experiments/llm_features.py Qwen/Qwen2.5-7B-Instruct          # GPU, writes experiments/feat/llm_*.csv
python experiments/topk.py --llm=7B,14B                              # CPU, feature pipeline
python experiments/finetune_cv.py Qwen/Qwen2.5-14B-Instruct --lora --4bit --epochs 12 --fold 0   # one fold per GPU
python experiments/assemble_folds.py Qwen2.5-14B-Instruct_lora_4bit_ep12_s72   # join per-fold .npy
python experiments/ensemble.py                                       # CPU, final numbers from experiments/feat/*.csv
```
**Saved models (`models/`):** two LoRA adapters for Qwen2.5-14B-Instruct (4-bit), trained on all 1,518 sentences
for the demo: `qwen14b_lora` (8 epochs) and `qwen14b_lora_gest` (+ GEST, 3 epochs). Epoch counts are the typical
best epochs from CV. Nothing was held out, so their accuracy is the CV estimate (average of the two: 0.833). The
base model downloads from Hugging Face on first use (~30 GB).

```bash
python experiments/finetune_cv.py Qwen/Qwen2.5-14B-Instruct --lora --4bit --epochs 8 --final models/qwen14b_lora
python experiments/predict.py models/qwen14b_lora models/qwen14b_lora_gest -- "Women are too emotional to be leaders."
```

**Neutral class:** the judge was trained only on stereotype vs anti-stereotype, so on its own it labels plain text
like "She ordered a coffee" a stereotype. A neutral detector now runs first (`predict.py --neutral`); only sentences
it calls stereotype-related go to the 14B models.

- Data: for each pair, Qwen2.5-7B writes one neutral sentence about the same people and setting
  (`gen_neutral.py` → `data/external/neutral_generated.csv`, 734 kept), so the detector must learn stereotype
  content rather than topic.
- Model: Snowflake-L embedding + logistic regression (`neutral_clf.py` → `models/neutral_lr.npz`, runs on CPU).
- Results: 0.924 grouped-CV accuracy (neutral vs stereotype-related); 97% of labeled sentences kept; 80% of 60
  hand-written neutral sentences (`neutral_sentences.csv`) called neutral. Control: 96.5% of stereotype sentences
  reworded by Qwen (`gen_neutral.py --paraphrase`) are still flagged, so it did not learn "LLM wording = neutral".
- Misses are mostly occupation sentences ("The mechanic said he would call…", score 0.5–0.7). The neutral test sentences
  were written by us/Claude, so treat 80% as a rough check.
- Held out, 97.0% of stereotype and 97.4% of anti-stereotype sentences get past the check. The 23 stereotype sentences
  it calls neutral are nearly all implicit, role-congruent descriptions ("The mother is home. She is in the kitchen
  cooking.", "He does the yardwork by himself."). This is the same gray area as the occupation misses above: whether a
  plain description that fits a gender role is a stereotype is a judgment call the datasets make inconsistently.
- Tried first: calling a sentence neutral when its gender-swapped version gets a similar score (`neutral_eval.py`).
  It failed: 54% of real labeled sentences were also called neutral, because many stereotypes do not flip with gender.

All out-of-fold predictions are committed in `experiments/feat/`, so `ensemble.py` reproduces the final table
without a GPU.

**3-class pipeline (replaces the neutral detector):** the neutral detector above called almost any general claim
stereotype-related ("Water is healthy." 0.81, "Dogs are loyal." 0.93, "Men like apples." 0.93), because every neutral
example it saw was a person doing an everyday thing. On a hand-written test set the old pipeline got 0.43.

- Data (`gen_neutral_v2.py` → `data/external/neutral_v2.csv`): Qwen2.5-14B writes the missing kinds of sentence (no
  people, animals, first person, non-gender groups, non-gender stereotypes, gendered neutral claims like "Women drive
  trucks.", gendered facts) plus gendered stereotype claims with their gender swaps (so writing style can't separate
  the classes). The same model checks each sentence; only agreeing ones are kept. The old neutral set is cleaned the
  same way (190 of 734 dropped as stereotype-related, `neutral_generated_dropped.csv`).
- Models: `gender_filter.py` (Snowflake-L + logistic regression, gender-related or not → `models/gender_filter.npz`;
  XGBoost compared) and `finetune_cv.py --three` (Qwen2.5-14B LoRA, 3 classes: anti / stereotype / neutral, on all
  data from `data3.py`; same 5 grouped folds → `models/qwen14b_lora_3class`, 5 epochs).
- Test set: `data/external/ood_test.csv`, 95 hand-written sentences in 9 categories that nothing trains on
  (3 overlapping with training data dropped). Written by Claude, so it should be reviewed and extended.

| | Old pipeline | New pipeline |
|---|---|---|
| Hand-written test set (92) | 0.43 | **0.92** |
| 3-class CV accuracy | – | 0.892 (sd 0.008) |
| Stereo vs anti on labeled sentences (CV) | 0.833 (2-model ensemble) | 0.822 |

By test category (new): non-gender sentences 0.86–1.00, gender facts 1.00, stereotypes 1.00, anti-stereotypes 0.78,
**gendered neutral claims 0.67** (weakest: "Men like apples." is still called anti-stereotype; only 138 such
sentences survived the check). The filter adds nothing measurable over the 3-class judge alone; it is kept as a cheap
first step. Full output: `outputs/three_class/eval.log`, `experiments/feat/ood_test_*.csv`.

```bash
python experiments/gen_neutral_v2.py                     # GPU, ~15 min
python experiments/gender_filter.py                      # GPU for embeddings, ~1 min
python experiments/finetune_cv.py Qwen/Qwen2.5-14B-Instruct --lora --4bit --three --epochs 6 --fold 0   # one per GPU
python experiments/finetune_cv.py Qwen/Qwen2.5-14B-Instruct --lora --4bit --three --epochs 5 --final models/qwen14b_lora_3class
python experiments/eval_three.py experiments/feat/ft_Qwen2.5-14B-Instruct_lora_4bit_3class_ep6_s72.csv \
    models/qwen14b_lora_3class models/qwen14b_lora models/qwen14b_lora_gest
```

**Interpretability (SHAP):** which words drive the Qwen judge, and whether it reads the *gender + attribute*
pairing rather than either word alone.

Results: [interactive interpretability report](https://claude.ai/artifact/48u3FWFt3Mesq4qFqPKXi7) (source: `docs/interpretability.html`,
open it in a browser; rebuild with `python experiments/shap_report.py`). Summary table: `outputs/shap/comparison.md`.

- `shap_qwen.py` (GPU): SHAP Partition explainer on the adapter's log-odds of "stereotype" (masked tokens → "...").
  It also runs a masking interaction for each sentence: G = its gendered words, a = a top non-gender word,
  `f(x) − f(x\G) − f(x\a) + f(x\G\a)`. A value > 0 means the attribute pushes toward "stereotype" more when the gender
  word is present.
- `shap_analyze.py` (CPU) outputs:
  - word importance (mean |SHAP|)
  - share of attribution on gender vs. attribute words
  - the same attribute's SHAP in male vs. female sentences
  - CrowS gender swaps: do the shared words keep their SHAP when only the gender changes?
  - interaction summary
- Caveats: the adapters were trained on all 1,518 sentences, so this explains learned behavior, not held-out
  generalization. Masked inputs are off-distribution.

```bash
python experiments/shap_qwen.py models/qwen14b_lora --max-evals 800 --batch 64   # GPU, all 1,518: ~50 min on an RTX 8000
python experiments/shap_analyze.py outputs/shap/qwen14b_lora                      # CPU, tables + charts + summary.txt
python experiments/shap_report.py                                                  # CPU, rebuilds docs/interpretability.html
python experiments/shap_qwen.py models/qwen14b_lora --data data/external/gest_pairs.csv --n 1500 --max-evals 800  # held out: GEST was not in this adapter's training data
python experiments/shap_qwen.py --neutral --max-evals 800 --batch 64   # neutral detector: its training data + held-out neutral sets, ~45 min
```

## Demo

`demo/` is a web page where you type sentences and the 3-class pipeline labels them (gender filter → 3-class 14B
judge), with each probability shown as a bar, a button that adds gender-swapped versions, and an **Explain** button
that computes live word-level SHAP values for the judge's predicted class (exact Shapley values up to 9 words, sampled
permutations beyond; ~0.1–0.4 s per short sentence).

```bash
python demo/server.py                  # on lisplab (GPU, ~10 GB VRAM): loads models once, serves localhost:8000
ssh -N -L 8000:localhost:8000 <you>@lisplab-1.thayer.dartmouth.edu   # on your laptop, then open http://localhost:8000
python demo/server.py --mock           # no GPU: fake scores, for editing the page
```

Options: `--gpu N`, `--port N`, `--host 0.0.0.0`, `--password-file FILE` (visitors must enter a shared password).

Shared link: Thayer's firewall blocks port 8000, so the lisplab copy (`/scratch2/f006j44/stereo-demo`) is shared through
a Cloudflare quick tunnel (`cloudflared tunnel --url http://localhost:8000`, tmux session `tunnel`) and protected by
`--password-file password.txt`. The trycloudflare.com URL changes whenever the tunnel restarts (see `tunnel.log`).

## Data

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

### Licensing and attribution

CrowS-Pairs (Nangia et al., 2020) and StereoSet (Nadeem et al., 2021) are released under CC BY-SA 4.0, so the
files derived from them (`data/raw/`, `data/processed/`) carry the same license and need attribution. GEST
(Pikuliak et al., 2024; Apache-2.0) is in `data/external/`. The neutral sentences in `data/external/` were
generated by Qwen2.5 (see Neutral class above).

### Output columns (`data/processed/merged_gender.csv`)
`pair_id`, `sentence`, `label`, `source` (`crows`, `stereoset_inter`, `stereoset_intra`)

## Exploration findings (descriptive only; not used to choose features)

Short version. The full analysis (sources, preprocessing, pair structure, label quality, word associations with
permutation tests, implications, draft Data section) is in [`DATANOTES.md`](DATANOTES.md).

- **Class balance:** exactly 759 / 759 overall and within each source, so a majority-class baseline scores 50%.
- **Sentence length:** a negligible label cue. Mean 12.08 tokens (stereotype) vs 12.27 (anti-stereotype); 71.5% of
  pairs have equal length. The stereotype sentence is about 0.6 tokens shorter in StereoSet intersentence
  (p = 0.005), the only source with a significant difference. Length differs far more by source
  (intrasentence ~8 tokens, intersentence ~16) but each source is 50/50.
- **Gendered words:** female words appear in 49% of stereotype vs 46% of anti-stereotype sentences; male words in
  53% vs 56%. The gender word alone barely predicts the label, which supports the expectation that a bag-of-words
  model will do poorly. (Word lists in `src/explore.py` are short, so treat these as approximate.)
- **Duplicates:** 5 sentences appear in more than one StereoSet context, always with the same label. Keep them
  in the same fold.
- **Label noise:** some StereoSet "stereotype" items are not really about gender (e.g. "My grandfather is old");
  CrowS annotators sometimes tagged gender pairs as another bias type. Useful for error analysis.

## Limitations

- Neutral text is handled by the three-class pipeline (see Neutral class above). Its neutral class is synthetic
  (generated and checked by Qwen), and the 92-sentence hand-written test set was written by Claude and has not
  been reviewed, so its 0.92 accuracy should be read with that in mind.
- Single-sentence judgements are unreliable: the baselines score ~0.60 and the best models 0.82–0.84, often with
  too much confidence. Use the judge as a flag for human review, not as a measure of bias.
- Labels are noisy (see Exploration findings), and StereoSet's public gender data is small (~500 pairs).

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/download_stereoset.py   # fetches StereoSet into data/raw/
python src/merge.py                # writes data/processed/merged_gender.csv
python src/explore.py              # prints summaries, writes outputs/
python src/split.py                # writes data/processed/folds.csv
python src/analyze_data.py         # dataset analysis behind DATANOTES.md, writes outputs/word_zscores.csv
python src/embed.py                # caches BERT + mpnet embeddings in data/embeddings/
python src/train.py                # trains and scores all models, writes outputs/
```
Scripts locate files relative to the repo root, so run them from anywhere inside it.

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
python analysis/10_three_class.py             # three-class results
python analysis/05_error_review.py            # sample of Qwen's errors to read and categorize
python analysis/11_error_analysis.py          # where the models fail
python analysis/03_make_audit_sheet.py        # blind audit sheets and the private answer key
python analysis/04_score_audit.py             # unblind the audit (first labeler)
python analysis/13_score_second_labeler.py    # second labeler's sheets vs dataset, first labeler, Qwen
python analysis/14_audit_vs_three_class.py    # three-class pipeline vs the audit labels
python analysis/12_report_figures.py          # Figures 1-8 -> outputs/report_figures/
```

The audit answer key (`outputs/audit/private/`) is deliberately not committed, so it stays hidden from labelers.
`03_make_audit_sheet.py` regenerates it (seed 72); steps 04, 13 and Figure 4 need it. The labeled sheets are in
`outputs/audit/` (`audit_results_*.csv` for the first labeler, `chris_labels_*.tsv` for the second). Tables are in
`outputs/metrics/`, and the manual error reviews are in `outputs/audit/`.
