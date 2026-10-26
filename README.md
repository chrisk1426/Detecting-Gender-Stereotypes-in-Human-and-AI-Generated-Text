# Detecting Gender Stereotypes in Human- and AI-Generated Text

COSC 72 course project (Julia Allos, Chris Kang). We build a "stereotype judge": a classifier trained on
human-labeled gender-stereotype sentences, then applied to language-model-generated text to measure
occupation-linked gender bias. See the project proposal for the full plan.

## Status

Done so far: dataset merge, exploration, grouped 5-fold split, Chris's Week 1–2 models, and the improved classifier
(0.838 accuracy, see [Improved classifier](#improved-classifier-experiments)).
Not yet started: TF-IDF + LR (Julia), generation pipeline, error analysis, scoring pipeline, demo.

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
Pipeline chart: `docs/pipeline.html`.

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
All out-of-fold predictions are committed in `experiments/feat/`, so `ensemble.py` reproduces the final table
without a GPU.

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

### Output columns (`data/processed/merged_gender.csv`)
`pair_id`, `sentence`, `label`, `source` (`crows`, `stereoset_inter`, `stereoset_intra`)

## Exploration findings (descriptive only; not used to choose features)

- **Class balance:** exactly 759 / 759 overall and within each source, so a majority-class baseline scores 50%.
- **Sentence length:** no label cue. Mean 12.08 tokens (stereotype) vs 12.27 (anti-stereotype); 71.5% of pairs
  have equal length. Length differs far more by source (intrasentence ~8 tokens, intersentence ~16) but each
  source is 50/50.
- **Gendered words:** female words appear in 49% of stereotype vs 46% of anti-stereotype sentences; male words in
  53% vs 56%. The gender word alone barely predicts the label, which supports the expectation that a bag-of-words
  model will do poorly. (Word lists in `src/explore.py` are short, so treat these as approximate.)
- **Duplicates:** 5 sentences appear in more than one StereoSet context, always with the same label. Keep them
  in the same fold.
- **Label noise:** some StereoSet "stereotype" items are not really about gender (e.g. "My grandfather is old");
  CrowS annotators sometimes tagged gender pairs as another bias type. Useful for error analysis.

## Limitations

- The judge only outputs stereotype vs anti-stereotype. Neutral text (e.g. LLM output like "The nurse checked the
  chart") will still be forced into one of the two labels. The neutral-class fix is still undecided.
- The baseline models (~0.60) are unreliable per sentence; the final ensemble (0.84) is better, but still aggregate
  scores over many generations rather than trusting single judgements.
- Labels are noisy (see Exploration findings), and StereoSet's public gender data is small (~500 pairs).

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/download_stereoset.py   # fetches StereoSet into data/raw/
python src/merge.py                # writes data/processed/merged_gender.csv
python src/explore.py              # prints summaries, writes outputs/
python src/split.py                # writes data/processed/folds.csv
python src/embed.py                # caches BERT + mpnet embeddings in data/embeddings/
python src/train.py                # trains and scores all models, writes outputs/
```
Scripts locate files relative to the repo root, so run them from anywhere inside it.
