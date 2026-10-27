# Metrics, Manual Audit and Error Analysis Plan

Owner: Julia. Chris handles the neutral detector and interpretability (SHAP). This file is a living log:
tick boxes and fill in the Progress Log as we go, so every number and chart can be reproduced.

Why this exists: in the Oct 26 TA meeting we were told to (1) audit the data and predictions by hand,
(2) report strong metrics (precision, recall, F1, ROC/AUC, confusion matrices) and (3) keep the scope
course-sized. Accuracy is already about 0.83-0.84 and does not need more tuning.

## Status

- [x] A. Combined predictions table (done 2026-10-27, `analysis/01_build_table.py`)
- [~] B. Metrics and charts: core B1-B8 done 2026-10-27 (`analysis/02_metrics.py`); extra metrics X1-X5 still to do
- [ ] C. Manual audit (blind labeling)
- [ ] D. Error analysis
- [ ] E. Three-class results with Chris's neutral detector (blocked until his part is ready)

## Ground rules

- No GPU needed for A-D. Every Qwen prediction is already saved in `experiments/feat/` (see Inputs).
- Only use **out-of-fold** predictions: each sentence was scored by a model that never trained on its fold.
  This is what makes the numbers honest.
- Every script is run from the repo root, writes to `outputs/`, and uses a fixed random seed (72, as in the
  existing fine-tuning runs).
- Label convention from the repo: `1` = stereotype-consistent, `0` = anti-stereotype. A model "predicts 1"
  when its probability is above 0.5.
- Nothing here changes the models or the existing results. We only read them.

## Inputs (already in the repo)

| File | What it is |
|---|---|
| `data/processed/merged_gender.csv` | 1,518 labeled sentences: `pair_id, sentence, label, source` |
| `data/processed/folds.csv` | Fold (0-4) for each `pair_id`; both sentences of a pair share a fold |
| `outputs/predictions.csv` | Out-of-fold predictions for the baseline models (majority, BERT+LR, mpnet+LR, mpnet+RF, fine-tuned BERT) |
| `tf-idf/results/predictions.csv` | Out-of-fold predictions for Julia's TF-IDF + logistic regression |
| `experiments/feat/ft_Qwen2.5-14B-Instruct_lora_4bit_ep12_s72.csv` | Best single model (Qwen 14B, LoRA, 12 epochs). Columns `prob, fold`, rows in the same order as `merged_gender.csv` |
| `experiments/feat/*.csv` (other `ft_*`, `oof_*`) | The other models that feed the stacked ensemble |
| `experiments/ensemble.py` | Builds the stacked ensemble (0.838). **It only prints a table; it does not save the ensemble's predictions** |

Assumption to verify in step A: the rows of each `experiments/feat/*.csv` line up with the rows of
`merged_gender.csv` (same length, 1,519 lines incl. header; the `fold` column should match `folds.csv`).

## A. Combined predictions table

Goal: one file with everything needed for metrics and the audit.

Steps:
1. Load `merged_gender.csv` and `folds.csv`; join on `pair_id`.
2. Add one probability column each for: TF-IDF + LR, best mpnet model, Qwen 14B (12 ep), stacked ensemble.
3. For the stacked ensemble, re-run the same logic as `experiments/ensemble.py` (logistic regression on the
   logits of the 9 models, fitted on 4 folds and scored on the 5th) and save the result, since the existing script
   does not.
4. Add a predicted-label column (`prob > 0.5`) per model.
5. Sanity checks: 1,518 rows; accuracy of the ensemble column is about 0.838 and Qwen 14B about 0.820 (the
   README numbers). If they do not match, stop and find out why before continuing.

Planned script: `analysis/01_build_table.py` -> `outputs/metrics/predictions_all.csv` (not written yet).

## B. Metrics and charts

Models compared: majority baseline, TF-IDF + LR, best classic model (mpnet + RF), Qwen 14B, stacked ensemble.

| # | Output | What it tells us | File (planned) |
|---|---|---|---|
| B1 | Table: accuracy, macro F1, fold sd per model | Overall ranking | `outputs/metrics/summary.csv` |
| B2 | Bar chart: accuracy per model (with 0.5 chance line) | The 0.57 -> 0.84 story | `outputs/metrics/accuracy_by_model.png` |
| B3 | 2x2 confusion matrix: Qwen 14B and ensemble | Which direction the errors lean | `outputs/metrics/confusion_*.png` |
| B4 | Table: precision, recall, F1 per label | Does it favor one label? | `outputs/metrics/per_label.csv` |
| B5 | ROC curve + AUC: Qwen, ensemble, TF-IDF | Quality of the confidence scores | `outputs/metrics/roc.png` |
| B6 | Accuracy by source (CrowS / StereoSet inter / intra) | Which data is hardest | `outputs/metrics/by_source.png` |
| B7 | Accuracy per fold | How stable the result is (about 0.02 sd) | `outputs/metrics/by_fold.png` |
| B8 | Pairwise accuracy (right sentence of a pair scored higher) | Compare with the README's 0.891 | in `summary.csv` |
| B9 | 3x3 confusion matrix + per-stage errors | Where in the pipeline mistakes happen | **later, needs Chris's neutral detector** |

Plain-language reminders:
- **Precision**: of what the model called stereotypical, the share that really was.
- **Recall**: of what really was stereotypical, the share the model found.
- **F1**: one number blending the two.
- **AUC**: 0.5 is a coin flip, 1.0 is perfect; measures whether the probabilities rank truly stereotypical
  sentences above the others.

Planned script: `analysis/02_metrics.py` (uses scikit-learn and matplotlib, both already in `requirements.txt`).

### B-extra. Additional metrics beyond the TA's list

The TA's list is the minimum. These all use the saved predictions only (no GPU). Tier 1 is cheap and worth
doing; Tier 2 is optional. The TA also warned against drifting into paper-scale work, so do Tier 1 first and
only add Tier 2 if time is left.

| Tier | # | Metric / chart | Why it helps | File (planned) |
|---|---|---|---|---|
| 1 | X1 | **Bootstrap confidence intervals** on accuracy, F1 and AUC (resample whole pairs, not single sentences) | With only 1,518 sentences, a score of 0.838 could really be 0.80 to 0.87. The interval says how much to trust each number. | `ci.csv` |
| 1 | X2 | **Significance of differences between models** (paired bootstrap or McNemar's test: Qwen vs ensemble, Qwen vs TF-IDF) | The README notes some leads are about one fold sd. This tells us which differences are real. | `model_comparison.csv` |
| 1 | X3 | **Calibration**: reliability diagram, Brier score, expected calibration error | The demo shows a confidence score. Calibration checks that "90% confident" means right about 90% of the time. | `calibration.png`, `calibration.csv` |
| 1 | X4 | **Error rate by confidence** (accuracy for low, medium and high confidence bins) | Shows whether the model's mistakes are the uncertain ones, or confident and wrong (which usually means a label problem). | `accuracy_by_confidence.png` |
| 1 | X5 | **Matthews correlation coefficient (MCC)** and **Cohen's kappa** between model and true labels | Single numbers that correct for chance. Kappa is also what we use to compare human labels in the audit. | in `summary.csv` |
| 2 | X6 | **Precision-recall curve** and average precision | A second view of the confidence scores, to sit beside ROC. | `pr_curve.png` |
| 2 | X7 | **Coverage vs accuracy curve** (what if the judge abstained on its least confident sentences?) | Relevant because the judge will score LLM text where single judgements are shaky. | `coverage_accuracy.png` |
| 2 | X8 | **Accuracy vs decision threshold** (0.5 vs the tuned one in `ensemble.py`) | Shows how sensitive the result is to the 0.5 cutoff. | `threshold_sweep.png` |
| 2 | X9 | **Seed spread** where multiple seeds exist (the 7B runs) | Shows how much results move from run to run. | in `summary.csv` |

## C. Manual audit (blind labeling)

Goal: find out whether errors come from the models or from the data, by reading real sentences.

Protocol:
1. **Sample.** Draw about 40 sentences with seed 72, spread across the three sources and balanced between
   label 1 and label 0, with no two sentences from the same pair. The sample is random, not chosen from known
   errors, so it is fair.
2. **Blind sheet.** Generate a file that shows only `audit_id` and the sentence. No true label, no model output.
3. **Label it yourself** (stereotypical / anti-stereotypical / neutral / unsure) and add a confidence (1-3)
   and a one-line reason. Do not look at any predictions until the whole sheet is done.
4. **Unblind.** Join your labels with the true label and the model predictions.
5. Record agreement: you vs true label, you vs Qwen, you vs ensemble, Qwen vs true label.

Note on difficulty: the dataset's labels are relative ("more stereotypical than its partner"), but the model
sees one sentence at a time, and so do you. Expect low agreement on some sentences. That is a finding, not a
failure, so mark those as "unsure" instead of forcing a guess.

Neutral sentences: neither dataset has them. Once Chris says where the neutral examples come from, add about
10 to the sheet. Until then the audit is two-class.

Planned scripts: `analysis/03_make_audit_sheet.py` -> `outputs/audit/audit_blind.csv` (sentences only) and a
private key file `outputs/audit/audit_key.csv` (true labels, predictions). `analysis/04_score_audit.py` reads
the filled-in sheet.

## D. Error analysis

Group every disagreement into the TA's categories (add new ones if the data needs them):

| Category | Meaning |
|---|---|
| Wrongly neutral | Sentence called neutral when it is not (needs Chris's detector; later) |
| Wrong direction | Biased sentence got the opposite stereotype label |
| Quote / negation / context | Meaning depends on a quote, "not", or missing context |
| Ambiguous label | Humans (including us) disagree or are not confident |
| Not really about gender | e.g. "My grandfather is old" labeled stereotype (noted in `DATANOTES.md`) |
| Other | Anything else; describe it |

Checks beyond the audit sample, using the full 1,518 predictions:
- Where do the Qwen errors concentrate? By source, sentence length, negation (`experiments/feat/negation.csv`),
  and whether the sentence mentions an occupation.
- Which sentences does TF-IDF get wrong but Qwen gets right, and the reverse? (This was promised in the proposal.)
- Do errors cluster on the sentences with the weakest annotator agreement (`DATANOTES.md`, label noise)?

Charts: bar chart of error categories; table of about 10 representative mistakes with a note on each.

Planned script: `analysis/05_error_analysis.py` -> `outputs/audit/error_breakdown.csv`, `error_categories.png`.

### D-extra. Additional error analyses

Same tiers as above. All use saved predictions plus the data files; none needs a GPU.

| Tier | # | Analysis | What it tells us |
|---|---|---|---|
| 1 | Y1 | **Pair outcomes**: for each pair, are both sentences right, one right, or both wrong? Also how often both sentences get the *same* label | A same-label pair means the model ignored the one thing that differs between the two sentences. This is the core failure the label-is-relative problem predicts. |
| 1 | Y2 | **Hard sentences**: sentences wrong for *every* model (TF-IDF, mpnet, Qwen, ensemble) vs wrong for only one | Sentences everyone gets wrong are the best candidates for label noise, so they feed the manual review. |
| 1 | Y3 | **Error overlap** between models (how many errors are shared; how much better an "oracle" that picks the right model per sentence would be) | Shows whether the ensemble is combining different strengths or just repeating the same errors. |
| 1 | Y4 | **Errors vs sentence properties**: length bins, gendered word present (female / male / none), negation, occupation word | Finds simple patterns in where mistakes happen. |
| 1 | Y5 | **Errors vs label quality**: accuracy for sentences with unanimous annotator votes vs split votes (CrowS `annotations`, StereoSet per-annotator labels; see `DATANOTES.md`) | Tests directly whether low accuracy comes from noisy labels. |
| 2 | Y6 | **Word patterns in errors**: words that appear more in misclassified than in correctly classified sentences (reuse the log-odds code from `src/explore.py` / `src/analyze_data.py`) | Highlights vocabulary the model struggles with, without needing SHAP. |
| 2 | Y7 | **Accuracy by StereoSet target term** (the 10 gender targets, e.g. mother, schoolboy) and by CrowS writer if available | `DATANOTES.md` warns folds share targets and writers, so scores may be optimistic. This checks how uneven they are. Needs a join with `data/raw/`. |
| 2 | Y8 | **Human ceiling**: have Chris label the same blind audit sheet, then compute Cohen's kappa between the two of us and each of us vs the true labels | Shows how well humans agree on this task. If we only agree about 80% of the time, a model at 0.84 is near the ceiling. |
| 2 | Y9 | **Gender-swap sensitivity** (swap he/she in a sentence and see if the prediction flips) | Directly tests whether the model reads the gender word. Needs the model, so it is Chris's to run, not ours. |

Planned scripts: add to `analysis/05_error_analysis.py` (Y1-Y7); Y8 goes in `analysis/04_score_audit.py`.

## E. With Chris's part (later)

- Neutral detector: needs its predictions (and the data it was trained on) to build the 3x3 matrix and
  per-stage error counts.
- SHAP: after the audit, ask Chris to explain a few of the error sentences, then test the highlights by
  removing or replacing the flagged words and re-running the model (TA's suggestion). This needs the model,
  so it is Chris's to run.

## What we expect to find (hypotheses, to be confirmed or rejected)

From the README and `DATANOTES.md`; none of this has been checked against the audit yet.
- CrowS-Pairs is the hardest source (its pairs differ mostly by one gender word). *Step B update: not clearly true for Qwen or the ensemble. The gaps between sources are 1 to 3 points, which is within noise (fold sd about 0.02); StereoSet intrasentence is the easiest source for every model. It was the hardest for the mpnet models in the README (0.573). Re-check with confidence intervals (X1).*
- A noticeable share of errors will turn out to be ambiguous or noisy labels, not model failures.
- The ensemble beats the single best Qwen model by a small margin that is close to one fold sd (about 0.02),
  so we should not over-claim it.

## Caveats to state in the write-up

- Many configurations were tried, so the top scores carry some selection bias.
- Qwen may have seen CrowS-Pairs and StereoSet during pretraining.
- Using an LLM to judge LLM-generated text is partly circular.
- StereoSet's public gender data is small (about 500 pairs) and CrowS gender pairs come from 40 writers, so
  cross-validation scores may be optimistic.

## Reproduce everything (once the scripts exist)

```bash
source .venv/bin/activate            # or the environment from requirements.txt
python analysis/01_build_table.py    # A: writes outputs/metrics/predictions_all.csv
python analysis/02_metrics.py        # B: tables and charts in outputs/metrics/
python analysis/03_make_audit_sheet.py   # C: writes the blind sheet and the private key
# ... label outputs/audit/audit_blind.csv by hand, then:
python analysis/04_score_audit.py    # C: agreement between us, the models and the true labels
python analysis/05_error_analysis.py # D: error categories and breakdowns
```

Already runnable today (prints the existing ensemble table, CPU only):

```bash
python experiments/ensemble.py
```

## Progress Log

| Date | Step | What we did | Result / numbers | Files |
|---|---|---|---|---|
| 2026-10-26 | - | Plan written after TA meeting | - | this file |
| 2026-10-26 | - | Added extra metrics (B-extra, X1-X9) and extra error analyses (D-extra, Y1-Y9) beyond the TA's list | - | this file |
| 2026-10-27 | A | Built the combined table. Script checks that every prediction file lines up row by row with `merged_gender.csv` (pair_id, sentence, label, fold), rebuilds the stacked ensemble, and compares accuracies with the README | Mean-of-folds accuracy: majority 0.500, TF-IDF 0.567, mpnet+RF 0.604, Qwen 14B 0.820, ensemble 0.838. All match the README exactly | `analysis/01_build_table.py`, `outputs/metrics/predictions_all.csv` |
| 2026-10-27 | B (core) | Computed B1-B8 from `predictions_all.csv`: summary table, per-label precision/recall/F1, 2x2 confusion matrices, ROC/AUC, accuracy by source and by fold, pairwise accuracy. Checked pairwise accuracy, fold sd and accuracy by source against the README (all within 0.01) | Accuracy: TF-IDF 0.567, mpnet+RF 0.604, Qwen 14B 0.820, ensemble 0.838. AUC: 0.583 / 0.667 / 0.888 / 0.915. Pairwise: 0.564 / 0.675 / 0.880 / 0.891. Qwen errors are balanced (133 anti called stereotype, 140 stereotype called anti); ensemble 119 / 127. Per-label precision and recall for Qwen all within 0.82 +/- 0.01. By source (ensemble): CrowS 0.840, inter 0.831, intra 0.843; (Qwen): 0.809 / 0.814 / 0.837. Ensemble beats Qwen in 4 of 5 folds (not fold 0: 0.805 vs 0.818) | `analysis/02_metrics.py`, `outputs/metrics/{summary,per_label,by_source,by_fold}.csv`, `accuracy_by_model.png`, `confusion_matrices.png`, `roc.png`, `by_source.png`, `by_fold.png` |
| | | | | |

## Open questions

- [ ] Where does Chris's neutral detector get its training data, and what does its prediction file look like?
- [x] Do the `experiments/feat/*.csv` rows really line up with `merged_gender.csv`? Yes: fold column matches row by row (checked in step A).
- [ ] This repo has no `.venv` of its own. Step A was run with the Python from `~/cosc72-stereotype-judge/.venv` (pandas 2.3.3, scikit-learn 1.6.1, matplotlib 3.9.4); `requirements.txt` pins scikit-learn 1.9.1. Results matched, but note it in the write-up or make a proper venv.
- [ ] Which single model counts as "the Qwen classifier" in the final write-up: the 14B 12-epoch run or the ensemble?
- [ ] Should the audit sample size be 30 or 50? (default 40)
