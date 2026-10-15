# Detecting Gender Stereotypes in Human- and AI-Generated Text

COSC 72 course project (Julia Allos, Chris Kang). We build a "stereotype judge": a classifier trained on
human-labeled gender-stereotype sentences, then applied to language-model-generated text to measure
occupation-linked gender bias. See the project proposal for the full plan.

## Status

Done so far: dataset merge and exploration. 
Not yet started: classifiers (majority baseline, TF-IDF + logistic
regression, BERT embeddings + logistic regression), train/test folds, generation pipeline, demo.

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
## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/download_stereoset.py   # fetches StereoSet into data/raw/
python src/merge.py                # writes data/processed/merged_gender.csv
python src/explore.py              # prints summaries, writes outputs/
```
Scripts locate files relative to the repo root, so run them from anywhere inside it.
