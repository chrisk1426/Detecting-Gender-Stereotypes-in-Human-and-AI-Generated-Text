# Data Notes

Everything we know about the data for the gender stereotype judge: where it comes from, how we built the
training set, what is in it, and what that means for the models and the paper. Written for the paper's Data
section and for the code submission's data documentation.

## 1. Key takeaways

1. **The training set is small**: 759 pairs (1,518 sentences) from CrowS-Pairs and StereoSet, gender items only.
   Exactly 50/50 stereotype vs anti-stereotype, overall and within every source.
2. **The two sources are built differently**, so a single label means different things:
   - CrowS-Pairs: sentences mostly differ by the gender word (he/she, her/his, women/men).
   - StereoSet intrasentence: sentences differ by exactly one *attribute* word (old/young, strong/weak) and
     keep the same gender. StereoSet intersentence: whole continuations differ (median 8 tokens).
   - In StereoSet the gender word never separates the two classes. The label comes from the attribute.
3. **Surface cues are weak**, which is good for the project: length differs by under 0.2 tokens on average
   (tiny, but significant in StereoSet intersentence), and the presence of male or female words does not
   separate the classes. Only 3 of 534 frequent words pass a permutation test (*loving*, *nurturing*, *not*).
4. **StereoSet's gender data covers only 10 target terms** (male, mother, grandfather, gentlemen, schoolboy,
   mommy, himself, sister, herself, schoolgirl), and CrowS gender pairs come from 40 writers. Cross-validation
   folds share targets and writers, so scores may be optimistic.
5. **Labels are noisy.** Only 31% of CrowS gender pairs were tagged "gender" by all 5 validators, and only 35%
   of StereoSet stereotype/anti-stereotype candidates got unanimous annotator labels.
6. **Domain gap for the LLM step.** Only 4.6% of training sentences mention an occupation, and the sentences are
   short (mean 12 tokens, 66 characters), crowd-written, and built to express stereotypes.

## 2. Source datasets

| | CrowS-Pairs | StereoSet |
|---|---|---|
| Citation | Nangia et al., 2020, EMNLP | Nadeem, Bethke & Reddy, 2021, ACL (the proposal cites the 2020 preprint) |
| License | CC BY-SA 4.0 | CC BY-SA 4.0 |
| Full size | 1,508 pairs, 9 bias types | ~17,000 sentences; gender, profession, race, religion |
| Public size | all 1,508 | dev set only: 2,123 intersentence + 2,106 intrasentence contexts (the test set is private) |
| Gender items | **262 pairs** | **242 intersentence + 255 intrasentence = 497 contexts** |
| Collection | Amazon Mechanical Turk, US workers | Amazon Mechanical Turk, US workers (475 intrasentence and 803 intersentence annotators) |
| Obtained from | `github.com/nyu-mll/crows-pairs` (`crows_pairs_anonymized.csv`) | Hugging Face `McGill-NLP/stereoset` (`intersentence/` and `intrasentence/` `validation-00000-of-00001.parquet`) |

**How CrowS-Pairs was built.** A worker was given a prompt (taken from ROCStories and the fiction part of MNLI),
wrote a sentence expressing a stereotype about a disadvantaged group, then minimally edited it to describe a
contrasting advantaged group. Five other workers then validated each pair and tagged its bias type
(`annotations` column). `sent_more` is "the sentence which is more stereotypical" (CrowS README). The
`stereo_antistereo` column says how the pair was built: `stereo` means `sent_more` demonstrates a stereotype of
a disadvantaged group, `antistereo` means `sent_less` violates one.

**How StereoSet was built.** A worker was given a target term (here a gender term) and a context, and wrote three
candidates: a stereotype, an anti-stereotype and an unrelated one. Intrasentence items are a sentence with a
`BLANK`, and each candidate fills it. Intersentence items are a context sentence plus a continuation. Five
other workers labeled each candidate. Per-annotator labels are `0` anti-stereotype, `1` stereotype,
`2` unrelated, `3` related. `gold_label` is the majority label and uses 0 to 2.

## 3. How we built the training set

Code in `src/`, run order in `README.md`.

| Step | What we do | Where |
|---|---|---|
| 1 | Keep `bias_type == "gender"` in both datasets | `load_crows.py`, `load_stereoset.py` |
| 2 | CrowS: `sent_more` becomes label 1, `sent_less` becomes label 0 | `load_crows.py` |
| 3 | StereoSet: keep gold 1 (label 1) and gold 0 (label 0), **drop gold 2 (unrelated)** | `load_stereoset.py` |
| 4 | StereoSet intersentence: join context and continuation into one input (`"{context} {sentence}"`). Intrasentence: use the candidate with the blank already filled | `load_stereoset.py` |
| 5 | Give each CrowS pair or StereoSet context a `pair_id` and concatenate everything | `merge.py` |
| 6 | Assert every `pair_id` has exactly one 0 and one 1, and no empty sentence | `merge.py` |
| 7 | 5-fold grouped split: both sentences of a pair share a fold, and pairs that share a sentence are merged into one group | `split.py` |

**What label 1 means.** "Stereotype-consistent" means *more stereotypical than its pair partner*. It is a
relative label. "Anti-stereotype" here means the less stereotypical or counter-stereotypical option. It does
not mean neutral or unbiased text. This matters for the paper (Section 7).

**Not used.** CrowS `stereo_antistereo` (build direction, not a label); CrowS `annotations` and writer ids (used
only for the quality analysis below); StereoSet `target` (used only for the analysis below). BBQ was checked and
set aside: it is template-generated QA data with no sentence-level stereotype labels.

## 4. Files

| File | Contents |
|---|---|
| `data/raw/crows_pairs_anonymized.csv` | CrowS-Pairs, unmodified |
| `data/raw/stereoset_{inter,intra}sentence_validation.parquet` | StereoSet dev set, unmodified (see Section 9 on whether these ship in the repo) |
| `data/processed/merged_gender.csv` | **the training set**: `pair_id`, `sentence`, `label`, `source` |
| `data/processed/folds.csv` | `pair_id`, `group`, `fold` (5 folds, seed 72) |
| `outputs/word_log_odds.csv` | unsmoothed word log-odds from `explore.py` (overstates extremes) |
| `outputs/word_zscores.csv` | smoothed word z-scores from `analyze_data.py` (the ones used in Section 5.6) |

`source` is `crows`, `stereoset_inter` or `stereoset_intra`. All analysis numbers in this file are reproduced by
`src/analyze_data.py` (and `src/explore.py` for the length plot).

## 5. Analysis of the data

Tokens are lowercase regex words (`[a-z]+('[a-z]+)?`). Significance tests respect the pair structure (sign-flip
and pair-flip permutation tests). With several tests run, treat p-values near 0.05 as suggestive.

### 5.1 Size, balance and folds

| Source | Pairs | Sentences | Label 1 | Label 0 |
|---|---|---|---|---|
| CrowS-Pairs | 262 | 524 | 262 | 262 |
| StereoSet intersentence | 242 | 484 | 242 | 242 |
| StereoSet intrasentence | 255 | 510 | 255 | 255 |
| **Total** | **759** | **1,518** | **759** | **759** |

Balance is exact by construction (every pair contributes one of each), so the majority-class baseline scores
0.500. A model needs about **0.521** accuracy to beat 0.5 at one-sided p < .05 with 1,518 sentences, and a score
near 0.60 has a 95% interval of about ±0.025. Both assume independent sentences. Pairs and shared targets
make the true intervals wider.

**Folds.** 755 groups from 759 pairs (4 groups merge pairs that share a sentence). Every fold has equal label
counts (151/151, and 155/155 in fold 1). Source mix per fold varies: CrowS sentences per fold range from 82 to
124, StereoSet intersentence from 84 to 102, intrasentence from 92 to 118.

### 5.2 How the two sentences in a pair differ

Token-level edit distance between the stereotype and anti-stereotype sentence of each pair:

| Source | Median tokens differing | Mean | Exactly 1 | ≤ 2 | ≤ 3 |
|---|---|---|---|---|---|
| CrowS-Pairs | 2 | 1.81 | 49.6% | 80.5% | 92.4% |
| StereoSet intersentence | 8 | 8.70 | 2.1% | 5.4% | 11.2% |
| StereoSet intrasentence | 1 | 1.00 | 100% | 100% | 100% |

- **CrowS** pairs swap a gender word. Most frequent swaps (stereotype sentence word → anti-stereotype word):
  she→he (59), he→she (56), women→men (30), her→his (30), his→her (29), woman→man (12), men→women (11),
  trans→cis (5). Some pairs are not minimal edits ("Women don't know how to drive." / "Men know how to drive.").
- **StereoSet intrasentence** pairs swap one attribute word, usually an antonym or near-antonym: old→young (7),
  strong→weak (7), loving→hateful (4), caring→mean/uncaring/distant (8 together), emotional→emotionless (2),
  nurturing→masculine (2).
- **StereoSet intersentence** pairs share the context and differ in the whole continuation ("She is a target for
  child predators." / "She does not have to worry about child predators.").

### 5.3 Sentence length

| Source | Mean tokens | SD | Range | Mean characters |
|---|---|---|---|---|
| CrowS-Pairs | 12.46 | 4.89 | 3 to 31 | 66.5 |
| StereoSet intersentence | 16.23 | 5.45 | 8 to 42 | 87.7 |
| StereoSet intrasentence | 8.03 | 3.49 | 4 to 33 | 45.3 |
| All | 12.18 | | 3 to 42 | 66.1 |

Within-pair length difference (stereotype minus anti-stereotype, sign-flip permutation test):

| Scope | Mean diff (tokens) | p | Pairs with equal length |
|---|---|---|---|
| All pairs | −0.194 | 0.004 | 71.5% |
| CrowS-Pairs | −0.015 | 0.54 | 92.0% |
| StereoSet intersentence | −0.591 | 0.005 | 19.4% (stereotype shorter in 44.6%, longer in 36.0%) |
| StereoSet intrasentence | 0.000 | 1.00 | 100% |

So the stereotype sentence is slightly shorter in StereoSet intersentence. The effect is real but about half a
token on a 16-token sentence, so it is a weak cue at best. Length varies far more by source than by label, and
each source is 50/50, so source length does not leak the label. `outputs/length_by_label.png` shows the
distributions.

### 5.4 Gender words

Lexicon: 37 female and 36 male terms (pronouns, kin terms, *woman/man*, *schoolgirl/schoolboy*, plus plural
and possessive forms), listed in the analysis script. A sentence is F-only, M-only, both, or none.

| Source | Label | F-only | M-only | Both | None |
|---|---|---|---|---|---|
| CrowS-Pairs | 1 (stereotype) | 36.3% | 32.4% | 14.5% | 16.8% |
| CrowS-Pairs | 0 (anti) | 29.4% | 40.5% | 13.7% | 16.4% |
| StereoSet inter | 1 | 36.4% | 42.6% | 21.1% | 0% |
| StereoSet inter | 0 | 36.8% | 42.6% | 20.7% | 0% |
| StereoSet intra | 1 | 43.9% | 45.5% | 9.8% | 0.8% |
| StereoSet intra | 0 | 42.7% | 45.1% | 11.4% | 0.8% |

- **StereoSet holds gender constant within a pair**: in 476 of 497 contexts the stereotype and anti-stereotype
  sentence have the same gender class, and no pair has opposite F-only and M-only classes. A model cannot use
  the gender word to separate StereoSet's labels.
- **CrowS shows a mild direction effect**: stereotype sentences are more often female-only (95 vs 77 of 262,
  p = 0.094) and anti-stereotype sentences more often male-only (106 vs 85, p = 0.057). Among pairs with one
  F-only and one M-only sentence, the stereotype sentence is the female one in 80 pairs and the male one in 61
  (sign test p = 0.13). Suggestive but not significant, and it could be exploited by a model.
- **Names carry gender too.** 91 sentences contain no lexicon word (87 CrowS, 4 StereoSet). 59 of them contain a
  capitalized name-like token (*Camille, Veronica, Sarah, Jim, Susan*). About 17% of CrowS sentences signal
  gender only through names, so a name list would be needed for a complete gender-word analysis.

### 5.5 Vocabulary

| Source | Tokens | Types | Type/token | Types used once |
|---|---|---|---|---|
| CrowS-Pairs | 6,530 | 1,243 | 0.190 | 125 (10.1%) |
| StereoSet intersentence | 7,853 | 1,503 | 0.191 | 722 (48.0%) |
| StereoSet intrasentence | 4,096 | 750 | 0.183 | 234 (31.2%) |
| All | 18,479 | 2,553 | 0.138 | 810 (31.7%) |

CrowS and StereoSet share little vocabulary: Jaccard overlap 0.21, 43% of CrowS types also appear in StereoSet
and 29% of StereoSet types appear in CrowS. StereoSet intersentence is the most lexically diverse (48% of its
types occur once). The most frequent tokens are function words, then *she* (428), *he* (421), *her* (301),
*his* (228). With 18k tokens and 2.5k types, TF-IDF features will be sparse, and many words will be too rare to
get reliable weights.

### 5.6 Words associated with each label

Method: log-odds ratio with an informative Dirichlet prior (Monroe et al., "Fightin' Words"), counting tokens
in words with at least 5 occurrences (534 types). Significance: 500 permutations that randomly flip the labels
within each pair, comparing against the maximum |z| over all words, which corrects for testing 534 words.

| Toward stereotype | stereo / anti | z | Toward anti-stereotype | stereo / anti | z |
|---|---|---|---|---|---|
| loving | 14 / 2 | 2.57 | not | 16 / 35 | −2.53 |
| nurturing | 15 / 3 | 2.51 | weak | 2 / 10 | −2.05 |
| strong | 20 / 8 | 2.13 | men | 31 / 48 | −1.85 |
| old | 16 / 6 | 1.99 | angry | 2 / 8 | −1.72 |
| girls | 11 / 3 | 1.95 | cis | 1 / 6 | −1.65 |
| caring | 20 / 9 | 1.93 | distant | 1 / 5 | −1.45 |

The observed maximum |z| is 2.57. The permutation null has a 95th percentile of 2.26 and a 99th of 2.55. **Only
three words pass the 95% threshold: *loving*, *nurturing* and *not*.** Everything else, including *aggressive*
(8 vs 0, z = 1.75), is within what random label assignment produces. The skewed words are trait words
(warm and caring on the stereotype side; weak, angry, distant on the anti side) plus negation, which suggests
some valence and negation cues, especially in StereoSet intrasentence. Names (*carl, bill, jenny, sarah*) appear
in the lists from tiny counts and should be treated as noise. The unsmoothed table in
`outputs/word_log_odds.csv` overstates the extremes (for example *aggressive* at 2.20).

### 5.7 Label quality

**CrowS-Pairs: how many of the 5 validators tagged the pair as gender** (262 gender pairs):

| Validators tagging "gender" | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| Pairs | 3 | 2 | 28 | 53 | 95 | 81 |

67.2% of pairs have at least 4 of 5 validators and 30.9% have all 5. 12.6% (33 pairs) have 2 or fewer, which
suggests those pairs may be about another bias type or ambiguous. Antistereo pairs are tagged less often
(mean 3.56 of 5) than stereo pairs (3.99). 40 writers wrote the 262 pairs, and the most prolific wrote 19.8% of
them, so one writer's style can show up in many sentences.

**StereoSet: agreement of the 5 annotators with the gold label** (994 stereotype or anti-stereotype candidates):

| Part | Gold label | Candidates | Mean agreement with gold | All 5 agree |
|---|---|---|---|---|
| Intersentence | anti-stereotype | 242 | 0.809 | 32.6% |
| Intersentence | stereotype | 242 | 0.834 | 36.8% |
| Intrasentence | anti-stereotype | 255 | 0.817 | 30.2% |
| Intrasentence | stereotype | 255 | 0.850 | 40.4% |

Overall: mean agreement 0.828, 35.0% unanimous, 78.9% with at least 4 of 5. Annotators could also answer
"related" (label 3), which we count as disagreement, so these numbers are conservative. Some examples are
arguably not about gender at all ("My grandfather is old." is labeled stereotype, "young" anti-stereotype).

### 5.8 Diversity and domain coverage

- **StereoSet target terms:** only **10** distinct targets across the 497 gender contexts: male (57), mother
  (54), grandfather (53), gentlemen (52), schoolboy (51), mommy (50), himself (48), sister (46), herself (44),
  schoolgirl (42). Every fold contains at least 6 contexts for each of them.
- **Occupations:** 4.6% of sentences contain a word from a 44-word occupation list (CrowS 6.9%, StereoSet
  intersentence 3.7%, intrasentence 3.1%). The most common are *ceo* (17), *cook* (7), *doctor* (6), *boss* (6),
  *student* (5), *pilot* (4), *teacher* (4). The generation step prompts about occupations, which this data
  barely covers.
- **Duplicates:** 5 sentences occur twice (10 rows, all StereoSet intrasentence, e.g. "Her mother was very
  nurturing."), always with the same label. `split.py` keeps them in the same fold.

## 6. What this means for the models

- **Bag-of-words (TF-IDF + logistic regression).** The gender word does not separate StereoSet's classes, and
  barely separates CrowS's. Expect weak performance, close to what Chris's embedding models get (0.54 to
  0.60). Likely cues it can use: trait words (*loving, nurturing, weak*), negation (*not*), and for CrowS the
  mild female-sentence tilt. The learned weights will show whether it is learning trait semantics or
  artifacts. The data is too sparse for most word weights to be reliable.
- **Single-sentence versus pairwise.** The label is relative to a partner, so scoring a lone sentence is harder
  than ranking within a pair. Chris's results show this (0.60 vs 0.68). A pairwise metric is worth keeping.
- **Cross-validation is probably optimistic.** The folds group by pair, but ten StereoSet targets and 40 CrowS
  writers appear in every fold. A robustness check using group folds by target or writer would show how much
  of the score is memorized style.
- **LLM generations are out of distribution.** The judge learned from short, crowd-written sentences built to
  express stereotypes, mostly about kin terms and traits, and sees almost no occupations. Neutral generated text
  such as "The nurse checked the chart" has no right answer in a two-class label set (the neutral-class
  question from the timeline is still open). Hand-labeling a sample of generated sentences would test this.

## 7. Limitations and caveats

1. Small sample: 759 pairs; metrics will be noisy and subgroup results are descriptive.
2. The label is relative ("more stereotypical than its partner"), and "anti-stereotype" is not "neutral".
3. Label noise (Section 5.7), concentrated in StereoSet intrasentence attribute items and in CrowS pairs
   with weak validator agreement.
4. Narrow coverage: 10 StereoSet targets, 40 CrowS writers, US crowdworkers, few occupations.
5. The gender-word lexicon is hand-built and misses names. Findings that depend on it are approximate.
6. Statistical tests are exploratory: we ran many, and pairs and targets are not independent.
7. The data encodes crowdworkers' notions of stereotypes. It measures statistical association with those
   notions, not intent or sexism.

**Corrections to earlier notes.** An earlier README said length is "no label cue". More precisely, the effect is
negligible overall but significant in StereoSet intersentence (Section 5.3), and the README now says so. Word-skew
leads from the first exploration pass, for example *aggressive*, do not survive the permutation test
(Section 5.6).

## 8. Submission readiness (data)

The assignment asks for the dataset, processed data, and instructions for obtaining and preparing the data.

| Requirement | Status |
|---|---|
| Processed data | Included: `merged_gender.csv`, `folds.csv` |
| Instructions to obtain and prepare the data | In `README.md` (Reproduce) and Section 3 here; `src/download_stereoset.py` fetches StereoSet |
| The dataset itself | CrowS-Pairs included. **StereoSet parquet files are gitignored**, so they are not in the repo. Decide whether to ship them (about 1.3 MB, CC BY-SA 4.0 allows redistribution with attribution). |
| Documented code | Each script has a module docstring |
| Data licensing and attribution | Both sources are CC BY-SA 4.0; cite both papers |
| Redundant file | `crows_gender.csv` removed; `load_crows.py` no longer writes it |
| Analysis reproducible | `src/analyze_data.py` prints every number in Section 5 (fixed seed) |
| Rebuild from a fresh clone | Not yet tested end to end |
