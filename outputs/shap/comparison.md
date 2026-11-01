# SHAP comparison across runs

From each run's `summary.txt` (`experiments/shap_analyze.py`). Qwen: f = logit(stereotype) − logit(anti-stereotype).
Neutral detector: f = logit of "stereotype-related" vs neutral. Max 800 SHAP evaluations per sentence.

| | Qwen `qwen14b_lora` | Qwen `qwen14b_lora_gest` | Qwen `qwen14b_lora`, held-out GEST | Neutral detector |
|---|---|---|---|---|
| Sentences | 1,518 (training data) | 1,518 (training data) | 1,500 (750 pairs, unseen) | 2,809 (training + held-out neutral) |
| Share of \|SHAP\|: gender / names / attribute / function / punct. | 24 / 4 / 43 / 23 / 6 % | 24 / 4 / 41 / 23 / 8 % | 17 / 0 / 36 / 30 / 17 % | 19 / 2 / 42 / 26 / 11 % |
| corr(attribute SHAP in male vs female sentences) | −0.26 | −0.30 | −0.11 | 0.95 |
| Gender swaps: corr(shared-word SHAP, stereo vs anti version) | −0.04 (CrowS) | −0.11 (CrowS) | 0.16 (GEST) | 0.99 (CrowS) |
| Gender swaps: shared words whose SHAP flips sign | 49% | 50% | 40% | 4% |
| Interaction, stereotype sentences: mean (share > 0) | +4.45 (66%) | +2.16 (66%) | +2.11 (61%) | −0.28 (31%) |
| Interaction, anti-stereotype sentences: mean (share > 0) | −10.70 (17%) | −4.05 (19%) | −5.67 (32%) | −0.22 (33%) |

Reading it:
- The Qwen judges are gender-dependent: the same attribute word pushes in opposite directions depending on the
  gender it appears with (correlations ≤ 0, ~half of shared words flip sign), and the gender × attribute
  interaction is positive for most stereotypes and negative for most anti-stereotypes. This holds on unseen GEST
  text, at smaller magnitude.
- The neutral detector is gender-blind (correlations 0.95–0.99, 4% flips, interaction ≈ 0): it detects
  stereotype-related content and leaves the direction to the judge, as designed.

Word types: first names (classified by hand in `experiments/shap_utils.py`) count as a gender cue, separate from
gender words. The masking interaction was computed before names were added, so a name can appear there as the
"attribute". Interactive version: `docs/interpretability.html`.
