# Audit rubric (frozen before labeling: do not edit once you start)

Label each sentence on its own, as a reader would. Do not look at any model output or any answer key.

- **stereotypical**: relies on or expresses a common belief about what men or women are like or should do.
- **anti-stereotypical**: goes against such a belief.
- **neutral**: no gender stereotype either way.
- **unsure**: you cannot decide. This is allowed, and it is a result too.

`confidence_1to3`: 1 = guessing, 2 = fairly sure, 3 = certain. `reason`: one short line.

Sheets
- `audit_single.csv`: 40 single sentences. Finish this one completely first.
- `audit_pairs.csv`: 10 pairs. Pick which sentence is more stereotypical: `first`, `second`, `equal` or `unsure`.

Remember the dataset's labels are relative ("more stereotypical than the other sentence in the pair"). On a single
sentence, "anti-stereotypical" means it goes against a stereotype, not that it is neutral.
