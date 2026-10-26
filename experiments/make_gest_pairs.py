"""Turn GEST (kinit/gest on Hugging Face, v1.1, Apache-2.0; Pikuliak et al. 2023) into gender-swapped pairs.

GEST sentences are first person and gender-neutral, each tagged with a stereotype id: 1-7 about women,
8-16 about men. Each becomes a pair with a gendered speaker, e.g.
  She said: "I am emotional."  -> 1 (stereotype 1 is about women)
  He said: "I am emotional."   -> 0
Speaker phrasings vary so the model can't key on one template. Used as extra TRAINING data only.
"""
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TEMPLATES = [("She said", "He said"), ("My sister told me", "My brother told me"),
             ("The woman explained", "The man explained"), ("My mother once said", "My father once said"),
             ("The girl admitted", "The boy admitted"), ("My wife told me", "My husband told me")]

g = pd.read_csv(ROOT / "data" / "external" / "gest_1.1.csv")
rng = np.random.default_rng(72)
rows = []
for i, r in g.iterrows():
    fem, mal = TEMPLATES[rng.integers(len(TEMPLATES))]
    s = r["sentence"].strip()
    female_stereo = r["stereotype"] <= 7
    rows.append({"pair_id": f"gest_{i}", "sentence": f'{fem}: "{s}"', "label": int(female_stereo)})
    rows.append({"pair_id": f"gest_{i}", "sentence": f'{mal}: "{s}"', "label": int(not female_stereo)})
out = pd.DataFrame(rows)
out.to_csv(ROOT / "data" / "external" / "gest_pairs.csv", index=False)
print(len(out), "rows,", out.pair_id.nunique(), "pairs; label balance", out.label.mean())
print(out.head(4).to_string(index=False, max_colwidth=90))
