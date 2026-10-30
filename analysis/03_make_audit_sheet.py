"""Step C of docs/MetricsAnalysisPlan.md: build the blind audit sheets and the private answer key.

Parts (seed 72, every part uses different pairs, so nobody sees a partner of an already-labeled sentence):
  C1  30 single sentences: 10 per source, balanced label 1 / 0
  C2  10 pairs for "which is more stereotypical?" (3 CrowS, 3 StereoSet inter, 4 StereoSet intra)
  C3  10 targeted single sentences from known weak spots: 4 CrowS pairs where <=2 of 5 validators said "gender",
      3 StereoSet pairs with weak annotator agreement, 3 sentences that all four models get wrong
C1 and C3 are mixed and shuffled into ONE sheet, so the labeler cannot tell which rows are the targeted ones.

Writes (nothing is written with --dry-run, which prints counts only):
  outputs/audit/audit_single.csv, audit_pairs.csv, RUBRIC.md       blind: sentences only
  outputs/audit/private/audit_key_single.csv, audit_key_pairs.csv  answers + model predictions: do NOT open these
                                                                   until every row of both sheets is labeled
"""
from pathlib import Path
import ast, sys
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs" / "audit"
DRY = "--dry-run" in sys.argv
rng = np.random.default_rng(72)

df = pd.read_csv(ROOT / "outputs" / "metrics" / "predictions_all.csv")
MODELS = ["tfidf_lr", "mpnet_rf", "qwen14b", "ensemble"]
SOURCES = ["crows", "stereoset_inter", "stereoset_intra"]
pairs = {pid: g for pid, g in df.groupby("pair_id")}
assert all(len(g) == 2 for g in pairs.values())

# ---- Weak-label scores per pair (same definitions as src/analyze_data.py) -----------------------------------
crows = pd.read_csv(RAW / "crows_pairs_anonymized.csv").rename(columns={"Unnamed: 0": "orig_idx"})
crows = crows[crows.bias_type == "gender"]
crows_validators = {f"crows_{r.orig_idx}": sum("gender" in a for a in ast.literal_eval(r.annotations))
                    for r in crows.itertuples()}  # how many of 5 validators tagged the pair "gender"
ss_agreement = {}  # weakest annotator agreement with the gold label among a pair's two kept candidates
for part in ("intersentence", "intrasentence"):
    ss = pd.read_parquet(RAW / f"stereoset_{part}_validation.parquet")
    for _, r in ss[ss.bias_type == "gender"].iterrows():
        ss_agreement[f"ss_{part[:5]}_{r['id']}"] = min(
            np.mean(np.array(lab["label"]) == gold) for gold, lab in zip(r.sentences["gold_label"], r.sentences["labels"])
            if gold != 2)

used_pairs, used_text = set(), set()


def pick(candidates, n, label=None):
    """Draw n pairs from candidates (a list of pair_ids) that are not used yet, with unseen sentence text.
    Returns [(pair_id, sentence_row)], choosing the sentence with `label` if given, else at random."""
    out = []
    for pid in rng.permutation(candidates):
        if len(out) == n:
            break
        if pid in used_pairs:
            continue
        g = pairs[pid]
        row = g[g.label == label].iloc[0] if label is not None else g.iloc[rng.integers(2)]
        if row.sentence in used_text or any(s in used_text for s in g.sentence):
            continue
        used_pairs.add(pid)
        used_text.update(g.sentence)
        out.append((pid, row))
    assert len(out) == n, f"only found {len(out)} of {n} pairs"
    return out


# ---- C1: 30 random single sentences, 10 per source, 5 with label 1 and 5 with label 0 ------------------------
c1 = []
for s in SOURCES:
    pool = [p for p, g in pairs.items() if g.source.iloc[0] == s]
    for lab, n in ((1, 5), (0, 5)):
        c1 += [("C1", "random", pid, row) for pid, row in pick(pool, n, label=lab)]

# ---- C2: 10 random pairs ---------------------------------------------------------------------------------------
c2 = []
for s, n in zip(SOURCES, (3, 3, 4)):
    pool = [p for p, g in pairs.items() if g.source.iloc[0] == s]
    for pid, _ in pick(pool, n):
        c2.append((pid, pairs[pid]))

# ---- C3: targeted weak-label sentences --------------------------------------------------------------------------
weak_crows = [p for p, v in crows_validators.items() if v <= 2 and p in pairs]
weak_ss = [p for p, v in ss_agreement.items() if v <= 0.6 and p in pairs]
hard = df[(df[[f"{m}_correct" for m in MODELS]] == 0).all(axis=1)]
hard_pairs = list(hard.pair_id.unique())
c3 = ([("C3", "crows, <=2 of 5 validators said gender", pid, row) for pid, row in pick(weak_crows, 4)]
      + [("C3", "stereoset, weak annotator agreement", pid, row) for pid, row in pick(weak_ss, 3)]
      + [("C3", "wrong for every model", pid, hard[hard.pair_id == pid].iloc[rng.integers((hard.pair_id == pid).sum())])
         for pid in [p for p, _ in pick(hard_pairs, 3)]])

# ---- Build sheets -----------------------------------------------------------------------------------------------
single = c1 + c3
order = rng.permutation(len(single))
single = [single[i] for i in order]
single_ids = [f"S{i + 1:02d}" for i in range(len(single))]
blind_single = pd.DataFrame({"audit_id": single_ids, "sentence": [r.sentence for _, _, _, r in single],
                             "your_label": "", "confidence_1to3": "", "reason": ""})
key_single = pd.DataFrame([{"audit_id": i, "part": part, "selection_reason": why, "pair_id": pid, "source": r.source,
                            "true_label": r.label, "sentence": r.sentence,
                            **{f"{m}_prob": r[f"{m}_prob"] for m in MODELS},
                            **{f"{m}_pred": r[f"{m}_pred"] for m in MODELS}}
                           for i, (part, why, pid, r) in zip(single_ids, single)])

order = rng.permutation(len(c2))
pair_rows, key_rows = [], []
for n, i in enumerate(order, 1):
    pid, g = c2[i]
    stereo, anti = g[g.label == 1].iloc[0], g[g.label == 0].iloc[0]
    first_is_stereo = bool(rng.integers(2))
    a, b = (stereo, anti) if first_is_stereo else (anti, stereo)
    aid = f"P{n:02d}"
    pair_rows.append({"audit_id": aid, "sentence_1": a.sentence, "sentence_2": b.sentence,
                      "your_pick_first_second_equal_unsure": "", "confidence_1to3": "", "reason": ""})
    key_rows.append({"audit_id": aid, "pair_id": pid, "source": stereo.source,
                     "true_more_stereotypical": 1 if first_is_stereo else 2,
                     **{f"{m}_prob_1": a[f"{m}_prob"] for m in MODELS}, **{f"{m}_prob_2": b[f"{m}_prob"] for m in MODELS}})
blind_pairs, key_pairs = pd.DataFrame(pair_rows), pd.DataFrame(key_rows)

RUBRIC = """# Audit rubric (frozen before labeling: do not edit once you start)

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
"""

# ---- Report (counts only: no sentences, no labels, no predictions) --------------------------------------------------
all_pairs = [pid for _, _, pid, _ in single] + [pid for pid, _ in c2]
assert len(all_pairs) == len(set(all_pairs)), "a pair appears in more than one part"
print(f"single sheet: {len(blind_single)} rows (C1 {len(c1)}, C3 {len(c3)}); pair sheet: {len(blind_pairs)} rows")
print("C1 by source and label:", pd.DataFrame([(r.source, r.label) for _, _, _, r in c1]).value_counts().sort_index().to_dict())
print("C2 by source:", pd.Series([g.source.iloc[0] for _, g in c2]).value_counts().to_dict())
print("C3 by reason:", pd.Series([why for _, why, _, _ in c3]).value_counts().to_dict())
print(f"weak-label pools: crows <=2 validators {len(weak_crows)}, stereoset agreement <=0.6 {len(weak_ss)}, wrong for every model {len(hard_pairs)} pairs")
print(f"distinct pairs across all parts: {len(set(all_pairs))} (no overlap)")

if DRY:
    print("\n--dry-run: nothing written.")
else:
    (OUT / "private").mkdir(parents=True, exist_ok=True)
    blind_single.to_csv(OUT / "audit_single.csv", index=False)
    blind_pairs.to_csv(OUT / "audit_pairs.csv", index=False)
    (OUT / "RUBRIC.md").write_text(RUBRIC)
    key_single.to_csv(OUT / "private" / "audit_key_single.csv", index=False)
    key_pairs.to_csv(OUT / "private" / "audit_key_pairs.csv", index=False)
    print(f"\nwrote blind sheets and RUBRIC.md to {OUT}, answer keys to {OUT / 'private'}")
