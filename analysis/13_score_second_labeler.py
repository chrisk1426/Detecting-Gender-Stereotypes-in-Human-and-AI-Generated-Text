"""Score the second labeler's blind audit sheets against the dataset, the first labeler and Qwen (Y8 of docs/MetricsAnalysisPlan.md).

Reads outputs/audit/chris_labels_{single,pairs}.tsv (copied from the labeled sheets) and the git-ignored answer key
outputs/audit/private/. Prints the numbers used in the Blind audit part of the report; see outputs/audit/second_labeler_scoring.txt.
"""
from pathlib import Path
import pandas as pd, numpy as np
from sklearn.metrics import cohen_kappa_score
ROOT = Path(__file__).resolve().parent.parent
A = ROOT / "outputs" / "audit"
S = A
def norm(x):
    x = str(x).strip().lower()
    return {"stereotypical":"stereo","anti-stereotypical":"anti","neutral":"neutral","unsure":"unsure"}.get(x, x)
cs = pd.read_csv(f"{S}/chris_labels_single.tsv", sep="\t").iloc[:, :5]
cs.columns = ["audit_id","sentence","chris","chris_conf","chris_reason"]
cs["chris"] = cs.chris.map(norm)
key = pd.read_csv(f"{A}/private/audit_key_single.csv")
jul = pd.read_csv(f"{A}/audit_results_single.csv")[["audit_id","your_label"]].rename(columns={"your_label":"julia"})
jul["julia"] = jul.julia.map(norm)
d = cs.merge(key[["audit_id","part","source","true_label","qwen14b_pred","ensemble_pred"]], on="audit_id").merge(jul, on="audit_id")
assert len(d) == 40
print("Chris single-sentence labels:", d.chris.value_counts().to_dict())
print("Julia single-sentence labels:", d.julia.value_counts().to_dict())
tr = d.true_label.map({1:"stereo",0:"anti"})
def vs_dataset(col, name):
    b = d[d[col].isin(["stereo","anti"])]
    acc = (b[col]==b.true_label.map({1:"stereo",0:"anti"})).mean()
    print(f"{name}: binary on {len(b)}/40 rows; agrees with dataset {int((b[col]==b.true_label.map({1:'stereo',0:'anti'})).sum())}/{len(b)} = {acc:.3f}")
    for p in ["C1","C3"]:
        bb = b[b.part==p]; print(f"   part {p}: {int((bb[col]==bb.true_label.map({1:'stereo',0:'anti'})).sum())}/{len(bb)}")
vs_dataset("chris","Chris"); vs_dataset("julia","Julia")
print("Chris vs Julia, all 4 categories: agree", int((d.chris==d.julia).sum()), "/40, kappa", round(cohen_kappa_score(d.chris, d.julia),3))
both = d[d.chris.isin(["stereo","anti"]) & d.julia.isin(["stereo","anti"])]
print("Chris vs Julia, rows both gave stereo/anti:", len(both), "agree", int((both.chris==both.julia).sum()), "kappa", round(cohen_kappa_score(both.chris, both.julia),3))
print("cross-tab Chris (rows) vs Julia (cols):"); print(pd.crosstab(d.chris, d.julia))
q = d.qwen14b_pred.map({1:"stereo",0:"anti"})
b = d[d.chris.isin(["stereo","anti"])]
print("Chris vs Qwen:", int((b.chris==b.qwen14b_pred.map({1:'stereo',0:'anti'})).sum()), "/", len(b))
b = d[d.julia.isin(["stereo","anti"])]
print("Julia vs Qwen:", int((b.julia==b.qwen14b_pred.map({1:'stereo',0:'anti'})).sum()), "/", len(b))
print("Qwen vs dataset on these 40:", int((d.qwen14b_pred==d.true_label).sum()), "/40")
print("\nrows where Chris and Julia differ:")
for r in d[d.chris!=d.julia].itertuples(): print(f" {r.audit_id} chris={r.chris} julia={r.julia} dataset={'stereo' if r.true_label else 'anti'} | {r.sentence[:90]}")
# pairs
cp = pd.read_csv(f"{S}/chris_labels_pairs.tsv", sep="\t").iloc[:, :6]
cp.columns = ["audit_id","s1","s2","chris","conf","reason"]; cp["chris"] = cp.chris.map(norm)
kp = pd.read_csv(f"{A}/private/audit_key_pairs.csv")[["audit_id","true_more_stereotypical","qwen14b_prob_1","qwen14b_prob_2"]]
jp = pd.read_csv(f"{A}/audit_results_pairs.csv")[["audit_id","your_pick_first_second_equal_unsure"]].rename(columns={"your_pick_first_second_equal_unsure":"julia"})
jp["julia"] = jp.julia.map(norm)
p = cp.merge(kp, on="audit_id").merge(jp, on="audit_id")
p["truth"] = p.true_more_stereotypical.map({1:"first",2:"second"})
p["qwen"] = np.where(p.qwen14b_prob_1 > p.qwen14b_prob_2, "first", "second")
print("\nPairs: Chris", p.chris.value_counts().to_dict(), "| Julia", p.julia.value_counts().to_dict())
print("Chris picks dataset's sentence:", int((p.chris==p.truth).sum()), "/10;  Julia:", int((p.julia==p.truth).sum()), "/10;  Qwen:", int((p.qwen==p.truth).sum()), "/10")
print("Chris vs Julia agree:", int((p.chris==p.julia).sum()), "/10")
print(p[["audit_id","chris","julia","truth","qwen"]].to_string(index=False))
