"""Compare the three-class pipeline (gender filter + three-class Qwen, out-of-fold) with the audit labels.

Reads experiments/feat/ft_..._3class_ep6_s72.csv, gender_filter_oof.csv, outputs/audit/audit_results_*.csv and the second
labeler's chris_labels_*.tsv. Prints the numbers used in the Neutral text and Blind audit parts of the report;
see outputs/audit/audit_vs_three_class.txt.
"""
from pathlib import Path
import pandas as pd, numpy as np
from sklearn.metrics import cohen_kappa_score
R = Path(__file__).resolve().parent.parent
S = R / "outputs" / "audit"
cv = pd.read_csv(f"{R}/experiments/feat/ft_Qwen2.5-14B-Instruct_lora_4bit_3class_ep6_s72.csv")
gf = pd.read_csv(f"{R}/experiments/feat/gender_filter_oof.csv")[["sentence","p_gender_lr"]]
cv = cv.merge(gf, on="sentence", how="left")
cv["qwen3"] = cv[["p0","p1","p2"]].to_numpy().argmax(1)
cv["pcls"] = np.where(cv.p_gender_lr < 0.5, 2, cv.qwen3)
cv["s_minus_a"] = cv.p1 - cv.p0
nm = {0:"anti",1:"stereo",2:"neutral"}
norm = lambda x: {"stereotypical":"stereo","anti-stereotypical":"anti","neutral":"neutral","unsure":"unsure"}.get(str(x).strip().lower(), str(x))
cs = pd.read_csv(f"{S}/chris_labels_single.tsv", sep="\t").iloc[:, :3]; cs.columns = ["audit_id","sentence","chris"]; cs["chris"]=cs.chris.map(norm)
jul = pd.read_csv(f"{R}/outputs/audit/audit_results_single.csv")[["audit_id","your_label","true_label","qwen14b_pred","ensemble_pred"]].rename(columns={"your_label":"julia"}); jul["julia"]=jul.julia.map(norm)
d = cs.merge(jul, on="audit_id").merge(cv[["sentence","p0","p1","p2","p_gender_lr","pcls","s_minus_a"]], on="sentence", how="left")
print("audit sentences found in 3-class OOF file:", d.p1.notna().sum(), "/ 40")
d["plab"] = d.pcls.map(nm)
d["ds"] = d.true_label.map({1:"stereo",0:"anti"})
print("\n3-class pipeline labels on the 40:", d.plab.value_counts().to_dict())
print("Chris:", d.chris.value_counts().to_dict(), "| Julia:", d.julia.value_counts().to_dict())
nn = d[d.plab!="neutral"]
print(f"\nNot called neutral by pipeline: {len(nn)}; agrees with dataset {int((nn.plab==nn.ds).sum())}/{len(nn)}")
print("Stereo-vs-anti by p1>p0 on all 40 vs dataset:", int((((d.p1>d.p0).astype(int))==d.true_label).sum()), "/40  (old 2-class Qwen:", int((d.qwen14b_pred==d.true_label).sum()), "/40)")
for who in ["chris","julia"]:
    print(f"\n{who}: agreement with pipeline (3-way, rows {who} gave stereo/anti/neutral):")
    b = d[d[who].isin(["stereo","anti","neutral"])]
    print("  agree", int((b[who]==b.plab).sum()), "/", len(b), " kappa", round(cohen_kappa_score(b[who], b.plab),3))
    print(pd.crosstab(b[who], b.plab).to_string())
print("\nPipeline neutral vs not, when Chris says neutral (23):", pd.crosstab(d.chris=="neutral", d.plab=="neutral").to_string())
print("Neutral calls: pipeline", int((d.plab=="neutral").sum()), " Chris", int((d.chris=="neutral").sum()), " Julia", int((d.julia=="neutral").sum()))
print("\nPipeline-neutral sentences (and what dataset/Julia/Chris say):")
for r in d[d.plab=="neutral"].itertuples(): print(f"  {r.audit_id} ds={r.ds} julia={r.julia} chris={r.chris} gender_p={r.p_gender_lr:.2f} | {r.sentence[:80]}")
# pairs
pr = pd.read_csv(f"{R}/outputs/audit/audit_results_pairs.csv")[["audit_id","sentence_1","sentence_2","true_more_stereotypical","your_pick_first_second_equal_unsure"]]
pr = pr.rename(columns={"your_pick_first_second_equal_unsure":"julia"})
cp = pd.read_csv(f"{S}/chris_labels_pairs.tsv", sep="\t").iloc[:, [0,3]]; cp.columns=["audit_id","chris"]
pr = pr.merge(cp, on="audit_id")
m1 = cv.set_index("sentence")[["pcls","s_minus_a"]]
pr["s1"] = pr.sentence_1.map(m1.s_minus_a); pr["s2"] = pr.sentence_2.map(m1.s_minus_a)
pr["pcls"] = np.where(pr.s1>pr.s2,"first","second"); pr["truth"] = pr.true_more_stereotypical.map({1:"first",2:"second"})
print("\nPairs found:", pr.s1.notna().sum(), "/ 10; 3-class pipeline picks dataset's sentence:", int((pr.pcls==pr.truth).sum()), "/10")
print(pr[["audit_id","truth","pcls","julia","chris"]].to_string(index=False))
