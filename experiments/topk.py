"""Top-k chi2 selection on the TF-IDF blocks of the best pipeline. Dense features (embedding, masked LM,
gender, optional LLM) are always kept. k is chosen per outer fold by grouped inner CV (no test-fold tuning)."""
import re, sys, numpy as np, pandas as pd, scipy.sparse as sp
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
HERE=Path(__file__).parent; sys.path.insert(0,str(HERE)); from mlm_features import F2M, M2F
F,M=set(F2M),set(M2F)
df=pd.read_csv("data/processed/merged_gender.csv").merge(pd.read_csv("data/processed/folds.csv"),on="pair_id")
y,f,grp=df.label.values,df.fold.values,df.group.values
toks=df.sentence.map(lambda s: re.findall(r"[a-z']+",s.lower()))
g=toks.map(lambda t:("both" if set(t)&F and set(t)&M else "F" if set(t)&F else "M" if set(t)&M else "none"))
STOP=ENGLISH_STOP_WORDS-F-M
inter=np.array([" ".join(f"{gg}_{w}" for w in t if w not in F|M and w not in STOP) for t,gg in zip(toks,g)])
S=df.sentence.values
E=np.load(HERE/"emb"/"snowflake-arctic-l-v2.0.npy")
DENSE=[pd.read_csv(HERE/"feat"/"mlm_roberta-large.csv").values, pd.get_dummies(g).values.astype(float)]
# --llm=7B,14B picks which LLM feature files to add (default 7B)
LLM_ARG=next((a for a in sys.argv if a.startswith("--llm")),None)
if LLM_ARG:
    for size in (LLM_ARG.split("=")[1].split(",") if "=" in LLM_ARG else ["7B"]):
        p_=HERE/"feat"/f"llm_Qwen2.5-{size}-Instruct.csv"; DENSE.append(pd.read_csv(p_).values); print("using",p_.name)
KS=[250,500,1000,2000,4000,None]
AUG="--aug" in sys.argv
from llm_features import swap_gender
def _gtag(t): return ("both" if set(t)&F and set(t)&M else "F" if set(t)&F else "M" if set(t)&M else "none")
SW=[swap_gender(x) for x in S]
SW_inter=np.array(["" if w is None else " ".join(f"{_gtag(tt)}_{x}" for x in tt if x not in F|M and x not in STOP)
                   for w,tt in ((w,re.findall(r"[a-z']+",w.lower()) if w else []) for w in SW)])
HAS=np.array([w is not None for w in SW])

def fit_score(tr,te,k):
    A,B=[],[]
    tr=np.where(tr)[0] if tr.dtype==bool else tr
    aug=tr[HAS[tr]] if AUG else tr[:0]
    ytr=np.r_[y[tr],1-y[aug]]
    for col,swc,kw in ((S,np.array([w or "" for w in SW]),dict(ngram_range=(1,2),sublinear_tf=True)),(inter,SW_inter,dict(token_pattern=r"\S+",sublinear_tf=True))):
        trtxt=np.r_[col[tr],swc[aug]]
        v=TfidfVectorizer(**kw).fit(trtxt); A.append(v.transform(trtxt)); B.append(v.transform(col[te]))
    A,B=sp.hstack(A).tocsr(),sp.hstack(B).tocsr()
    if k is not None and k<A.shape[1]:
        sel=SelectKBest(chi2,k=k).fit(A,ytr); A,B=sel.transform(A),sel.transform(B)
    if AUG or "--textonly" in sys.argv: return text_only_score(A,B,ytr,te)
    for X,w in [(E,0.05)]+[(D,1) for D in DENSE]:
        sc=StandardScaler().fit(X[tr]); A=sp.hstack([A,sp.csr_matrix(sc.transform(X[tr])*w)]); B=sp.hstack([B,sp.csr_matrix(sc.transform(X[te])*w)])
    m=LogisticRegression(C=1,max_iter=5000).fit(A.tocsr(),y[tr]); LAST_PROB[0]=m.predict_proba(B.tocsr())[:,1]; return (m.predict(B.tocsr())==y[te]).mean()

def text_only_score(A,B,ytr,te):
    m=LogisticRegression(C=1,max_iter=5000).fit(A,ytr); return (m.predict(B)==y[te]).mean()

LAST_PROB=[None]; OOF=np.zeros(len(y))
fixed={k:[] for k in KS}; nested=[]; chosen=[]
for k_out in range(5):
    tr_o=np.where(f!=k_out)[0]; te_o=f==k_out
    inner={k:np.mean([fit_score(tr_o[a],tr_o[b],k) for a,b in GroupKFold(4).split(tr_o,groups=grp[tr_o])]) for k in KS}
    best=max(inner,key=inner.get); chosen.append(best)
    trm=np.zeros(len(y),bool); trm[tr_o]=True
    for k in KS:
        fixed[k].append(fit_score(trm,te_o,k))
        if k==best: OOF[te_o]=LAST_PROB[0]
    nested.append(fixed[best][-1]); print(f"fold {k_out}: inner picks k={best}",flush=True)
print("\nfixed k (for reference; picking the best of these by eye is test-fold tuning):")
for k in KS: print(f"  k={str(k):5s} {np.mean(fixed[k]):.3f} (sd {np.std(fixed[k]):.3f})")
print(f"\nNESTED (honest) top-k: {np.mean(nested):.3f} (sd {np.std(nested):.3f})   chosen k per fold: {chosen}")

if LLM_ARG and not AUG:
    tag=LLM_ARG.split("=")[1].replace(",","+") if "=" in LLM_ARG else "7B"
    pd.DataFrame({"prob":OOF,"fold":f}).to_csv(HERE/"feat"/f"oof_feature_pipeline_{tag}.csv",index=False); print("saved OOF probs",tag)
