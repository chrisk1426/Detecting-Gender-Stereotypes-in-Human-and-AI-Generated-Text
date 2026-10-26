"""Score engineered features alone and on top of the best model so far (Snowflake-L + TF-IDF + gender x word)."""
import re, sys, numpy as np, pandas as pd, scipy.sparse as sp
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
HERE=Path(__file__).parent; FEAT=HERE/"feat"
R="data/processed/"
df=pd.read_csv(R+"merged_gender.csv").merge(pd.read_csv(R+"folds.csv"),on="pair_id")
y,f=df.label.values,df.fold.values
sys.path.insert(0,str(HERE)); from mlm_features import F2M, M2F
F,M=set(F2M),set(M2F)
toks=df.sentence.map(lambda s: re.findall(r"[a-z']+",s.lower()))
g=toks.map(lambda t:("both" if set(t)&F and set(t)&M else "F" if set(t)&F else "M" if set(t)&M else "none"))
G=pd.get_dummies(g).values.astype(float)
STOP=list(ENGLISH_STOP_WORDS-F-M)
inter=np.array([" ".join(f"{gg}_{w}" for w in t if w not in F|M and w not in STOP) for t,gg in zip(toks,g)])
E=np.load(HERE/"emb"/"snowflake-arctic-l-v2.0.npy")
neg=pd.read_csv(FEAT/"negation.csv").values; pos=pd.read_csv(FEAT/"sentiment.csv").values if (FEAT/"sentiment.csv").exists() else None
mlm={p.stem[4:]:pd.read_csv(p) for p in sorted(FEAT.glob("mlm_*.csv"))}

print("== single-feature check: MLM gender agreement ==")
for n,m in mlm.items():
    a=m.agree.values; ok=m.n_gender_words.values>0
    pr=df.assign(a=a).pivot_table(index="pair_id",columns="label",values="a")
    print(f"{n:22s} coverage {ok.mean():.0%}  AUC {roc_auc_score(y[ok],a[ok]):.3f}  pairwise (stereo > anti) {(pr[1]>pr[0]).mean():.3f}")
print("negation AUC (higher = anti):",round(1-roc_auc_score(y,neg[:,0]),3))
if pos is not None: print("sentiment AUC (higher pos = anti):",round(1-roc_auc_score(y,pos[:,0]),3))

def run(dense=(), text=True, emb=True):
    acc=[]
    for k in range(5):
        tr,te=f!=k,f==k; A,B=[],[]
        if text:
            for col,kw in ((df.sentence.values,dict(ngram_range=(1,2),sublinear_tf=True)),(inter,dict(token_pattern=r"\S+",sublinear_tf=True))):
                v=TfidfVectorizer(**kw).fit(col[tr]); A.append(v.transform(col[tr])); B.append(v.transform(col[te]))
        if emb:
            sc=StandardScaler().fit(E[tr]); A.append(sp.csr_matrix(sc.transform(E[tr])*0.05)); B.append(sp.csr_matrix(sc.transform(E[te])*0.05))
        for X in dense:
            sc=StandardScaler().fit(X[tr]); A.append(sp.csr_matrix(sc.transform(X[tr]))); B.append(sp.csr_matrix(sc.transform(X[te])))
        m=LogisticRegression(C=1,max_iter=5000).fit(sp.hstack(A).tocsr(),y[tr]); acc.append((m.predict(sp.hstack(B).tocsr())==y[te]).mean())
    return np.mean(acc),np.std(acc)

gsign=np.where(g=="F",1,np.where(g=="M",-1,0))[:,None]
rows=[("best so far: Snowflake-L + TF-IDF + gender x word",dict())]
for n,m in mlm.items():
    X=m.values
    rows+=[(f"MLM {n} only",dict(dense=(X,G),text=False,emb=False)),
           (f"best + MLM {n}",dict(dense=(X,G)))]
rows+=[("best + negation",dict(dense=(neg,)))]
if pos is not None: rows+=[("best + sentiment (+ x gender)",dict(dense=(pos,pos*gsign)))]
allm=np.hstack([m.values for m in mlm.values()]) if mlm else np.zeros((len(df),0))
rows+=[("best + all MLM + negation" + (" + sentiment" if pos is not None else ""),dict(dense=(allm,G,neg)+((pos,pos*gsign) if pos is not None else ())))]
out=[]
for name,kw in rows:
    a,s=run(**kw); out.append(dict(features=name,accuracy=round(a,3),fold_sd=round(s,3))); print(f"{name:55s} {a:.3f} (sd {s:.3f})",flush=True)
pd.DataFrame(out).to_csv("outputs/engineered_features.csv",index=False)
