"""Does the best pipeline need its TF-IDF blocks? Drop each block and re-score on the same 5 folds."""
import re, sys, numpy as np, pandas as pd, scipy.sparse as sp
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
HERE=Path(__file__).parent; sys.path.insert(0,str(HERE)); from mlm_features import F2M, M2F
F,M=set(F2M),set(M2F)
df=pd.read_csv("data/processed/merged_gender.csv").merge(pd.read_csv("data/processed/folds.csv"),on="pair_id")
y,f=df.label.values,df.fold.values
toks=df.sentence.map(lambda s: re.findall(r"[a-z']+",s.lower()))
g=toks.map(lambda t:("both" if set(t)&F and set(t)&M else "F" if set(t)&F else "M" if set(t)&M else "none"))
G=pd.get_dummies(g).values.astype(float)
STOP=ENGLISH_STOP_WORDS-F-M
inter=np.array([" ".join(f"{gg}_{w}" for w in t if w not in F|M and w not in STOP) for t,gg in zip(toks,g)])
E=np.load(HERE/"emb"/"snowflake-arctic-l-v2.0.npy"); MLM=pd.read_csv(HERE/"feat"/"mlm_roberta-large.csv").values

def run(words, gxw, emb, min_df=1):
    acc=[];n=0
    for k in range(5):
        tr,te=f!=k,f==k; A,B=[],[]
        if words:
            v=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,min_df=min_df).fit(df.sentence[tr]); A.append(v.transform(df.sentence[tr])); B.append(v.transform(df.sentence[te]))
        if gxw:
            v=TfidfVectorizer(token_pattern=r"\S+",sublinear_tf=True,min_df=min_df).fit(inter[tr]); A.append(v.transform(inter[tr])); B.append(v.transform(inter[te]))
        dense=[MLM,G]+([E] if emb else [])
        for X in dense:
            sc=StandardScaler().fit(X[tr]); w=0.05 if X is E else 1
            A.append(sp.csr_matrix(sc.transform(X[tr])*w)); B.append(sp.csr_matrix(sc.transform(X[te])*w))
        Xa=sp.hstack(A).tocsr(); n=Xa.shape[1]
        m=LogisticRegression(C=1,max_iter=5000).fit(Xa,y[tr]); acc.append((m.predict(sp.hstack(B).tocsr())==y[te]).mean())
    return np.mean(acc),np.std(acc),n
for name,kw in [("all (current best)",dict(words=1,gxw=1,emb=1)),
                ("no plain TF-IDF",dict(words=0,gxw=1,emb=1)),
                ("no gender x word",dict(words=1,gxw=0,emb=1)),
                ("no TF-IDF at all",dict(words=0,gxw=0,emb=1)),
                ("all, rare words dropped (min_df=2)",dict(words=1,gxw=1,emb=1,min_df=2)),
                ("TF-IDF blocks only (no embedding)",dict(words=1,gxw=1,emb=0))]:
    a,s,n=run(**kw); print(f"{name:38s} {a:.3f} (sd {s:.3f})  features {n:,}")
