import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
R="data/"
df=pd.read_csv(R+"processed/merged_gender.csv").merge(pd.read_csv(R+"processed/folds.csv"),on="pair_id")
y=df.label.values; f=df.fold.values
for name in ["bert","mpnet"]:
    X=np.load(R+f"embeddings/{name}.npy"); Xn=X/np.linalg.norm(X,axis=1,keepdims=True)
    # 1. within-pair vs random cosine
    a=df[df.label==1].index; b=df[df.label==0].set_index("pair_id").loc[df.loc[a,"pair_id"]]["sentence"]
    idx0=df[df.label==0].reset_index().set_index("pair_id").loc[df.loc[a,"pair_id"],"index"].values
    cos=(Xn[a]*Xn[idx0]).sum(1)
    rnd=(Xn[a]*Xn[np.random.default_rng(0).permutation(idx0)]).sum(1)
    print(f"\n[{name}] within-pair cos by source:", pd.Series(cos,index=df.loc[a,'source'].values).groupby(level=0).mean().round(3).to_dict(), "random pair cos:", rnd.mean().round(3))
    # 2. learning curve, single-sentence LR
    lc={}
    for frac in [0.25,0.5,0.75,1.0]:
        acc=[]
        for k in range(5):
            tr=np.where(f!=k)[0]; te=f==k
            pids=df.pair_id.values[tr]; up=np.unique(pids); keep=np.random.default_rng(k).choice(up,int(len(up)*frac),replace=False)
            tr=tr[np.isin(pids,keep)]
            m=make_pipeline(StandardScaler(),LogisticRegression(C=0.1,max_iter=5000)).fit(X[tr],y[tr])
            acc.append((m.predict(X[te])==y[te]).mean())
        lc[frac]=round(np.mean(acc),3)
    print(f"[{name}] learning curve (frac train -> acc):",lc)
    # train acc
    m=make_pipeline(StandardScaler(),LogisticRegression(C=0.1,max_iter=5000)).fit(X[f!=0],y[f!=0])
    print(f"[{name}] train acc fold0: {(m.predict(X[f!=0])==y[f!=0]).mean():.3f}")
    # 3. pairwise difference model: x_a - x_b -> is a the stereotype?
    pa=X[a]-X[idx0]; pf=df.loc[a,"fold"].values
    D=np.vstack([pa,-pa]); Y=np.r_[np.ones(len(pa)),np.zeros(len(pa))]; F=np.r_[pf,pf]
    acc=[]
    for k in range(5):
        m=LogisticRegression(C=1,max_iter=5000).fit(D[F!=k],Y[F!=k]); acc.append((m.predict(D[F==k])==Y[F==k]).mean())
    print(f"[{name}] pairwise-difference LR acc: {np.mean(acc):.3f}")
