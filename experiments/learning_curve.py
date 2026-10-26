"""Learning curve: train on a fraction of each training fold's PAIRS, test on the full held-out fold.
Rising at 100% = more data would likely help; flat = it would not."""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from classifiers import feats, y, f, df, LogisticRegression, RandomForestClassifier
FRACS=[0.25,0.5,0.75,1.0]; pids=df.pair_id.values
res={("LR",fr):[] for fr in FRACS}|{("RF",fr):[] for fr in FRACS}
for k in range(5):
    te=f==k; up=np.unique(pids[f!=k])
    for fr in FRACS:
        for rep in range(3 if fr<1 else 1):
            keep=np.random.default_rng(100*k+rep).choice(up,int(len(up)*fr),replace=False)
            tr=(f!=k)&np.isin(pids,keep)
            A,B,dA,dB,_,_=feats(tr,te)
            res[("LR",fr)].append((LogisticRegression(C=1,max_iter=5000).fit(A,y[tr]).predict(B)==y[te]).mean())
            res[("RF",fr)].append((RandomForestClassifier(500,max_features=0.1,n_jobs=-1,random_state=72).fit(dA,y[tr]).predict(dB)==y[te]).mean())
    print("fold",k,"done",flush=True)
for m in ("LR","RF"):
    print(m," ".join(f"{int(fr*100)}%={np.mean(res[(m,fr)]):.3f}" for fr in FRACS))
