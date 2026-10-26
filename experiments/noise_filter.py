"""#7: drop suspected mislabeled pairs from TRAINING folds only, retrain, score on untouched test folds.

Suspicion is computed inside each outer training fold (inner 4-fold OOF of the LR pipeline), so the
test fold never influences which rows are dropped. A pair is suspect when the model confidently ranks
the 'anti-stereotype' sentence above the 'stereotype' one.
"""
import sys, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
sys.path.insert(0,str(Path(__file__).parent))
from classifiers import feats, y, f, df, LogisticRegression, RandomForestClassifier
grp,pid=df.group.values,df.pair_id.values
DROPS=[0,0.05,0.10,0.20]
res={(m,d):[] for m in ("LR","RF") for d in DROPS}
for k in range(5):
    trn=np.where(f!=k)[0]; te=f==k
    inner=np.zeros(len(y))
    for a,b in GroupKFold(4).split(trn,groups=grp[trn]):
        ma=np.zeros(len(y),bool); ma[trn[a]]=True; mb=np.zeros(len(y),bool); mb[trn[b]]=True
        A,B,_,_,_,_=feats(ma,mb); inner[mb]=LogisticRegression(C=1,max_iter=5000).fit(A,y[ma]).predict_proba(B)[:,1]
    d=pd.DataFrame({"pid":pid[trn],"p":inner[trn],"y":y[trn]})
    margin=d[d.y==1].set_index("pid").p - d[d.y==0].set_index("pid").p   # >0 = model agrees with label
    order=margin.sort_values().index                                      # most contradicted first
    for drop in DROPS:
        bad=set(order[:int(len(order)*drop)])
        tr=(f!=k)&~np.isin(pid,list(bad))
        A,B,dA,dB,_,_=feats(tr,te)
        res[("LR",drop)].append((LogisticRegression(C=1,max_iter=5000).fit(A,y[tr]).predict(B)==y[te]).mean())
        res[("RF",drop)].append((RandomForestClassifier(500,max_features=0.1,n_jobs=-1,random_state=72).fit(dA,y[tr]).predict(dB)==y[te]).mean())
    print("fold",k,"done",flush=True)
for m in ("LR","RF"):
    print(m,"  ".join(f"drop {int(d*100)}%={np.mean(res[(m,d)]):.3f}" for d in DROPS))
