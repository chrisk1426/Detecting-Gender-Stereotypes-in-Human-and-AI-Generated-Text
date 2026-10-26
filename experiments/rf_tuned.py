"""#5: random forest on the dense features, settings chosen per outer fold by grouped inner CV. Saves OOF probs."""
import sys, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import GridSearchCV, GroupKFold
sys.path.insert(0,str(Path(__file__).parent))
from classifiers import feats, y, f, df, RandomForestClassifier
grid={"max_features":["sqrt",0.05,0.1,0.3],"min_samples_leaf":[1,3,10]}
grp=df.group.values; p=np.zeros(len(y))
for k in range(5):
    tr,te=f!=k,f==k; _,_,dA,dB,_,_=feats(tr,te)
    gs=GridSearchCV(RandomForestClassifier(500,random_state=72,n_jobs=-1),grid,cv=GroupKFold(4),scoring="accuracy").fit(dA,y[tr],groups=grp[tr])
    p[te]=gs.predict_proba(dB)[:,1]
    print(f"fold {k}: picks {gs.best_params_}  inner {gs.best_score_:.3f}  test {((p[te]>0.5)==y[te]).mean():.3f}",flush=True)
acc=[((p[f==k]>0.5)==y[f==k]).mean() for k in range(5)]
print(f"RF tuned (nested): {np.mean(acc):.3f} (sd {np.std(acc):.3f})")
pd.DataFrame({"prob":p,"fold":f}).to_csv(Path(__file__).parent/"feat"/"oof_rf_tuned.csv",index=False)
