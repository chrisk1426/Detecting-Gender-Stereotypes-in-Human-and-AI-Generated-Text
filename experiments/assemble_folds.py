"""Assemble per-fold prediction files (ft_<tag>.fold{k}.npy) into one out-of-fold CSV and print accuracy."""
import sys, numpy as np, pandas as pd
from pathlib import Path
HERE=Path(__file__).parent
df=pd.read_csv(HERE.parent/"data/processed/merged_gender.csv").merge(pd.read_csv(HERE.parent/"data/processed/folds.csv"),on="pair_id")
y,f=df.label.values,df.fold.values
for tag in sys.argv[1:]:
    parts=[HERE/"feat"/f"ft_{tag}.fold{k}.npy" for k in range(5)]
    if not all(p.exists() for p in parts): print(tag,"missing folds",[k for k,p in enumerate(parts) if not p.exists()]); continue
    p=np.zeros(len(y))
    for k in range(5): p[f==k]=np.load(parts[k])
    a=[((p[f==k]>0.5)==y[f==k]).mean() for k in range(5)]
    pd.DataFrame({"prob":p,"fold":f}).to_csv(HERE/"feat"/f"ft_{tag}.csv",index=False)
    print(f"{tag}: {np.mean(a):.3f} (sd {np.std(a):.3f})  folds {[round(x,3) for x in a]}")
