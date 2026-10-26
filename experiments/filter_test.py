import re, numpy as np, pandas as pd
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
R="data/processed/"
df=pd.read_csv(R+"merged_gender.csv").merge(pd.read_csv(R+"folds.csv"),on="pair_id")
y,f=df.label.values,df.fold.values
GENDER={"he","she","his","her","hers","him","himself","herself","man","woman","men","women","boy","girl","boys","girls",
        "male","female","father","mother","son","daughter","husband","wife","brother","sister","gentleman","lady","guy","guys",
        "mom","dad","grandfather","grandmother","boyfriend","girlfriend","mr","mrs","ms","sir","ma'am","uncle","aunt"}
tok=lambda s: re.findall(r"[a-z']+",s.lower())
enc=SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
lr=lambda: make_pipeline(StandardScaler(),LogisticRegression(C=0.1,max_iter=5000))
base=np.load("data/embeddings/mpnet.npy")
acc_b,acc_f=[],[]
for k in range(5):
    tr,te=f!=k,f==k
    tops=[{w for w,_ in Counter(w for s in df.sentence[tr&(y==lab)] for w in tok(s)).most_common(50)} for lab in (0,1)]
    stop=(tops[0]&tops[1])-GENDER
    filt=df.sentence.map(lambda s:" ".join(w for w in tok(s) if w not in stop))
    if k==0:
        print(f"fold0 removes {len(stop)} words:",sorted(stop))
        for i in (0,1,600): print(" ",df.sentence[i],"\n   ->",filt[i])
    X=enc.encode(list(filt),batch_size=64,normalize_embeddings=True)
    acc_b.append((lr().fit(base[tr],y[tr]).predict(base[te])==y[te]).mean())
    acc_f.append((lr().fit(X[tr],y[tr]).predict(X[te])==y[te]).mean())
print(f"\nmpnet+LR  baseline {np.mean(acc_b):.3f} (sd {np.std(acc_b):.3f})  |  filtered {np.mean(acc_f):.3f} (sd {np.std(acc_f):.3f})")
