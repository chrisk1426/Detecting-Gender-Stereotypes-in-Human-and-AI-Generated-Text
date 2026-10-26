import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
R="data/processed/"
df=pd.read_csv(R+"merged_gender.csv").merge(pd.read_csv(R+"folds.csv"),on="pair_id")
X,y,f=df.sentence.values,df.label.values,df.fold.values
for ng in [(1,1),(1,2)]:
    for C in (0.3,1,3,10):
        acc=[(make_pipeline(TfidfVectorizer(ngram_range=ng,sublinear_tf=True),LogisticRegression(C=C,max_iter=5000))
              .fit(X[f!=k],y[f!=k]).predict(X[f==k])==y[f==k]).mean() for k in range(5)]
        print(f"ngrams {ng}  C={C:<4} acc {np.mean(acc):.3f} (sd {np.std(acc):.3f})")
