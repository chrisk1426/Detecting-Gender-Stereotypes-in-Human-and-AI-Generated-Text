import re, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
R="data/processed/"
df=pd.read_csv(R+"merged_gender.csv").merge(pd.read_csv(R+"folds.csv"),on="pair_id")
y,f=df.label.values,df.fold.values
F={"she","her","hers","herself","woman","women","girl","girls","female","mother","mom","mommy","daughter","wife","sister","lady","ladies","grandmother","girlfriend","aunt","mrs","ms","schoolgirl"}
M={"he","his","him","himself","man","men","boy","boys","male","father","dad","son","husband","brother","gentleman","gentlemen","guy","guys","grandfather","boyfriend","uncle","mr","sir","schoolboy"}
def gender_tag(s):
    w=set(re.findall(r"[a-z']+",s.lower())); g="F" if w&F and not w&M else "M" if w&M and not w&F else "X"
    # prefix every non-gender word with the sentence's gender: "nurturing" -> "F_nurturing"
    return " ".join(f"{g}_{t}" for t in re.findall(r"[a-z']+",s.lower()) if t not in F|M)
def cv(texts,ngram=(1,1),C=1.0):
    acc=[]
    for k in range(5):
        m=make_pipeline(TfidfVectorizer(ngram_range=ngram,min_df=1,sublinear_tf=True),LogisticRegression(C=C,max_iter=5000))
        acc.append((m.fit(texts[f!=k],y[f!=k]).predict(texts[f==k])==y[f==k]).mean())
    return np.mean(acc),np.std(acc)
raw=df.sentence.values; tagged=df.sentence.map(gender_tag).values
both=np.array([a+" "+b for a,b in zip(raw,tagged)])
for name,t,ng in [("TF-IDF words",raw,(1,1)),("TF-IDF words+bigrams",raw,(1,2)),
                  ("gender x word only",tagged,(1,1)),("TF-IDF + gender x word",both,(1,1))]:
    for C in (0.3,1,3):
        m,s=cv(t,ng,C); print(f"{name:26s} C={C:<4} acc {m:.3f} (sd {s:.3f})")
