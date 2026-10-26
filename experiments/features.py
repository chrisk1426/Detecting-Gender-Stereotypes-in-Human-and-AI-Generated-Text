"""Feature ablation: stop words, gender one-hot, source one-hot, gender x word, length. Same 5 grouped folds."""
import re, numpy as np, pandas as pd, scipy.sparse as sp
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
R="data/processed/"; EMB=Path(__file__).parent/"emb"
df=pd.read_csv(R+"merged_gender.csv").merge(pd.read_csv(R+"folds.csv"),on="pair_id")
y,f=df.label.values,df.fold.values
F={"she","her","hers","herself","woman","women","girl","girls","female","mother","mom","mommy","daughter","wife","sister","lady","ladies",
   "grandmother","girlfriend","aunt","mrs","ms","schoolgirl","sisters","mothers","wives","daughters","bride","queen","actress","waitress"}
M={"he","his","him","himself","man","men","boy","boys","male","father","dad","son","husband","brother","gentleman","gentlemen","guy","guys",
   "grandfather","boyfriend","uncle","mr","sir","schoolboy","brothers","fathers","husbands","sons","groom","king","actor","waiter"}
STOP=list(ENGLISH_STOP_WORDS-F-M)
toks=df.sentence.map(lambda s: re.findall(r"[a-z']+",s.lower()))
def gender(t):
    w=set(t); return ("both" if w&F and w&M else "F" if w&F else "M" if w&M else "none")
g=toks.map(gender)
print("gender counts:",g.value_counts().to_dict())
print("label rate by gender:",df.groupby(g).label.mean().round(3).to_dict())
G=pd.get_dummies(g).values.astype(float)
S=pd.get_dummies(df.source).values.astype(float)
L=toks.map(len).values[:,None].astype(float)
inter=np.array([" ".join(f"{gg}_{w}" for w in t if w not in F|M and w not in STOP) for t,gg in zip(toks,g)])
emb={p.stem:np.load(p) for p in EMB.glob("*.npy") if p.stem in ("snowflake-arctic-l-v2.0","mpnet_current")}

def run(text_stop=None, use_text=True, extras=(), use_inter=False, emb_name=None):
    acc=[]
    for k in range(5):
        tr,te=f!=k,f==k; blocks_tr,blocks_te=[],[]
        if use_text:
            v=TfidfVectorizer(stop_words=text_stop,ngram_range=(1,2),sublinear_tf=True).fit(df.sentence[tr])
            blocks_tr.append(v.transform(df.sentence[tr])); blocks_te.append(v.transform(df.sentence[te]))
        if use_inter:
            v=TfidfVectorizer(token_pattern=r"\S+",sublinear_tf=True).fit(inter[tr])
            blocks_tr.append(v.transform(inter[tr])); blocks_te.append(v.transform(inter[te]))
        if emb_name:
            sc=StandardScaler().fit(emb[emb_name][tr]); blocks_tr.append(sp.csr_matrix(sc.transform(emb[emb_name][tr])*0.05)); blocks_te.append(sp.csr_matrix(sc.transform(emb[emb_name][te])*0.05))
        for X in extras:
            sc=StandardScaler().fit(X[tr]); blocks_tr.append(sp.csr_matrix(sc.transform(X[tr]))); blocks_te.append(sp.csr_matrix(sc.transform(X[te])))
        m=LogisticRegression(C=1,max_iter=5000).fit(sp.hstack(blocks_tr).tocsr(),y[tr])
        acc.append((m.predict(sp.hstack(blocks_te).tocsr())==y[te]).mean())
    return np.mean(acc),np.std(acc)

rows=[
 ("TF-IDF (baseline)",                       dict()),
 ("+ stop words removed",                    dict(text_stop=STOP)),
 ("+ stop words + gender",                   dict(text_stop=STOP,extras=(G,))),
 ("+ stop words + gender + source",          dict(text_stop=STOP,extras=(G,S))),
 ("+ stop words + gender + source + length", dict(text_stop=STOP,extras=(G,S,L))),
 ("+ gender x word",                         dict(text_stop=STOP,extras=(G,S,L),use_inter=True)),
 ("gender x word only",                      dict(use_text=False,use_inter=True)),
]
for e in emb: rows+= [(f"{e} + gender + source + length", dict(use_text=False,extras=(G,S,L),emb_name=e)),
                      (f"{e} + all TF-IDF features",       dict(text_stop=STOP,extras=(G,S,L),use_inter=True,emb_name=e))]
out=[]
for name,kw in rows:
    m,s=run(**kw); out.append(dict(features=name,accuracy=round(m,3),fold_sd=round(s,3))); print(f"{name:55s} {m:.3f} (sd {s:.3f})",flush=True)
pd.DataFrame(out).to_csv("outputs/feature_ablation.csv",index=False)
