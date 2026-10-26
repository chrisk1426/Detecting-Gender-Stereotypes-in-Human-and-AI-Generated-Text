import re, pandas as pd
from collections import Counter
df=pd.read_csv("data/processed/merged_gender.csv")
F={"she","her","hers","herself","woman","women","girl","girls","female","mother","mom","mommy","daughter","wife","sister","lady","ladies","grandmother","girlfriend","aunt","mrs","ms","schoolgirl"}
M={"he","his","him","himself","man","men","boy","boys","male","father","dad","son","husband","brother","gentleman","gentlemen","guy","guys","grandfather","boyfriend","uncle","mr","sir","schoolboy"}
tok=lambda s: re.findall(r"[a-z']+",s.lower())
rows=[]
for pid,g in df.groupby("pair_id"):
    s1=g[g.label==1].sentence.iloc[0]; s0=g[g.label==0].sentence.iloc[0]
    c1,c0=Counter(tok(s1)),Counter(tok(s0))
    d1=list((c1-c0).elements()); d0=list((c0-c1).elements())
    kind="gender swap" if set(d1+d0)<=F|M|set() or (set(d1)&(F|M) and set(d0)&(F|M) and len(d1)<=3) else "attribute swap"
    rows.append(dict(pair_id=pid,source=g.source.iloc[0],only1=d1,only0=d0,n_diff=len(d1)+len(d0),kind=kind,
                     g1=("F" if set(d1)&F else "M" if set(d1)&M else "-"),g0=("F" if set(d0)&F else "M" if set(d0)&M else "-")))
p=pd.DataFrame(rows)
print("== what differs within a pair ==")
print(pd.crosstab(p.source,p.kind))
print("\nmedian # differing words:",p.groupby("source").n_diff.median().to_dict())
gs=p[p.kind=="gender swap"]
print("\n== gender swaps: gender of the STEREOTYPE sentence ==")
print(pd.crosstab(gs.source,gs.g1))
at=p[p.kind=="attribute swap"]
c1=Counter(w for l in at.only1 for w in l); c0=Counter(w for l in at.only0 for w in l)
print("\n== attribute swaps: most common words ONLY in stereotype side ==\n",c1.most_common(40))
print("\n== ...ONLY in anti-stereotype side ==\n",c0.most_common(40))
neg={"not","no","never","didn't","wasn't","isn't","don't","doesn't","can't","couldn't","won't"}
print("\nnegation words: stereotype side",sum(c1[w] for w in neg),"| anti side",sum(c0[w] for w in neg))
pd.set_option("display.max_colwidth",60); pd.set_option("display.width",200)
print("\nsample attribute swaps:\n",at.sample(15,random_state=3)[["source","only1","only0"]].to_string(index=False))
