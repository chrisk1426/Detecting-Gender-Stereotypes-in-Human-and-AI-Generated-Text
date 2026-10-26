"""PCA / truncated SVD on the high-dimensional blocks, fitted on training folds only. Same folds and features."""
import sys, warnings, numpy as np, scipy.sparse as sp
from pathlib import Path
from sklearn.decomposition import PCA, TruncatedSVD
warnings.filterwarnings("ignore")
sys.path.insert(0,str(Path(__file__).parent))
from classifiers import feats, y, f, E, SMALL, S, inter, LogisticRegression, RandomForestClassifier, StandardScaler, TfidfVectorizer
NS=[16,32,64,128,256,None]
rf=lambda: RandomForestClassifier(500,max_features=0.1,n_jobs=-1,random_state=72)
res={}
for k in range(5):
    tr,te=f!=k,f==k
    A,B,_,_,sA,sB=feats(tr,te)                       # A/B: top-k TF-IDF + scaled dense (LR input)
    se=StandardScaler().fit(E[tr]); eA,eB=se.transform(E[tr]),se.transform(E[te])
    for n in NS:
        if n: p=PCA(n,random_state=72).fit(eA); pA,pB=p.transform(eA),p.transform(eB)
        else: pA,pB=eA,eB
        res.setdefault(("RF, PCA on embedding",n),[]).append((rf().fit(np.hstack([pA,sA]),y[tr]).predict(np.hstack([pB,sB]))==y[te]).mean())
        sc=StandardScaler().fit(pA); w=0.05*np.sqrt(1024/pA.shape[1])
        LA=sp.hstack([A[:,:-(1024+SMALL.shape[1])],sp.csr_matrix(sc.transform(pA)*w),sp.csr_matrix(sA)]).tocsr()
        LB=sp.hstack([B[:,:-(1024+SMALL.shape[1])],sp.csr_matrix(sc.transform(pB)*w),sp.csr_matrix(sB)]).tocsr()
        res.setdefault(("LR, PCA on embedding",n),[]).append((LogisticRegression(C=1,max_iter=5000).fit(LA,y[tr]).predict(LB)==y[te]).mean())
    # truncated SVD on the full TF-IDF blocks (sparse PCA), for LR
    T=[]
    for col,kw in ((S,dict(ngram_range=(1,2),sublinear_tf=True)),(inter,dict(token_pattern=r"\S+",sublinear_tf=True))):
        v=TfidfVectorizer(**kw).fit(col[tr]); T.append((v.transform(col[tr]),v.transform(col[te])))
    TA,TB=sp.hstack([t[0] for t in T]).tocsr(),sp.hstack([t[1] for t in T]).tocsr()
    for n in NS[:-1]:
        sv=TruncatedSVD(n,random_state=72).fit(TA); ss=StandardScaler().fit(sv.transform(TA))
        XA=np.hstack([ss.transform(sv.transform(TA)),eA*0.05,sA]); XB=np.hstack([ss.transform(sv.transform(TB)),eB*0.05,sB])
        res.setdefault(("LR, SVD on TF-IDF",n),[]).append((LogisticRegression(C=1,max_iter=5000).fit(XA,y[tr]).predict(XB)==y[te]).mean())
        res.setdefault(("RF, SVD TF-IDF + embedding",n),[]).append((rf().fit(np.hstack([sv.transform(TA),eA,sA]),y[tr]).predict(np.hstack([sv.transform(TB),eB,sB]))==y[te]).mean())
    print("fold",k,"done",flush=True)
for name in dict.fromkeys(m for m,_ in res):
    print(f"{name:28s}"+"  ".join(f"{('all' if n is None else n)}={np.mean(res[(name,n)]):.3f}" for (m,n) in res if m==name))
