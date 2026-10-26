"""Same features as the best pipeline (TF-IDF top-k + Snowflake + masked LM + gender + Qwen 7B/14B), swap the classifier."""
import re, sys, warnings, numpy as np, pandas as pd, scipy.sparse as sp
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
HERE=Path(__file__).parent; sys.path.insert(0,str(HERE)); from mlm_features import F2M, M2F
F,M=set(F2M),set(M2F)
df=pd.read_csv("data/processed/merged_gender.csv").merge(pd.read_csv("data/processed/folds.csv"),on="pair_id")
y,f=df.label.values,df.fold.values
toks=df.sentence.map(lambda s: re.findall(r"[a-z']+",s.lower()))
g=toks.map(lambda t:("both" if set(t)&F and set(t)&M else "F" if set(t)&F else "M" if set(t)&M else "none"))
STOP=ENGLISH_STOP_WORDS-F-M
inter=np.array([" ".join(f"{gg}_{w}" for w in t if w not in F|M and w not in STOP) for t,gg in zip(toks,g)])
S=df.sentence.values; E=np.load(HERE/"emb"/"snowflake-arctic-l-v2.0.npy")
SMALL=np.hstack([pd.read_csv(HERE/"feat"/"mlm_roberta-large.csv").values, pd.get_dummies(g).values.astype(float),
                 pd.read_csv(HERE/"feat"/"llm_Qwen2.5-7B-Instruct.csv").values, pd.read_csv(HERE/"feat"/"llm_Qwen2.5-14B-Instruct.csv").values])
K=2000

def feats(tr,te):
    A,B=[],[]
    for col,kw in ((S,dict(ngram_range=(1,2),sublinear_tf=True)),(inter,dict(token_pattern=r"\S+",sublinear_tf=True))):
        v=TfidfVectorizer(**kw).fit(col[tr]); A.append(v.transform(col[tr])); B.append(v.transform(col[te]))
    A,B=sp.hstack(A).tocsr(),sp.hstack(B).tocsr(); sel=SelectKBest(chi2,k=K).fit(A,y[tr]); A,B=sel.transform(A),sel.transform(B)
    se,ss=StandardScaler().fit(E[tr]),StandardScaler().fit(SMALL[tr])
    dA=np.hstack([se.transform(E[tr])*0.05,ss.transform(SMALL[tr])]); dB=np.hstack([se.transform(E[te])*0.05,ss.transform(SMALL[te])])
    dA2=np.hstack([se.transform(E[tr]),ss.transform(SMALL[tr])]); dB2=np.hstack([se.transform(E[te]),ss.transform(SMALL[te])])
    return sp.hstack([A,sp.csr_matrix(dA)]).tocsr(), sp.hstack([B,sp.csr_matrix(dB)]).tocsr(), dA2, dB2, ss.transform(SMALL[tr]), ss.transform(SMALL[te])

MODELS={
 "logistic regression (current)": ("all",   lambda: LogisticRegression(C=1,max_iter=5000)),
 "MLP 1 hidden layer (256)":      ("all",   lambda: MLPClassifier((256,),alpha=1e-2,early_stopping=True,max_iter=300,random_state=72)),
 "MLP 2 hidden layers (256,64)":  ("all",   lambda: MLPClassifier((256,64),alpha=1e-2,early_stopping=True,max_iter=300,random_state=72)),
 "SVM, RBF kernel":               ("all",   lambda: SVC(C=1,kernel="rbf")),
 "random forest (dense feats)":   ("dense", lambda: RandomForestClassifier(500,max_features=0.1,n_jobs=-1,random_state=72)),
 "gradient boosting (dense)":     ("dense", lambda: HistGradientBoostingClassifier(max_iter=300,learning_rate=0.05,random_state=72)),
 "gradient boosting (LLM/MLM/gender only)": ("small", lambda: HistGradientBoostingClassifier(max_iter=200,learning_rate=0.05,max_depth=3,random_state=72)),
}
if __name__=="__main__":
    cache=[feats(f!=k,f==k) for k in range(5)]
    for name,(kind,mk) in MODELS.items():
        acc,tra=[],[]
        for k in range(5):
            A,B,dA,dB,sA,sB=cache[k]; Xa,Xb={"all":(A,B),"dense":(dA,dB),"small":(sA,sB)}[kind]
            m=mk().fit(Xa,y[f!=k]); acc.append((m.predict(Xb)==y[f==k]).mean()); tra.append((m.predict(Xa)==y[f!=k]).mean())
        print(f"{name:42s} test {np.mean(acc):.3f} (sd {np.std(acc):.3f})   train {np.mean(tra):.3f}",flush=True)
