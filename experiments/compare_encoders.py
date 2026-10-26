"""Compare frozen encoders on the same 5 grouped folds: LR (same as train.py) and a random forest."""
import os, sys; sys.path.insert(0,"src")
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from embed import bert_mean_pool, DEVICE
CACHE=Path(__file__).parent/"emb"; CACHE.mkdir(exist_ok=True)
R="data/processed/"
df=pd.read_csv(R+"merged_gender.csv").merge(pd.read_csv(R+"folds.csv"),on="pair_id")
S,y,f=list(df.sentence),df.label.to_numpy(),df.fold.to_numpy()

def st(name):
    def enc(s):
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(name,device=DEVICE,trust_remote_code=True).encode(s,batch_size=32,normalize_embeddings=True)
    return enc
def openai_enc(model):
    def enc(s):
        from openai import OpenAI
        c=OpenAI(); out=[]
        for i in range(0,len(s),500): out+= [d.embedding for d in c.embeddings.create(model=model,input=s[i:i+500]).data]
        return np.array(out)
    return enc

ENC={
 "bert-base (current)":   lambda s: np.load("data/embeddings/bert.npy"),
 "mpnet (current)":       lambda s: np.load("data/embeddings/mpnet.npy"),
 "bert-large":            lambda s: bert_mean_pool(s,"bert-large-uncased"),
 "roberta-base":          lambda s: bert_mean_pool(s,"roberta-base"),
 "roberta-large":         lambda s: bert_mean_pool(s,"roberta-large"),
 "deberta-v3-base":       lambda s: bert_mean_pool(s,"microsoft/deberta-v3-base"),
 "snowflake-arctic-m-v1.5": st("Snowflake/snowflake-arctic-embed-m-v1.5"),
 "snowflake-arctic-l-v2.0": st("Snowflake/snowflake-arctic-embed-l-v2.0"),
}
if os.environ.get("OPENAI_API_KEY"):
    ENC["openai-3-small"]=openai_enc("text-embedding-3-small"); ENC["openai-3-large"]=openai_enc("text-embedding-3-large")

def cv(model,X):
    return np.mean([(model.fit(X[f!=k],y[f!=k]).predict(X[f==k])==y[f==k]).mean() for k in range(5)])
rows=[]
for name,enc in ENC.items():
    p=CACHE/(name.replace(" ","_").replace("(","").replace(")","")+".npy")
    try:
        X=np.load(p) if p.exists() else enc(S); np.save(p,X)
    except Exception as e:
        print(f"{name}: FAILED {e}",flush=True); continue
    lr=cv(make_pipeline(StandardScaler(),LogisticRegression(C=0.1,max_iter=5000)),X)
    rf=cv(RandomForestClassifier(n_estimators=300,max_features=0.1,random_state=72,n_jobs=-1),X)
    rows.append(dict(encoder=name,dim=X.shape[1],lr_acc=round(lr,3),rf_acc=round(rf,3))); print(rows[-1],flush=True)
out=pd.DataFrame(rows).sort_values("rf_acc",ascending=False)
out.to_csv("outputs/encoder_comparison.csv",index=False); print("\n",out.to_string(index=False))
