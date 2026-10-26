import sys; sys.path.insert(0,"src")
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from finetune import BertFineTuned
from embed import DEVICE
EPOCHS=8
R="data/processed/"
df=pd.read_csv(R+"merged_gender.csv").merge(pd.read_csv(R+"folds.csv"),on="pair_id")
X,y,f=df.sentence.to_numpy(),df.label.to_numpy(),df.fold.to_numpy()
rows=[]
for k in range(5):
    tr,te=f!=k,f==k
    m=BertFineTuned(epochs=EPOCHS); torch.manual_seed(m.seed)
    from transformers import AutoTokenizer
    m.tok=AutoTokenizer.from_pretrained(m.model_name)
    m.model=AutoModelForSequenceClassification.from_pretrained(m.model_name,num_labels=2).to(DEVICE)
    loader=DataLoader(list(zip(X[tr],y[tr])),batch_size=m.batch_size,shuffle=True,generator=torch.Generator().manual_seed(m.seed),
                      collate_fn=lambda b:(m._encode([t for t,_ in b]),torch.tensor([l for _,l in b])))
    opt=torch.optim.AdamW(m.model.parameters(),lr=m.lr,weight_decay=0.01)
    steps=EPOCHS*len(loader); sched=get_linear_schedule_with_warmup(opt,int(0.1*steps),steps)
    for ep in range(1,EPOCHS+1):
        m.model.train(); losses=[]
        for batch,labels in loader:
            loss=m.model(**batch.to(DEVICE),labels=labels.to(DEVICE)).loss
            loss.backward(); opt.step(); sched.step(); opt.zero_grad(); losses.append(loss.item())
        tra=(m.predict(X[tr])==y[tr]).mean(); tea=(m.predict(X[te])==y[te]).mean()
        rows.append(dict(fold=k,epoch=ep,train_loss=np.mean(losses),train_acc=tra,test_acc=tea))
        print(rows[-1],flush=True)
out=pd.DataFrame(rows); out.to_csv("outputs/bert_ft_epoch_curve.csv",index=False)
print("\n",out.groupby("epoch")[["train_loss","train_acc","test_acc"]].mean().round(3))
