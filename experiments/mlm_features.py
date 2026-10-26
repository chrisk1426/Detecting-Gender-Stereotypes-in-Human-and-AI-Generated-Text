"""Engineered features per sentence, cached to scratchpad/feat/<name>.csv.

mlm gender agreement: mask each gender word, ask a masked LM log P(female form) - log P(male form)
  ("lean"), multiply by the word's actual gender (+1 F, -1 M), average over gender words.
  High = the content expects the gender the sentence actually has = reads as stereotypical.
negation count, sentiment P(positive).
"""
import sys, re, numpy as np, pandas as pd, torch
from pathlib import Path
sys.path.insert(0,"src")
from embed import DEVICE
OUT=Path(__file__).parent/"feat"; OUT.mkdir(exist_ok=True)
PAIRS=[("she","he"),("her","his"),("her","him"),("hers","his"),("herself","himself"),("woman","man"),("women","men"),
 ("girl","boy"),("girls","boys"),("female","male"),("mother","father"),("mothers","fathers"),("mom","dad"),("mommy","daddy"),
 ("daughter","son"),("daughters","sons"),("wife","husband"),("wives","husbands"),("sister","brother"),("sisters","brothers"),
 ("lady","gentleman"),("ladies","gentlemen"),("grandmother","grandfather"),("girlfriend","boyfriend"),("aunt","uncle"),
 ("mrs","mr"),("schoolgirl","schoolboy"),("queen","king"),("bride","groom"),("niece","nephew"),("grandma","grandpa"),("females","males")]
F2M={}; M2F={}
for f,m in PAIRS: F2M.setdefault(f,m); M2F.setdefault(m,f)
NEG={"not","no","never","nothing","nobody","none","neither","nor","cannot"}

@torch.no_grad()
def mlm_agreement(sentences, model_name):
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    tok=AutoTokenizer.from_pretrained(model_name); mdl=AutoModelForMaskedLM.from_pretrained(model_name).to(DEVICE).eval()
    vocab=tok.get_vocab()
    agree,lean_abs,n_g=[],[],[]
    for s in sentences:
        vals,leans=[],[]
        for mt in re.finditer(r"[A-Za-z']+",s):
            w=mt.group().lower()
            if w in F2M: fw,mw,sign=w,F2M[w],1
            elif w in M2F: fw,mw,sign=M2F[w],w,-1
            else: continue
            if "roberta" in model_name:
                key=(lambda x: x.capitalize()) if mt.start()==0 else (lambda x: "\u0120"+x)
            else:
                key=lambda x: x
            fw,mw=key(fw),key(mw)
            if fw not in vocab or mw not in vocab: continue
            masked=s[:mt.start()]+tok.mask_token+s[mt.end():]
            enc=tok(masked,return_tensors="pt").to(DEVICE)
            pos=(enc.input_ids[0]==tok.mask_token_id).nonzero()[0,0]
            lp=torch.log_softmax(mdl(**enc).logits[0,pos],-1)
            lean=(lp[vocab[fw]]-lp[vocab[mw]]).item()
            vals.append(lean*sign); leans.append(abs(lean))
        agree.append(np.mean(vals) if vals else 0.0); lean_abs.append(np.mean(leans) if leans else 0.0); n_g.append(len(vals))
    return pd.DataFrame({"agree":agree,"lean_abs":lean_abs,"n_gender_words":n_g})

def sentiment(sentences):
    from transformers import pipeline
    clf=pipeline("sentiment-analysis",model="distilbert-base-uncased-finetuned-sst-2-english",device=DEVICE)
    return pd.DataFrame({"pos":[r["score"] if r["label"]=="POSITIVE" else 1-r["score"] for r in clf(list(sentences),batch_size=32)]})

if __name__=="__main__":
    df=pd.read_csv("data/processed/merged_gender.csv"); S=df.sentence.tolist()
    pd.DataFrame({"neg":[sum(w in NEG or w.endswith("n't") for w in re.findall(r"[a-z']+",s.lower())) for s in S]}).to_csv(OUT/"negation.csv",index=False)
    for name in ["bert-base-uncased","bert-large-uncased","roberta-large"]:
        p=OUT/f"mlm_{name}.csv"
        if not p.exists(): mlm_agreement(S,name).to_csv(p,index=False)
        print(name,"done",flush=True)
    sentiment(S).to_csv(OUT/"sentiment.csv",index=False); print("sentiment done",flush=True)
