"""Sentence embeddings for the classifiers, cached to data/embeddings/<name>.npy (row order = merged CSV).

bert:  bert-base-uncased, mean of last-layer token vectors over non-padding tokens (768-d).
mpnet: sentence-transformers/all-mpnet-base-v2, its own mean pooling + L2 normalisation (768-d).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
EMB = ROOT / "data" / "embeddings"
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def bert_mean_pool(sentences, model_name="bert-base-uncased", batch_size=32):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(DEVICE).eval()
    out = []
    for i in range(0, len(sentences), batch_size):
        batch = tok(sentences[i:i + batch_size], padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        hidden = model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).float()
        out.append(((hidden * mask).sum(1) / mask.sum(1)).cpu().numpy())
    return np.vstack(out)


def mpnet(sentences, model_name="sentence-transformers/all-mpnet-base-v2", batch_size=32):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device=DEVICE)
    return model.encode(sentences, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)


ENCODERS = {"bert": bert_mean_pool, "mpnet": mpnet}


def load_embeddings(name, sentences):
    """Return cached embeddings for `name`, computing them if missing or stale."""
    path = EMB / f"{name}.npy"
    if path.exists():
        X = np.load(path)
        if len(X) == len(sentences):
            return X
    EMB.mkdir(parents=True, exist_ok=True)
    X = ENCODERS[name](list(sentences))
    np.save(path, X)
    return X


if __name__ == "__main__":
    df = pd.read_csv(ROOT / "data" / "processed" / "merged_gender.csv")
    for name in ENCODERS:
        X = load_embeddings(name, df["sentence"])
        print(f"{name}: {X.shape} -> data/embeddings/{name}.npy")
