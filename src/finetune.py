"""Fine-tuned BERT classifier with a scikit-learn style interface, so train.py can cross-validate it.

bert-base-uncased + a linear head on [CLS], all weights updated. Hyperparameters are the standard BERT
fine-tuning defaults (Devlin et al., 2019), not tuned on this data.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from embed import DEVICE


class BertFineTuned:
    def __init__(self, model_name="bert-base-uncased", epochs=4, lr=2e-5, batch_size=16, max_len=64, seed=72):
        self.model_name, self.epochs, self.lr = model_name, epochs, lr
        self.batch_size, self.max_len, self.seed = batch_size, max_len, seed
        self.classes_ = np.array([0, 1])

    def _encode(self, texts):
        return self.tok(list(texts), padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        self.tok = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=2).to(DEVICE)
        data = list(zip(X, y))
        loader = DataLoader(data, batch_size=self.batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(self.seed),
                            collate_fn=lambda b: (self._encode([t for t, _ in b]), torch.tensor([l for _, l in b])))
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        steps = self.epochs * len(loader)
        sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
        self.model.train()
        for _ in range(self.epochs):
            for batch, labels in loader:
                loss = self.model(**batch.to(DEVICE), labels=labels.to(DEVICE)).loss
                loss.backward()
                opt.step()
                sched.step()
                opt.zero_grad()
        return self

    @torch.no_grad()
    def predict_proba(self, X):
        self.model.eval()
        out = []
        for i in range(0, len(X), 64):
            logits = self.model(**self._encode(X[i:i + 64]).to(DEVICE)).logits
            out.append(torch.softmax(logits, -1).cpu().numpy())
        return np.vstack(out)

    def predict(self, X):
        return self.predict_proba(X).argmax(1)
