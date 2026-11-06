"""Interactive demo: a web page where you type sentences and the stereotype judge labels them, with live SHAP.

Pipeline:
  1. gender filter (Snowflake-L + logistic regression, models/gender_filter.npz)  ->  "not gender-related" if < 0.5
  2. 3-class Qwen2.5-14B LoRA judge (models/qwen14b_lora_3class)  ->  stereotype / anti-stereotype / neutral

/api/explain: word-level Shapley values for the judge's log-odds of its predicted class. A masked word is replaced by
"..." (as in experiments/shap_qwen.py). Up to EXACT_MAX words every subset is scored (exact Shapley values); longer
sentences use sampled permutations within a fixed budget of model calls.

The base model is loaded once (4-bit, ~10 GB VRAM). Stdlib HTTP server only.

  python demo/server.py                      # GPU (lisplab), serves http://localhost:8000
  python demo/server.py --port 8080 --gpu 1  # pick port / GPU
  python demo/server.py --mock               # no models; fake scores, for working on the page locally
  python demo/server.py --password-file pw.txt   # require a shared password (sent as X-Demo-Password)

From a laptop, reach it through an SSH tunnel:  ssh -N -L 8000:localhost:8000 <you>@lisplab-1...  ->  http://localhost:8000
"""
import argparse, hmac, json, math, re, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PAGE = Path(__file__).resolve().parent / "index.html"
ADAPTER = ROOT / "models" / "qwen14b_lora_3class"
FILTER = ROOT / "models" / "gender_filter.npz"
CLASSES = ["anti-stereotype", "stereotype", "neutral"]
MAX_SENTENCES, MAX_CHARS, MAX_WORDS = 20, 400, 40
EXACT_MAX, BUDGET, MASK = 9, 600, "..."


class Judge:
    def __init__(self, gpu):
        import torch
        from peft import PeftModel
        from sentence_transformers import SentenceTransformer
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
        self.torch, self.dev = torch, f"cuda:{gpu}"
        cfg = json.loads((ADAPTER / "adapter_config.json").read_text())
        self.tok = AutoTokenizer.from_pretrained(ADAPTER)
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        base = AutoModelForSequenceClassification.from_pretrained(
            cfg["base_model_name_or_path"], num_labels=3, quantization_config=q, dtype=torch.float16,
            device_map={"": gpu})
        base.config.pad_token_id = self.tok.pad_token_id
        self.model = PeftModel.from_pretrained(base, ADAPTER).eval()
        w = np.load(FILTER)
        self.coef, self.intercept = w["coef"], w["intercept"]
        self.emb = SentenceTransformer("Snowflake/snowflake-arctic-embed-l-v2.0", device=self.dev)

    def gender(self, texts):
        E = self.emb.encode(texts, normalize_embeddings=True)
        return 1 / (1 + np.exp(-(E @ self.coef + self.intercept)))

    def probs(self, texts, batch=64):
        torch, out = self.torch, []
        for i in range(0, len(texts), batch):
            enc = self.tok(texts[i:i + batch], padding=True, truncation=True, max_length=96,
                           return_tensors="pt").to(self.dev)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                out.append(torch.softmax(self.model(**enc).logits.float(), -1).cpu().numpy())
        return np.concatenate(out)


class MockJudge:
    """Deterministic fake scores so the page can be developed without a GPU. Not the real model."""
    G = re.compile(r"\b(she|he|her|his|women|men|woman|man|girls?|boys?|mother|father)\b", re.I)

    def gender(self, texts):
        return np.array([0.95 if self.G.search(t) else 0.1 for t in texts])

    def probs(self, texts):
        out = []
        for t in texts:
            r = np.random.default_rng(abs(hash(t.lower())) % 2**32).random(3) + [0, 0.3 * t.count("emotional"), 0]
            out.append(r / r.sum())
        return np.array(out)


def classify(judge, texts):
    pg, P = judge.gender(texts), judge.probs(texts)
    rows = []
    for t, g, p in zip(texts, pg, P):
        label = "not gender-related" if g < 0.5 else CLASSES[int(p.argmax())]
        rows.append({"text": t, "p_gender": float(g), "probs": dict(zip(CLASSES, map(float, p))), "label": label,
                     "judge_label": CLASSES[int(p.argmax())]})
    return rows


def masked_text(words, keep):
    out, prev_mask = [], False
    for w, k in zip(words, keep):
        if k:
            out.append(w); prev_mask = False
        elif not prev_mask:
            out.append(MASK); prev_mask = True
    return " ".join(out)


def explain(judge, text, seed=0):
    """Shapley values of each word for the judge's log-odds of its predicted class."""
    words = text.split()[:MAX_WORDS]
    n = len(words)
    full = judge.probs([" ".join(words)])[0]
    c = int(full.argmax())
    cache = {}

    def value(masks):
        todo = [m for m in dict.fromkeys(masks) if m not in cache]
        if todo:
            P = judge.probs([masked_text(words, [(m >> i) & 1 for i in range(n)]) for m in todo])
            p = np.clip(P[:, c], 1e-6, 1 - 1e-6)
            cache.update(zip(todo, np.log(p) - np.log1p(-p)))
        return [cache[m] for m in masks]

    phi = np.zeros(n)
    if n <= EXACT_MAX:
        allm = list(range(2 ** n))
        value(allm)
        for i in range(n):
            for m in allm:
                if not (m >> i) & 1:
                    s = bin(m).count("1")
                    w = math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n)
                    phi[i] += w * (cache[m | 1 << i] - cache[m])
        method = "exact"
    else:
        rng = np.random.default_rng(seed)
        perms = []
        while len(perms) * (n + 1) < BUDGET:  # antithetic pairs: a permutation and its reverse
            p = rng.permutation(n); perms += [p, p[::-1]]
        for p in perms:
            masks = [0]
            for i in p:
                masks.append(masks[-1] | 1 << int(i))
            v = value(masks)
            for j, i in enumerate(p):
                phi[i] += v[j + 1] - v[j]
        phi /= len(perms)
        method = f"sampled ({len(perms)} permutations)"
    base, fx = value([0, 2 ** n - 1])
    return {"words": words, "values": phi.round(4).tolist(), "target": CLASSES[c], "base": float(base),
            "fx": float(fx), "method": method, "calls": len(cache)}


def make_handler(judge, mock, password=None):
    lock = threading.Lock()  # one GPU, one request at a time

    class H(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/health":
                self._send(200, {"ok": True, "mock": mock, "password": password is not None})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path not in ("/api/classify", "/api/explain"):
                return self._send(404, {"error": "not found"})
            if password is not None and not hmac.compare_digest(
                    self.headers.get("X-Demo-Password", "").encode(), password.encode()):
                return self._send(401, {"error": "wrong password"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            except ValueError:
                return self._send(400, {"error": "bad JSON"})
            t0 = time.time()
            if self.path == "/api/explain":
                text = str(body.get("text", "")).strip()[:MAX_CHARS]
                if not text:
                    return self._send(400, {"error": "no text"})
                with lock:
                    out = explain(judge, text)
                return self._send(200, {**out, "mock": mock, "seconds": round(time.time() - t0, 2)})
            texts = [str(t).strip()[:MAX_CHARS] for t in body.get("texts", []) if str(t).strip()][:MAX_SENTENCES]
            if not texts:
                return self._send(400, {"error": "no text"})
            with lock:
                rows = classify(judge, texts)
            self._send(200, {"results": rows, "mock": mock, "seconds": round(time.time() - t0, 2)})

        def log_message(self, fmt, *args):
            print(f"[{self.log_date_time_string()}] {fmt % args}")

    return H


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to accept connections from other machines")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--mock", action="store_true", help="fake scores, no GPU (page development only)")
    ap.add_argument("--password-file", help="file holding a shared password visitors must enter")
    a = ap.parse_args()
    password = Path(a.password_file).read_text().strip() if a.password_file else None
    judge = MockJudge() if a.mock else Judge(a.gpu)
    if not a.mock:  # warm-up so the first visitor doesn't wait
        classify(judge, ["She is a nurse."]); explain(judge, "She is a nurse.")
    print(f"Serving on http://{a.host}:{a.port}" + ("  (MOCK scores)" if a.mock else ""))
    ThreadingHTTPServer((a.host, a.port), make_handler(judge, a.mock, password)).serve_forever()
