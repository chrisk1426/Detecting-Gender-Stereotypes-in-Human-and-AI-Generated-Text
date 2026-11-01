"""Collect the SHAP results of all runs into one JSON for the interpretability report page (docs/interpretability.html).

  python experiments/shap_report.py      # reads outputs/shap/<run>/, writes docs/interpretability_data.json
                                         # and embeds it in docs/interpretability.html
"""
import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shap_analyze import word_table

ROOT = Path(__file__).resolve().parent.parent
S = ROOT / "outputs" / "shap"
RUNS = {"qwen": ("qwen14b_lora", "Qwen judge", "training data (CrowS-Pairs + StereoSet)"),
        "qwen_gest": ("qwen14b_lora_gest", "Qwen judge (+GEST adapter)", "training data (CrowS-Pairs + StereoSet)"),
        "gest": ("qwen14b_lora__gest_pairs", "Qwen judge on GEST", "held out: GEST, never seen in training"),
        "neutral": ("neutral_lr", "Neutral detector", "training data + held-out neutral sets")}
LABELS = {1: "stereotype", 0: "anti-stereotype", -1: "neutral"}
KINDS = ["gender", "name", "attribute", "function", "punct"]
EXAMPLES = [  # (run, sentence, why it is shown)
    ("qwen", "Men are naturally timid.",
     "Each word alone looks stereotypical to the model; only the pairing makes it anti-stereotypical."),
    ("qwen", "Girls are hopeless at outdoor activities.",
     "A stereotype: the gender word and the attribute push the same way and reinforce each other."),
    ("qwen", "The male psychic was overly feminine.",
     "'feminine' counts toward stereotype for women and against it for men."),
    ("qwen", "The mother takes care of the plumbing at home.",
     "An occupation-style attribute: 'plumbing' flips the verdict once it is paired with 'mother'."),
    ("gest", None, "Held-out GEST pair: the same quoted sentence, only the speaker's gender changes."),
]


def r3(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 3)


def words_of(W, sid):
    w = W[W.sid == sid]
    return [[r.word, r3(r.shap), r.kind] for r in w.itertuples() if r.word]


def main():
    out = {"runs": {}, "by_label": {}, "position": {}, "hist": {}}
    W, I = {}, {}
    for k, (d, name, data) in RUNS.items():
        t = pd.read_csv(S / d / "tokens.csv", keep_default_na=False)
        W[k] = word_table(t)
        I[k] = pd.read_csv(S / d / "interactions.csv")
        w = W[k]
        tot = w.assign(a=w.shap.abs()).groupby(["sid", "kind"]).a.sum().unstack(fill_value=0)
        tot = tot[(tot.get("gender", 0) + tot.get("name", 0)) > 0]
        share = tot.div(tot.sum(1), axis=0).mean()
        top1 = I[k][I[k].attr_rank == 0]
        summary = (S / d / "summary.txt").read_text()
        out["runs"][k] = dict(
            name=name, data=data, sentences=int(t.sid.nunique()),
            share={c: r3(share.get(c, 0)) for c in KINDS},
            inter={LABELS[l]: dict(mean=r3(g.interaction.mean()), pos=r3((g.interaction > 0).mean()), n=len(g))
                   for l, g in top1.groupby("label")},
            summary=summary)
        ag = pd.read_csv(S / d / "attr_by_gender.csv")
        out["runs"][k]["corr_mf"] = r3(np.corrcoef(ag.mean_male, ag.mean_female)[0, 1])
        cs = pd.read_csv(S / d / "crows_swap.csv")
        out["runs"][k]["swap"] = dict(corr=r3(np.corrcoef(cs.shap_stereo, cs.shap_anti)[0, 1]),
                                      flip=r3((np.sign(cs.shap_stereo) != np.sign(cs.shap_anti)).mean()),
                                      pairs=int(cs.pair_id.nunique()))

        # importance by label: share of |SHAP| per word kind, and the top words
        bl = {}
        for l, g in w[w.kind != "punct"].groupby("label"):
            sh = g.assign(a=g.shap.abs()).groupby("kind").a.sum()
            imp = (g.groupby("key").agg(n=("sid", "nunique"), mean_abs=("shap", lambda s: s.abs().mean()),
                                         mean=("shap", "mean"), kind=("kind", "first"))
                   .loc[lambda x: x.n >= 4].sort_values("mean_abs", ascending=False).head(15))
            bl[LABELS[l]] = dict(share={c: r3(sh.get(c, 0) / sh.sum()) for c in KINDS if c != "punct"},
                                 top=[[i, r3(r.mean_abs), r3(r["mean"]), r.kind, int(r.n)] for i, r in imp.iterrows()])
        out["by_label"][k] = bl

        # position: share of the sentence's |SHAP| at each relative word position (10 bins), by label
        p = w[w.kind != "punct"].copy()
        p["i"] = p.groupby("sid").cumcount()
        p["n"] = p.groupby("sid").sid.transform("size")
        p = p[p.n >= 3]
        p["bin"] = np.minimum((p.i / (p.n - 1) * 10).astype(int), 9)
        p["a"] = p.shap.abs() / p.groupby("sid").shap.transform(lambda s: s.abs().sum())
        out["position"][k] = {LABELS[l]: [r3(v) for v in g.groupby("bin").a.mean().reindex(range(10)).to_numpy()]
                              for l, g in p.groupby("label")}
        kinds_pos = {}
        for kind in ("gender", "name", "attribute"):
            q = p[p.kind == kind]
            kinds_pos[kind] = [int(v) for v in q.groupby("bin").size().reindex(range(10), fill_value=0)]
        out["position"][k]["_kind_counts"] = kinds_pos

        edges = np.linspace(-40, 40, 33)
        out["hist"][k] = dict(edges=edges.tolist(), **{
            LABELS[l]: np.histogram(g.interaction.clip(-40, 40), bins=edges)[0].tolist()
            for l, g in top1.groupby("label")})

    # words that lean toward women / men
    out["gender_words"] = {}
    for k in ("qwen", "gest", "neutral"):
        ag = pd.read_csv(S / RUNS[k][0] / "attr_by_gender.csv")
        ag = ag[(ag.count_male >= 3) & (ag.count_female >= 3)].sort_values("female_minus_male")
        row = lambda r: [r.key, r3(r.mean_female), r3(r.mean_male), r3(r.female_minus_male),
                         int(r.count_female), int(r.count_male)]
        out["gender_words"][k] = dict(women=[row(r) for r in ag.iloc[::-1].head(15).itertuples()],
                                      men=[row(r) for r in ag.head(15).itertuples()])

    # gender x attribute combinations with the strongest interaction (top attribute per sentence)
    out["combos"] = {}
    for k in ("qwen", "gest"):
        t = I[k][I[k].attr_rank == 0].copy()
        t["gender"] = t.gender_words.str.split().str[0]
        rows = lambda d: [[r.gender_words, r.attr, r3(r.interaction), r.sentence, LABELS[r.label]]
                          for r in d.itertuples()]
        short = t[t.sentence.str.len() < 90]
        out["combos"][k] = dict(pos=rows(short.nlargest(10, "interaction")), neg=rows(short.nsmallest(10, "interaction")))

    # worked examples: word-level SHAP from the judge and from the neutral detector, plus the masking numbers
    ex = []
    sent = lambda k: W[k].groupby("sid").word.apply(lambda s: " ".join(s))
    for run, text, why in EXAMPLES:
        if text is None:  # GEST pair with the largest gender-driven swing
            g = I[run][I[run].attr_rank == 0]
            pid = g.groupby("pair_id").f_full.agg(lambda s: s.max() - s.min()).loc[
                lambda s: s.index.isin(g[g.sentence.str.len() < 70].pair_id)].idxmax()
            sids = sorted(g[g.pair_id == pid].sid.unique())
        else:
            m = I[run][(I[run].sentence == text) & (I[run].attr_rank == 0)]
            if m.empty:
                continue
            sid = int(m.sid.iat[0])
            pid = m.pair_id.iat[0]
            sids = sorted(I[run][I[run].pair_id == pid].sid.unique()) or [sid]
        items = []
        for sid in sids:
            m = I[run][(I[run].sid == sid) & (I[run].attr_rank == 0)]
            if m.empty:
                continue
            r = m.iloc[0]
            item = dict(sentence=r.sentence, label=LABELS[r.label], attr=r.attr, gender=r.gender_words,
                        f_full=r3(r.f_full), f_no_gender=r3(r.f_no_gender), f_no_attr=r3(r.f_no_attr),
                        f_no_both=r3(r.f_no_both), interaction=r3(r.interaction), qwen=words_of(W[run], sid))
            if run == "qwen":  # same sid in the neutral run (its first 1,518 rows are merged_gender.csv)
                item["neutral"] = words_of(W["neutral"], sid)
            items.append(item)
        ex.append(dict(run=run, why=why, items=items))
    out["examples"] = ex

    dest = ROOT / "docs" / "interpretability_data.json"
    blob = json.dumps(out, separators=(",", ":"))
    dest.write_text(blob)
    page = ROOT / "docs" / "interpretability.html"  # embed the data in the report page
    html = re.sub(r'(<script id="data" type="application/json">).*?(</script>)',
                  lambda m: m.group(1) + blob.replace("</", "<\\/") + m.group(2), page.read_text(), flags=re.S)
    page.write_text(html)
    print(f"wrote {dest} ({dest.stat().st_size / 1024:.0f} KB) and embedded it in {page}")


if __name__ == "__main__":
    main()
