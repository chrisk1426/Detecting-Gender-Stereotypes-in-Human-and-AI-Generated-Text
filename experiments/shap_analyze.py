"""Tables and charts from shap_qwen.py output (CPU).

  python experiments/shap_analyze.py outputs/shap/qwen14b_lora

Writes into the same folder:
  word_importance.csv / .png   global importance: mean |SHAP| per word (n >= MIN_N), signed mean, gender or not
  attribution_share.csv        share of each sentence's total |SHAP| on gender words / attribute words / other
  attr_by_gender.csv / .png    same attribute word in male vs female sentences: mean SHAP in each, and the gap
  crows_swap.csv               CrowS-Pairs (pairs differ only in gender): SHAP of the shared words in both versions
  interaction_summary.csv/.png gender x attribute masking interaction (see shap_qwen.py), by label and source
  summary.txt                  headline numbers
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shap_utils import GENDER, NAMES, PUNCT, gender_of, group_words, key

MIN_N = 5        # word must appear in this many sentences for word_importance
SWAP_SOURCES = ("crows", "gest")  # datasets whose pairs differ only in the gender word
MIN_N_GENDER = 3 # ... and this many sentences of each gender for attr_by_gender


def word_table(tokens):
    """One row per word per sentence: summed token SHAP, word type and the sentence's gender."""
    rows = []
    for sid, t in tokens.sort_values(["sid", "pos"]).groupby("sid"):
        vals = t.shap.to_numpy()
        words = group_words(t.token.tolist())
        keys = [key(w) for w, _ in words]
        g = gender_of(keys)
        for (w, idx), k in zip(words, keys):
            kind = ("punct" if not k or PUNCT.match(w) else "gender" if k in GENDER else "name" if k in NAMES
                    else "function" if k in ENGLISH_STOP_WORDS else "attribute")
            rows.append((sid, t.pair_id.iat[0], t.label.iat[0], t.source.iat[0], g, w, k, kind, vals[idx].sum(),
                         t.fx.iat[0]))
    return pd.DataFrame(rows, columns=["sid", "pair_id", "label", "source", "gender", "word", "key", "kind", "shap",
                                       "fx"])


def barh(ax, labels, vals, colors, title, xlabel):
    y = np.arange(len(labels))[::-1]
    ax.barh(y, vals, color=colors)
    ax.set_yticks(y, labels)
    ax.set_title(title, loc="left")
    ax.set_xlabel(xlabel)
    ax.axvline(0, color="#888", lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main(d):
    d = Path(d)
    read = lambda name, **kw: pd.concat([pd.read_csv(p, **kw) for p in sorted(d.glob(f"{name}*.csv"))
                                         if p.stat().st_size > 1] or [pd.DataFrame()])
    tokens = read("tokens", keep_default_na=False)
    W = word_table(tokens)
    out = []
    neutral = (tokens.label == -1).any()  # neutral-detector run: f = logit of "stereotype-related" vs neutral
    axis = "<- neutral   |   stereotype-related ->" if neutral else "<- anti-stereotype   |   stereotype ->"

    # 1. global word importance
    imp = (W[W.kind != "punct"].groupby("key")
           .agg(n=("sid", "nunique"), mean_abs=("shap", lambda s: s.abs().mean()), mean=("shap", "mean"),
                kind=("kind", "first"))
           .loc[lambda x: x.n >= MIN_N].sort_values("mean_abs", ascending=False))
    imp.to_csv(d / "word_importance.csv")
    top = imp.head(25)
    fig, ax = plt.subplots(1, 2, figsize=(11, 7))
    col = {"gender": "#c0392b", "name": "#e59866", "attribute": "#2e86c1", "function": "#aab7b8"}
    barh(ax[0], top.index, top.mean_abs, [col[k] for k in top.kind], "Top words: mean |SHAP|",
         "mean |SHAP| (log-odds)")
    signed = imp[imp.kind == "attribute"].sort_values("mean")
    s = pd.concat([signed.head(12), signed.tail(12)]).iloc[::-1]
    barh(ax[1], s.index, s["mean"], ["#c0392b" if v > 0 else "#2e86c1" for v in s["mean"]],
         "Attribute words: signed mean SHAP", axis)
    ax[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in col.values()], labels=list(col),
                 frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(d / "word_importance.png", dpi=150)
    out.append(f"Top 10 words by mean |SHAP|: {', '.join(f'{k} ({v:.2f})' for k, v in imp.mean_abs.head(10).items())}")

    # 2. how much of each sentence's attribution goes to gender vs attribute words
    share = W.assign(a=W.shap.abs()).pivot_table(index=["sid", "label", "source"], columns="kind", values="a",
                                                 aggfunc="sum", fill_value=0)
    share = share.div(share.sum(1), axis=0).reset_index()
    share.to_csv(d / "attribution_share.csv", index=False)
    has_g = share[(share.get("gender", 0) + share.get("name", 0)) > 0]
    m = has_g[[c for c in ("gender", "name", "attribute", "function", "punct") if c in has_g]].mean()
    out.append("Share of |SHAP| (sentences with a gender word or first name): " + ", ".join(f"{k} {v:.0%}" for k, v in m.items()))

    # 3. attribute x gender: does the same attribute word count differently for men and women?
    A = W[(W.kind == "attribute") & W.gender.isin(["male", "female"])]
    ag = A.pivot_table(index="key", columns="gender", values="shap", aggfunc=["mean", "count"])
    ag.columns = [f"{a}_{g}" for a, g in ag.columns]
    ag = ag.reindex(columns=["mean_female", "mean_male", "count_female", "count_male"]).dropna()
    ag = ag[(ag.count_male >= MIN_N_GENDER) & (ag.count_female >= MIN_N_GENDER)]
    ag["female_minus_male"] = ag.mean_female - ag.mean_male
    ag = ag.sort_values("female_minus_male")
    ag.to_csv(d / "attr_by_gender.csv")
    if len(ag):
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].scatter(ag.mean_male, ag.mean_female, s=12 + 3 * np.sqrt(ag.count_male + ag.count_female),
                      color="#2e86c1", alpha=0.6)
        lim = np.abs(ag[["mean_male", "mean_female"]].to_numpy()).max() * 1.1
        ax[0].plot([-lim, lim], [-lim, lim], color="#888", lw=0.8, ls="--")
        ax[0].axhline(0, color="#ccc", lw=0.6); ax[0].axvline(0, color="#ccc", lw=0.6)
        for k, r in pd.concat([ag.head(6), ag.tail(6)]).iterrows():
            ax[0].annotate(k, (r.mean_male, r.mean_female), fontsize=8)
        ax[0].set(xlabel="mean SHAP in male sentences", ylabel="mean SHAP in female sentences",
                  title="Same attribute, different gender\n(off the diagonal = gender-dependent)")
        s = pd.concat([ag.head(12), ag.tail(12)]).drop_duplicates().iloc[::-1]
        barh(ax[1], s.index, s.female_minus_male, ["#c0392b" if v > 0 else "#2e86c1" for v in s.female_minus_male],
             "Attribute SHAP gap", "<- more stereotypical for men   |   for women ->")
        fig.tight_layout()
        fig.savefig(d / "attr_by_gender.png", dpi=150)
        r = np.corrcoef(ag.mean_male, ag.mean_female)[0, 1] if len(ag) > 2 else np.nan
        out.append(f"Attribute words seen with both genders: {len(ag)}; corr(male SHAP, female SHAP) = {r:.2f} "
                   f"(1 = gender-independent, < 0 = flips with gender)")
        out.append("Most stereotypical for women: " + ", ".join(ag.index[::-1][:8]))
        out.append("Most stereotypical for men:   " + ", ".join(ag.index[:8]))

    # 4. CrowS-Pairs gender swaps: same words, other gender. Do shared words keep their SHAP?
    C = W[W.source.isin(SWAP_SOURCES) & (W.kind == "attribute")]
    rows = []
    for pid, p in C.groupby("pair_id"):
        if p.sid.nunique() != 2:
            continue
        st, an = (g.assign(i=g.groupby("key").cumcount()) for _, g in
                  sorted(p.groupby("sid"), key=lambda x: -x[1].label.iat[0]))  # stereotype version first
        if st.label.iat[0] == an.label.iat[0]:
            continue
        m = st.merge(an, on=["key", "i"], suffixes=("_stereo", "_anti"))
        rows.append(m.assign(pair_id=pid)[["pair_id", "key", "shap_stereo", "shap_anti", "gender_stereo",
                                           "gender_anti"]])
    cs = pd.concat(rows) if rows else pd.DataFrame()
    cs_sources = sorted(set(C.source))
    if len(cs):
        cs.to_csv(d / "crows_swap.csv", index=False)
        flip = (np.sign(cs.shap_stereo) != np.sign(cs.shap_anti)).mean()
        r = np.corrcoef(cs.shap_stereo, cs.shap_anti)[0, 1]
        out.append(f"Gender swaps ({', '.join(cs_sources)}): {cs.pair_id.nunique()} pairs, {len(cs)} shared attribute words; "
                   f"corr(SHAP stereo version, SHAP anti version) = {r:.2f}; sign flips {flip:.0%}; "
                   f"mean SHAP {cs.shap_stereo.mean():+.3f} (stereo) vs {cs.shap_anti.mean():+.3f} (anti)")

    # 5. masking interaction between gender words and top attribute words
    I = read("interactions")
    if len(I):
        I["label_name"] = I.label.map({1: "stereotype", 0: "anti-stereotype", -1: "neutral"})
        top1 = I[I.attr_rank == 0]
        summ = (top1.assign(gender_effect=top1.f_full - top1.f_no_gender, attr_effect=top1.f_full - top1.f_no_attr,
                            positive=top1.interaction > 0)
                .groupby(["source", "label_name"])
                .agg(n=("sid", "size"), interaction=("interaction", "mean"), frac_positive=("positive", "mean"),
                     gender_effect=("gender_effect", "mean"), attr_effect=("attr_effect", "mean")))
        summ.to_csv(d / "interaction_summary.csv")
        (I.sort_values("interaction", ascending=False)
          [["sentence", "label", "gender_words", "attr", "interaction", "f_full", "f_no_gender", "f_no_attr",
            "f_no_both"]].head(30).to_csv(d / "top_interactions.csv", index=False))
        fig, ax = plt.subplots(figsize=(7, 4))
        for lab, c in (("stereotype", "#c0392b"), ("anti-stereotype", "#2e86c1"), ("neutral", "#7f8c8d")):
            ax.hist(top1[top1.label_name == lab].interaction, bins=40, alpha=0.55, color=c, label=lab)
        ax.axvline(0, color="#555", lw=0.8)
        ax.set(xlabel="gender x top-attribute interaction (log-odds)", ylabel="sentences",
               title="Does the attribute count more when the gender word is there?")
        ax.legend(frameon=False)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        fig.savefig(d / "interaction.png", dpi=150)
        out.append("Interaction (top attribute, mean): " + "; ".join(
            f"{lab} {g.interaction.mean():+.2f} ({(g.interaction > 0).mean():.0%} > 0, n={len(g)})"
            for lab, g in top1.groupby("label_name")))

    (d / "summary.txt").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "outputs/shap/qwen14b_lora")
