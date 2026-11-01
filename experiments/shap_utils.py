"""Shared helpers for the SHAP analysis: gender lexicon and token -> word grouping."""
import re

MALE = set("""he him his himself man men boy boys male males father fathers dad dads daddy son sons husband husbands
brother brothers gentleman gentlemen guy guys grandfather grandfathers grandpa grandson uncle uncles nephew mr sir
king groom boyfriend boyfriends schoolboy schoolboys fiance stepfather""".split())
FEMALE = set("""she her hers herself woman women girl girls female females mother mothers mom moms mommy daughter
daughters wife wives sister sisters lady ladies gal grandmother grandmothers grandma granddaughter aunt aunts niece
mrs ms miss madam queen bride girlfriend girlfriends schoolgirl schoolgirls fiancee stepmother""".split())
GENDER = MALE | FEMALE

PUNCT = re.compile(r"^\W+$")


def key(word):
    """Normalized word: lowercase, letters/digits/apostrophes only ("Women," -> "women", "twin's" -> "twin's")."""
    return re.sub(r"[^a-z0-9']", "", word.lower()).strip("'")


def group_words(tokens):
    """Group subword tokens into words. Returns a list of (word_text, [token indices]).
    A new word starts at a token with leading whitespace, at punctuation, or right after punctuation."""
    groups = []
    for i, t in enumerate(tokens):
        new = (not groups or t[:1].isspace() or PUNCT.match(t.strip() or "x")
               or PUNCT.match(tokens[i - 1].strip() or "x"))
        if new:
            groups.append([t, [i]])
        else:
            groups[-1][0] += t
            groups[-1][1].append(i)
    return [(w.strip(), idx) for w, idx in groups]


def gender_of(keys):
    """'male' / 'female' by majority of gendered words, 'mixed' on a tie, 'none' if no gendered words."""
    m = sum(k in MALE for k in keys)
    f = sum(k in FEMALE for k in keys)
    return "none" if m == f == 0 else "male" if m > f else "female" if f > m else "mixed"
