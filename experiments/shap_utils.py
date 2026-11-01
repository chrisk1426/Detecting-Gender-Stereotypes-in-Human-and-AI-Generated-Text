"""Shared helpers for the SHAP analysis: gender lexicon and token -> word grouping."""
import re

MALE = set("""he him his himself man men boy boys male males father fathers dad dads daddy son sons husband husbands
brother brothers gentleman gentlemen guy guys grandfather grandfathers grandpa grandson uncle uncles nephew mr sir
king groom boyfriend boyfriends schoolboy schoolboys fiance stepfather""".split())
FEMALE = set("""she her hers herself woman women girl girls female females mother mothers mom moms mommy daughter
daughters wife wives sister sisters lady ladies gal grandmother grandmothers grandma granddaughter aunt aunts niece
mrs ms miss madam queen bride girlfriend girlfriends schoolgirl schoolgirls fiancee stepmother""".split())
GENDER = MALE | FEMALE

# First names in CrowS-Pairs / StereoSet (all always-capitalized words in the data, classified by hand; unisex names
# such as Alex, Jamie, Kim and Sam are left out). Names are a gender cue, kept separate from the words above.
MALE_NAMES = set("""adam alan alexander andrew anse austin ben bill billy bob brad carl charles craig daniel david denny
drew dylan earl ed eric gary george greg harold jack jacob jake james jason jeff jeffrey jeremy jim jimmy joe john josh
julius kenny kevin kyle laquan lars mack mark martin marty mathew matt michael mike paul robert ron ronnie ryan scott
shane shawn silas steve stewart tim tom tommy tony wesley william""".split())
FEMALE_NAMES = set("""alena anna beth brenda camille candace candice carol carrie cathy celia charlene cheryl cynthia
emily greta heather jane jasmine jen jenna jennifer jenny jessica jill joyce judy juliette kassandra kassidy kate katherine
kathy kelly lara lena lisa liz lucy lynn maggie marie martha mary mattie maya megan melissa missy molly monica naomi
priya sally samantha sara sarah shannon sharon shelby shelly sue susan susie tammy veronica wendy""".split())
NAMES = MALE_NAMES | FEMALE_NAMES

PUNCT = re.compile(r"^\W+$")


def key(word):
    """Normalized word: lowercase, letters/digits/apostrophes only ("Women," -> "women", "twin's" -> "twin's")."""
    return re.sub(r"[^a-z0-9']", "", word.lower()).strip("'")


def group_words(tokens):
    """Group subword tokens into words. Returns a list of (word_text, [token indices]).
    A new word starts at a token with leading whitespace (Qwen: " put"), after a token with trailing whitespace
    (XLM-R / Snowflake: "put "), at punctuation, or right after punctuation."""
    groups = []
    for i, t in enumerate(tokens):
        new = (not groups or t[:1].isspace() or tokens[i - 1][-1:].isspace() or PUNCT.match(t.strip() or "x")
               or PUNCT.match(tokens[i - 1].strip() or "x"))
        if new:
            groups.append([t, [i]])
        else:
            groups[-1][0] += t
            groups[-1][1].append(i)
    return [(w.strip(), idx) for w, idx in groups]


def gender_of(keys):
    """'male' / 'female' by majority of gendered words and first names, 'mixed' on a tie, 'none' if neither."""
    m = sum(k in MALE or k in MALE_NAMES for k in keys)
    f = sum(k in FEMALE or k in FEMALE_NAMES for k in keys)
    return "none" if m == f == 0 else "male" if m > f else "female" if f > m else "mixed"
