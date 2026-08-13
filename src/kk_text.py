"""
Kazakh text normalization plus a rough morphological stemmer.

Why this is a separate module: on Kazakh, a large chunk of WER isn't an
acoustic-model error at all. It's (a) a punctuation/case mismatch, or (b)
one wrong affix that makes the whole word count as a substitution. We want
to split these apart and show the numbers.
"""

import re
import unicodedata

# Kazakh-specific letters and the Russian look-alikes a model trained mostly
# on Russian most often collapses them into.
KK_FOLD_MAP = {
    "ә": "а",
    "ғ": "г",
    "қ": "к",
    "ң": "н",
    "ө": "о",
    "ұ": "у",
    "ү": "у",
    "һ": "х",
    "і": "и",
}

PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
SPACE_RE = re.compile(r"\s+")

# Most frequent Kazakh affixes (case, plural, possessive, predicate, some
# verb forms). Deliberately incomplete and rough: this is a heuristic for
# classifying errors, not a real morphological analyzer.
SUFFIXES = [
    "дікі", "тікі", "нікі",
    "дағы", "дегі", "тағы", "тегі", "ндағы", "ндегі",
    "ларда", "лерде", "дарда", "дерде", "тарда", "терде",
    "лардан", "лерден", "дардан", "дерден", "тардан", "терден",
    "ларға", "лерге", "дарға", "дерге", "тарға", "терге",
    "лардың", "лердің", "дардың", "дердің", "тардың", "тердің",
    "ларды", "лерді", "дарды", "дерді", "тарды", "терді",
    "лар", "лер", "дар", "дер", "тар", "тер",
    "ның", "нің", "дың", "дің", "тың", "тің",
    "мын", "мін", "сың", "сің", "быз", "біз", "сыз", "сіз",
    "нан", "нен", "дан", "ден", "тан", "тен",
    "мыз", "міз", "ңыз", "ңіз",
    "ған", "ген", "қан", "кен",
    "атын", "етін", "йтын", "йтін",
    "ады", "еді", "йды", "йді",
    "ып", "іп", "п",
    "қа", "ке", "ға", "ге", "на", "не", "а", "е",
    "ды", "ді", "ты", "ті", "ны", "ні",
    "да", "де", "та", "те", "нда", "нде",
    "ым", "ім", "м", "ың", "ің", "ң", "сы", "сі", "ы", "і",
]
SUFFIXES = sorted(set(SUFFIXES), key=len, reverse=True)

MIN_STEM = 3


def normalize(text: str, fold_kk: bool = False) -> str:
    """Bring text to a comparable form: NFC, lowercase, no punctuation.

    fold_kk=True additionally folds ә/ө/ұ/ү/і/ң/қ/ғ into their Russian
    look-alikes. That gives a "lenient" WER, and the difference from the
    regular one shows how many errors are down to Kazakh-specific graphemes.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.lower().replace("ё", "е").replace("-", " ")
    text = PUNCT_RE.sub(" ", text)
    if fold_kk:
        text = "".join(KK_FOLD_MAP.get(ch, ch) for ch in text)
    return SPACE_RE.sub(" ", text).strip()


def stem(word: str) -> str:
    """Roughly strip affixes. Iterative, up to 3 passes."""
    for _ in range(3):
        for suf in SUFFIXES:
            if word.endswith(suf) and len(word) - len(suf) >= MIN_STEM:
                word = word[: -len(suf)]
                break
        else:
            break
    return word


def stem_text(text: str) -> str:
    return " ".join(stem(w) for w in text.split())


def has_kk_letter(word: str) -> bool:
    return any(ch in KK_FOLD_MAP for ch in word)
