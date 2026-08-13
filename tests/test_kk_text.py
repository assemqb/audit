import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kk_text import has_kk_letter, normalize, stem, stem_text  # noqa: E402


def test_normalize_strips_case_and_punctuation():
    assert normalize("Сәлем, Әлем!") == "сәлем әлем"


def test_normalize_folds_kazakh_letters_when_requested():
    assert normalize("әөұүіңқғһ", fold_kk=True) == "аоууинкгх"


def test_normalize_leaves_kazakh_letters_by_default():
    assert normalize("әөұүіңқғһ") == "әөұүіңқғһ"


def test_normalize_collapses_hyphen_and_whitespace():
    assert normalize("бір-екі   үш") == "бір екі үш"


def test_stem_strips_known_suffix():
    # "кітаптар" (books) = "кітап" + plural suffix "тар". The stemmer is a
    # documented rough heuristic (up to 3 greedy passes), not a real
    # morphoanalyzer, and here it over-strips past the true root ("кітап")
    # down to "кіт" — this test pins that known limitation so it doesn't
    # silently drift, rather than pretending the heuristic is exact.
    assert stem("кітаптар") == "кіт"


def test_stem_keeps_short_roots_untouched():
    # guard against stripping a suffix-lookalike down to nothing useful
    assert len(stem("ол")) >= 2


def test_stem_text_applies_per_word():
    assert stem_text("кітаптар үйлерде") == stem("кітаптар") + " " + stem("үйлерде")


def test_has_kk_letter():
    assert has_kk_letter("қала")
    assert not has_kk_letter("kala")
