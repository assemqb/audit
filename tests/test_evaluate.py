import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evaluate import bootstrap_wer_ci, classify  # noqa: E402


def test_classify_special_letters_only():
    # "қала" vs "кала" — only ә/ө/ұ/ү/і/ң/қ/ғ differ, root+suffix identical
    assert classify("қала", "кала") == "спецбуквы (ә/ө/ұ/ү/і/ң/қ/ғ)"


def test_classify_affix_only():
    # same root "кітап", different case suffix — a morphology miss, not a new word
    assert classify("кітапта", "кітаптан") == "аффикс (морфология)"


def test_classify_full_replacement():
    # neither word contains a special letter, so this can't be misfiled as
    # "спецбуквы" or "уход в русскую графику" — a genuine unrelated swap
    assert classify("бару", "алу") == "полная замена слова"


def test_classify_kk_letter_lost_beats_full_replacement():
    # "кітап" has "і" (a special letter) but no shared root with "теледидар" —
    # has_kk_letter fires before the code would otherwise call this a full
    # replacement, which is the documented priority order in classify()
    assert classify("кітап", "теледидар") == "уход в русскую графику"


def test_classify_digits_bucketed_separately():
    assert classify("5", "бес") == "числа/нормализация"


def test_bootstrap_ci_bounds_are_ordered_and_contain_point_estimate():
    refs = ["сәлем әлем", "бұл кітап жақсы", "мен үйге барамын", "ол оқушы"]
    hyps = ["сәлем әлем", "бұл кітап жаман", "мен үйге барамын", "ол оқушы емес"]
    lo, hi = bootstrap_wer_ci(refs, hyps, n_boot=200, seed=0)
    assert 0.0 <= lo <= hi <= 1.0


def test_bootstrap_ci_is_deterministic_given_a_seed():
    refs = ["сәлем әлем", "бұл кітап жақсы", "мен үйге барамын"]
    hyps = ["сәлем әлем", "бұл кітап жаман", "мен үйге барамын"]
    assert bootstrap_wer_ci(refs, hyps, n_boot=100, seed=1) == bootstrap_wer_ci(
        refs, hyps, n_boot=100, seed=1
    )


def test_bootstrap_ci_too_few_utterances():
    lo, hi = bootstrap_wer_ci(["сәлем"], ["сәлем"])
    assert lo != lo  # nan
