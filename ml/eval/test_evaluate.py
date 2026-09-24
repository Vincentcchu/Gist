"""Tests for the ACOS metrics. Every expected value here was worked out by hand.

pytest ml/eval/test_evaluate.py
"""

import json

import pytest

from evaluate import max_matching, relaxed_counts, score, spans_overlap


def quad(term: str, category: str, polarity: str, opinion: str) -> dict[str, str]:
    return {
        "term": term,
        "category": category,
        "polarity": polarity,
        "opinion": opinion,
    }


def run(text: str, gold: list[dict], predicted: list[dict] | str) -> dict:
    output = predicted if isinstance(predicted, str) else json.dumps(predicted)
    return score([{"text": text, "quads": gold}], [output])["views"]


# --- spans_overlap -----------------------------------------------------------------------


def test_identical_spans_overlap():
    assert spans_overlap("炒蛋凍晒", "凍晒", "凍晒")


def test_short_span_inside_longer_span_overlaps():
    # The boundary disagreement that motivated the relaxed view.
    assert spans_overlap("個炒蛋已經凍晒", "凍晒", "已經凍晒")


def test_null_matches_only_null():
    assert spans_overlap("好正！", "NULL", "NULL")
    assert not spans_overlap("好正！", "NULL", "好正")


def test_overlap_is_by_position_not_shared_characters():
    # 好正 and 好鹹 share 好 but sit in different parts of the review.
    assert not spans_overlap("叉燒好正。湯好鹹。", "好正", "好鹹")


def test_repeated_span_matches_through_any_occurrence():
    # 好正 occurs twice; only its second occurrence lies inside 甜品都好正.
    assert spans_overlap("好正！甜品都好正", "好正", "甜品都好正")
    # 好正！ (start) and 都好正 (end) never cover the same characters.
    assert not spans_overlap("好正！甜品都好正", "好正！", "都好正")


def test_longer_spans_touching_at_one_character_do_not_overlap():
    # Shared only 字; half of the shorter (5-character) span would be 3 characters.
    assert not spans_overlap("個湯得個咸字牛肉又韌", "湯得個咸字", "字牛肉又韌")


def test_spans_missing_from_the_review_never_overlap():
    assert not spans_overlap("叉燒好正", "好正", "唔在文中")


# --- max_matching -------------------------------------------------------------------------


def test_max_matching_finds_the_largest_pairing_regardless_of_order():
    # Prediction A can match gold X or Y; B can only match X. Greedy first-fit would give A->X
    # and leave B unmatched (1); the maximum pairing is A->Y, B->X (2).
    predicted = [{"id": "A"}, {"id": "B"}]
    gold = [{"id": "X"}, {"id": "Y"}]
    allowed = {("A", "X"), ("A", "Y"), ("B", "X")}
    assert (
        max_matching(predicted, gold, lambda p, g: (p["id"], g["id"]) in allowed) == 2
    )


# --- relaxed scoring ----------------------------------------------------------------------

TEXT = "個炒蛋已經凍晒，好失望。"
GOLD = [quad("炒蛋", "Food", "negative", "已經凍晒")]


def test_boundary_disagreement_is_a_strict_miss_but_a_relaxed_match():
    views = run(TEXT, GOLD, [quad("炒蛋", "Food", "negative", "凍晒")])
    assert views["full quad"]["f1"] == 0.0
    assert views["full quad (overlap)"]["f1"] == 1.0


def test_relaxed_match_still_requires_exact_polarity_and_category():
    wrong_polarity = run(TEXT, GOLD, [quad("炒蛋", "Food", "positive", "凍晒")])
    wrong_category = run(TEXT, GOLD, [quad("炒蛋", "Service", "negative", "凍晒")])
    assert wrong_polarity["full quad (overlap)"]["f1"] == 0.0
    assert wrong_category["full quad (overlap)"]["f1"] == 0.0


def test_each_gold_quad_is_credited_at_most_once():
    # One prediction overlaps both gold quads: 1 true positive, 1 missed gold quad.
    text = "個炒蛋已經凍晒又冇味"
    gold = [
        quad("炒蛋", "Food", "negative", "已經凍晒"),
        quad("炒蛋", "Food", "negative", "凍晒又冇味"),
    ]
    predicted = [quad("炒蛋", "Food", "negative", "凍晒")]
    assert relaxed_counts(
        text, predicted, gold, ("term", "category", "polarity", "opinion")
    ) == (
        1,
        0,
        1,
    )


def test_malformed_predictions_are_skipped_not_crashed_on():
    # For the full-quad view, a quad missing category/polarity/opinion is malformed.
    full = ("term", "category", "polarity", "opinion")
    assert relaxed_counts(TEXT, [{"term": "炒蛋"}, "not a dict"], GOLD, full) == (
        0,
        0,
        1,
    )


# --- strict scoring is unchanged ----------------------------------------------------------

REVIEW = "西多士一流，奶茶好滑，但係侍應好慢。"
REVIEW_GOLD = [
    quad("西多士", "Food", "positive", "一流"),
    quad("奶茶", "Food", "positive", "好滑"),
    quad("侍應", "Service", "negative", "好慢"),
]


def test_perfect_prediction_scores_one_everywhere():
    views = run(REVIEW, REVIEW_GOLD, REVIEW_GOLD)
    assert views["full quad"]["f1"] == 1.0
    assert views["full quad (overlap)"]["f1"] == 1.0


def test_mixed_prediction_matches_the_hand_computed_scores():
    # One right, one with the wrong polarity, one invented; 侍應 missed.
    predicted = [
        REVIEW_GOLD[0],
        quad("奶茶", "Food", "negative", "好滑"),
        quad("價錢", "Price", "positive", "抵"),
    ]
    views = run(REVIEW, REVIEW_GOLD, predicted)
    assert views["full quad"]["f1"] == pytest.approx(1 / 3)
    assert views["term"]["f1"] == pytest.approx(2 / 3)
    assert views["term+polarity"]["f1"] == pytest.approx(1 / 3)


def test_unparseable_output_scores_zero():
    views = run(REVIEW, REVIEW_GOLD, '[{"term":"西多士","category":"Food"')
    assert views["full quad"]["f1"] == 0.0
    assert views["full quad (overlap)"]["f1"] == 0.0
