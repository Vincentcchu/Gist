"""Tests for the v2 data path: verify_dataset's rules, prepare_dataset's pass-through, and the
long-review chunking in prompt_format. Expected values are worked out by hand.

pytest ml/training/test_data_v2.py
"""

import copy
import json
import sys
from pathlib import Path

import pytest

from prepare_dataset import prepare_v2_review
from prompt_format import QUAD_FIELDS, build_target, merge_quads, split_for_inference
from verify_dataset import check_v2_example

GENERATOR_REPO = Path("/Users/vincent_c/projects/Synethic_review")
V2_RAW = Path("ml/data/raw/hk_restaurant_absa.v2.jsonl")

TEXT = "奶茶好滑，但下次都會再嚟。個waiter好黑面"


def quad(term: str, description: str, polarity: str, opinion: str) -> dict[str, str]:
    return {
        "term": term,
        "description": description,
        "polarity": polarity,
        "opinion": opinion,
    }


def good_example() -> dict:
    return {
        "text": TEXT,
        "quads": [
            quad("奶茶", "drink texture", "positive", "好滑"),
            quad("NULL", "revisit intent", "positive", "下次都會再嚟"),
            quad("waiter", "staff attitude", "negative", "好黑面"),
        ],
    }


# --- verify_dataset: one failing example per rule ------------------------------------------


def test_valid_example_passes():
    assert check_v2_example(good_example(), synthetic=True) == []


@pytest.mark.parametrize(
    "mutate, reason",
    [
        (lambda q: q.update(polarity="good"), "bad polarity"),
        (lambda q: q.update(term="凍檸茶"), "term not verbatim"),
        (lambda q: q.update(opinion="好甜"), "opinion not verbatim"),
        (lambda q: q.update(term="NULL", opinion="NULL"), "both NULL"),
        (lambda q: q.update(opinion="奶茶好滑"), "term inside opinion"),
        (lambda q: q.update(description="Drink Texture"), "bad description format"),
        (
            lambda q: q.update(description="drink texture of the milk tea"),
            "bad description format",
        ),
        (
            lambda q: q.update(description="overall experience"),
            "general verdict with a named term",
        ),
        (lambda q: q.update(category="Food"), "unexpected quad keys"),
        (lambda q: q.update(term=" "), "empty term"),
    ],
)
def test_each_rule_rejects(mutate, reason):
    example = good_example()
    mutate(example["quads"][0])
    problems = check_v2_example(example, synthetic=True)
    assert any(reason in p for p in problems), problems


def test_english_term_inside_a_longer_word_is_not_inside_the_opinion():
    # "tea" appears in "instead" but not as a word, so the opinion doesn't contain the term.
    example = {
        "text": "tea was weak, they gave us water instead",
        "quads": [quad("tea", "drink taste", "negative", "they gave us water instead")],
    }
    assert check_v2_example(example, synthetic=True) == []


def test_duplicate_quad_rejected():
    example = good_example()
    example["quads"].append(dict(example["quads"][0]))
    assert "duplicate quad" in check_v2_example(example, synthetic=True)


def test_record_limits_apply_to_synthetic_only():
    long_text = "好" * 601
    example = {
        "text": long_text,
        "quads": [quad("NULL", "food taste", "positive", "好")],
    }
    assert any("over 600" in p for p in check_v2_example(example, synthetic=True))
    assert check_v2_example(example, synthetic=False) == []


@pytest.mark.skipif(
    not (GENERATOR_REPO / "schema.py").exists() or not V2_RAW.exists(),
    reason="needs the generator repo and the v2 raw file",
)
def test_agrees_with_generator_validate_on_the_full_v2_file():
    sys.path.insert(0, str(GENERATOR_REPO))
    sys.dont_write_bytecode = (
        True  # the generator repo is read-only from here: no __pycache__
    )
    import schema  # the generator's validator

    records = [json.loads(line) for line in V2_RAW.open(encoding="utf-8")]
    for record in records:
        ours = check_v2_example(
            {"text": record["text"], "quads": record["aspects"]}, True
        )
        assert schema.validate(record)[0] == (ours == []), record["text"][:40]


# --- prepare_dataset: v2 quads pass through unchanged --------------------------------------


def raw_record() -> dict:
    return {
        "text": TEXT,
        "aspects": good_example()["quads"],
        "overall_sentiment": "mixed",
        "meta": {
            "language_mode": "cantonese_colloquial",
            "style": "short_note",
            "orthography": "typical",
            "emoji_density": "none",
            "venue_type": "cha chaan teng",
        },
    }


def test_prepare_passes_quads_through_unchanged():
    record = raw_record()
    example = prepare_v2_review(copy.deepcopy(record))
    assert example["quads"] == record["aspects"]
    assert example["venue_type"] == "cha chaan teng"
    assert example["language_mode"] == "cantonese_colloquial"


def test_prepare_does_not_split_opinions():
    # v1 would have split this at the comma; v2 keeps the one continuous stretch.
    record = raw_record()
    record["aspects"] = [quad("NULL", "revisit intent", "positive", "但下次都會再嚟")]
    assert prepare_v2_review(record)["quads"][0]["opinion"] == "但下次都會再嚟"


def test_prepare_raises_on_a_non_verbatim_span():
    record = raw_record()
    record["aspects"][0]["opinion"] = "好甜"
    with pytest.raises(ValueError):
        prepare_v2_review(record)


# --- prompt_format: target and chunking -----------------------------------------------------


def test_target_uses_v2_fields_in_order():
    target = json.loads(
        build_target([quad("奶茶", "drink texture", "positive", "好滑")])
    )
    assert QUAD_FIELDS == ("term", "description", "polarity", "opinion")
    assert list(target[0]) == list(QUAD_FIELDS)


def test_short_review_is_one_chunk():
    assert split_for_inference("好食", max_chars=600) == ["好食"]


def test_paragraphs_are_packed_greedily():
    # Three 4-character paragraphs with max 10: the first two fit together (4 + 1 + 4 = 9).
    text = "aaaa\nbbbb\ncccc"
    assert split_for_inference(text, max_chars=10) == ["aaaa\nbbbb", "cccc"]


def test_long_paragraph_is_cut_at_sentence_ends():
    text = "炒蛋好滑。多士好脆。奶茶好甜。"
    assert split_for_inference(text, max_chars=10) == [
        "炒蛋好滑。多士好脆。",
        "奶茶好甜。",
    ]


def test_decimal_point_is_not_a_sentence_end():
    text = "rated 3.5 out of 5. good"
    chunks = split_for_inference(text, max_chars=20)
    assert chunks == ["rated 3.5 out of 5.", "good"]


def test_overlong_sentence_is_cut_at_max_chars():
    assert split_for_inference("一" * 25, max_chars=10) == [
        "一" * 10,
        "一" * 10,
        "一" * 5,
    ]


def test_chunks_are_slices_within_the_limit():
    text = "第一段好長" * 30 + "。\n\n" + "第二段" * 50 + "！\n" + "最後一句好食"
    chunks = split_for_inference(text, max_chars=100)
    assert all(len(c) <= 100 and c in text for c in chunks)
    # Nothing lost: only line breaks at chunk edges are stripped.
    assert "".join(chunks).replace("\n", "") == text.replace("\n", "")


def test_merge_drops_exact_duplicates_and_counts_unparseable_chunks():
    a = quad("NULL", "revisit intent", "positive", "下次都會再嚟")
    b = quad("奶茶", "drink texture", "positive", "好滑")
    outputs = [json.dumps([a, b]), "not json", json.dumps([a])]
    merged, unparsed = merge_quads(outputs)
    assert merged == [a, b]
    assert unparsed == 1


# --- reviews with no judgment: an empty quads list is valid in v2 ---------------------------


def test_empty_review_passes_in_v2():
    from verify_dataset import check_split

    examples = [good_example(), {"text": "叫咗西多士同奶茶", "quads": []}]
    assert check_split("gold", examples, synthetic=False) == []
    assert check_split("synthetic", examples, synthetic=True) == []


def test_empty_review_still_fails_in_v1():
    from verify_dataset import check_split

    v1_quad = {
        "term": "奶茶",
        "category": "Food",
        "polarity": "positive",
        "opinion": "好滑",
    }
    examples = [{"text": TEXT, "quads": [v1_quad]}, {"text": "正", "quads": []}]
    assert any("empty quads" in f for f in check_split("v1", examples))


def test_format_detection_skips_reviews_with_no_labels():
    from prepare_dataset import label_format

    empty = dict(raw_record(), aspects=[])
    assert label_format([empty, raw_record()]) == "v2"


def test_no_judgment_target_is_an_empty_array():
    assert build_target([]) == "[]"
