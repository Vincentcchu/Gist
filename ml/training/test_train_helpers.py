"""Tests for train.py's prediction assembly and longest-first selection. No GPU or model needed.

pytest ml/training/test_train_helpers.py
"""

import json

from prompt_format import split_for_inference
from train import assemble_predictions, longest_first


def test_one_chunk_review_keeps_the_raw_output():
    records = assemble_predictions([["好食"]], ["not even json"])
    assert records == [{"output": "not even json"}]


def test_multi_chunk_review_is_merged():
    a = {
        "term": "炒蛋",
        "description": "food texture",
        "polarity": "positive",
        "opinion": "好滑",
    }
    b = {
        "term": "NULL",
        "description": "revisit intent",
        "polarity": "positive",
        "opinion": "會再嚟",
    }
    chunked = [["短"], ["第一段", "第二段", "第三段"]]
    outputs = ["[]", json.dumps([a]), json.dumps([a, b]), "broken"]
    records = assemble_predictions(chunked, outputs)
    assert records[0] == {"output": "[]"}
    assert json.loads(records[1]["output"]) == [a, b]
    assert records[1]["chunks"] == 3 and records[1]["unparsed_chunks"] == 1


def test_a_long_review_goes_through_chunking_end_to_end():
    text = (
        "炒蛋好滑。" * 80 + "\n" + "奶茶好香。" * 80
    )  # 800 characters, two paragraphs
    chunked = [split_for_inference(text)]
    assert len(chunked[0]) == 2
    records = assemble_predictions(chunked, ["[]", "[]"])
    assert records[0]["chunks"] == 2


class CharTokenizer:
    """Stands in for a real tokenizer: one token per character."""

    eos_token_id = 0

    def apply_chat_template(self, messages, **_):
        return "".join(m["content"] for m in messages)

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text)))


def test_longest_first_sorts_by_token_length_and_skips_overlong():
    examples = [{"text": "a" * n, "quads": []} for n in (5, 50, 20, 5000)]
    ordered = longest_first(examples, CharTokenizer(), max_seq_len=1000)
    assert [len(e["text"]) for e in ordered[:3]] == [50, 20, 5]
    assert len(ordered[-1]["text"]) == 5000  # too long to train on: sorted last
