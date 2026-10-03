"""Tests for clean_text and the --reclean safety lock. Expected values are worked out by hand.

pytest ml/labeling/test_clean_text.py
"""

import json
import sqlite3

import pytest

from export_for_validation import clean_text, clean_with_counts, reclean


def test_number_line_between_paragraphs_is_removed():
    raw = "叉燒幾好食👍🏻\n\n1\n\n\n食到咁上下，侍應會問你飲咩"
    assert clean_text(raw) == "叉燒幾好食👍🏻\n\n\n\n食到咁上下，侍應會問你飲咩"


def test_trailing_counters_are_removed_and_counted():
    text, counts = clean_with_counts("西多士一流\n\n114\n15\n38")
    assert text == "西多士一流"
    assert counts["trailing counter lines"] == 3


def test_prices_ratings_and_numbers_inside_sentences_stay():
    raw = "$58茶餐\n凍飲加2蚊\n8/10\n等咗10分鐘\n2024"
    # "2024" alone on the last line is a trailing counter by the old rule, so it goes; the rest
    # carries a $, 蚊, a slash or words and stays.
    assert clean_text(raw) == "$58茶餐\n凍飲加2蚊\n8/10\n等咗10分鐘"


def test_rating_lines_and_captions_stay():
    raw = "味道：🔅🔅🔅🔅\n\n📍叉燒湯意粉\n\n炒蛋真係有香又嫩又滑👍"
    assert clean_text(raw) == raw


def test_truncated_preview_is_rejected():
    assert clean_text("好好食…查看更多") is None


def make_db(path, raw_texts: dict[int, str]) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE reviews (id INTEGER PRIMARY KEY, raw_text TEXT)"
        )
        connection.executemany("INSERT INTO reviews VALUES (?, ?)", raw_texts.items())


def write_rows(path, rows) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def test_reclean_rewrites_text_and_keeps_everything_else(tmp_path):
    db, out = tmp_path / "db.sqlite", tmp_path / "real_test.jsonl"
    make_db(db, {7: "好食\n\n1\n\n抵食", 3: "正"})
    write_rows(
        out,
        [
            {"review_id": 7, "venue": "A", "text": "old", "quads": []},
            {"review_id": 3, "venue": "B", "text": "正", "quads": []},
        ],
    )
    reclean(out, db)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["review_id"] for r in rows] == [7, 3]
    assert rows[0] == {
        "review_id": 7,
        "venue": "A",
        "text": "好食\n\n\n抵食",
        "quads": [],
    }
    assert rows[1]["text"] == "正"


def test_reclean_refuses_once_anything_is_labeled(tmp_path):
    db, out = tmp_path / "db.sqlite", tmp_path / "real_test.jsonl"
    make_db(db, {7: "好食"})
    labeled = {
        "review_id": 7,
        "venue": "A",
        "text": "好食",
        "quads": [
            {
                "term": "NULL",
                "description": "food taste",
                "polarity": "positive",
                "opinion": "好食",
            }
        ],
    }
    write_rows(out, [labeled])
    before = out.read_text()
    with pytest.raises(SystemExit):
        reclean(out, db)
    assert out.read_text() == before
