"""Export real OpenRice reviews as a hand-labeling template for the real test set.

    python ml/labeling/export_for_validation.py --n 50
    python ml/labeling/export_for_validation.py --reclean    # re-derive real_test's texts

Writes ml/data/real/real_test.jsonl - one review per line with `quads` empty, ready to label by
hand (see ml/data/real/README.md). Refuses to overwrite an existing file: once labeling starts,
that file holds hours of hand work.

Scraper defects shape what gets exported (measured on the 312 reviews scraped so far):

- 87 reviews - every 富臨飯店 review - are collapsed preview cards ending in "…查看更多"
  ("…see more"), not full text. Labeling a truncated review would bake the truncation into the
  test set, so they are excluded.
- Like/comment and photo counters are scraped in as lines holding only a number: at the end
  ("\n\n114\n15\n38", 48 reviews) and between paragraphs (51 of the 175 non-test reviews).
  clean_text strips both.

clean_text leaves everything else alone. Typed rating lines (味道：🔅🔅🔅🔅) and photo captions are
the reviewer's own text and get labeled (ml/data/real/README.md). Labels are exact copies of the
text, so the text must be final before labeling: --reclean refreshes real_test.jsonl's texts
with the current clean_text, and refuses once any review carries labels.

Sampling is stratified by length tertile, so short, medium and long reviews are equally
represented rather than following whatever the scrape happened to return.
"""

import argparse
import json
import random
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

TRUNCATION_MARKER = "查看更多"
TRAILING_COUNTERS = re.compile(r"(\s*\n\s*\d+)+\s*$")
# A line holding only a 1-4 digit number, anywhere in the review: counters, not prose. Prices
# keep their $ or 蚊 and ratings their /10, so they never match.
NUMBER_LINE = re.compile(r"^[ \t]*\d{1,4}[ \t]*(?:\n|$)", re.MULTILINE)

DEFAULT_DB = Path("app/local.db")
DEFAULT_OUT = Path("ml/data/real/real_test.jsonl")


def clean_with_counts(raw_text: str) -> tuple[str, Counter]:
    """Clean a review's text and count what each rule removed (in lines)."""
    counts: Counter = Counter()
    trailing = TRAILING_COUNTERS.search(raw_text)
    if trailing:
        counts["trailing counter lines"] = trailing.group(0).strip().count("\n") + 1
        raw_text = raw_text[: trailing.start()]
    text, removed = NUMBER_LINE.subn("", raw_text)
    counts["number-only lines mid-text"] = removed
    return text.strip(), counts


def clean_text(raw_text: str) -> str | None:
    """Return labelable text, or None if the review is a truncated preview."""
    if TRUNCATION_MARKER in raw_text:
        return None
    return clean_with_counts(raw_text)[0]


def load_reviews(db_path: Path, max_chars: int) -> list[dict[str, Any]]:
    query = """
        SELECT r.id, v.name, r.raw_text
        FROM reviews r JOIN venues v ON v.id = r.venue_id
        WHERE r.source = 'openrice'
        ORDER BY r.id
    """
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(query).fetchall()

    reviews = []
    for review_id, venue, raw_text in rows:
        text = clean_text(raw_text)
        if text is None or len(text) > max_chars:
            continue
        reviews.append({"review_id": review_id, "venue": venue, "text": text})
    return reviews


def stratified_sample(
    reviews: list[dict[str, Any]], n: int, seed: int
) -> list[dict[str, Any]]:
    """Draw n reviews, as evenly as possible from each length tertile."""
    ordered = sorted(reviews, key=lambda r: len(r["text"]))
    third = len(ordered) // 3
    tertiles = [ordered[:third], ordered[third : 2 * third], ordered[2 * third :]]

    rng = random.Random(seed)
    sample: list[dict[str, Any]] = []
    for index, tertile in enumerate(tertiles):
        take = n // 3 + (1 if index < n % 3 else 0)
        sample.extend(rng.sample(tertile, min(take, len(tertile))))
    rng.shuffle(sample)
    return sample


def reclean(path: Path, db_path: Path) -> None:
    """Re-derive the texts of an existing, still unlabeled export from the DB, in place.

    Same review_ids, same order, other fields untouched. Refuses if any review has quads:
    changing text under existing labels would break their exact-copy spans. Prints counts only,
    never review text, because this runs on the test set.
    """
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    labeled = sum(1 for row in rows if row.get("quads"))
    if labeled:
        raise SystemExit(
            f"{path}: {labeled} reviews already have quads - refusing to change their text, "
            "which would break the labels' spans."
        )

    with sqlite3.connect(db_path) as connection:
        raw = dict(connection.execute("SELECT id, raw_text FROM reviews").fetchall())

    totals: Counter = Counter()
    changed = 0
    for row in rows:
        text, counts = clean_with_counts(raw[row["review_id"]])
        totals.update(counts)
        changed += text != row["text"]
        row["text"] = text

    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(f"{path}: {len(rows)} reviews re-cleaned in place, {changed} texts changed")
    for rule, count in sorted(totals.items()):
        print(f"  {rule}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--reclean",
        action="store_true",
        help="re-derive the texts of the existing --out file with the current clean_text "
        "(same reviews and order); refuses once any review has quads",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=1000,
        help="skip longer reviews: costly to hand-label and past anything in training "
        "(excludes 3 of 225 today)",
    )
    args = parser.parse_args()

    if args.reclean:
        reclean(args.out, args.db)
        return

    if args.out.exists():
        raise SystemExit(
            f"{args.out} already exists and may hold hand labels - refusing to overwrite. "
            "Move it aside first if you really want a fresh template."
        )

    reviews = load_reviews(args.db, args.max_chars)
    sample = stratified_sample(reviews, args.n, args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for review in sample:
            handle.write(json.dumps({**review, "quads": []}, ensure_ascii=False) + "\n")

    lengths = sorted(len(r["text"]) for r in sample)
    print(f"usable reviews: {len(reviews)}  ->  sampled {len(sample)} to {args.out}")
    print(
        f"length (chars): min {lengths[0]}  median {lengths[len(lengths) // 2]}  "
        f"max {lengths[-1]}"
    )


if __name__ == "__main__":
    main()
