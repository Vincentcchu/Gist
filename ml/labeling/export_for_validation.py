"""Export real OpenRice reviews as a hand-labeling template for the real test set.

    python ml/labeling/export_for_validation.py --n 50

Writes ml/data/real/real_test.jsonl - one review per line with `quads` empty, ready to label by
hand (see ml/data/real/README.md). Refuses to overwrite an existing file: once labeling starts,
that file holds hours of hand work.

Two scraper defects shape what gets exported (measured on the 312 reviews scraped so far):

- 87 reviews - every 富臨飯店 review - are collapsed preview cards ending in "…查看更多"
  ("…see more"), not full text. Labeling a truncated review would bake the truncation into the
  test set, so they are excluded.
- 48 end with like/comment counters scraped in as text ("\n\n114\n15\n38"). Those are stripped;
  the stored text is the cleaned one, so hand labels stay verbatim against it.

Sampling is stratified by length tertile, so short, medium and long reviews are equally
represented rather than following whatever the scrape happened to return.
"""

import argparse
import json
import random
import re
import sqlite3
from pathlib import Path
from typing import Any

TRUNCATION_MARKER = "查看更多"
TRAILING_COUNTERS = re.compile(r"(\s*\n\s*\d+)+\s*$")

DEFAULT_DB = Path("app/local.db")
DEFAULT_OUT = Path("ml/data/real/real_test.jsonl")


def clean_text(raw_text: str) -> str | None:
    """Return labelable text, or None if the review is a truncated preview."""
    if TRUNCATION_MARKER in raw_text:
        return None
    return TRAILING_COUNTERS.sub("", raw_text).strip()


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--max-chars",
        type=int,
        default=1000,
        help="skip longer reviews: costly to hand-label and past anything in training "
        "(excludes 3 of 225 today)",
    )
    args = parser.parse_args()

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
