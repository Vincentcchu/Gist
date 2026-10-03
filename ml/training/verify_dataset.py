"""Audit the processed splits. Independent of prepare_dataset.py by design.

Re-declares the expected schema rather than importing it, so a wrong constant in the
normalizer cannot validate itself. Exits non-zero on any failure.

    python ml/training/verify_dataset.py --dir ml/data/processed_v2      # v2 splits
    python ml/training/verify_dataset.py                                  # v1 splits
    python ml/training/verify_dataset.py --files ml/data/real/real_test.jsonl --allow-empty

The label format is detected from the quads: v2 quads carry a `description`, v1 quads a
`category`. The v2 rules match the generator's schema.validate() (Synethic_review, commit
a5cd94e) rule for rule, so a record that passed generation passes here and vice versa. They are
written out again here rather than imported, for the same reason as above.
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# --- v2: term, description, polarity, opinion ---------------------------------------------
EXPECTED_V2_KEYS = {"term", "description", "polarity", "opinion"}
# 1-4 lowercase words, letters only: "food taste", "revisit intent".
DESCRIPTION_FORMAT = re.compile(r"[a-z]+( [a-z]+){0,3}")
# Verdicts about the whole experience. Their term is always NULL.
GENERAL_DESCRIPTIONS = {"overall experience", "revisit intent", "recommendation"}
# Limits the generator held synthetic records to. Real reviews are labeled in full, so these
# apply to the synthetic splits only, never to --files (the hand-labeled gold).
MAX_QUADS = 14
MAX_TEXT_CHARS = 600

# --- v1: term, category, polarity, opinion ------------------------------------------------
EXPECTED_CATEGORIES = {
    "Food",
    "Service",
    "Price",
    "Ambience",
    "Hygiene",
    "Waiting Time",
}
EXPECTED_POLARITIES = {"positive", "negative", "neutral"}
EXPECTED_QUAD_KEYS = {"term", "category", "polarity", "opinion"}
NULL = "NULL"

SPLITS = ("train", "val", "synthetic_test")


def load(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def label_format(examples: list[dict[str, Any]]) -> str:
    """'v2' if the labeled quads carry a description, else 'v1'."""
    for example in examples:
        for quad in example["quads"]:
            return "v2" if isinstance(quad, dict) and "description" in quad else "v1"
    return "v2"  # nothing labeled yet: new files are v2


def contains_span(haystack: str, needle: str) -> bool:
    """Substring test that won't match an English word inside a longer one ("tea" in "instead")."""
    pattern = re.escape(needle)
    if needle[:1].isascii() and needle[:1].isalnum():
        pattern = r"(?<![A-Za-z0-9])" + pattern
    if needle[-1:].isascii() and needle[-1:].isalnum():
        pattern += r"(?![A-Za-z0-9])"
    return re.search(pattern, haystack) is not None


def check_v2_quad(quad: Any, text: str) -> str | None:
    """Return why a v2 quad is invalid, or None if it's fine."""
    if not isinstance(quad, dict) or set(quad) != EXPECTED_V2_KEYS:
        return (
            f"unexpected quad keys {sorted(quad) if isinstance(quad, dict) else quad!r}"
        )
    term, description = quad["term"], quad["description"]
    polarity, opinion = quad["polarity"], quad["opinion"]

    if polarity not in EXPECTED_POLARITIES:
        return f"bad polarity {polarity!r}"
    for field, value in (("term", term), ("opinion", opinion)):
        if not isinstance(value, str) or not value.strip():
            return f"empty {field}: {value!r}"
        if value != NULL and value not in text:
            return f"{field} not verbatim: {value!r}"
    if term == NULL and opinion == NULL:
        return "term and opinion both NULL: nothing is grounded in the text"
    if term != NULL and opinion != NULL and contains_span(opinion, term):
        return f"term inside opinion: {term!r} in {opinion!r}"
    if not isinstance(description, str) or not DESCRIPTION_FORMAT.fullmatch(
        description
    ):
        return f"bad description format: {description!r}"
    if description in GENERAL_DESCRIPTIONS and term != NULL:
        return f"general verdict with a named term: {term!r}"
    return None


def check_v2_example(example: dict[str, Any], synthetic: bool) -> list[str]:
    """Problems with one v2 review. Record limits apply only to synthetic data."""
    problems: list[str] = []
    quads = example["quads"]
    for quad in quads:
        reason = check_v2_quad(quad, example["text"])
        if reason:
            problems.append(reason)
    if not problems:
        keys = [
            tuple(quad[field] for field in sorted(EXPECTED_V2_KEYS)) for quad in quads
        ]
        if len(set(keys)) != len(keys):
            problems.append("duplicate quad")
    if synthetic:
        if len(quads) > MAX_QUADS:
            problems.append(
                f"{len(quads)} quads, over the synthetic cap of {MAX_QUADS}"
            )
        if len(example["text"]) > MAX_TEXT_CHARS:
            problems.append(
                f"text over {MAX_TEXT_CHARS} characters: {len(example['text'])}"
            )
    return problems


def check_split(
    name: str,
    examples: list[dict[str, Any]],
    allow_empty: bool = False,
    synthetic: bool = True,
) -> list[str]:
    failures: list[str] = []
    quads_seen = 0
    v2 = label_format(examples) == "v2"

    for index, example in enumerate(examples):
        text = example["text"]
        quads = example["quads"]
        where = f"{name}[{index}]"

        if not quads and not allow_empty:
            failures.append(f"{where}: empty quads")

        if v2:
            quads_seen += len(quads)
            failures.extend(
                f"{where}: {p}" for p in check_v2_example(example, synthetic)
            )
            continue

        for quad in quads:
            quads_seen += 1
            if set(quad) != EXPECTED_QUAD_KEYS:
                failures.append(f"{where}: unexpected quad keys {sorted(quad)}")
                continue
            if quad["category"] not in EXPECTED_CATEGORIES:
                failures.append(f"{where}: bad category {quad['category']!r}")
            if quad["polarity"] not in EXPECTED_POLARITIES:
                failures.append(f"{where}: bad polarity {quad['polarity']!r}")
            for field in ("term", "opinion"):
                value = quad[field]
                if value == NULL:
                    continue
                if not value:
                    failures.append(f"{where}: empty {field}")
                elif value not in text:
                    failures.append(f"{where}: {field} not verbatim: {value!r}")

    print(
        f"  {name:16s} {len(examples):5d} reviews  {quads_seen:6d} quads  "
        f"({'v2' if v2 else 'v1'} labels)"
    )
    return failures


def check_files(paths: list[Path], allow_empty: bool) -> None:
    failures: list[str] = []
    print("=== files ===")
    for path in paths:
        examples = load(path)
        failures.extend(check_split(path.name, examples, allow_empty, synthetic=False))
        if allow_empty:
            done = sum(1 for e in examples if e["quads"])
            print(f"  progress: {done}/{len(examples)} reviews labeled")
        if len({e["text"] for e in examples}) != len(examples):
            failures.append(f"{path.name}: duplicate review texts")
    report(failures, "PASS: all spans verbatim or NULL, schema valid")


def report(failures: list[str], success: str) -> None:
    if failures:
        print(f"\nFAILED: {len(failures)} problems")
        for failure in failures[:20]:
            print(f"  - {failure}")
        sys.exit(1)
    print(f"\n{success}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("ml/data/processed"),
        help="split directory: ml/data/processed (v1) or ml/data/processed_v2",
    )
    parser.add_argument(
        "--files",
        type=Path,
        nargs="+",
        help="check these files alone (e.g. the hand-labeled real test set) - schema and "
        "verbatim spans only, no cross-split leakage or stratification checks",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="with --files: don't fail on unlabeled reviews, so a half-finished labeling "
        "session surfaces only real span errors",
    )
    args = parser.parse_args()

    if args.files:
        check_files(args.files, args.allow_empty)
        return

    print("=== splits ===")
    data = {name: load(args.dir / f"{name}.jsonl") for name in SPLITS}
    failures: list[str] = []
    for name, examples in data.items():
        failures.extend(check_split(name, examples))

    print("\n=== leakage ===")
    texts = {name: {e["text"] for e in examples} for name, examples in data.items()}
    for left, right in (
        ("train", "val"),
        ("train", "synthetic_test"),
        ("val", "synthetic_test"),
    ):
        overlap = texts[left] & texts[right]
        print(f"  {left} ∩ {right}: {len(overlap)}")
        if overlap:
            failures.append(f"{len(overlap)} reviews shared between {left} and {right}")

    for name, examples in data.items():
        if len(texts[name]) != len(examples):
            failures.append(f"{name}: duplicate review texts within split")

    print("\n=== stratification (language_mode %) ===")
    for name, examples in data.items():
        counts = Counter(e["language_mode"] for e in examples)
        shares = "  ".join(
            f"{mode}={100*count/len(examples):.0f}%"
            for mode, count in sorted(counts.items())
        )
        print(f"  {name:16s} {shares}")

    report(failures, "PASS: all spans verbatim, no leakage, schema valid")


if __name__ == "__main__":
    main()
