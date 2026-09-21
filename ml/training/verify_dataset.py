"""Audit the processed splits. Independent of prepare_dataset.py by design.

Re-declares the expected schema rather than importing it, so a wrong constant in the
normalizer cannot validate itself. Exits non-zero on any failure.

    python ml/training/verify_dataset.py
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

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


def check_split(
    name: str, examples: list[dict[str, Any]], allow_empty: bool = False
) -> list[str]:
    failures: list[str] = []
    quads_seen = 0

    for index, example in enumerate(examples):
        text = example["text"]
        quads = example["quads"]
        where = f"{name}[{index}]"

        if not quads and not allow_empty:
            failures.append(f"{where}: empty quads")

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

    print(f"  {name:16s} {len(examples):5d} reviews  {quads_seen:6d} quads")
    return failures


def check_files(paths: list[Path], allow_empty: bool) -> None:
    failures: list[str] = []
    print("=== files ===")
    for path in paths:
        examples = load(path)
        failures.extend(check_split(path.name, examples, allow_empty))
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
    parser.add_argument("--dir", type=Path, default=Path("ml/data/processed"))
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
