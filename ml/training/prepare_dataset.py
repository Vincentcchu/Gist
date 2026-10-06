"""Turn a raw synthetic dataset into span-verified train/val/synthetic_test splits.

Two label formats, detected from the records:

- **v2** (current): term, description, polarity, opinion; no category.
      python ml/training/prepare_dataset.py --input ml/data/raw/hk_restaurant_absa.v2.jsonl
  writes `ml/data/processed_v2/`. Every v2 opinion is already an exact copy of one continuous
  stretch of the review (the generator's schema.validate() enforced it), so quads are checked
  and passed through unchanged. None of the v1 normalization below runs on them: splitting a v2
  opinion would break a span the labels deliberately keep whole.

- **v1** (kept so v1 results stay reproducible): ACOS = (Aspect, Category, Opinion,
  Sentiment). Reads `ml/data/raw/hk_restaurant_absa.jsonl` and writes `ml/data/processed/`.

The raw file's `opinion_words` is not a verbatim substring of the review in 18% of labels
(4,289 / 23,908) - it frequently stitches fragments together across a clause boundary. This
script enforces one invariant:

    Every emitted `opinion` is either the original label verbatim, or a punctuation-delimited
    fragment of it, verbatim. Nothing is synthesized, paraphrased, or extended.

Labels that cannot satisfy that are dropped, not repaired. Run with --help for options; the
stats report it prints is the audit trail for what was kept, split, and discarded.

Quad fields map onto AspectExtraction (app/models.py) as:
    term -> aspect_text, polarity -> sentiment, opinion -> span, category -> category (new column)
"""

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, NamedTuple

# Clause boundaries. Deliberately punctuation only - splitting on conjunctions like 又/同/and
# would cut inside single opinions ("肥瘦均勻又唔會太肥膩" is one judgment, not two).
SEPARATORS = re.compile(r"[，,。、；;！!？?…]+")

WHITESPACE = re.compile(r"\s+")
CJK = re.compile(r"[一-鿿]")

# A span containing one of these expresses ambivalence, and the ambivalence IS the opinion.
# Splitting it would hand both halves a polarity that belongs to neither:
#   "仲係咁貴唔平，不過都算值得" is labeled neutral precisely because it balances out.
CONTRAST_MARKERS = ("不過", "但係", "雖然", "可惜", "但", "however", "but ", "唔係話")

CATEGORIES = frozenset(
    {"Food", "Service", "Price", "Ambience", "Hygiene", "Waiting Time"}
)
POLARITIES = frozenset({"positive", "negative", "neutral"})

# Canonical ACOS sentinel for an implicit aspect or opinion. The synthetic data contains none,
# but the teacher pass on real reviews will, so the validator accepts it from the start.
NULL = "NULL"

DEFAULT_INPUT = Path("ml/data/raw/hk_restaurant_absa.jsonl")
DEFAULT_OUTDIRS = {"v1": Path("ml/data/processed"), "v2": Path("ml/data/processed_v2")}
V2_FIELDS = ("term", "description", "polarity", "opinion")


class Quad(NamedTuple):
    term: str
    category: str
    polarity: str
    opinion: str


def has_contrast(opinion: str) -> bool:
    return any(marker in opinion for marker in CONTRAST_MARKERS)


def split_opinion(opinion: str) -> list[str]:
    """Split an opinion into its separate judgments.

    Casual Cantonese typing often uses a space where punctuation would go
    ("脆咗少少 一般般" is two verdicts), so whitespace is a clause boundary too - but only
    when every resulting part contains a Chinese character. English opinions are
    space-separated by nature and must stay whole ("was quick", "good value").
    """
    fragments: list[str] = []
    for chunk in SEPARATORS.split(opinion):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p for p in WHITESPACE.split(chunk) if p]
        if len(parts) > 1 and all(CJK.search(p) for p in parts):
            fragments.extend(parts)
        else:
            fragments.append(chunk)
    return fragments


def locate_ordered(text: str, fragments: Iterable[str]) -> list[int] | None:
    """Return each fragment's start offset, searching left to right.

    Each fragment must appear at or after the end of the previous match, so fragments that
    occur in the text but in the wrong order are rejected rather than silently accepted.
    """
    starts: list[int] = []
    position = 0
    for fragment in fragments:
        start = text.find(fragment, position)
        if start == -1:
            return None
        starts.append(start)
        position = start + len(fragment)
    return starts


def normalize_label(text: str, label: dict[str, Any]) -> tuple[list[Quad], str, int]:
    """Turn one raw label into zero or more verbatim quads.

    Returns (quads, branch_name, fragments_dropped). The branch name is recorded so the
    stats report can be diffed against the numbers this algorithm was validated on.
    """
    term = label["term"]
    category = label["category"]
    polarity = label["polarity"]
    opinion = (label.get("opinion_words") or "").strip()

    if category not in CATEGORIES:
        raise ValueError(f"unexpected category: {category!r}")
    if polarity not in POLARITIES:
        raise ValueError(f"unexpected polarity: {polarity!r}")

    def quad(span: str) -> Quad:
        return Quad(term=term, category=category, polarity=polarity, opinion=span)

    if not opinion:
        return [], "drop_empty_opinion", 0

    if has_contrast(opinion):
        if opinion in text:
            return [quad(opinion)], "keep_contrast_whole", 0
        return [], "drop_contrast_nonverbatim", 0

    fragments = split_opinion(opinion)

    if len(fragments) == 1:
        if opinion in text:
            return [quad(opinion)], "keep_single", 0
        return [], "drop_single_paraphrase", 0

    if locate_ordered(text, fragments) is not None:
        return [quad(f) for f in fragments], "split_full", 0

    verbatim = [f for f in fragments if f in text]
    if verbatim:
        return (
            [quad(f) for f in verbatim],
            "split_partial",
            len(fragments) - len(verbatim),
        )
    return [], "drop_no_fragment_match", 0


def normalize_review(record: dict[str, Any], stats: Counter) -> dict[str, Any] | None:
    """Normalize one raw review. Returns None if no quad survived."""
    text = record["text"]
    quads: list[Quad] = []

    for label in record["aspects"]:
        label_quads, branch, dropped = normalize_label(text, label)
        stats[branch] += 1
        stats["fragments_dropped"] += dropped
        quads.extend(label_quads)

    if not quads:
        stats["reviews_dropped"] += 1
        return None

    # Order quads by where their opinion appears, so the generation target runs left to right
    # over the review. A monotonic target is easier to learn than the raw file's arbitrary
    # label order, and makes output diffs readable during error analysis.
    quads.sort(key=lambda q: (text.find(q.opinion), text.find(q.term)))

    meta = record.get("meta", {})
    return {
        "text": text,
        "quads": [q._asdict() for q in quads],
        "overall_sentiment": record.get("overall_sentiment"),
        "language_mode": meta.get("language_mode"),
        "style": meta.get("style"),
        "orthography": meta.get("orthography"),
        "emoji_density": meta.get("emoji_density"),
    }


def label_format(records: list[dict[str, Any]]) -> str:
    """'v2' if the raw labels carry a description, 'v1' if they carry a category.

    Looks at the first review that has labels: v2 reviews may have none (no judgment).
    """
    first = next(r["aspects"][0] for r in records if r["aspects"])
    return "v2" if "description" in first else "v1"


def check_v2_quad(text: str, quad: dict[str, Any]) -> str | None:
    """Why a v2 quad can't be trained on, or None. verify_dataset.py runs the full audit."""
    if set(quad) != set(V2_FIELDS):
        return f"keys {sorted(quad)}"
    if quad["polarity"] not in POLARITIES:
        return f"polarity {quad['polarity']!r}"
    for field in ("term", "opinion"):
        value = quad[field]
        if value != NULL and value not in text:
            return f"{field} not verbatim: {value!r}"
    return None


def prepare_v2_review(record: dict[str, Any]) -> dict[str, Any]:
    """One v2 review, quads unchanged and in their original order.

    Raises on an invalid quad instead of dropping it: every record passed the generator's
    validator, so a failure here means something is wrong with the file, not with one label.
    """
    text = record["text"]
    for quad in record["aspects"]:
        problem = check_v2_quad(text, quad)
        if problem:
            raise ValueError(f"invalid v2 quad ({problem}) in review: {text[:60]!r}")

    meta = record["meta"]
    return {
        "text": text,
        "quads": [
            {field: quad[field] for field in V2_FIELDS} for quad in record["aspects"]
        ],
        "overall_sentiment": record["overall_sentiment"],
        "language_mode": meta["language_mode"],
        "style": meta["style"],
        "orthography": meta["orthography"],
        "emoji_density": meta["emoji_density"],
        "venue_type": meta["venue_type"],
    }


def report_v2(examples: list[dict[str, Any]]) -> None:
    quads = [q for e in examples for q in e["quads"]]
    per_review = sorted(len(e["quads"]) for e in examples)

    print("\n=== v2 output (quads passed through unchanged) ===")
    print(f"  reviews                     {len(examples):6d}")
    print(f"  quads                       {len(quads):6d}")
    print(
        f"  quads/review                mean {len(quads)/len(examples):.1f}  "
        f"median {per_review[len(per_review)//2]}  max {per_review[-1]}"
    )
    print(f"  NULL term                   {sum(q['term'] == NULL for q in quads):6d}")
    print(
        f"  NULL opinion                {sum(q['opinion'] == NULL for q in quads):6d}"
    )
    empty = sum(1 for e in examples if not e["quads"])
    print(f"  reviews with no quads       {empty:6d}  ({100*empty/len(examples):.1f}%)")

    counts = Counter(q["polarity"] for q in quads)
    print("\n  polarity:")
    for value, count in counts.most_common():
        print(f"    {value:14s} {count:6d}  {100*count/len(quads):5.1f}%")
    descriptions = Counter(q["description"] for q in quads)
    print(f"\n  descriptions: {len(descriptions)} distinct; most common:")
    for value, count in descriptions.most_common(10):
        print(f"    {value:22s} {count:6d}")


def verify(examples: list[dict[str, Any]]) -> None:
    """Fail loudly if any span is not verbatim. This is the invariant, not a nicety."""
    violations = 0
    for example in examples:
        text = example["text"]
        for quad in example["quads"]:
            for field in ("term", "opinion"):
                value = quad[field]
                if value != NULL and value not in text:
                    violations += 1
    if violations:
        raise AssertionError(f"{violations} non-verbatim spans survived normalization")


def stratum(example: dict[str, Any], separate_empty: bool) -> tuple:
    """The group a review is split within: its (language_mode, style).

    With separate_empty (v2), reviews with no quads form one group of their own instead. There
    are few of them (3% of v2), so spread across the language/style groups each group would hold
    ~7 and round to zero val/test reviews; as one group, val and test each get their share.
    """
    if separate_empty and not example["quads"]:
        return ("no judgment",)
    return (example["language_mode"], example["style"])


def stratified_split(
    examples: list[dict[str, Any]],
    val_size: int,
    test_size: int,
    seed: int,
    separate_empty: bool = False,
) -> tuple[list, list, list]:
    """Split by stratum so val and test mirror the training distribution."""
    groups: dict[tuple, list] = defaultdict(list)
    for example in examples:
        groups[stratum(example, separate_empty)].append(example)

    rng = random.Random(seed)
    total = len(examples)
    val: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    train: list[dict[str, Any]] = []

    for key in sorted(groups, key=str):
        members = groups[key]
        rng.shuffle(members)
        n_val = round(val_size * len(members) / total)
        n_test = round(test_size * len(members) / total)
        val.extend(members[:n_val])
        test.extend(members[n_val : n_val + n_test])
        train.extend(members[n_val + n_test :])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def write_jsonl(path: Path, examples: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def report(examples: list[dict[str, Any]], stats: Counter, raw_labels: int) -> None:
    quads = [q for e in examples for q in e["quads"]]
    per_review = [len(e["quads"]) for e in examples]

    print("\n=== normalization branches ===")
    for branch in sorted(stats):
        print(f"  {branch:28s} {stats[branch]:6d}")

    print("\n=== output ===")
    growth = 100 * len(quads) / raw_labels - 100
    print(f"  raw labels                  {raw_labels:6d}")
    print(f"  quads                       {len(quads):6d}  ({growth:+.0f}%)")
    print(f"  reviews retained            {len(examples):6d}")
    print(
        f"  quads/review                mean {sum(per_review)/len(per_review):.1f}  "
        f"median {sorted(per_review)[len(per_review)//2]}  max {max(per_review)}"
    )

    for field in ("polarity", "category"):
        counts = Counter(q[field] for q in quads)
        print(f"\n  {field}:")
        for value, count in counts.most_common():
            print(f"    {value:14s} {count:6d}  {100*count/len(quads):5.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="default: ml/data/processed_v2 for v2 input, ml/data/processed for v1",
    )
    parser.add_argument("--val-size", type=int, default=250)
    parser.add_argument("--test-size", type=int, default=250)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="normalize and report, but write nothing",
    )
    args = parser.parse_args()

    records = read_jsonl(args.input)
    fmt = label_format(records)
    outdir = args.outdir or DEFAULT_OUTDIRS[fmt]
    print(f"{args.input}: {len(records)} reviews, {fmt} labels -> {outdir}/")

    if fmt == "v2":
        examples = [prepare_v2_review(record) for record in records]
        verify(examples)
        report_v2(examples)
    else:
        raw_labels = sum(len(r["aspects"]) for r in records)
        stats: Counter = Counter()
        examples = [
            example
            for record in records
            if (example := normalize_review(record, stats)) is not None
        ]
        verify(examples)
        report(examples, stats, raw_labels)

    if args.dry_run:
        print("\n(dry run - nothing written)")
        return

    train, val, test = stratified_split(
        examples,
        args.val_size,
        args.test_size,
        args.seed,
        separate_empty=fmt == "v2",
    )
    write_jsonl(outdir / "train.jsonl", train)
    write_jsonl(outdir / "val.jsonl", val)
    write_jsonl(outdir / "synthetic_test.jsonl", test)

    print(
        f"\nwrote train={len(train)} val={len(val)} synthetic_test={len(test)} "
        f"to {outdir}/"
    )


if __name__ == "__main__":
    main()
