"""Score a fine-tuned adapter on ACOS quad extraction.

    python ml/eval/evaluate.py --adapter ml/outputs/qwen3-4b
    python ml/eval/evaluate.py --adapter ml/outputs/qwen3-4b --errors 10
    python ml/eval/evaluate.py --adapter ml/outputs/qwen3-4b --from-predictions   # re-score only

Generation is slow (an hour-plus for the 8B over every test set), so raw model outputs are cached
to <adapter>/eval/<dataset>.predictions.jsonl and scoring runs from that file. Re-scoring after a
metrics change never needs the model again.

## The metric

Exact-match micro precision / recall / F1 over full quads - the standard ACOS metric. A predicted
quad counts only if term, category, polarity and opinion all match a gold quad. Exact match is
well-defined here because term and opinion are verbatim spans of the input, not free text.

Partial views localize *where* the model fails:
- term                       did it find the right aspects at all?
- term + category            ...and file them correctly?
- term + polarity            ...and get the sentiment right?
- category + polarity        what the dashboard actually aggregates ("Food: 70% positive")
- full quad                  the headline number

An unparseable output predicts nothing, so every gold quad in that review is a miss.

Run it on synthetic_test and real_test side by side: the gap between them, not the synthetic
number, is the finding.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "ml" / "training"))

from prompt_format import (  # noqa: E402
    generation_metrics,
    parse_quads,
    prompt_contract,
    render_prompt,
)

FULL = ("term", "category", "polarity", "opinion")
VIEWS = {
    "term": ("term",),
    "term+category": ("term", "category"),
    "term+polarity": ("term", "polarity"),
    "category+polarity": ("category", "polarity"),
    "full quad": FULL,
}
DEFAULT_DATASETS = (
    Path("ml/data/processed/synthetic_test.jsonl"),
    Path("ml/data/real/real_test.jsonl"),
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def quad_keys(quads: list[Any], fields: tuple[str, ...]) -> Counter:
    """Multiset of quads projected onto `fields`. Malformed entries are skipped."""
    keys: Counter = Counter()
    for quad in quads:
        if isinstance(quad, dict) and all(isinstance(quad.get(f), str) for f in fields):
            keys[tuple(quad[f] for f in fields)] += 1
    return keys


def match_counts(predicted: Counter, gold: Counter) -> tuple[int, int, int]:
    """(true positives, false positives, false negatives) between two multisets."""
    tp = sum((predicted & gold).values())
    return tp, sum(predicted.values()) - tp, sum(gold.values()) - tp


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def predicted_quads(output: str) -> list[Any]:
    return parse_quads(output) or []


def score(examples: list[dict[str, Any]], outputs: list[str]) -> dict[str, Any]:
    totals = {name: [0, 0, 0] for name in VIEWS}
    by_category: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    by_polarity: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])

    for example, output in zip(examples, outputs):
        predicted = predicted_quads(output)
        for name, fields in VIEWS.items():
            counts = match_counts(
                quad_keys(predicted, fields), quad_keys(example["quads"], fields)
            )
            totals[name] = [a + b for a, b in zip(totals[name], counts)]

        full_pred = quad_keys(predicted, FULL)
        full_gold = quad_keys(example["quads"], FULL)
        for group, index in ((by_category, 1), (by_polarity, 2)):
            labels = {key[index] for key in full_pred} | {
                key[index] for key in full_gold
            }
            for label in labels:
                pred = Counter(
                    {k: v for k, v in full_pred.items() if k[index] == label}
                )
                gold = Counter(
                    {k: v for k, v in full_gold.items() if k[index] == label}
                )
                counts = match_counts(pred, gold)
                group[label] = [a + b for a, b in zip(group[label], counts)]

    polarity_f1 = {label: prf(*c)["f1"] for label, c in by_polarity.items()}
    return {
        "reviews": len(examples),
        "gold_quads": sum(len(e["quads"]) for e in examples),
        "views": {name: prf(*counts) for name, counts in totals.items()},
        "by_category": {label: prf(*c) for label, c in sorted(by_category.items())},
        "by_polarity": {label: prf(*c) for label, c in sorted(by_polarity.items())},
        "polarity_macro_f1": (
            sum(polarity_f1.values()) / len(polarity_f1) if polarity_f1 else 0.0
        ),
        "generation": generation_metrics(
            [(output, e["text"]) for e, output in zip(examples, outputs)]
        ),
    }


def worst_examples(
    examples: list[dict[str, Any]], outputs: list[str], n: int
) -> list[dict[str, Any]]:
    ranked = []
    for example, output in zip(examples, outputs):
        counts = match_counts(
            quad_keys(predicted_quads(output), FULL), quad_keys(example["quads"], FULL)
        )
        ranked.append((prf(*counts)["f1"], example, output))
    ranked.sort(key=lambda item: item[0])
    return [
        {"f1": f1, "text": e["text"], "gold": e["quads"], "output": out}
        for f1, e, out in ranked[:n]
    ]


def load_model(adapter: Path, model_path: str | None) -> tuple[Any, Any]:
    """Load base + adapter with MLX and refuse to run if the prompt contract has drifted."""
    from mlx_lm import load

    config = json.loads((adapter / "adapter_config.json").read_text())
    if model_path is None:
        # train_mlx.py stores local models relative to the repo root; hub ids pass through.
        local = REPO_ROOT / config["model"]
        model_path = str(local) if local.exists() else config["model"]
    model, tokenizer = load(model_path, adapter_path=str(adapter))

    saved = json.loads((adapter / "prompt_contract.json").read_text(encoding="utf-8"))
    current = prompt_contract(tokenizer)
    if saved != current:
        raise SystemExit(
            "Prompt contract mismatch: this tokenizer or prompt_format.py renders the prompt "
            "differently from training. Scores would reflect the drift, not the adapter."
        )
    return model, tokenizer


def generate_outputs(
    model: Any, tokenizer: Any, examples: list[dict[str, Any]], max_tokens: int
) -> list[str]:
    from mlx_lm import generate

    outputs = []
    for index, example in enumerate(examples, 1):
        prompt = tokenizer.encode(
            render_prompt(tokenizer, example["text"]), add_special_tokens=False
        )
        outputs.append(
            generate(
                model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False
            )
        )
        if index % 10 == 0 or index == len(examples):
            print(f"    generated {index}/{len(examples)}", flush=True)
    return outputs


def print_report(name: str, result: dict[str, Any]) -> None:
    gen = result["generation"]
    print(
        f"\n=== {name}: {result['reviews']} reviews, {result['gold_quads']} gold quads ==="
    )
    print(
        f"  parse rate {gen['json_parse_rate']:.3f}   verbatim {gen['span_verbatim_rate']:.3f}"
        f"   quads/review {gen['quads_per_review']:.1f}"
        f" (gold {result['gold_quads'] / max(result['reviews'], 1):.1f})"
    )
    print(f"  {'view':20s} {'P':>6s} {'R':>6s} {'F1':>6s}")
    for view, m in result["views"].items():
        print(f"  {view:20s} {m['precision']:6.3f} {m['recall']:6.3f} {m['f1']:6.3f}")
    print(
        "  full-quad F1 by category: "
        + "  ".join(f"{c} {m['f1']:.2f}" for c, m in result["by_category"].items())
    )
    print(
        "  full-quad F1 by polarity: "
        + "  ".join(f"{p} {m['f1']:.2f}" for p, m in result["by_polarity"].items())
        + f"   (macro {result['polarity_macro_f1']:.3f})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--data", type=Path, nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument(
        "--limit", type=int, default=None, help="first N reviews per dataset"
    )
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument(
        "--errors", type=int, default=0, help="print the N worst reviews"
    )
    parser.add_argument(
        "--from-predictions",
        action="store_true",
        help="re-score cached predictions instead of generating",
    )
    args = parser.parse_args()

    eval_dir = args.adapter / "eval"
    eval_dir.mkdir(exist_ok=True)
    model = tokenizer = None
    results = {}

    for path in args.data:
        if not path.exists():
            print(f"\n(skipping {path}: not found)")
            continue
        examples = read_jsonl(path)[: args.limit]
        unlabeled = sum(1 for e in examples if not e["quads"])
        if unlabeled:
            print(f"\n(skipping {path}: {unlabeled} reviews not labeled yet)")
            continue

        name = path.stem
        cache = eval_dir / f"{name}.predictions.jsonl"
        if args.from_predictions:
            outputs = [row["output"] for row in read_jsonl(cache)][: len(examples)]
            if len(outputs) < len(examples):
                # score() zips outputs with examples, so a short cache would silently score
                # the first N reviews while counting gold quads from all of them.
                raise SystemExit(
                    f"{cache.name} has {len(outputs)} predictions but {path.name} has "
                    f"{len(examples)} reviews. If the job was capped with --max-examples, "
                    f"score it with --limit {len(outputs)}."
                )
        else:
            if model is None:
                model, tokenizer = load_model(args.adapter, args.model_path)
            print(f"\n  generating {name} ({len(examples)} reviews)...")
            outputs = generate_outputs(model, tokenizer, examples, args.max_tokens)
            with cache.open("w", encoding="utf-8") as handle:
                for output in outputs:
                    handle.write(
                        json.dumps({"output": output}, ensure_ascii=False) + "\n"
                    )

        results[name] = score(examples, outputs)
        print_report(name, results[name])
        if args.errors:
            print(f"\n  --- {args.errors} worst ({name}) ---")
            for item in worst_examples(examples, outputs, args.errors):
                print(f"  F1 {item['f1']:.2f}  {item['text'][:120]!r}")
                print(
                    f"    gold:   {json.dumps(item['gold'], ensure_ascii=False)[:300]}"
                )
                print(f"    output: {item['output'][:300]!r}")

    (eval_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nresults written to {eval_dir / 'results.json'}")


if __name__ == "__main__":
    main()
