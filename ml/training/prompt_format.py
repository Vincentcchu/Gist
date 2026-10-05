"""The single definition of the prompt, target format, tokenization and long-review chunking.

Both trainers (train.py for PyTorch/SageMaker, train_mlx.py for local MLX), generation-eval and
serving import from here. If the token sequence drifts between training and inference - or
between the two frameworks - the adapter degrades silently in a way that looks like a bad
fine-tune, so there is exactly one copy of it.

The system prompt is deliberately terse. A fine-tuned model learns the task from thousands of
examples; the prompt only has to carry the hard constraints, and every token of it is paid on
every training example and every inference call. The labeling conventions themselves live in
ml/data/real/README.md; labeling happens outside this repo.
"""

import json
import re
from typing import Any

# v2 label format: term, description, polarity, opinion. There is no category; categories are
# applied after the model runs, from the description. v1 adapters (term, category, polarity,
# opinion) keep their own prompt in the prompt_contract.json saved next to them, and their
# cached predictions are scored without re-rendering, so changing this prompt doesn't touch them.
#
# Kept terse on purpose (see the module docstring), and within the generator's token budget:
# the synthetic records were sized for max_seq_len 1280 with <=200 tokens of system prompt plus
# chat template (Synethic_review/schema.py, TOKEN_BUDGET). This one measures 158.
SYSTEM_PROMPT = """Extract opinion quads from a Hong Kong restaurant review (Cantonese, English, or mixed). Output a JSON array of {"term","description","polarity","opinion"} objects.
- term, opinion: exact substrings of the review, never translated or reworded. Use "NULL" when implied but not stated.
- description: the quality judged, lowercase English, 1-4 words ("food taste", "queue time", "revisit intent").
- polarity: positive, negative, or neutral.
- One quad per judgment. General verdicts (overall experience, revisit intent, recommendation) take term "NULL".
Output only the JSON array, [] if the review judges nothing."""

QUAD_FIELDS = ("term", "description", "polarity", "opinion")

# Training reviews are at most 600 characters. Longer real reviews are cut into pieces of at
# most this size, run one by one, and their quads merged (split_for_inference / merge_quads).
MAX_CHUNK_CHARS = 600


def build_messages(text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]


def build_target(quads: list[dict[str, Any]]) -> str:
    ordered = [{field: quad[field] for field in QUAD_FIELDS} for quad in quads]
    return json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))


def render_prompt(tokenizer: Any, text: str) -> str:
    """Render the chat prompt exactly as training does.

    `enable_thinking=False` matters for Qwen3, whose template injects an empty <think></think>
    block in non-thinking mode. Tokenizers without that flag ignore it.
    """
    return tokenizer.apply_chat_template(
        build_messages(text),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def encode_example(
    example: dict[str, Any], tokenizer: Any, max_seq_len: int
) -> tuple[list[int], int] | None:
    """Tokenize one example into (token_ids, prompt_length).

    Loss is computed only from `prompt_length` onward - on the JSON target, never on the review
    the model is reading. Both frameworks express that mask differently (-100 labels in PyTorch,
    an offset in MLX), but they derive it from this one return value.

    Over-length examples return None rather than being truncated: a truncated target is
    malformed JSON, and training on it teaches the model to emit malformed JSON. (mlx-lm's own
    batcher truncates silently, so this has to happen before data reaches it.)
    """
    prompt_ids = tokenizer.encode(
        render_prompt(tokenizer, example["text"]), add_special_tokens=False
    )
    target_ids = tokenizer.encode(
        build_target(example["quads"]), add_special_tokens=False
    ) + [tokenizer.eos_token_id]

    if len(prompt_ids) + len(target_ids) > max_seq_len:
        return None
    return prompt_ids + target_ids, len(prompt_ids)


CANARY_TEXT = "個叉燒好正，不過個waiter好慢。"


def prompt_contract(tokenizer: Any) -> dict[str, str]:
    """The exact rendered prompt, saved next to every adapter.

    Serving re-renders CANARY_TEXT and asserts it matches byte for byte. Qwen3's template injects
    an empty <think></think> block when thinking is off, so a serving stack that renders even
    slightly differently loses quality in a way that is easy to misread as a bad adapter.
    """
    return {
        "system_prompt": SYSTEM_PROMPT,
        "canary_text": CANARY_TEXT,
        "rendered_example": render_prompt(tokenizer, CANARY_TEXT),
    }


def parse_quads(generated: str) -> list[dict[str, Any]] | None:
    """Inverse of build_target. None if the output isn't a JSON array."""
    try:
        quads = json.loads(generated)
    except json.JSONDecodeError:
        return None
    return quads if isinstance(quads, list) else None


def generation_metrics(results: list[tuple[str, str]]) -> dict[str, float]:
    """Score (generated_output, review_text) pairs on what loss can't see.

    - json_parse_rate: share of outputs that parse as a JSON array.
    - span_verbatim_rate: share of term/opinion values that are NULL or copied exactly from the
      review. A falling loss with a flat verbatim rate means the model learned the JSON shape
      while inventing the content.
    - quads_per_review: over parsed outputs; the synthetic training data has none below 2, so
      this drifting high on real reviews is expected and worth watching.

    Shared by both trainers so the PyTorch and MLX runs report comparable numbers.
    """
    parsed = spans_total = spans_verbatim = predicted = 0
    for generated, review in results:
        quads = parse_quads(generated)
        if quads is None:
            continue
        parsed += 1
        predicted += len(quads)
        for quad in quads:
            if not isinstance(quad, dict):
                continue
            for field in ("term", "opinion"):
                value = quad.get(field)
                if not isinstance(value, str):
                    continue
                spans_total += 1
                if value == "NULL" or value in review:
                    spans_verbatim += 1
    return {
        "json_parse_rate": parsed / len(results) if results else 0.0,
        "span_verbatim_rate": spans_verbatim / spans_total if spans_total else 0.0,
        "quads_per_review": predicted / parsed if parsed else 0.0,
    }


# Where a sentence ends: Chinese and English end punctuation. A period counts only before
# whitespace, so "$58.5" and "3.5" are never cut.
SENTENCE_END = re.compile(r"[。！？!?]+|\.(?=\s)")


def _pieces(
    text: str, start: int, end: int, boundary: re.Pattern
) -> list[tuple[int, int]]:
    """Cut text[start:end] right after each boundary match, as (start, end) offsets."""
    cuts = [m.end() for m in boundary.finditer(text, start, end)]
    edges = [start] + [c for c in cuts if start < c < end] + [end]
    return [(a, b) for a, b in zip(edges, edges[1:]) if a < b]


def split_for_inference(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Cut a long review into chunks of at most max_chars, at the most natural breaks available.

    1. A review of max_chars or less is one chunk.
    2. Otherwise it's cut after every line break, and consecutive paragraphs are packed
       greedily into chunks of at most max_chars.
    3. A paragraph longer than max_chars is cut at sentence ends instead, and a sentence longer
       than max_chars at every max_chars characters.

    Every chunk is a slice of the original text, so a span copied from a chunk is also verbatim
    in the full review. Chunks are stripped of surrounding whitespace; blank ones are dropped.
    """
    if len(text) <= max_chars:
        return [text]

    units: list[tuple[int, int]] = []
    for para in _pieces(text, 0, len(text), re.compile(r"\n+")):
        if para[1] - para[0] <= max_chars:
            units.append(para)
            continue
        for sentence in _pieces(text, *para, SENTENCE_END):
            a, b = sentence
            units.extend((i, min(i + max_chars, b)) for i in range(a, b, max_chars))

    chunks: list[str] = []
    start, end = units[0]
    for a, b in units[1:]:
        if b - start <= max_chars:
            end = b
        else:
            chunks.append(text[start:end])
            start, end = a, b
    chunks.append(text[start:end])
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def merge_quads(chunk_outputs: list[str]) -> tuple[list[Any], int]:
    """Merge the model's outputs for the chunks of one review.

    Returns (quads in chunk order with exact duplicates dropped, number of unparseable chunks).
    An unparseable chunk contributes no quads; callers count it against the parse rate.
    """
    merged: list[Any] = []
    seen: set[str] = set()
    unparsed = 0
    for output in chunk_outputs:
        quads = parse_quads(output)
        if quads is None:
            unparsed += 1
            continue
        for quad in quads:
            key = json.dumps(quad, ensure_ascii=False, sort_keys=True)
            if key not in seen:
                seen.add(key)
                merged.append(quad)
    return merged, unparsed
