"""The single definition of the ACOS prompt, target format and tokenization.

Both trainers (train.py for PyTorch/SageMaker, train_mlx.py for local MLX), generation-eval and
serving import from here. If the token sequence drifts between training and inference - or
between the two frameworks - the adapter degrades silently in a way that looks like a bad
fine-tune, so there is exactly one copy of it.

The system prompt is deliberately terse. A fine-tuned model learns the task from thousands of
examples; the prompt only has to carry the hard constraints, and every token of it is paid on
every training example and every inference call. The teacher prompt in ml/labeling/prompts.py
is verbose on purpose - it drives a general model few-shot, where the explanation earns its
tokens. Don't "sync" the two.
"""

import json
from typing import Any

SYSTEM_PROMPT = """Extract ACOS quads from a Hong Kong restaurant review (Cantonese, English, or mixed). Output a JSON array of {"term","category","polarity","opinion"} objects.
- term, opinion: exact substrings of the review, never translated or reworded. Use "NULL" when implied but not stated.
- category: Food, Service, Price, Ambience, Hygiene, or Waiting Time.
- polarity: positive, negative, or neutral.
- One quad per judgment: a term judged for two reasons gets two quads.
- A phrase weighing good against bad stays one quad with its overall polarity.
Output only the JSON array."""

QUAD_FIELDS = ("term", "category", "polarity", "opinion")


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
