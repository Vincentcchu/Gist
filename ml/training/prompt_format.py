"""The single definition of the ACOS prompt and target format.

Training, generation-eval and serving all import from here. If the prompt drifts between
training and inference the adapter degrades silently in a way that looks like a bad fine-tune,
so there is exactly one copy of it.
"""

import json
from typing import Any

SYSTEM_PROMPT = """You extract aspect-based sentiment from Hong Kong restaurant reviews. Reviews may be in colloquial Cantonese, English, standard written Chinese, or a mix, and often switch language mid-sentence.

Output a JSON array. Each element describes one (aspect, category, opinion, sentiment) quad:

- "term": the aspect being judged, copied verbatim from the review in its original language.
- "category": exactly one of Food, Service, Price, Ambience, Hygiene, Waiting Time.
- "polarity": exactly one of positive, negative, neutral.
- "opinion": the phrase expressing the judgment, copied verbatim from the review.

Rules:
1. "term" and "opinion" must be exact substrings of the review. Never translate, paraphrase or reword them.
2. One quad per judgment. If a review praises a dish for two separate reasons, emit two quads sharing the same term.
3. If a single phrase balances a positive against a negative ("平但唔好食"), keep it as one quad and judge the overall polarity.
4. Use "NULL" for "term" when the aspect is implied but never named, and for "opinion" when a judgment is clear but no phrase states it.
5. Judge the reviewer's intent, not surface wording. Sarcasm inverts the literal meaning.

Output only the JSON array. No markdown fences, no commentary."""

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
