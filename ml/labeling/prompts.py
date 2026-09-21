"""Teacher-labeling prompts for ACOS quad extraction on real OpenRice reviews.

Output schema is identical to the fine-tuning target in ml/training/prompt_format.py -
same four fields, same category set, same verbatim-span requirement. If these two drift
apart, teacher labels stop being comparable to model predictions and the eval numbers
become meaningless.

This prompt carries the two things the synthetic training set structurally cannot teach:
implicit aspects and implicit opinions (NULL). Real reviews are full of both.
"""

import json
from typing import Any

CATEGORIES = ("Food", "Service", "Price", "Ambience", "Hygiene", "Waiting Time")

SYSTEM_PROMPT = """You are labeling Hong Kong restaurant reviews for aspect-based sentiment analysis (ACOS: aspect, category, opinion, sentiment).

Reviews come from OpenRice. They are written in colloquial Cantonese (咗/嘅/㗎/喺/乜/嘢), standard written Chinese, English, or a mix that switches language mid-sentence ("食材quality簡直一流"). Read the whole review, including all Chinese text, before extracting anything.

Output a JSON array. Each element is one quad:

- "term": the aspect being judged, copied verbatim from the review in its original language. Use "NULL" when the reviewer clearly judges something they never name.
- "category": exactly one of Food, Service, Price, Ambience, Hygiene, Waiting Time.
- "polarity": exactly one of positive, negative, neutral.
- "opinion": the phrase expressing the judgment, copied verbatim from the review. Use "NULL" when the judgment is unmistakable but no phrase states it.

Rules:
1. "term" and "opinion" must be exact substrings of the review - same characters, same casing, same punctuation. Never translate, paraphrase, reword, or stitch together fragments from different parts of the review. If you cannot copy it exactly, use "NULL" or omit the quad.
2. One quad per judgment. If a reviewer praises one dish for two separate reasons, emit two quads sharing that term. Do not merge them.
3. If a single phrase balances positive against negative ("平但唔好食", "貴但值得"), keep it as one quad and judge the overall polarity - usually neutral. Do not split it, because neither half carries the reviewer's actual verdict.
4. Judge intent, not surface wording. Sarcasm inverts the literal meaning: "真不愧為垃圾餐廳" is negative, "一流啦" said after describing a 40-minute wait is negative. A sarcastic restatement is almost always re-expressing a complaint you already captured - extend that quad's span rather than emitting a new one.
5. Reputation and occasion framing is context, not an aspect. "試吓呢間殿堂級中式fine dining" is why they came; "同媽咪慶祝生日" is the occasion. Extract the verdicts the reviewer actually reaches.
6. Never invent an aspect that isn't grounded in the text.
7. If the review reaches no aspect-specific verdict at all, emit a single quad with "term": "NULL", the best-fitting category, your reading of the tone, and the verbatim phrase as "opinion".

Output only the JSON array. No markdown fences, no commentary."""

FEW_SHOT_EXAMPLES: list[dict[str, Any]] = [
    # English, one term carrying two separate judgments -> two quads, not one merged span.
    {
        "review": (
            "Service was quick and the staff were friendly, but the congee was "
            "lukewarm and way too salty."
        ),
        "quads": [
            {
                "term": "Service",
                "category": "Service",
                "polarity": "positive",
                "opinion": "was quick",
            },
            {
                "term": "staff",
                "category": "Service",
                "polarity": "positive",
                "opinion": "were friendly",
            },
            {
                "term": "congee",
                "category": "Food",
                "polarity": "negative",
                "opinion": "lukewarm",
            },
            {
                "term": "congee",
                "category": "Food",
                "polarity": "negative",
                "opinion": "way too salty",
            },
        ],
    },
    # Word-level code-switching. 殿堂級中式fine dining is reputation framing, not a verdict.
    {
        "review": "同媽咪慶祝生日🎂試吓呢間殿堂級中式fine dining，叉燒同燒鵝都好正，不過個waiter成晚黑面，好掃興。",
        "quads": [
            {
                "term": "叉燒同燒鵝",
                "category": "Food",
                "polarity": "positive",
                "opinion": "都好正",
            },
            {
                "term": "個waiter",
                "category": "Service",
                "polarity": "negative",
                "opinion": "成晚黑面，好掃興",
            },
        ],
    },
    # Implicit opinion: the wait is stated as fact, the complaint is unmistakable but unworded.
    {
        "review": "訂咗六點位，到咗仲要企喺門口等四十分鐘。啲點心就真係好食。",
        "quads": [
            {
                "term": "等四十分鐘",
                "category": "Waiting Time",
                "polarity": "negative",
                "opinion": "NULL",
            },
            {
                "term": "啲點心",
                "category": "Food",
                "polarity": "positive",
                "opinion": "真係好食",
            },
        ],
    },
    # Implicit aspect: "抵食" judges value without naming price or any dish.
    {
        "review": "兩個人食咗成餐先百六蚊，真係抵食。不過個位好逼。",
        "quads": [
            {
                "term": "NULL",
                "category": "Price",
                "polarity": "positive",
                "opinion": "真係抵食",
            },
            {
                "term": "個位",
                "category": "Ambience",
                "polarity": "negative",
                "opinion": "好逼",
            },
        ],
    },
    # Sarcasm. The literal words are praise; the verdict is negative. One quad, not two -
    # the sarcastic line restates the same complaint rather than raising a new one.
    {
        "review": "叫個夥計等埋朋友去完洗手間，佢話唔得要即刻埋單出去等。真不愧為垃圾餐廳。",
        "quads": [
            {
                "term": "個夥計",
                "category": "Service",
                "polarity": "negative",
                "opinion": "話唔得要即刻埋單出去等。真不愧為垃圾餐廳",
            },
        ],
    },
    # Balanced phrase -> one neutral quad. Splitting would hand each half a wrong polarity.
    {
        "review": "呢間嘢食貴但都算值得，環境就麻麻。",
        "quads": [
            {
                "term": "嘢食",
                "category": "Price",
                "polarity": "neutral",
                "opinion": "貴但都算值得",
            },
            {
                "term": "環境",
                "category": "Ambience",
                "polarity": "negative",
                "opinion": "麻麻",
            },
        ],
    },
    # One-liner with no specific aspect -> single NULL-term quad (rule 7).
    {
        "review": "正！下次再嚟。",
        "quads": [
            {
                "term": "NULL",
                "category": "Food",
                "polarity": "positive",
                "opinion": "正",
            },
        ],
    },
    # Long queue framed as a positive: speed despite the line. Tests that the labeler reads
    # the verdict rather than keying off 條隊好長.
    {
        "review": "澳牛嘅速度真係唔講得少，唔好睇佢條隊好長，基本上等5分鐘就入得。個西多士一流。",
        "quads": [
            {
                "term": "澳牛嘅速度",
                "category": "Waiting Time",
                "polarity": "positive",
                "opinion": "真係唔講得少",
            },
            {
                "term": "條隊",
                "category": "Waiting Time",
                "polarity": "positive",
                "opinion": "基本上等5分鐘就入得",
            },
            {
                "term": "個西多士",
                "category": "Food",
                "polarity": "positive",
                "opinion": "一流",
            },
        ],
    },
]


def build_messages(review_text: str) -> list[dict[str, str]]:
    """Few-shot messages for one review, as alternating user/assistant turns."""
    messages: list[dict[str, str]] = []
    for example in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": example["review"]})
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps(example["quads"], ensure_ascii=False),
            }
        )
    messages.append({"role": "user", "content": review_text})
    return messages
