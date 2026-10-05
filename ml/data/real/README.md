# Real test set: the gold labels

`real_test.jsonl` is the only evaluation data in this project that the training process didn't
generate. Every "how good is it on real reviews" number comes from here. This file is its spec:
what the data is, the label format, and the rules every label follows.

**Labeling itself happens outside this repo, in the annotator.** Gist exports the text, checks
the labels that come back, and scores models against them.

The file is:
- **Not in git.** It holds the full text of scraped OpenRice user reviews, which don't belong in a
  public repo. Once labeled it's hours of hand work that can't be regenerated, so **back it up
  yourself** (somewhere private, not this repo).
- **Never used for training, tuning, or prompt iteration.** Look at it only to report final
  numbers. The moment it influences a decision, it stops measuring generalization.

## How it was built

`python ml/labeling/export_for_validation.py --n 50` sampled 50 of the 312 scraped reviews,
stratified by length tertile. It refuses to overwrite this file once it exists.

The texts were later re-derived with `--reclean`, while no review had labels yet, after
`clean_text` learned to strip counter lines between paragraphs (13 of the 50 texts changed).
**Labels are exact copies of the text, so the text is final from here on:** `--reclean` refuses
once any review has quads. Label against what's in the file.

Known limits, to state next to every number from it:
- **One venue only (澳洲牛奶公司).** All 87 富臨飯店 reviews were excluded: the scraper saved
  OpenRice's collapsed preview cards ("…查看更多"), not the full text.
- **Almost no short reviews.** Only 1 usable review is under 80 characters.

**Growing it:** new gold comes from new venues held out of training. Whole venues are assigned to
training or gold at random before anything is labeled. A venue is never in both. 澳牛's 175
non-test reviews stay out of the gold, because they were the synthetic generator's real-review
reference.

## Format

One review per line. `quads` holds the labels:

```json
{"review_id": 123, "venue": "澳洲牛奶公司", "text": "…",
 "quads": [{"term": "奶茶", "description": "drink texture", "polarity": "positive", "opinion": "好滑"},
           {"term": "NULL", "description": "revisit intent", "polarity": "positive", "opinion": "下次都會再嚟"}]}
```

Every label has exactly four fields, in this order: **term, description, polarity, opinion**. There
is no category; categories are applied afterwards from the description. Implicit fields are the
string `"NULL"`. Quad order doesn't matter, because scoring is set-based.

These are the same rules the synthetic training data was generated under (generator repo
`Synethic_review`, `schema.py` and README "Label format (v2)"). `verify_dataset.py` enforces the
mechanical ones.

## The rules

**Which labels get the gold.** Gold labels are made **by hand, with no LLM pre-labels**.
Pre-labels would pull the gold toward the labeler that produced the training data and inflate the
real-test score.

<!-- guidelines:begin -->
### 1. Description: what quality is judged
- Lowercase English, 1–4 words, letters only, shaped "<kind of thing> <quality>". Never the item
  itself: "drink texture", not "milk tea texture".
- **Pick from the seed list.** Write a new phrase only when none fits, in the same shape:

  | Area | Seeds |
  |---|---|
  | Food | food overall, food taste, food texture, food temperature, food freshness, food doneness, food portion, food presentation, menu variety |
  | Drinks | drink overall, drink taste, drink sweetness, drink texture, drink temperature |
  | Staff and service | service overall, staff attitude, staff attentiveness, serving speed, food wait time, order accuracy, dining time limit, payment methods |
  | Price | price level, value for money, service charge |
  | Getting in | queue time, table availability |
  | Room | seating space, seat comfort, noise level, decor, ambience, table sharing, crowding, room temperature, lighting |
  | Cleanliness | table cleanliness, tableware cleanliness, restroom cleanliness, food hygiene, restaurant cleanliness |
  | General | overall experience, revisit intent, recommendation |
  | Other | location convenience, opening hours, parking |

- **When a verdict names no particular quality**, use its area's "overall" description: food
  overall, drink overall, service overall.
  - 西多士一流 → `西多士` | food overall. 奶茶正 → `奶茶` | drink overall. 服務好 → `服務` |
    service overall.
  - A word that names a quality decides the description instead: 好食, 好味, delicious → food
    taste.
  - A general word next to a named quality is one label with the named quality: 一流，外脆內軟 →
    food texture.
  - For the room and for cleanliness, the overall descriptions are ambience and restaurant
    cleanliness.
  - The whole visit is different: overall experience, revisit intent and recommendation (rule 4).
- **Never interchangeable:**
  - queue time (waiting to get in) / food wait time (waiting for food after ordering) / serving
    speed (how quickly staff move and respond);
  - value for money / price level / service charge (加一);
  - table sharing (搭枱) / crowding;
  - overall experience / revisit intent / recommendation.
- "drink …" is for beverages. Soups take "food …" (Cantonese "drinks" soup, 飲湯).

### 2. Term: the thing judged
- The shortest phrase naming it, **copied exactly** (copy-paste, don't retype), with no verdict
  words and no leading classifier or determiner: `waiter`, not `個waiter`.
- Never evidence about the thing:
  - a price's term is the priced item if one is named (雙拼飯七十幾蚊 → `雙拼飯`), else `NULL`,
    never the amount;
  - a wait's term is what was waited for (`上菜`, `外賣`, `等位`), else `NULL`, never the duration.
- `NULL` when the thing is never named (好抵食), and always for general verdicts.
- Two things judged by one verdict stay one term (`叉燒同燒鵝`).
- **A word that only points back to an item named elsewhere in the review** (飲料, 個包, "it", a
  shortened name) is not the term: use the item's name, copied from where the review names it.
  叫咗菠蘿包同凍檸茶，個包好鬆軟，杯嘢飲太甜 → `菠蘿包` | `好鬆軟` + `凍檸茶` | `太甜`.
  - If it isn't clear which item is meant, keep the word the reviewer used.
  - A word for the whole category stays the term: 食物一般 → `食物`.
  - This is different from a part of a dish (below): a part is a different thing from its dish
    and is still the term when judged in its own clause (`火腿`). A stand-in word is the same item
    under a vaguer name.
- **Parts of a dish** (皮, 餡, 肉, 汁, 湯底…):
  - A part named with its dish in the same clause keeps the dish as the term, and the part stays
    in the opinion: 雲吞麵個湯底好鮮 → `雲吞麵` | `個湯底好鮮`.
  - A part judged in its own clause is the term: …，啲火腿都唔係求其嗰啲 → `火腿`;
    油雞髀都滑，雞皮薄薄地 → `油雞髀` | `都滑` + `雞皮` | `薄薄地`.
  - A soup base that is itself the item ordered (麻辣湯底, 豚骨湯底) is simply the term.
  - Clauses are separated by punctuation, line breaks or spaces.

### 3. Opinion: the words carrying the verdict
- One continuous stretch, **copied exactly**, that doesn't contain the term. When the reviewer
  restates the verdict, extend the opinion over the restatement.
- When a stated fact carries the verdict on its own, the opinion is that stretch:
  訂咗六點位，到咗仲要企喺門口等四十分鐘 → term `NULL`, `queue time`, negative, opinion
  `到咗仲要企喺門口等四十分鐘`.
- `NULL` only when no words state or carry the verdict. Term and opinion are never both `NULL`.

### 4. General verdicts
Term `NULL`, one label per kind: overall experience, revisit intent, recommendation.
Restatements of one kind extend one label.
- 總括嚟講都OK，下次都會再嚟 → overall experience (總括嚟講都OK) + revisit intent (下次都會再嚟).
- 正！下次再嚟 → overall experience (正) + revisit intent (下次再嚟).
- 值得一試 / 推介大家 → recommendation.
- **Recommending or warning against one dish is a verdict on that dish, not "recommendation"**:
  西多士必食 → `西多士` | food overall | positive | `必食`. Overall experience, revisit intent and
  recommendation are only for the whole visit, and always have term `NULL`.

### 5. Good against bad: count the verdicts
- Each half a complete verdict on a different quality → one label per quality:
  偏鹹咗少少，不過夠滑 → food taste negative + food texture positive; 平但唔好食 → price level
  positive + food taste negative.
- Halves weighing up one conclusion about one quality → one label, usually neutral:
  貴，但一分錢一分貨 → value for money.
- A drawback reframed as a plus (…不過好有街坊feel，我反而鍾意) → one verdict, labeled as the
  reviewer means it.

### 6. Polarity
positive, negative or neutral. **Neutral** is a lukewarm or mixed verdict (一般, 中規中矩,
唔平唔貴). A purely factual mention with no evaluation (what they ordered) is context and isn't
labeled. Judge intent, not wording: sarcasm (真不愧為垃圾餐廳) is negative.

### 7. Typed rating lines (味道：🔅🔅🔅🔅)
One label per rated item:
- Term `NULL`. The description comes from the rated label:

  | Label | Description |
  |---|---|
  | 味道 | food taste |
  | 衛生 | restaurant cleanliness |
  | 速度 | serving speed |
  | 份量 | food portion |
  | 服務 | staff attitude |
  | 抵食度 | value for money |
  | 環境 | ambience |
  | 整體評分 | overall experience |

- This table is fixed. The "overall" rule from rule 1 doesn't apply here: 服務 stays staff attitude.
- The opinion is the label and its score, copied exactly (`味道：🔅🔅🔅🔅`).
- Polarity by score, out of 5: 1–2 negative, 3 neutral, 4–5 positive.
- Commentary on the same line is labeled as ordinary labels.
- OpenRice's own star rating is a separate field, not text, and isn't labeled.

### 8. Photo captions
Short lone lines between blank lines stay in the text. A lone dish name with no verdict is context.
A caption that carries a verdict (炒蛋真係有香又嫩又滑👍) is labeled like any other text.

### 9. No cap
Real reviews are labeled in full: no limit on the number of labels, and no length limit. (The
synthetic data's 14-label, 600-character limits don't apply here; long reviews are split into
chunks at inference.)
<!-- guidelines:end -->

## Checking the labels

While labeling (unlabeled lines are allowed, and progress is reported):

```bash
python ml/training/verify_dataset.py --files ml/data/real/real_test.jsonl --allow-empty
```

When finished (every review must have at least one label):

```bash
python ml/training/verify_dataset.py --files ml/data/real/real_test.jsonl
```

The checks:
- exact field names;
- a valid polarity;
- term and opinion copied exactly or `NULL`, never both `NULL`;
- the term not inside the opinion;
- the description format;
- general descriptions with a `NULL` term;
- no duplicate labels.

## Double annotation

10–15 reviews are labeled twice: by a second person, or by yourself again at least a week later
without looking at the first labels. Gist scores one version against the other (exact match,
span overlap, and description group). That agreement is reported next to the model scores as the
human ceiling.

Budget ~2–4 hours for all 50. Stopping at 25 is fine: a small real test set beats none.
