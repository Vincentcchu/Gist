# Real test set — hand-labeled OpenRice reviews

`real_test.jsonl` is the only evaluation data in this project that the model's training process
didn't generate. Every "how good is it on real reviews" number comes from here. It is:

- **Committed to git** (unlike every other `.jsonl` under `ml/data/`) — small, and hours of
  hand work that can't be regenerated.
- **Never used for training, tuning, or prompt iteration.** Look at it only to report final
  numbers. The moment it influences a decision, it stops measuring generalization.

## How it was built

`python ml/labeling/export_for_validation.py --n 50` sampled 50 of the 312 scraped reviews,
stratified by length tertile. It refuses to overwrite this file once it exists.

What the sample can and can't tell you — record these in the model card:

- **One venue only (澳洲牛奶公司).** All 87 富臨飯店 reviews were excluded: the scraper saved
  OpenRice's collapsed preview cards ("…查看更多"), not the full text.
- **Almost no short reviews.** Only 1 usable review is under 80 characters, so the synthetic
  set's biggest gap (it has no 0- or 1-quad reviews) is barely tested here.
- 48 reviews had like/comment counters scraped in as trailing text (`\n\n114\n15`); those were
  stripped, and the stored `text` is the cleaned version. Label against what's in the file.

## Labeling

Fill in `quads` on each line. Same schema as training:

```json
{"term": "西多士", "category": "Food", "polarity": "positive", "opinion": "一流"}
```

- **`category`**: exactly one of `Food`, `Service`, `Price`, `Ambience`, `Hygiene`,
  `Waiting Time`.
- **`polarity`**: `positive`, `negative`, or `neutral`.
- **`term`** and **`opinion`**: **copied verbatim** from `text`. Copy-paste, don't retype —
  retyping is how `很滑` ends up where the review says `好滑`, and the verifier will reject it.

Rules — the same ones `ml/labeling/prompts.py` gives the teacher model:

1. **One quad per judgment.** A dish praised for two separate reasons gets two quads sharing
   the term.
2. **A phrase that weighs good against bad stays one quad** ("貴但值得"), with its overall
   polarity — usually `neutral`.
3. **`"NULL"` for implicit parts.** Judged but never named → `"term": "NULL"` ("真係抵食" judges
   price without naming it). Clear verdict but no phrase states it → `"opinion": "NULL"`.
4. **No aspect-specific verdict at all** ("正！下次再嚟") → one quad, `"term": "NULL"`.
5. **Judge intent, not wording.** Sarcasm ("真不愧為垃圾餐廳") is negative.
6. **Reputation and occasion are context, not aspects.** "香港人的集體回憶" and "帶朋友慶祝"
   explain why they came; extract only what they concluded.

Quad order doesn't matter — scoring is set-based.

These reviews are the implicit-aspect, code-switched, sarcastic kind the synthetic training set
barely contains. That's the point: if a review is hard to label, it's a review the model will
find hard too. Label what's there; don't simplify it.

## Checking your work

While labeling (unlabeled lines are allowed, progress is reported):

```bash
python ml/training/verify_dataset.py --files ml/data/real/real_test.jsonl --allow-empty
```

When finished (every review must have at least one quad):

```bash
python ml/training/verify_dataset.py --files ml/data/real/real_test.jsonl
```

Budget ~2–4 hours for all 50. Stopping at 25 is fine — a small real test set beats none.
