# Raw data provenance

## `hk_restaurant_absa.jsonl`

5,000 synthetic Hong Kong restaurant reviews with ACOS labels, teacher-generated (Claude).

**Not tracked in git** (see `.gitignore`), so a fresh clone will not have it. Neither are the
derived splits in `ml/data/processed/`.

Canonical copy: synthetic_review repo

To rebuild everything from it, drop the file at this path and run:

```bash
python ml/training/prepare_dataset.py   # writes train/val/synthetic_test to ml/data/processed/
python ml/training/verify_dataset.py    # asserts every span is verbatim, no split leakage
```

Splits are deterministic (stratified by language mode and style, `--seed 13`), so the same raw
file always reproduces the same three splits. That makes this file the single point of failure
for reproducing any result — keep a copy somewhere other than this working tree.

**Provenance is not reproducible from this repo.** The file was generated in a separate project;
the generator was not retained here. Treat the file as a fixed input, not as build output — there
is no script in this repo that regenerates it, and re-generating it elsewhere would produce
different reviews.

Record this in `docs/model_card.md`: the training data is synthetic, and its generation is not
reproducible in-repo.

## Schema

One JSON object per line:

```json
{
  "text": "<review text>",
  "aspects": [
    {
      "term": "<aspect phrase, verbatim from text, source language>",
      "category": "Food | Service | Price | Ambience | Hygiene | Waiting Time",
      "polarity": "positive | negative | neutral",
      "opinion_words": "<opinion phrase>"
    }
  ],
  "overall_sentiment": "positive | negative | neutral | mixed",
  "meta": {
    "language_mode": "cantonese_colloquial | code_switch | english | written_chinese",
    "style": "standard | short_note | long_writeup | one_liner | rant",
    "orthography": "clean | typical | messy",
    "emoji_density": "none | light | heavy"
  }
}
```

## Known quality issues

Measured across all 5,000 reviews / 23,908 aspect labels:

- **`term` is verbatim in `text` 100% of the time.** Safe to use as-is.
- **`opinion_words` is NOT verbatim in 18% of labels** (4,289). Most are fragments stitched across
  a clause boundary — e.g. `"即叫即切，肉汁鎖得很好"` where the text reads
  `"…即叫即切，師傅落刀準繩，肉汁鎖得很好…"`. A minority are genuine paraphrases with no
  recoverable span.

  `ml/training/prepare_dataset.py` normalizes this. **Do not train on this file directly.**

- **No implicit aspects or opinions.** Canonical ACOS permits `NULL` for either; this set has zero
  of both, so a model trained only on it cannot learn to emit them. Real reviews contain them
  frequently ("好正！").
- **No 0- or 1-aspect reviews.** Minimum is 2 aspects (mean 4.8, max 6 pre-normalization), so a
  model trained only on this will over-predict on short real reviews.
- **Polarity skews positive**: 11,051 positive / 6,771 neutral / 6,086 negative.

## Downstream

`ml/training/prepare_dataset.py` reads this file and writes span-verified quads to
`ml/data/processed/{train,val,synthetic_test}.jsonl`.
