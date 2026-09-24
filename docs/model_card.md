# Model card — Cantonese restaurant-review ACOS extractor (Qwen3 QLoRA)

Fine-tuned Qwen3 models that read a Hong Kong restaurant review — colloquial Cantonese, English,
written Chinese, or a mix of them mid-sentence — and extract every opinion in it as a structured
**(aspect, category, opinion, sentiment)** quad.

> **Read this first.** Every model here was trained **and** tested on *synthetic* reviews
> (generated with Claude). The real-review test — 50 scraped OpenRice reviews, labeled by hand — is
> pending, and it's the number that matters. Section 10 is reserved for it. Treat the scores below
> as "how well the models learned the task as the synthetic data defines it", not as real-world
> accuracy.

## 0. Summary

Synthetic test set: 251 reviews, 1,770 gold quads. Each model's best epoch, selected on validation
loss. Strict = exact match on all four fields; overlap = term and opinion only need to cover the
same stretch of the review ([§4](#4-evaluation-method)).

| Model | Full-quad F1, strict | 95% interval | Full-quad F1, overlap | Category + polarity F1 | Training cost |
|---|---|---|---|---|---|
| Qwen3-4B-Instruct-2507 | 0.512 | [0.483, 0.540] | 0.732 | 0.843 | $4.77 |
| Qwen3-8B | 0.521 | [0.492, 0.549] | 0.737 | 0.845 | $4.63 |
| **Qwen3-14B** | **0.542** | [0.510, 0.573] | **0.755** | **0.854** | $6.98 |

- **The 14B is reliably better than the 4B:** +3.0 points strict, paired-bootstrap 95% interval
  [+1.1, +5.0]; the 14B was ahead in all 2,000 resamples. **The 8B can't be told apart from the
  4B** (+0.9, [−1.1, +2.9]).
- **The 14B's advantage is concentrated in colloquial Cantonese** (+4.9 points over the 4B) rather
  than English (+0.3) — the case the project cares about most.
- **About 40% of strict "misses" are span-boundary disagreements**, not errors: relaxing boundary
  matching adds ~22 points for every model without changing their order. A third of all misses had
  the right aspect, category and sentiment, and differed only in where the opinion span ends.
- All three models always produce valid JSON and almost never invent text: 100% parse rate, ≥99.8%
  of extracted spans copied verbatim from the review.
- Total training compute for the whole study: **$19.26** on AWS SageMaker.

## 1. Task

**Aspect-Category-Opinion-Sentiment (ACOS)** extraction. For each judgment in a review, output:

| Field | Meaning | Constraint |
|---|---|---|
| `term` | the aspect being judged | copied **verbatim** from the review, or `NULL` if implied |
| `category` | one of Food, Service, Price, Ambience, Hygiene, Waiting Time | fixed set |
| `polarity` | positive, negative, neutral | fixed set |
| `opinion` | the phrase expressing the judgment | copied **verbatim**, or `NULL` if implied |

A worked example — a code-switched review from the synthetic test set, and the 14B's output, which
matches the gold labels exactly:

> 一個人等friend食lunch，本來諗住食完一定唔會再黎，點知個清蒸石斑肉質好滑好鮮，一百八十幾蚊真係唔貴，
> 椒鹽豆腐就普通啲油，不過都OK，最後我改左主意，下次仲要book位帶friend黎。

```json
[{"term": "清蒸石斑", "category": "Food",  "polarity": "positive", "opinion": "肉質好滑好鮮"},
 {"term": "一百八十幾蚊", "category": "Price", "polarity": "positive", "opinion": "真係唔貴"},
 {"term": "椒鹽豆腐", "category": "Food",  "polarity": "neutral",  "opinion": "普通啲油，不過都OK"}]
```

Three things the model had to get right: the opening expectation ("本來諗住食完一定唔會再黎") is
context, not a verdict, so it produces no quad; a price with no named dish is still an aspect; and
"普通啲油，不過都OK" balances a complaint against acceptance, so it stays **one** neutral quad rather
than being split into a negative and a positive.

## 2. Data

### 2.1 Synthetic training data

5,000 Hong Kong restaurant reviews with ACOS labels, generated with Claude **outside this
repository** — the generator isn't retained here, so the raw file can't be regenerated from this
repo ([`ml/data/raw/README.md`](../ml/data/raw/README.md)).

| Language mode | Reviews | Share |
|---|---|---|
| Colloquial Cantonese | 1,915 | 38% |
| Cantonese–English code-switching | 1,341 | 27% |
| English | 965 | 19% |
| Written (standard) Chinese | 779 | 16% |

**Normalization** ([`prepare_dataset.py`](../ml/training/prepare_dataset.py)). In the raw labels,
18% of opinion spans (4,289 of 23,908) were not verbatim substrings of their review — mostly
fragments stitched together across clauses. Training on those would teach the model to invent
text. One invariant was enforced instead:

> Every opinion in the training data is either the original label verbatim, or a
> punctuation-delimited fragment of it, verbatim. Nothing is synthesized or extended.

Compound opinions are split into one quad per judgment (commas, full stops, and spaces between
Chinese phrases, as casual Cantonese typing uses them), except when the span contains a contrast
marker (不過, 但係, however, …): a span like "貴但值得" *is* the judgment, and splitting it would give
each half the wrong sentiment.

| | Count |
|---|---|
| Raw labels | 23,908 |
| Quads after normalization | **35,390** (+48%) |
| Labels dropped (no verbatim span recoverable) | 426 (1.8%) |
| Non-verbatim spans remaining | **0** |
| Reviews retained | 5,000 / 5,000 |

Resulting balance: polarity 48% positive / 26% negative / 26% neutral; category Food 44%, Service
14%, Price 12%, Ambience 12%, Waiting Time 10%, Hygiene 8%.

**Splits**, stratified by language mode and style (seed 13), with zero text overlap between them:

| Split | Reviews | Quads |
|---|---|---|
| train | 4,498 | 31,775 |
| val | 251 | 1,845 |
| synthetic_test | 251 | 1,770 |

### 2.2 Real test set (pending)

50 real OpenRice reviews, hand-labeled against the same schema, **evaluation-only** — never used for
training, tuning or model selection. Not in the repository (it holds third-party review text).

Built with [`export_for_validation.py`](../ml/labeling/export_for_validation.py) from 312 scraped
reviews: 87 were excluded because the scraper saved collapsed preview cards ("…查看更多") instead of
full text; 48 had scraped like/comment counters stripped; 3 over 1,000 characters were excluded;
50 were sampled, stratified by length tertile (80–693 characters, median 250).

Known gaps, to keep in mind when its results arrive: every review comes from **one venue**, and
only **one** is under 80 characters — so short reviews are barely tested.

## 3. Models and training

**Base models.** Qwen3 was chosen over Llama for two task-specific reasons: its tokenizer encodes
traditional Chinese at roughly 1–1.5 characters per token versus ~2–3 tokens *per character* for
Llama-3.1 — halving sequence length and making verbatim span copying far more reliable — and its
pretraining covers 119 languages and dialects, which matters for colloquial Cantonese (咗, 嘅, 㗎, 喺).

**Why the lineup stops at 14B.** Training used QLoRA on a single 24 GB GPU. Memory at the longest
training example, computed from each model's configuration:

| Model | Full fine-tune | LoRA (bf16 base) | QLoRA (used) | Measured peak |
|---|---|---|---|---|
| 4B | 64 GB | 12.2 GB | 6.8 GB | 14.4 GB (batch 4) |
| 8B | 131 GB | 21.0 GB | 10.7 GB | 15.6 GB (batch 2) |
| 14B | 236 GB | 35.0 GB | 15.4 GB | 19.2 GB (batch 1) |
| 32B | 524 GB | 73.5 GB | 27.2 GB | — does not fit |

**Hyperparameters** ([`config.yaml`](../ml/training/config.yaml)), identical across models:

| | |
|---|---|
| Quantization | 4-bit NF4, double quantization, bf16 compute |
| LoRA | rank 32, alpha 64 (scale 2.0), dropout 0.05, on q/k/v/o/gate/up/down projections of every layer |
| Trainable parameters | 66.1M (4B, 1.62%) · 87.3M (8B, 1.05%) · 128.5M (14B, 0.86%) |
| Optimizer | AdamW, learning rate 1e-4, cosine decay, 3% warmup |
| Effective batch | 16 (micro-batch × accumulation: 4×4, 2×8, 1×16) |
| Max sequence | 1,024 tokens (longest example: 902) |
| Seed | 13 |

**Prompt and loss.** A 148-token system prompt states the schema and rules; the review is the user
turn; the JSON array is the target. Qwen3's "thinking" mode is disabled, and the exact rendered
prompt is saved next to every adapter so serving can verify it renders identically. Loss is
computed **only on the JSON target** — never on the prompt or review — so every gradient goes to
the extraction itself.

**Model selection.** Each epoch ends with a validation pass and a checkpoint; the final adapter is
the epoch with the lowest **synthetic validation loss**. The test sets are never consulted. The 4B
ran 3 epochs; its third overfit (§5.2), so the 8B and 14B ran 2.

**Infrastructure.** AWS SageMaker training jobs on `ml.g5.2xlarge` (NVIDIA A10G, 24 GB), PyTorch 2.10
container, Hugging Face Transformers + PEFT + bitsandbytes. A parallel local path on Apple silicon
via MLX ([`train_mlx.py`](../ml/training/train_mlx.py)) tokenizes identically and was verified,
but was too slow for these runs (4B: 0.56 iterations/s on an M5 MacBook Air).

## 4. Evaluation method

Implemented in [`ml/eval/evaluate.py`](../ml/eval/evaluate.py); its behaviour is pinned by
hand-computed tests in [`test_evaluate.py`](../ml/eval/test_evaluate.py).

### 4.1 Strict (headline)

Micro-averaged precision, recall and F1 over quads. A predicted quad counts as correct only if all
four fields match a gold quad **exactly**; each gold quad can be matched once. This is the standard
ACOS metric, and exact match is well-defined here because term and opinion are verbatim spans of
the input, not free text. An output that isn't valid JSON predicts nothing.

Partial views isolate where errors happen:

| View | Question it answers |
|---|---|
| term | Did it find the right aspects? |
| term + category | …and file them under the right category? |
| term + polarity | …and read the sentiment correctly? |
| category + polarity | What a dashboard aggregates ("Food: 70% positive") |
| full quad | The headline |

### 4.2 Overlap (secondary)

Exact match is strict about span **boundaries**, which are partly arbitrary: a model answering
"凍晒" for a gold "已經凍晒" (both: the scrambled eggs had gone cold) scores a total miss. The overlap
views accept a match when:

1. category and polarity match **exactly**;
2. term and opinion each **overlap** their gold span — occupying the same stretch of the review,
   judged by position (not shared characters), with the overlap covering at least half the
   shorter span. `NULL` matches only `NULL`;
3. predictions and gold are paired **one-to-one** by maximum bipartite matching, so one prediction
   overlapping two gold quads is credited once, and the result doesn't depend on output order.

Validation: of a random sample of 12 matches the overlap view accepts but strict rejects, all 12
were the same judgment with different boundaries — `冇乜特別` vs `唔差 但冇乜特別`, `Our server` vs
`server`, `田雞` vs `田雞煲仔飯`. Strict stays the headline; overlap explains it.

### 4.3 Health metrics

Tracked per epoch during training and on the test set: **JSON parse rate**, **verbatim-span rate**
(share of predicted terms and opinions copied exactly from the review — the direct measure of
hallucination), and quads predicted per review.

### 4.4 Uncertainty

95% intervals from a bootstrap over reviews (2,000 resamples). Model comparisons use a **paired**
bootstrap — both models scored on the same resampled reviews — which is much tighter than comparing
two independent intervals.

### 4.5 Not yet measured

Real-review quality (§10) and serving latency (Phase 7).

## 5. Results — synthetic test set

### 5.1 All views

Precision / recall / F1:

| View | 4B | 8B | 14B |
|---|---|---|---|
| term | 0.637 / 0.668 / **0.652** | 0.629 / 0.663 / **0.646** | 0.652 / 0.683 / **0.667** |
| term + category | 0.635 / 0.667 / **0.651** | 0.628 / 0.662 / **0.644** | 0.652 / 0.682 / **0.667** |
| term + polarity | 0.609 / 0.639 / **0.624** | 0.610 / 0.644 / **0.626** | 0.631 / 0.660 / **0.645** |
| category + polarity | 0.823 / 0.863 / **0.843** | 0.823 / 0.868 / **0.845** | 0.835 / 0.875 / **0.854** |
| **full quad** | 0.500 / 0.525 / **0.512** | 0.508 / 0.536 / **0.521** | 0.530 / 0.555 / **0.542** |
| term (overlap) | 0.789 / 0.828 / **0.808** | 0.783 / 0.825 / **0.803** | 0.802 / 0.840 / **0.821** |
| full quad (overlap) | 0.715 / 0.750 / **0.732** | 0.718 / 0.757 / **0.737** | 0.738 / 0.773 / **0.755** |

Reading down a column: term → term + category costs almost nothing (≤0.002), so **when a model
finds the right aspect it nearly always files it correctly**. The large drop is to full quad —
the opinion span (§6).

| Health (test set) | 4B | 8B | 14B | Gold |
|---|---|---|---|---|
| JSON parse rate | 1.000 | 1.000 | 1.000 | — |
| Verbatim-span rate | 0.999 | 0.998 | 0.999 | 1.000 |
| Quads per review | 7.40 | 7.44 | 7.39 | 7.05 |

### 5.2 Training curves

Validation after each epoch (loss on all 251 val reviews; generation metrics on a 16-review sample,
gold average 7.35 quads per review):

| Model | Epoch | Val loss | Parse rate | Verbatim rate | Quads / review |
|---|---|---|---|---|---|
| 4B | 1 | 0.0768 | 1.00 | 1.000 | 8.06 |
| 4B | 2 | **0.0719** | 1.00 | 1.000 | 7.69 |
| 4B | 3 | 0.0762 | 1.00 | 0.996 | 7.63 |
| 8B | 1 | 0.0757 | 1.00 | 1.000 | 7.94 |
| 8B | 2 | **0.0717** | 1.00 | 1.000 | 7.88 |
| 14B | 1 | 0.0690 | 1.00 | 1.000 | 7.81 |
| 14B | 2 | **0.0661** | 1.00 | 1.000 | 8.00 |

- **The 4B overfit in epoch 3**: validation loss rose, and it produced its first invented spans.
  Keeping every epoch's checkpoint meant the final 4B adapter is epoch 2's, and the extra epoch cost
  only time.
- **The 14B was still improving at epoch 2**, and after one epoch was already below either smaller
  model's best. It may be under-trained at 2 epochs — a cheap follow-up to test.

### 5.3 By category

Full-quad F1, strict → overlap, with median gold span lengths in characters:

| Category | Gold quads | Median term / opinion | 4B | 14B |
|---|---|---|---|---|
| Food | 784 | 4 / 4 | 0.660 → 0.808 | 0.668 → 0.825 |
| Price | 221 | 7 / 5 | 0.540 → 0.860 | 0.591 → 0.859 |
| Service | 267 | 3 / 7 | 0.454 → 0.764 | 0.539 → 0.786 |
| Hygiene | 119 | 3 / 6 | 0.352 → 0.680 | 0.431 → 0.740 |
| Ambience | 166 | 5 / 7 | 0.290 → 0.460 | 0.318 → 0.531 |
| Waiting Time | 213 | 9 / 7 | 0.265 → 0.520 | 0.257 → 0.533 |

- **Price is mostly a boundary problem**: 0.54 → 0.86 once boundaries are forgiven. Whether the
  aspect is "七十八蚊" or "照收七十八蚊" is a judgment call.
- **Waiting Time and Ambience are weak even under overlap** (~0.52–0.53). Their spans are the
  longest, and in them the line between aspect and opinion genuinely blurs — "等咗大約十分鐘 同 app
  寫嘅差唔多" contains both the wait and the verdict on it (§6.3).
- The 14B's largest gains over the 4B are in Service (+8.5 strict) and Hygiene (+7.9).

### 5.4 By sentiment

Full-quad F1, strict:

| Polarity | 4B | 8B | 14B |
|---|---|---|---|
| positive | 0.573 | 0.573 | 0.595 |
| neutral | 0.507 | 0.545 | 0.549 |
| negative | 0.419 | 0.416 | 0.449 |
| macro average | 0.500 | 0.511 | 0.531 |

Negative quads are hardest for every model, even though a quarter of the data is negative.

### 5.5 By language mode

Full-quad F1, strict → overlap. Small cells — treat differences of a few points as indicative.

| Language mode | Test reviews | 4B | 14B | 14B − 4B (strict) |
|---|---|---|---|---|
| Colloquial Cantonese | 96 | 0.529 → 0.725 | 0.578 → 0.754 | **+4.9** |
| Code-switching | 67 | 0.517 → 0.759 | 0.543 → 0.766 | +2.6 |
| Written Chinese | 39 | 0.547 → 0.745 | 0.563 → 0.762 | +1.6 |
| English | 49 | 0.436 → 0.696 | 0.439 → 0.733 | +0.3 |

- **English is the hardest mode**, not Cantonese — its spans are longer and wordier ("which was such
  a win" vs "such a win"), so exact boundaries are harder to hit; the overlap view narrows the gap.
- **Model size pays off most for colloquial Cantonese.** That matches why Qwen was chosen, and it's
  encouraging for the real test set, which is predominantly Cantonese.

## 6. Error analysis

### 6.1 Opinion boundaries: the largest single source

Of the gold quads the 14B missed under strict matching, **33%** (263 of 788) had a prediction with
the *same* term, category and polarity — only the opinion span differed. For the 4B it's 32% (272
of 841). A random sample of the 14B's cases — the aspect is identical, and the opinions differ
only in where they start or end:

| Aspect | Model's opinion | Gold opinion |
|---|---|---|
| 素鵝 | 入面豆腐皮好香 | 豆腐皮好香 |
| 蒜香焗蟶子 | 蒜蓉唔會搶味 | 唔會搶味 |
| 侍應 | 冇道歉就拎返入去 | 冇道歉 |
| brownie | 今次真係失望 | 失望 |
| 廁所 | 濕漉漉的 | 地板濕漉漉的 |

Neither side is consistently longer: the model sometimes includes a lead-in the gold label
omits, and sometimes leaves out a subject the gold label includes.

### 6.2 Other boundary disagreements

Aspect boundaries account for much of the rest of the overlap gain — often together with an
opinion boundary: `Our server → nice enough touch` vs `server → a nice enough touch`, `田雞` vs
`田雞煲仔飯`, `lunch set` vs `lunch set 每人一百二十蚊`. Overall, strict → overlap adds ~22 points of
full-quad F1 for every model.

### 6.3 Genuine errors

- **Aspect attribution.** Given "個湯得個咸字 牛肉又韌", the 14B attached "牛肉又韌" (the beef is tough)
  to the aspect "個湯" (the soup) — the right sentiment and a verbatim span, but the wrong target.
  This is the kind of error the smaller models' worst cases rarely showed; theirs were mostly
  boundaries.
- **Over-generation.** Every model predicts slightly more quads than gold (7.4 vs 7.05 per review).
  Of the 14B's predictions with no gold counterpart even under overlap (466), 70% repeat the aspect,
  category and sentiment of another prediction in the same review: an extra opinion attached to an
  aspect already covered, rather than an invented aspect.

### 6.4 Noise in the gold labels

The synthetic labels are imperfect, which caps the achievable score:

- For a review about a crowded hotel buffet, the gold labels treat "追追逐逐" (kids running around) as
  the *aspect* and "好嘈" (noisy) as the opinion, where the aspect is arguably the noise or the room
  itself. Every model "missed" these.
- For "等咗大約十分鐘 同 app 寫嘅差唔多", the gold label puts almost the whole phrase in the aspect and
  just "差唔多" in the opinion. The 14B split it as aspect "等咗大約十分鐘", opinion "同 app 寫嘅差唔多"
  — arguably better, but a strict miss.

## 7. Limitations

- **Synthetic-only.** Training and test data come from the same generator, so these scores measure
  in-distribution fit. Real-review performance is untested until §10.
- **Noisy gold labels** (§6.4) put a ceiling on synthetic scores and add noise to comparisons.
- **No implicit quads in training.** The data contains no `NULL` aspects or opinions, although real
  reviews are full of them ("正！"). The models have never seen one.
- **No very short reviews in training** — every synthetic review has at least 2 quads — so the models
  may over-extract from one-line reviews.
- **The real test set is small and narrow**: 50 reviews, one venue, almost no short reviews.
- **Test-set size**: individual scores carry ±~3 points of uncertainty (§0). Per-category and
  per-language cells (39–784 quads) are noisier still.
- **Exact match understates quality** (§6.1–6.2); overlap may slightly overstate it.
- **No serving measurements yet**: latency and cost per review come with Phase 7.

## 8. Compute and cost

AWS SageMaker, `ml.g5.2xlarge` on-demand, $1.515/hour, us-east-1. Billed time from each job's
`BillableTimeInSeconds`.

| Job | Outcome | Billed | Cost |
|---|---|---|---|
| Smoke test, 0.6B | failed: container had no default AWS region | 3.5 min | $0.09 |
| Smoke test, 0.6B | passed — pipeline proven end to end | 17.6 min | $0.44 |
| Fit/speed check, 14B | passed | 11.8 min | $0.30 |
| **4B**, 3 epochs | completed | 3.15 h | **$4.77** |
| 8B, first attempt | failed: out of memory at first evaluation | 1.35 h | $2.05 |
| **8B**, 2 epochs | completed | 3.06 h | **$4.63** |
| **14B**, 2 epochs | completed | 4.61 h | **$6.98** |
| **Total** | | | **$19.26** |

Training throughput (train runtime ÷ examples processed, including per-epoch evaluation): 4B 0.73 s,
8B 1.04 s, 14B 1.62 s per example.

The two failures were worth their $2.14: the first surfaced that the training container sets no AWS
region; the second, that evaluation ran at the Trainer's default batch of 8 — about 4.4 GB of
output logits with Qwen's 151,936-token vocabulary — and would otherwise have ended the 14B run
after ~2.5 hours. The fix evaluates at the training micro-batch.

## 9. Reproducing

```bash
# Data (needs the raw file - see ml/data/raw/README.md)
python ml/training/prepare_dataset.py
python ml/training/verify_dataset.py

# Train on SageMaker (needs an execution role; see launch_sagemaker.py)
python ml/training/launch_sagemaker.py --model-id Qwen/Qwen3-4B-Instruct-2507 --batch-size 4 --epochs 3
python ml/training/launch_sagemaker.py --model-id Qwen/Qwen3-8B --batch-size 2 --epochs 2
python ml/training/launch_sagemaker.py --model-id Qwen/Qwen3-14B --batch-size 1 --epochs 2

# Fetch a job's model.tar.gz into ml/outputs/<job>/ (the launcher prints the command), then score
python ml/eval/evaluate.py --adapter ml/outputs/<job> --from-predictions
pytest ml/eval/test_evaluate.py
```

Each training job writes the final adapter, one checkpoint per epoch, the prompt contract, and
predictions for every test file into its `model.tar.gz`. Scoring never needs the model again.
Not reproducible from this repository alone: the raw synthetic data (generated elsewhere) and the
real test set (third-party text, kept out of git).

## 10. Real-review results — pending

To be filled once `ml/data/real/real_test.jsonl` is hand-labeled. Every model already saved
predictions for these 50 reviews, so no retraining is needed:

```bash
python ml/training/verify_dataset.py --files ml/data/real/real_test.jsonl
python ml/eval/evaluate.py --adapter ml/outputs/<job> --from-predictions
```

| | 4B | 8B | 14B |
|---|---|---|---|
| Full-quad F1, strict | — | — | — |
| Full-quad F1, overlap | — | — | — |
| Category + polarity F1 | — | — | — |
| **Gap vs synthetic (strict)** | — | — | — |

This section answers the question the synthetic results can't: whether the models generalize to
real reviews, and whether the 14B's edge grows there. A clearly larger gap closed by the 14B would
justify training a 32B (which needs a 48 GB GPU); a gap that no size closes would point at the data
instead.
