# Build Guide — Review ABSA MVP (Free-Text Aspects, Fine-Tuned Generative LLM)

Each phase produces something visible/testable before you move on. Don't skip ahead — the point of this order is that you're never debugging two unfamiliar things at once.

---

## Phase 0 — Setup (half a day)

- [ ] GitHub repo, structure from `PROJECT_STRUCTURE.md`
- [ ] Railway or Render account, Postgres instance provisioned
- [ ] Local Python env (3.11+), `poetry` or plain `venv`
- [ ] W&B account (free tier)
- [ ] Pick your teacher model (Claude/GPT via API) and get an API key
- [x] Base model decided: **Qwen3-8B**. Picked on task fit, not availability:
  - **Tokenizer economics.** Qwen's 151k vocab encodes traditional Chinese at ~1-1.5 chars/token; Llama-3.1 needs ~2-3 tokens *per char*. That decides whether a p99 review fits in a 1024 sequence or needs 2048 — roughly 2x on training time and per-review serving cost — and verbatim span copying is far more reliable over clean tokens than byte-fallback fragments.
  - **Cantonese, not just Chinese.** The corpus is dense in 咗/嘅/㗎/喺, rare in Mandarin-dominant pretraining. Qwen3 covers 119 languages/dialects vs Qwen2.5's 29. This is the biggest quality lever available, and it lives in the base model rather than the LoRA.
  - **JSON adherence.** A malformed generation scores zero on every quad in that review, so format reliability shows up as F1 whether or not you attribute it correctly.
  - Rejected: **Llama-3.1-8B** (strictly dominated — worse tokenizer *and* weaker Chinese); **XLM-R token tagging** (BIO can't represent free-text aspects with overlapping multi-clause opinions, let alone 4 slots); **mT5/GAS seq2seq**, the academic ACOS default (weaker colloquial Cantonese and format control, no offsetting gain).
  - **Sharp edge:** Qwen3 is a hybrid-thinking model whose template injects an empty `<think></think>` block when thinking is off. Training and inference must render the prompt identically or quality drops in a way that looks like a bad adapter. `ml/training/prompt_format.py` is the single source of truth, and `train.py` writes a `prompt_contract.json` next to the adapter to assert against.
  - Chosen deliberately over a 1.5-3B model to get real LoRA-on-a-real-sized-model experience and a stronger quality ceiling; the tradeoff is higher serving cost/latency later. **Qwen3-4B-Instruct-2507** is trained alongside as the cost/quality baseline, so the model card reports a measured tradeoff rather than an assumed one.
- [ ] AWS account set up for SageMaker training (see Phase 5a for the full setup list)

**Checkpoint**: `docker run` a "hello world" FastAPI container locally, confirm it runs. Don't touch cloud yet.

---

## Phase 1 — Skeleton product with fake data (2-3 days)

Goal: something on screen, before any scraping or ML.

- [ ] `app/models.py` — SQLAlchemy models: `Hotel`, `Review` (raw text, source, scraped_at, status), `AspectExtraction` (review_id, aspect_text, sentiment, span, model_version)
- [ ] `app/main.py` — endpoints: `GET /hotels/{id}/reviews`, `GET /hotels/{id}/aspects`, `GET /health`
- [ ] Seed the DB with 10-20 hand-written fake reviews + fake aspect results
- [ ] Minimal `frontend/` — a table/list showing reviews and their aspects, pointed at your local API
- [ ] Deploy `app/` to Railway/Render, point frontend at the deployed URL

**Checkpoint**: You can open a real URL and see (fake) data rendered. This is your safety net — everything after this point is *improving* a working thing, not building from zero.

---

## Phase 2 — Real data in (3-5 days)

- [ ] `scraper/scrape.py` — Playwright script, one hotel/restaurant source to start (don't generalize yet)
- [ ] Write raw reviews to `Review` table, `status = 'pending'`
- [ ] Deploy as a scheduled job (Railway cron / GitHub Actions scheduled workflow)
- [ ] Manually trigger it, confirm real reviews land in Postgres

**Checkpoint**: Real reviews visible in your dashboard (even with no aspect data yet — that column can be empty for now).

---

## Phase 3 — Teacher labeling of the real reviews (3-4 days) — ⏸ DEFERRED

> **Deferred by scope decision.** Scraping real reviews proved harder than planned, so the
> product is being built end to end on the synthetic set first (Phase 5) and data quality is
> upgraded afterwards. This phase is the upgrade path, not cancelled. In the meantime, 50 real
> reviews were hand-labeled as an eval-only test set — see Phase 4.
>
> Fix the scraper before resuming: every 富臨飯店 review in the DB is a truncated preview card
> ending in "…查看更多", and 48 reviews carry scraped like/comment counters as trailing text.

Training data is currently 100% synthetic; the 312 scraped OpenRice reviews in the DB are still
`status='pending'`.

- [x] `ml/labeling/prompts.py` — ACOS extraction prompt, output identical to the fine-tuning
  target in `ml/training/prompt_format.py`:
  ```json
  [{"term": "個waiter", "category": "Service", "polarity": "negative", "opinion": "成晚黑面"}]
  ```
  Eight few-shot examples covering English, word-level code-switching, sarcasm, a one-liner with
  no stated aspect, one term carrying two judgments, and a balanced phrase kept as one quad.
- [ ] `ml/labeling/label_with_teacher.py` — pulls `pending` reviews, calls teacher, writes to
  `AspectExtraction` with `model_version = 'teacher-v1'`. Needs a `category` column added first
  (see Phase 1 note below).
- [ ] **Reject and re-prompt any quad whose `term`/`opinion` isn't a verbatim substring.** The
  synthetic set failed this 18% of the time; assume the teacher will too unless checked.
- [ ] Deliberately cover **implicit quads** (`term: "NULL"` / `opinion: "NULL"`). The synthetic
  set contains zero of either, so this pass is the only source of that supervision.
- [ ] Run on ~50 reviews first, read every output by eye. Fix the prompt before scaling — cheap
  now, expensive after you've labeled 1000 reviews with a bad prompt.
- [ ] Run on your full backlog

**Checkpoint**: real reviews with verbatim-verified ACOS quads attached, teacher-generated.

> `AspectExtraction` currently has no `category` column. The quad maps as
> `term -> aspect_text`, `polarity -> sentiment`, `opinion -> span`, plus a new `category`.

---

## Phase 4 — Human validation (2-3 days, don't rush this) — ◐ REDUCED

This step is what makes your eval numbers later actually mean something.

**Done instead, for now:** a 50-review hand-labeled real test set, eval-only, at
`ml/data/real/real_test.jsonl` — kept **out of git**, since it holds the full text of scraped
OpenRice reviews and the repo is public (labeling rules and known limits in
`ml/data/real/README.md`).
`export_for_validation.py` builds the template: excludes truncated previews, strips scraped
counters, stratifies by length tertile, and refuses to overwrite labeled work. Validate with
`verify_dataset.py --files ml/data/real/real_test.jsonl`. The full teacher-validation flow
below resumes with Phase 3.

- [ ] `ml/labeling/export_for_validation.py` — export ~150-200 examples (stratified: some short, some long, some code-switched) to a spreadsheet or a simple labeling tool
- [ ] Go through by hand: mark each teacher-generated aspect as correct/incorrect, add any aspects the teacher missed, fix sentiment errors
- [ ] Split this validated set: ~100 for a held-out **test set** (never touched again until final eval), ~50-100 as a **validation set** for tuning during training
- [ ] Everything else (teacher-only, unvalidated) becomes your **training set**

**Checkpoint**: Three clearly separated files — `train.jsonl`, `val.jsonl`, `test.jsonl`. This separation is one of the most interview-relevant things you'll do in this whole project — be ready to explain why the test set is untouched.

---

## Phase 5 — Fine-tune the generative model (1-1.5 weeks, most of it waiting on training runs)

**Scope:** synthetic data + SageMaker now → scraped real data + SageMaker later (Phases 3-4).
Always **QLoRA** — a frozen 4-bit base with trained LoRA adapters, never full fine-tuning.

### Shared foundation (both training paths)

- [x] `ml/training/prepare_dataset.py` — normalizes the raw synthetic set into span-verified ACOS quads and writes stratified train/val/synthetic_test splits. Enforces one invariant: **every emitted `opinion` is either the original label verbatim or a punctuation-delimited fragment of it, verbatim** — nothing synthesized. 23,908 raw labels → 35,390 quads (+48%), 5,000/5,000 reviews retained, 0 verbatim violations. (Whitespace is also a clause boundary when every part is CJK — casual Cantonese uses spaces for punctuation.)
- [x] `ml/training/verify_dataset.py` — independent audit (re-declares the schema rather than importing it, so a bug in the normalizer can't validate itself): verbatim spans, split leakage, stratification.
- [x] `ml/training/prompt_format.py` — the single definition of prompt, target, tokenization and generation metrics, imported by both trainers so they train on byte-identical sequences (500/500 identical between the HF and MLX tokenizers). System prompt trimmed 304 → 148 tokens: mean example 627 → 471 tokens, −25% compute on every step and every inference call.
- [x] Completion-only loss masking — prompt tokens are excluded, so the gradient goes to the JSON target rather than to reproducing review text. Over-length examples are dropped, never truncated, since a truncated target is malformed JSON.

### 5a — PyTorch/PEFT on SageMaker (main path, now)

**Why 14B is the ceiling on `ml.g5.2xlarge` (A10G, 24GB, $1.515/hr).** QLoRA memory at the
longest example (902 tokens, batch 1), computed from the real Qwen3 configs:

| Model | Full fine-tune | LoRA (bf16 base) | **QLoRA** |
|---|---|---|---|
| 4B | 64 GB | 12.2 GB | **6.8 GB** |
| 8B | 131 GB | 21.0 GB | **10.7 GB** |
| 14B | 236 GB | 35.0 GB | **15.4 GB** |
| 32B | 524 GB | 73.5 GB | **27.2 GB** |

Conservative by ~2-3 GB versus measured MLX peaks. 32B needs a 48 GB GPU even under QLoRA.

- [x] `train.py` — per-epoch eval and checkpoint (adapter weights only), every epoch kept, best
  by **synthetic** val loss loaded at the end; post-training predictions on each test file,
  written in `evaluate.py`'s cache format; W&B key read from Secrets Manager by name; gradient
  accumulation derived from `effective_batch_size: 16`.
- [x] `launch_sagemaker.py` — rewritten for **SageMaker Python SDK v3** (`ModelTrainer`; the v2
  `PyTorch` estimator is gone). PyTorch **2.10** training container — the newest available;
  `2.14`, which the old launcher asked for, doesn't exist. torch is unpinned in
  `requirements.txt` so the container's CUDA build runs. Separate `train`/`val`/`eval` channels,
  so the training container never receives test data. `--dry-run` validates a request for free;
  `--max-hours` caps runtime so a hung job can't bill indefinitely.
- [x] AWS setup (us-east-1): GPU quota *"ml.g5.2xlarge for training job usage"* → 1 (approved);
  IAM admin user `vincent-admin` for daily use (root has MFA and stays in the drawer), CLI via
  `aws login` — the Python side needs `botocore[crt]` for those credentials; $50/month budget
  alarm `review-absa-monthly`; execution role `review-absa-sagemaker-training`
  (`AmazonSageMakerFullAccess` + `GetSecretValue` on the W&B secret only — verified with the IAM
  policy simulator: that secret allowed, any other denied); W&B key in Secrets Manager as
  `review-absa/wandb-api-key`, stored from a separate terminal so it never entered this repo or
  a chat transcript.
- [x] `--dry-run` validated a full job request against the account (image, role, channels,
  output path) without launching anything.
- [x] Smoke job: Qwen3-0.6B, 200 examples — container, requirements, bitsandbytes on real CUDA,
  S3 channels, W&B, and the `model.tar.gz` → local `evaluate.py` round trip. First attempt died on
  `NoRegionError` (the container sets no default AWS region; now passed in); second passed.
- [x] 14B smoke: 64 examples — fits (20.1 GB of 23.7 GB at micro-batch 1), ~1.6 s per training
  example. **Missed an eval-memory bug** because capping the data also capped the eval batch.
- [x] Sweep: 4B (3 epochs) → 8B → 14B (2 epochs each, after the 4B's epoch 3 overfit). The 8B's
  first run ran out of memory at its first end-of-epoch eval: `per_device_eval_batch_size` had
  been left at the Trainer default of 8, ~4.4 GB of logits at Qwen's 151,936-token vocabulary.
  Fixed by evaluating at the training micro-batch.
- [x] `eval_json_parse_rate` and `eval_span_verbatim_rate` logged per epoch: 1.00 / ≥0.996
  throughout for all three models — no hallucinated structure, and the one dip in verbatim rate
  (4B, epoch 3) coincided with its overfitting.
- [ ] Out of scope for now: spot instances (need checkpoint-resume; worth it only once runs get
  expensive), hyperparameter variants, 32B.

**Results** (synthetic test, 251 reviews; each model's best epoch by val loss):

| | 4B | 8B | 14B |
|---|---|---|---|
| Full-quad F1 | 0.512 | 0.521 | **0.542** |
| Category + polarity F1 | 0.843 | 0.845 | **0.854** |
| Best val loss | 0.0719 | 0.0717 | **0.0661** |
| Cost | $4.77 | $4.63 | $6.98 |

Size helps modestly on synthetic data (~+3 F1 points, 4B → 14B); the 14B was still improving at
epoch 2. Whether the gap grows on real reviews — and so whether 32B is justified — waits on the
hand-labeled `real_test`; every job already saved predictions for it.

**Cost, measured:** $19.26 of training compute for the whole phase, including smoke tests ($0.83)
and the failed 8B run ($2.05) — under the pre-sweep estimate of $25-40.

**Checkpoint** ✅: three adapters in S3 with per-epoch checkpoints, training curves in W&B, cached
predictions for both test sets.

### 5b — Local QLoRA via MLX (verified alternative)

Built and verified first, then superseded as the main path: too slow for the sweep, and 14B is
out of reach locally. Kept for quick local experiments and local inference of 4-bit models.

- [x] 4-bit conversion from the official Qwen weights, identical settings for every model:
  `mlx_lm.convert --hf-path <repo> --mlx-path ml/models/<name> -q --q-bits 4 --q-group-size 64 --q-mode affine`.
  With huggingface_hub 1.32, run `snapshot_download(<repo>)` first — mlx-lm's save step
  otherwise fails on an incomplete cache.
- [x] `train_mlx.py` uses mlx-lm's trainer directly, not the `mlx_lm.lora` CLI, because four CLI
  defaults silently diverge from the PyTorch run (read from mlx-lm 0.31.3 source): its chat
  dataset omits `enable_thinking=False` (Qwen3 would learn to emit an empty `<think>` block), its
  batcher silently truncates over-length JSON targets, LoRA goes on only the last 16 layers, and
  `scale` defaults to 20 where our PEFT alpha/rank is 2.0.
- [x] Smoke test (Qwen3-0.6B, 25 optimizer steps): val loss 0.77 → 0.19, JSON parse rate
  0 → 0.81, verbatim span rate 0.98.
- [x] Measured on the M5 MacBook Air: 4B **0.556 it/s, 5.0GB peak**; 8B **0.283 it/s, 7.8GB
  peak** — ~2.25h per epoch for the 4B, which is what moved the main runs to SageMaker.
- MLX and PEFT adapters aren't interchangeable.

---

## Phase 6 — Evaluation (2-3 days — do not skip this for a demo-only fine-tune)

- [x] `ml/eval/evaluate.py` — **exact-match micro P/R/F1 over full quads**, the standard ACOS
  metric. The semantic-similarity matching originally planned here was designed for free-text
  *English* aspects; ours are verbatim spans of the input, so exact match is well-defined and
  stricter. Partial views localize failures: term / term+category / term+polarity /
  category+polarity (what the dashboard aggregates) / full quad, plus per-category and
  per-polarity F1. Generations are cached, so re-scoring never needs the model.
- [x] **Relaxed span-overlap views** (secondary — strict stays the headline). Reading the worst
  cases showed many strict "misses" were span-*boundary* disagreements: `炒蛋 → 凍晒` against a
  gold `炒蛋 → 已經凍晒` scored as a total miss. The `term (overlap)` and `full quad (overlap)`
  views accept term/opinion spans covering the same stretch of the review (at least half the
  shorter span, judged by position so a shared character elsewhere doesn't count), with category
  and polarity still exact, and pair predictions with gold one-to-one via maximum bipartite
  matching. A random sample of relaxed-only matches were all the same judgment with different
  boundaries (`冇乜特別` vs `唔差 但冇乜特別`, `Our server` vs `server`).

  | Synthetic test | 4B | 8B | 14B |
  |---|---|---|---|
  | Full quad, strict | 0.512 | 0.521 | 0.542 |
  | Full quad, overlap | 0.732 | 0.737 | 0.755 |
  | Term, strict | 0.652 | 0.646 | 0.667 |
  | Term, overlap | 0.808 | 0.803 | 0.821 |

  ~40% of strict misses are boundary disagreement, and the model ranking is unchanged — so the
  relaxed view explains the strict number rather than flattering any model.
- [x] `ml/eval/test_evaluate.py` — the project's first tests (15, all hand-computed): overlap by
  position, NULL handling, the half-span threshold, one-to-one crediting (including a case where
  greedy matching would under-count), and the strict scores unchanged.
- [ ] Run on `synthetic_test` and `real_test` side by side — **the gap is the headline
  finding**, not the synthetic number.
- [ ] `ml/eval/error_analysis.py` — pull the worst 15-20 examples, read them, categorize failure types (misses short reviews? struggles with code-switching? invents aspects not in the text?)
- [ ] Compare cost/latency: teacher API cost per review vs. your fine-tuned model's inference cost per review (this is your resume metric — note that a 7-8B model narrows the cost/latency gap vs. teacher compared to a smaller model, so be precise about the actual numbers rather than assuming a dramatic win; this is a real, discussable tradeoff, not a problem to hide)

**Checkpoint**: A metrics table (precision/recall/F1, cost per review, latency) and a short written error analysis. This phase is what turns the project into something you can defend under questioning.

---

## Phase 7 — Serve the fine-tuned model (2-3 days)

- [ ] `ml/serving/serve.py` — FastAPI wrapping the model (load base model + LoRA adapter, expose a `/extract` endpoint)
- [ ] At 7-8B, check whether Railway/Render's available instance sizes/GPU support fit your hosting needs — you may need a GPU-backed host (e.g. a small SageMaker endpoint, or a GPU instance on Railway/Lambda Labs) rather than a plain CPU container; quantization (e.g. 4-bit via bitsandbytes) is worth evaluating here to keep serving cost reasonable
- [ ] Containerize, deploy as its own service
- [ ] Update `app/` to call this service for new reviews instead of the teacher API directly
- [ ] Add basic latency logging on the serving endpoint

**Checkpoint**: New scraped reviews get processed by *your* model, not the teacher, end to end.

---

## Phase 8 — Polish for portfolio (2-3 days)

- [ ] `docs/model_card.md` — training data size/source, prompt/taxonomy approach, eval metrics, known failure modes, cost/latency comparison vs. teacher
- [ ] README with the architecture diagram, setup instructions, and a "why these decisions" section (why distillation, why LoRA, why this base model)
- [ ] Clean commit history / basic CI (lint + test on push) if you haven't already
- [ ] Optional: a short write-up (blog post or README section) walking through the distillation results — this is genuinely good interview prep material, since you'll basically have pre-written your answer to "tell me about a project"

---

## Rough total timeline

3-4 weeks part-time, working through phases in order. Phases 3-6 (labeling → validation → training → eval) are the ML-engineer core — if time gets tight, that's where to protect your time, and the infra phases (0-2, 7) are where you can move fastest since they're more mechanical.

## If you get stuck mid-phase

Come back with: which phase, what you tried, what broke. Happy to debug specific errors or review code as you go rather than only planning ahead of time.
