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
- [ ] AWS account set up, since fine-tuning will run on SageMaker (see Phase 5)

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

## Phase 3 — Teacher labeling of the real reviews (3-4 days)

**This is the gating work for every number the project reports.** Training data is currently
100% synthetic; the 312 scraped OpenRice reviews in the DB are still `status='pending'`. Until
they're labeled and validated, there is no honest headline metric.

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

## Phase 4 — Human validation (2-3 days, don't rush this)

This step is what makes your eval numbers later actually mean something.

- [ ] `ml/labeling/export_for_validation.py` — export ~150-200 examples (stratified: some short, some long, some code-switched) to a spreadsheet or a simple labeling tool
- [ ] Go through by hand: mark each teacher-generated aspect as correct/incorrect, add any aspects the teacher missed, fix sentiment errors
- [ ] Split this validated set: ~100 for a held-out **test set** (never touched again until final eval), ~50-100 as a **validation set** for tuning during training
- [ ] Everything else (teacher-only, unvalidated) becomes your **training set**

**Checkpoint**: Three clearly separated files — `train.jsonl`, `val.jsonl`, `test.jsonl`. This separation is one of the most interview-relevant things you'll do in this whole project — be ready to explain why the test set is untouched.

---

## Phase 5 — Fine-tune the generative model on SageMaker (1-1.5 weeks, most of it waiting on training runs)

- [x] `ml/training/prepare_dataset.py` — normalizes the raw synthetic set into span-verified ACOS quads and writes stratified train/val/synthetic_test splits. Enforces one invariant: **every emitted `opinion` is either the original label verbatim or a punctuation-delimited fragment of it, verbatim** — nothing synthesized. 23,908 raw labels → 32,283 quads (+35%), 4,987/5,000 reviews retained, 0 verbatim violations.
- [x] `ml/training/verify_dataset.py` — independent audit (re-declares the schema rather than importing it, so a bug in the normalizer can't validate itself): verbatim spans, split leakage, stratification.
- [ ] Set up a SageMaker execution role (IAM) with S3 read/write and ECR pull permissions
- [x] `ml/training/train.py` — LoRA fine-tune via PEFT on Qwen3-8B. One script for every environment: reads `SM_CHANNEL_*` / `SM_MODEL_DIR` when present, local paths otherwise. Key detail is **completion-only loss masking** (prompt tokens set to `-100`) — training on the prompt spends most of the gradient reproducing review text the model already sees. Over-length examples are dropped, never truncated, since a truncated target is malformed JSON.
- [x] `ml/training/launch_sagemaker.py` — thin PyTorch-estimator wrapper around the same `train.py`.
- [ ] Instance choice: `ml.g5.2xlarge` (A10G, 24GB) for standard runs; `ml.g5.12xlarge` or `ml.p4d.24xlarge` (A100, 40GB+) if you want faster iteration or larger batch sizes
- [ ] If you hit OOM, lower `per_device_train_batch_size` before anything else and buy the effective batch back with `gradient_accumulation_steps`. Qwen's vocab is 151,936 tokens, so the logits tensor (batch × seq × vocab) dominates memory rather than the weights — at seq 1024 that is ~1.2GB per copy in bf16, before the loss upcasts to fp32. This is why the local smoke test needs `--batch-size 1` even on a 0.5B model.
- [ ] Log to W&B: loss curves, hyperparameters, LoRA rank/alpha, learning rate, plus two
  generation-based metrics that loss cannot see — **`eval_json_parse_rate`** and
  **`eval_span_verbatim_rate`** (is the model copying spans, or inventing them?). A falling loss
  with a flat verbatim rate means it's learning the format and hallucinating the content.
- [ ] Start with a small SageMaker run (1 epoch, subset of data) to confirm the training script + container + S3 I/O all work end-to-end before committing to a full run — debugging a failed SageMaker job is slower than local, so validate the pipeline cheaply first
- [ ] Run 2-3 hyperparameter variants (learning rate, LoRA rank) as separate SageMaker jobs, compare via W&B
- [ ] Checkpoints land in S3 automatically via the SageMaker training job output path

**Checkpoint**: A saved LoRA checkpoint in S3, training curves logged, loss decreasing sensibly (not flat, not diverging). Bonus resume line: you now have hands-on SageMaker training job experience (IAM roles, S3 data flow, instance selection) in addition to the fine-tuning itself.

**Cost note**: `g5.2xlarge` runs roughly $1.20-1.50/hr on-demand. A LoRA run on a few thousand examples for 2-3 epochs is typically a few hours — expect $5-20 per experiment, ~$50-100 total across your sweep.

---

## Phase 6 — Evaluation (2-3 days — do not skip this for a demo-only fine-tune)

Free-text aspect extraction is harder to score than fixed-category classification, since your model's aspect phrasing won't exactly match the teacher's. Handle this properly:

- [ ] `ml/eval/evaluate.py` — run your fine-tuned model on the held-out **test set**
- [ ] Since aspects are free text, use **semantic similarity matching** (e.g., embed both teacher-aspect and model-aspect with a sentence embedding model, match if cosine similarity > threshold) rather than exact string match, to compute precision/recall/F1
- [ ] Score sentiment classification separately (this one *can* be exact-match: positive/negative/neutral)
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
