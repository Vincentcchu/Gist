# Build Guide — Review ABSA MVP (Free-Text Aspects, Fine-Tuned Generative LLM)

Each phase produces something visible/testable before you move on. Don't skip ahead — the point of this order is that you're never debugging two unfamiliar things at once.

---

## Phase 0 — Setup (half a day)

- [ ] GitHub repo, structure from `PROJECT_STRUCTURE.md`
- [ ] Railway or Render account, Postgres instance provisioned
- [ ] Local Python env (3.11+), `poetry` or plain `venv`
- [ ] W&B account (free tier)
- [ ] Pick your teacher model (Claude/GPT via API) and get an API key
- [ ] Decide your base model now so you're not switching later: **Qwen2.5-7B-Instruct** or **Llama-3.1-8B-Instruct** — strong multilingual (including Chinese) instruction-following, well-supported by LoRA tooling (PEFT/Unsloth), and a meaningfully higher quality ceiling than smaller models for handling code-switching/sarcasm. Chosen deliberately over a 1.5-3B model to get real LoRA-on-a-real-sized-model experience and a stronger quality guarantee; the tradeoff is higher serving cost/latency later (worth noting in the model card).
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

## Phase 3 — Teacher labeling for free-text aspects (3-4 days)

Since you want *free-text* aspects (not fixed categories), your prompt design matters more here than in the fixed-taxonomy version — you're teaching the small model to imitate an open-ended extraction pattern, so the teacher's output format needs to be consistent.

- [ ] `ml/labeling/prompts.py` — design the extraction prompt. Ask for structured JSON output like:
  ```json
  [{"aspect": "check-in speed", "sentiment": "negative", "span": "waited 40 minutes to check in"}]
  ```
  Give 5-8 few-shot examples covering: English reviews, Cantonese-English code-switched reviews, short reviews, reviews with multiple aspects, reviews with sarcasm/mixed sentiment.
- [ ] `ml/labeling/label_with_teacher.py` — pulls `pending` reviews, calls teacher, writes results to `AspectExtraction` with `model_version = 'teacher-v1'`
- [ ] Run on ~50 reviews first, read every output by eye. Fix the prompt before scaling — cheap now, expensive after you've labeled 1000 reviews with a bad prompt.
- [ ] Run on your full backlog

**Checkpoint**: A table of reviews with free-text aspect/sentiment pairs attached, teacher-generated.

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

- [ ] `ml/training/prepare_dataset.py` — format each example as an instruction-tuning pair: input = review text + instruction, output = the JSON aspect list. Match the format your base model expects (check Qwen/Llama's chat template). Upload the prepared train/val splits to S3.
- [ ] Set up a SageMaker execution role (IAM) with S3 read/write and ECR pull permissions
- [ ] `ml/training/train.py` — LoRA fine-tune via PEFT (or Unsloth) on Qwen2.5-7B-Instruct or Llama-3.1-8B-Instruct, packaged as a SageMaker training job (HuggingFace estimator or a custom container)
- [ ] Instance choice: `ml.g5.2xlarge` (A10G, 24GB) for standard runs; `ml.g5.12xlarge` or `ml.p4d.24xlarge` (A100, 40GB+) if you want faster iteration or larger batch sizes
- [ ] Log to W&B: loss curves, hyperparameters, LoRA rank/alpha, learning rate
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
