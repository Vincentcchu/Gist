# review-absa

Aspect-based sentiment analysis for restaurant reviews (Hong Kong market,
Cantonese-English code-switching). Portfolio project targeting applied ML Engineer roles.

## Stack
- Backend: FastAPI, SQLAlchemy, DB via `DATABASE_URL` env var (SQLite locally by
  default, Postgres in prod on Railway/Render — same code, no branching needed)
- Frontend: Next.js on Vercel
- ML: synthetic labeled data (generated with Claude in a separate repo) -> LoRA
  fine-tune of **Qwen3-8B** (PEFT) for quad extraction in the **v2 label format** —
  term, description, polarity, opinion. There is no category: the description is
  an open English phrase ("drink texture"), and categories are applied afterwards
  from editable maps (description → group → level 1). Qwen over Llama for two task-specific reasons: its tokenizer
  encodes traditional Chinese at ~1-1.5 chars/token vs Llama-3.1's ~2-3 tokens
  per char (halves sequence length, and verbatim span copying is far more
  reliable over clean tokens), and Qwen3's pretraining covers 119
  languages/dialects vs Qwen2.5's 29, which matters for colloquial Cantonese
  (咗/嘅/㗎/喺). Qwen3-4B-Instruct-2507 is trained alongside as a cost/quality
  baseline for the model card. Chosen over a 1.5-3B model deliberately: real
  LoRA-on-a-real-sized-model experience + stronger quality ceiling, at the cost
  of higher serving latency/cost later (tracked in the model card, not hidden).
  Note Qwen3 is a hybrid-thinking model — the chat template must be identical
  between training and inference (see `ml/training/prompt_format.py`).
- Training: QLoRA (4-bit frozen base + LoRA adapters), never full fine-tuning. Two
  paths sharing one tokenizer (`prompt_format.py`), so both train on identical sequences:
  - **Main — PyTorch/PEFT on SageMaker** (`train.py`, `launch_sagemaker.py`, SDK v3):
    `ml.g5.2xlarge` (A10G, 24GB), PyTorch 2.10 training container, us-east-1. Model
    lineup **Qwen3-4B / 8B / 14B** — 14B is the ceiling on 24GB even under QLoRA (32B
    needs ~25GB+); 32B is a follow-up only if 8B→14B measurably shrinks the
    synthetic-vs-real gap.
  - **Local alternative — MLX** (`train_mlx.py`) on the M5 MacBook Air: built and
    verified, too slow for the main runs (4B 0.56 it/s, 8B 0.28 it/s). MLX and PEFT
    adapters aren't interchangeable.
- Experiment tracking: Weights & Biases
- Deployment: Docker containers, no Kubernetes/Terraform for the app layer

## Conventions
- Python 3.11+ (use the conda env `gist`, not a nested venv), type
  hints everywhere, black formatting
- Plain `requirements.txt` per service, no Poetry
- SQLAlchemy models in app/models.py, one class per table
- Never commit secrets — use .env locally, Railway/AWS secrets in prod
- Keep functions small and testable; this is a learning project, prefer
  explicit/readable code over clever code

## Repo scope
Gist does **no labeling**. The generator repo (`Synethic_review`) makes the synthetic
data; the annotator repo (`llm-absa-annotator`) produces every hand label, including
the gold. Gist exports and cleans scraped text, verifies data, trains, scores any
predictions file against gold, and serves. Label conventions: `ml/data/real/README.md`.

## Current phase
**v2 data is in (2026-10-04).** The synthetic set was regenerated in the v2 format
(5,000 reviews, 43,507 quads; provenance in `ml/data/raw/README.md`) and prepared into
`ml/data/processed_v2/`. `real_test.jsonl` was re-cleaned and is ready for hand
labeling. Retraining on v2 is on hold. The plan: train on synthetic and score on gold,
then compare with training on hand-labeled real reviews from new venues (no venue in
both training and gold).

Before v2: Phase 5 done → Phase 6 (evaluation). The 4B/8B/14B QLoRA sweep on the **synthetic
set** is complete on SageMaker: synthetic-test full-quad F1 0.512 / 0.521 / 0.542,
$19.26 total. Adapters and predictions live under `ml/outputs/<job>/` (gitignored;
originals in S3). Those v1 models output the six old categories. Full results, method and error analysis: `docs/model_card.md`
(fill in its §10 when the real-test scores exist).

**Scope:** synthetic data + SageMaker now → scraped real data + SageMaker later.
Phases 3-4 (teacher-labeling real reviews) are deferred, not skipped — scraping
real reviews proved harder than planned, so the product gets built end to end on
synthetic data first and data quality is upgraded afterwards. Consequences to
keep honest in the model card:
- Training data is 100% synthetic. The hand-labeled real set is eval-only.
- Headline numbers are the synthetic-vs-real *gap*, never synthetic F1 alone.
- The scraper has known defects: 富臨飯店 reviews are truncated previews
  (`…查看更多`) and some reviews carry scraped UI counters. See
  `ml/data/real/README.md`.

See docs/BUILD_GUIDE.md for the full phase plan.

## What NOT to build yet
No auth, no SQS/queues (use a `status` column), no Terraform/VPC/IAM for the
app layer. AWS is scoped specifically to SageMaker for training (deferred scale-up
stage), not the whole app's infra.

## How to work with me on this
- Use plan mode before implementing anything non-trivial — I want to review
  the approach before code gets written, since I'm learning as I go
- Explain non-obvious parts with comments or a short walkthrough after
  implementing — don't just execute silently
- `/clear` between phases; `/compact` if a session starts feeling slow
