# review-absa

Aspect-based sentiment analysis for restaurant reviews (Hong Kong market,
Cantonese-English code-switching). Portfolio project targeting applied ML Engineer roles.

## Stack
- Backend: FastAPI, SQLAlchemy, DB via `DATABASE_URL` env var (SQLite locally by
  default, Postgres in prod on Railway/Render — same code, no branching needed)
- Frontend: Next.js on Vercel
- ML: teacher-labeled data (Claude API) -> LoRA fine-tune of **Qwen3-8B**
  (PEFT) for **ACOS** quad extraction — aspect term, category, opinion span,
  sentiment. Qwen over Llama for two task-specific reasons: its tokenizer
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
- Training, two deliberate paths sharing one tokenizer (`prompt_format.py`):
  - **Now — local MLX QLoRA** on the M5 MacBook Air (24GB): `ml/training/train_mlx.py`
    on 4-bit models converted with `mlx_lm.convert`. Measured: 4B 0.56 it/s / 5.0GB,
    8B 0.28 it/s / 7.8GB peak.
  - **Later — PyTorch/PEFT on SageMaker** (`train.py`, `launch_sagemaker.py`):
    `ml.g5.2xlarge` (A10G, 24GB) default. Deferred, not cancelled — having both
    frameworks is intentional experience, not duplication.
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

## Current phase
Phase 5 — training on the **synthetic set only**, locally via MLX. Span-verified
ACOS quads in `ml/data/processed/`; the real test set (50 hand-labeled OpenRice
reviews) in `ml/data/real/`.

**Scope decision:** Phases 3-4 (teacher-labeling real reviews) are *deferred*,
not skipped — scraping real reviews proved harder than planned, so the product
gets built end to end on synthetic data first and data quality is upgraded
afterwards. Consequences to keep honest in the model card:
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
