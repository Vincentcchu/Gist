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
- Training: AWS SageMaker training jobs (not local/RunPod) — `ml.g5.2xlarge`
  (A10G, 24GB) as the default instance, stepping up to `ml.g5.12xlarge` /
  `ml.p4d` (A100) if iteration speed matters. Data/checkpoints via S3.
- Experiment tracking: Weights & Biases
- Deployment: Docker containers, no Kubernetes/Terraform for the app layer

## Conventions
- Python 3.11+ (use the conda env `review-absa`, not a nested venv), type
  hints everywhere, black formatting
- Plain `requirements.txt` per service, no Poetry
- SQLAlchemy models in app/models.py, one class per table
- Never commit secrets — use .env locally, Railway/AWS secrets in prod
- Keep functions small and testable; this is a learning project, prefer
  explicit/readable code over clever code

## Current phase
Phase 5 — LoRA training pipeline. `ml/data/processed/` holds span-verified ACOS
quads from the synthetic set; `ml/training/` has prepare/verify/train/launch.

Phases 3-4 on *real* data are still outstanding and are the gating work for any
headline metric: the 312 scraped OpenRice reviews in `app/local.db` are still
`status='pending'`, so the only labeled data is synthetic. A synthetic-test F1
is not a result — the real number needs the teacher pass
(`ml/labeling/prompts.py`) plus a human-validated split.

See docs/BUILD_GUIDE.md for the full phase plan.

## What NOT to build yet
No auth, no SQS/queues (use a `status` column), no Terraform/VPC/IAM for the
app layer. AWS is scoped specifically to SageMaker for training (Phase 5), not
the whole app's infra.

## How to work with me on this
- Use plan mode before implementing anything non-trivial — I want to review
  the approach before code gets written, since I'm learning as I go
- Explain non-obvious parts with comments or a short walkthrough after
  implementing — don't just execute silently
- `/clear` between phases; `/compact` if a session starts feeling slow
