# review-absa

Aspect-based sentiment analysis for hotel/restaurant reviews (Hong Kong market,
Cantonese-English code-switching). Portfolio project targeting applied ML Engineer roles.

## Stack
- Backend: FastAPI, SQLAlchemy, DB via `DATABASE_URL` env var (SQLite locally by
  default, Postgres in prod on Railway/Render — same code, no branching needed)
- Frontend: Next.js on Vercel
- ML: teacher-labeled data (Claude/GPT API) -> LoRA fine-tune of
  Qwen2.5-7B-Instruct or Llama-3.1-8B-Instruct (PEFT/Unsloth) for free-text
  aspect + sentiment extraction. Chosen over a smaller (1.5-3B) model
  deliberately: real LoRA-on-a-real-sized-model experience + stronger quality
  ceiling, at the cost of higher serving latency/cost later (tracked in the
  model card, not hidden).
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
Phase 1 — SQLAlchemy models, FastAPI endpoints, seed script with fake data.
See BUILD_GUIDE.md for the full phase plan.

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
