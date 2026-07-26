# Project Structure — Applied ML Engineer MVP

## Repo layout

```
Gist/
├── app/                        # main backend, deployed on Railway/Render
│   ├── main.py                 # FastAPI app: endpoints for dashboard
│   ├── models.py                # SQLAlchemy models (Review, Aspect, Hotel)
│   ├── db.py                    # DB connection/session setup
│   ├── health.py                 # /health endpoint
│   ├── requirements.txt
│   └── Dockerfile
│
├── scraper/                     # scheduled job, separate container
│   ├── scrape.py                # Playwright scraping logic
│   ├── requirements.txt
│   └── Dockerfile
│
├── ml/                          # everything ML-engineer-relevant lives here
│   ├── labeling/
│   │   ├── label_with_teacher.py     # calls teacher LLM, writes labels to DB
│   │   ├── prompts.py                 # your aspect taxonomy + few-shot examples
│   │   └── export_for_validation.py   # dumps a sample to CSV/label-studio for human review
│   │
│   ├── training/
│   │   ├── prepare_dataset.py         # validated labels -> train/val/test splits
│   │   ├── train.py                    # fine-tuning script (XLM-R or LoRA)
│   │   ├── config.yaml                 # hyperparameters
│   │   └── wandb_setup.py
│   │
│   ├── eval/
│   │   ├── evaluate.py                 # precision/recall/F1 per aspect, vs teacher
│   │   └── error_analysis.py           # confusion matrix, worst-case examples
│   │
│   └── serving/
│       ├── serve.py                     # FastAPI wrapping the fine-tuned model
│       ├── model_loader.py              # loads best checkpoint
│       ├── requirements.txt
│       └── Dockerfile
│
├── frontend/                     # Next.js dashboard, deployed on Vercel
│   └── ...
│
└── docs/
    ├── architecture.mermaid
    └── model_card.md             # document your model's training data, metrics, limitations
```

## Suggested build order

1. **`app/` + `frontend/` with fake/seed data** — get the dashboard rendering something, even hardcoded, so you have a visible product almost immediately. Motivation matters on a first solo project.
2. **`scraper/`** — get real reviews flowing into Postgres with `status = pending`.
3. **`ml/labeling/`** — prompt the teacher LLM on a handful of reviews, inspect outputs by eye before scaling up.
4. **Human validation pass** — correct ~50-100 labeled examples. This becomes your held-out eval set later, so keep it separate from what you bulk-label next.
5. **Bulk labeling** — run the validated prompt across your full pending set.
6. **`ml/training/`** — fine-tune XLM-R (or LoRA on a small LLM) on the labeled set. Log every run to W&B — hyperparameters, loss curves, final metrics.
7. **`ml/eval/`** — score the fine-tuned model against your human-validated held-out set. This is what turns "I fine-tuned a model" into "I fine-tuned a model and proved it works" — the second one is what gets you hired.
8. **`ml/serving/`** — wrap the winning checkpoint in FastAPI, deploy as its own service, point `app/` at it for new reviews instead of calling the teacher LLM directly.
9. **`docs/model_card.md`** — a short writeup: training data size/source, aspect taxonomy, eval metrics, known limitations (e.g., "underperforms on code-switched Cantonese-English reviews with <10 words"). This single file is a disproportionately good portfolio artifact — it signals rigor that most bootcamp-style projects skip entirely.

## What NOT to build right now

- Terraform / IaC
- VPC, IAM roles, ALB
- SQS (the `status` column on your reviews table does this job at your scale)
- Auth/OAuth (still scraping for now)

You can revisit the AWS/Terraform version later as a v2 migration story once the model + product actually work — that's a stronger narrative than starting there.
