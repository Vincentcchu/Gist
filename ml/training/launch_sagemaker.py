"""Launch train.py as a SageMaker training job (SageMaker Python SDK v3).

A thin wrapper - all training logic lives in train.py, which runs identically here and locally.
This file runs on your laptop only:

    pip install sagemaker==3.22.1 "botocore[crt]"   # crt: needed for `aws login` credentials
    aws login --region us-east-1                      # as an IAM user, never root
    export SAGEMAKER_ROLE_ARN=arn:aws:iam::<ACCOUNT_ID>:role/review-absa-sagemaker-training

Validate the job request without launching anything (free):
    python ml/training/launch_sagemaker.py \\
        --model-id Qwen/Qwen3-0.6B --max-examples 200 --epochs 1 --dry-run

Pipeline smoke test (~$0.50), then a 14B fit/speed check before any full run:
    ... --model-id Qwen/Qwen3-0.6B --max-examples 200 --epochs 1 --wait
    ... --model-id Qwen/Qwen3-14B --batch-size 1 --max-examples 64 --epochs 1 --wait

The sweep (one at a time - the GPU quota is 1):
    ... --model-id Qwen/Qwen3-4B-Instruct-2507 --batch-size 4
    ... --model-id Qwen/Qwen3-8B --batch-size 2
    ... --model-id Qwen/Qwen3-14B --batch-size 1

When a job finishes, the printed commands fetch its model.tar.gz into ml/outputs/<job>/ and
score its cached predictions with evaluate.py --from-predictions.
"""

import argparse
import os
import shutil
import tempfile
from pathlib import Path

from botocore.exceptions import (
    ConnectionClosedError,
    EndpointConnectionError,
    ReadTimeoutError,
)
from sagemaker.core import image_uris
from sagemaker.core.helper.session_helper import Session
from sagemaker.core.training.configs import (
    Compute,
    InputData,
    SourceCode,
    StoppingCondition,
)
from sagemaker.train.model_trainer import ModelTrainer

SOURCE_DIR = Path(__file__).resolve().parent
REPO_ROOT = SOURCE_DIR.parents[1]

# Newest SageMaker PyTorch training container (verified against the SDK's bundled image
# config and AWS's published list). torch is deliberately unpinned in requirements.txt so
# this container's CUDA build is the one that runs.
PYTORCH_VERSION = "2.10.0"
PY_VERSION = "py313"

CHANNEL_FILES = {
    "train": ["ml/data/processed/train.jsonl"],
    "val": ["ml/data/processed/val.jsonl"],
    # Test sets go in their own channel. train.py only generates on them after training; the
    # train and val channels never contain them, so nothing in training can read test data.
    "eval": [
        "ml/data/processed/synthetic_test.jsonl",
        "ml/data/real/real_test.jsonl",
    ],
}


def stage_channels(staging: Path) -> list[InputData]:
    """Copy each channel's files into its own directory; the SDK uploads each one to S3."""
    inputs = []
    for channel, files in CHANNEL_FILES.items():
        channel_dir = staging / channel
        channel_dir.mkdir(parents=True)
        for relative in files:
            source = REPO_ROOT / relative
            if not source.exists():
                raise SystemExit(
                    f"missing {relative} - run prepare_dataset.py / export_for_validation.py"
                )
            shutil.copy(source, channel_dir / source.name)
        inputs.append(InputData(channel_name=channel, data_source=str(channel_dir)))
    return inputs


def job_basename(model_id: str) -> str:
    """'Qwen/Qwen3-4B-Instruct-2507' -> 'absa-qwen3-4b-instruct-2507' (SageMaker name rules)."""
    name = model_id.split("/")[-1].lower().replace(".", "-").replace("_", "-")
    return f"absa-{name}"[:50]


def hyperparameters(args: argparse.Namespace) -> dict[str, object]:
    """Passed to train.py as `--key value` CLI arguments by the SDK's container driver."""
    values: dict[str, object] = {"model-id": args.model_id}
    optional = {
        "batch-size": args.batch_size,
        "epochs": args.epochs,
        "learning-rate": args.learning_rate,
        "max-examples": args.max_examples,
    }
    values.update({key: value for key, value in optional.items() if value is not None})
    return values


def environment(args: argparse.Namespace) -> dict[str, str]:
    # The training container sets no default AWS region, so boto3 inside it (used to fetch
    # the W&B secret) has to be told. Without this the first smoke job died on NoRegionError.
    env = {"AWS_DEFAULT_REGION": args.region}
    if args.no_wandb:
        return {**env, "WANDB_MODE": "disabled"}
    # The secret's name, never the key: train.py fetches the value from Secrets Manager
    # inside the container, so it doesn't appear in the job definition.
    return {
        **env,
        "WANDB_PROJECT": "review-absa",
        "WANDB_SECRET_NAME": args.wandb_secret,
    }


def print_fetch_commands(job_name: str, artifact_uri: str | None) -> None:
    uri = artifact_uri or (
        f"$(aws sagemaker describe-training-job --training-job-name {job_name} "
        "--query ModelArtifacts.S3ModelArtifacts --output text)"
    )
    print(
        "\nfetch and score:\n"
        f"  mkdir -p ml/outputs/{job_name}\n"
        f"  aws s3 cp {uri} - | tar -xz -C ml/outputs/{job_name}\n"
        f"  python ml/eval/evaluate.py --adapter ml/outputs/{job_name} --from-predictions"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--role",
        default=os.environ.get("SAGEMAKER_ROLE_ARN"),
        help="execution role ARN (or set SAGEMAKER_ROLE_ARN)",
    )
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--instance-type", default="ml.g5.2xlarge")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument(
        "--max-hours",
        type=float,
        default=12,
        help="hard runtime cap - SageMaker stops the job (and the billing) after this",
    )
    parser.add_argument("--wandb-secret", default="review-absa/wandb-api-key")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument(
        "--wait", action="store_true", help="stream logs until the job ends"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="validate the request without launching"
    )
    args = parser.parse_args()
    if not args.role:
        parser.error("--role is required (or set SAGEMAKER_ROLE_ARN)")
    return args


def main() -> None:
    args = parse_args()
    import boto3

    session = Session(boto_session=boto3.Session(region_name=args.region))
    image = image_uris.retrieve(
        framework="pytorch",
        region=args.region,
        version=PYTORCH_VERSION,
        py_version=PY_VERSION,
        instance_type=args.instance_type,
        image_scope="training",
    )
    print(f"image: {image}")

    trainer = ModelTrainer(
        sagemaker_session=session,
        role=args.role,
        base_job_name=job_basename(args.model_id),
        training_image=image,
        source_code=SourceCode(
            source_dir=str(SOURCE_DIR),
            entry_script="train.py",
            requirements="requirements.txt",
        ),
        compute=Compute(
            instance_type=args.instance_type,
            instance_count=1,
            # Headroom for the base model download: Qwen3-14B is ~30GB in bf16.
            volume_size_in_gb=100,
        ),
        stopping_condition=StoppingCondition(
            max_runtime_in_seconds=int(args.max_hours * 3600)
        ),
        hyperparameters=hyperparameters(args),
        environment=environment(args),
    )

    watched_to_end = args.wait
    with tempfile.TemporaryDirectory() as staging:
        try:
            trainer.train(
                input_data_config=stage_channels(Path(staging)),
                wait=args.wait,
                logs=args.wait,
                dry_run=args.dry_run,
            )
        except (
            ConnectionClosedError,
            EndpointConnectionError,
            ReadTimeoutError,
        ) as error:
            # --wait streams CloudWatch logs from this laptop for the whole run. If that
            # connection drops, only the watcher is gone: once CreateTrainingJob succeeded,
            # the job runs entirely on AWS and is unaffected.
            if trainer._latest_training_job is None:
                raise
            print(
                f"\nlost the connection while watching ({type(error).__name__}). "
                "The job itself is unaffected and keeps running on AWS."
            )
            watched_to_end = False

    if args.dry_run:
        print("\ndry run: request validated, nothing launched")
        return

    job = trainer._latest_training_job
    print(f"\njob: {job.training_job_name}")
    if not watched_to_end:
        print(
            "follow it with:\n"
            f"  aws sagemaker describe-training-job --training-job-name {job.training_job_name}"
            " --query '[TrainingJobStatus,SecondaryStatus,FailureReason]'\n"
            "  aws logs tail /aws/sagemaker/TrainingJobs"
            f" --log-stream-name-prefix {job.training_job_name} --follow"
        )
    artifact = None
    if watched_to_end:
        job.refresh()
        artifact = job.model_artifacts.s3_model_artifacts
        print(f"status: {job.training_job_status}  artifact: {artifact}")
    print_fetch_commands(job.training_job_name, artifact)


if __name__ == "__main__":
    main()
