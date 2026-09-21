"""Launch train.py as a SageMaker training job.

This is a thin wrapper - all the real logic lives in train.py, which runs identically here
and locally. Run the local smoke test first; debugging a failed SageMaker job is far slower
than debugging the same bug on a laptop.

    pip install sagemaker boto3
    python ml/training/launch_sagemaker.py --role arn:aws:iam::<account>:role/<SageMakerRole>

Start small, then scale:
    --subset 500 --epochs 1        # validate container + S3 I/O cheaply
    --instance-type ml.g5.12xlarge # faster iteration once it works
"""

import argparse
from pathlib import Path

import sagemaker
from sagemaker.pytorch import PyTorch

SOURCE_DIR = Path(__file__).parent
LOCAL_DATA = Path("ml/data/processed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--role", type=str, required=True, help="SageMaker execution role ARN"
    )
    parser.add_argument("--instance-type", type=str, default="ml.g5.2xlarge")
    parser.add_argument(
        "--framework-version",
        type=str,
        default="2.14.0",
        help="SageMaker PyTorch DLC version; keep matched to the torch pin",
    )
    parser.add_argument("--s3-prefix", type=str, default="review-absa")
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument(
        "--subset", type=int, default=None, help="cap training examples"
    )
    parser.add_argument("--job-name", type=str, default=None)
    parser.add_argument(
        "--wait", action="store_true", help="stream logs until the job ends"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    session = sagemaker.Session()

    data_uri = session.upload_data(
        path=str(LOCAL_DATA),
        bucket=session.default_bucket(),
        key_prefix=f"{args.s3_prefix}/data",
    )
    print(f"data uploaded to {data_uri}")

    hyperparameters: dict[str, object] = {}
    if args.model_id:
        hyperparameters["model-id"] = args.model_id
    if args.epochs is not None:
        hyperparameters["epochs"] = args.epochs
    if args.learning_rate is not None:
        hyperparameters["learning-rate"] = args.learning_rate
    if args.subset is not None:
        hyperparameters["max-examples"] = args.subset

    estimator = PyTorch(
        entry_point="train.py",
        source_dir=str(SOURCE_DIR),
        role=args.role,
        instance_type=args.instance_type,
        instance_count=1,
        # Keep this matched to the torch pin in requirements.txt. If they diverge, pip
        # reinstalls torch inside the container on every job - slow, and it can land a
        # build that doesn't match the instance's CUDA. Verify the tag exists first:
        #   aws sagemaker list-images / the DLC release notes on GitHub.
        framework_version=args.framework_version,
        py_version="py311",
        hyperparameters=hyperparameters,
        base_job_name=args.job_name or "review-absa-acos",
        # W&B needs its key inside the container: store it in Secrets Manager or pass
        # WANDB_API_KEY via environment= before the first real run.
        environment={"WANDB_PROJECT": "review-absa"},
    )

    estimator.fit({"train": data_uri, "val": data_uri}, wait=args.wait)
    print(f"job: {estimator.latest_training_job.name}")
    if args.wait:
        print(f"adapter artifacts: {estimator.model_data}")


if __name__ == "__main__":
    main()
