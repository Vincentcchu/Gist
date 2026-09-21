"""LoRA fine-tune for ACOS quad extraction.

One script for every environment. Data and output paths come from SageMaker's SM_CHANNEL_*
and SM_MODEL_DIR when present, and fall back to local paths otherwise, so the same file runs
on a laptop, a rented GPU, or a SageMaker training job with no branching.

Local smoke test on Apple silicon (verified end to end):
    python ml/training/train.py --model-id Qwen/Qwen2.5-0.5B-Instruct \\
        --max-examples 30 --epochs 2 --batch-size 1 --no-wandb

What a run produces in the output dir (SM_MODEL_DIR on SageMaker, so it all lands in
model.tar.gz):
    adapter_model.safetensors, adapter_config.json   the best epoch, by synthetic val loss
    checkpoints/checkpoint-*/                          every epoch's adapter, for comparison
    prompt_contract.json                               exact rendered prompt, for serving
    eval/<test set>.predictions.jsonl                  generations on each test file, in the
                                                       format evaluate.py --from-predictions reads

`--batch-size 1` is not optional on MPS. Qwen's 151,936-token vocabulary makes the logits
tensor (batch x seq x vocab) the memory bottleneck rather than the weights, and batch 4 at
seq 1024 OOMs a 0.5B model on a 24GB M5 MacBook Air.

This script is the PyTorch/PEFT path for CUDA (SageMaker, rented GPUs). Local QLoRA on
Apple silicon goes through train_mlx.py instead - see that file for why.

Full run:
    python ml/training/train.py
"""

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from prompt_format import (
    encode_example,
    generation_metrics,
    prompt_contract,
    render_prompt,
)

CONFIG_PATH = Path(__file__).parent / "config.yaml"


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    data_dir = Path(os.environ.get("SM_CHANNEL_TRAIN", args.data_dir))
    val_dir = Path(os.environ.get("SM_CHANNEL_VAL", args.data_dir))
    output_dir = Path(os.environ.get("SM_MODEL_DIR", args.output_dir))
    return data_dir / "train.jsonl", val_dir / "val.jsonl", output_dir


def resolve_eval_files(args: argparse.Namespace) -> list[Path]:
    """Test files to generate predictions for after training.

    On SageMaker, every .jsonl in the `eval` channel; locally, --eval-files that exist.
    Only review text is read, so an unlabeled real test set works fine here.
    """
    channel = os.environ.get("SM_CHANNEL_EVAL")
    if channel:
        return sorted(Path(channel).glob("*.jsonl"))
    return [path for path in args.eval_files if path.exists()]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_dataset(
    examples: list[dict[str, Any]], tokenizer: Any, max_seq_len: int, name: str
) -> list[dict[str, list[int]]]:
    kept: list[dict[str, list[int]]] = []
    for example in examples:
        encoded = encode_example(example, tokenizer, max_seq_len)
        if encoded is None:
            continue
        token_ids, prompt_length = encoded
        kept.append(
            {
                "input_ids": token_ids,
                "labels": [-100] * prompt_length + token_ids[prompt_length:],
            }
        )
    dropped = len(examples) - len(kept)
    print(f"  {name}: {len(kept)} examples ({dropped} dropped as over-length)")
    return kept


@dataclass
class PadCollator:
    pad_token_id: int

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        width = max(len(f["input_ids"]) for f in features)
        batch: dict[str, list[list[int]]] = {
            "input_ids": [],
            "labels": [],
            "attention_mask": [],
        }
        for feature in features:
            pad = width - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [self.pad_token_id] * pad)
            batch["labels"].append(feature["labels"] + [-100] * pad)
            batch["attention_mask"].append([1] * len(feature["input_ids"]) + [0] * pad)
        return {key: torch.tensor(value) for key, value in batch.items()}


class GenerationEval(TrainerCallback):
    """Measure what loss cannot: does the output parse, and are the spans really copied?

    Logs through the trainer rather than by adding to `metrics`: Trainer.evaluate() logs its
    metrics *before* calling on_evaluate, so anything added to that dict here is never logged.
    """

    def __init__(
        self,
        trainer: Trainer,
        model: Any,
        tokenizer: Any,
        examples: list[dict[str, Any]],
        max_new_tokens: int,
    ) -> None:
        self.trainer = trainer
        self.model = model
        self.tokenizer = tokenizer
        self.examples = examples
        self.max_new_tokens = max_new_tokens

    def _generate(self, text: str) -> str:
        inputs = self.tokenizer(
            render_prompt(self.tokenizer, text), return_tensors="pt"
        ).to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        return self.tokenizer.decode(
            output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

    def on_evaluate(self, args, state, control, metrics=None, **kwargs) -> None:
        was_training = self.model.training
        self.model.eval()

        results = [(self._generate(e["text"]), e["text"]) for e in self.examples]
        self.trainer.log(
            {
                f"eval_{name}": value
                for name, value in generation_metrics(results).items()
            }
        )

        if was_training:
            self.model.train()


def gradient_accumulation(effective_batch: int, micro_batch: int) -> int:
    """Derive accumulation so the optimizer always sees training.effective_batch_size.

    Each model can then use the largest micro-batch its memory allows (--batch-size) without
    changing the optimization. train_mlx.py derives it the same way.
    """
    if effective_batch % micro_batch:
        raise ValueError(
            f"micro-batch {micro_batch} doesn't divide effective batch {effective_batch}"
        )
    return effective_batch // micro_batch


def load_wandb_key() -> None:
    """On SageMaker, fetch the W&B key from Secrets Manager into the environment.

    The launcher passes only the secret's *name*. The key itself never appears in the repo, the
    job definition (visible to anyone who can describe the job), or the logs.
    """
    secret_name = os.environ.get("WANDB_SECRET_NAME")
    if os.environ.get("WANDB_API_KEY") or not secret_name:
        return
    import boto3

    secret = boto3.client("secretsmanager").get_secret_value(SecretId=secret_name)
    os.environ["WANDB_API_KEY"] = secret["SecretString"].strip()


def generate_batch(
    model: Any, tokenizer: Any, texts: list[str], max_new_tokens: int
) -> list[str]:
    """Greedy generation for several reviews at once.

    Left padding keeps every prompt's last token adjacent to where generation starts; with right
    padding the shorter prompts would continue from pad tokens.
    """
    previous_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        inputs = tokenizer(
            [render_prompt(tokenizer, text) for text in texts],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
    finally:
        tokenizer.padding_side = previous_side
    return tokenizer.batch_decode(
        output[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )


def write_predictions(
    model: Any,
    tokenizer: Any,
    eval_files: list[Path],
    output_dir: Path,
    batch_size: int,
    max_new_tokens: int,
    limit: int | None,
) -> None:
    """Generate on each test file and save in evaluate.py's prediction-cache format.

    Scoring happens later, locally: `evaluate.py --adapter <run> --from-predictions`. Nothing
    here looks at labels, so no decision is ever made on the test sets.
    """
    model.eval()
    eval_dir = output_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    for path in eval_files:
        texts = [example["text"] for example in read_jsonl(path)][:limit]
        outputs: list[str] = []
        for start in range(0, len(texts), batch_size):
            outputs.extend(
                generate_batch(
                    model, tokenizer, texts[start : start + batch_size], max_new_tokens
                )
            )
            print(f"  predictions {path.stem}: {len(outputs)}/{len(texts)}", flush=True)
        with (eval_dir / f"{path.stem}.predictions.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for output in outputs:
                handle.write(json.dumps({"output": output}, ensure_ascii=False) + "\n")


def compute_warmup_steps(
    num_examples: int, batch_size: int, grad_accum: int, epochs: float, ratio: float
) -> int:
    """transformers 5.x dropped `warmup_ratio`, so derive the step count here.

    Keeping the ratio in config means a 50-example smoke run and a 4,498-example full run both
    warm up over the same fraction of training, instead of a fixed count that would overshoot
    the short run entirely.
    """
    steps_per_epoch = math.ceil(num_examples / (batch_size * grad_accum))
    total_steps = math.ceil(steps_per_epoch * epochs)
    return max(1, int(total_steps * ratio))


def load_model(config: dict[str, Any], model_id: str) -> Any:
    on_cuda = torch.cuda.is_available()
    kwargs: dict[str, Any] = {"attn_implementation": "sdpa"}

    if on_cuda:
        kwargs["dtype"] = torch.bfloat16
        if config["quantization"]["load_in_4bit"]:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
    else:
        # No 4-bit path off CUDA here. Local quantized training uses train_mlx.py.
        kwargs["dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if "quantization_config" in kwargs:
        model = prepare_model_for_kbit_training(model)

    lora = config["lora"]
    model = get_peft_model(
        model,
        LoraConfig(
            r=lora["r"],
            lora_alpha=lora["alpha"],
            lora_dropout=lora["dropout"],
            target_modules=lora["target_modules"],
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()
    return model


def save_prompt_contract(output_dir: Path, tokenizer: Any) -> None:
    (output_dir / "prompt_contract.json").write_text(
        json.dumps(prompt_contract(tokenizer), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--data-dir", type=str, default="ml/data/processed")
    parser.add_argument("--output-dir", type=str, default="ml/outputs/run")
    parser.add_argument(
        "--eval-files",
        type=Path,
        nargs="*",
        default=[
            Path("ml/data/processed/synthetic_test.jsonl"),
            Path("ml/data/real/real_test.jsonl"),
        ],
        help="test files to predict on after training (SageMaker uses the eval channel)",
    )
    parser.add_argument("--run-name", type=str, default=None, help="W&B run name")
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--no-generation-eval", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())

    model_id = args.model_id or config["model_id"]
    train_cfg = config["training"]
    epochs = args.epochs if args.epochs is not None else train_cfg["epochs"]
    lr = (
        args.learning_rate
        if args.learning_rate is not None
        else float(train_cfg["learning_rate"])
    )
    batch_size = args.batch_size or train_cfg["per_device_train_batch_size"]
    grad_accum = gradient_accumulation(train_cfg["effective_batch_size"], batch_size)

    set_seed(config["seed"])
    train_path, val_path, output_dir = resolve_paths(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_wandb = config["wandb"]["enabled"] and not args.no_wandb
    if use_wandb:
        os.environ.setdefault("WANDB_PROJECT", config["wandb"]["project"])
        load_wandb_key()

    print(f"model: {model_id}")
    print(f"train: {train_path}\nval:   {val_path}\nout:   {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_raw = read_jsonl(train_path)
    val_raw = read_jsonl(val_path)
    if args.max_examples:
        train_raw = train_raw[: args.max_examples]
        val_raw = val_raw[: max(4, args.max_examples // 10)]

    print("tokenizing:")
    max_seq_len = config["max_seq_len"]
    train_ds = build_dataset(train_raw, tokenizer, max_seq_len, "train")
    val_ds = build_dataset(val_raw, tokenizer, max_seq_len, "val")

    model = load_model(config, model_id)

    on_cuda = torch.cuda.is_available()
    warmup_steps = compute_warmup_steps(
        num_examples=len(train_ds),
        batch_size=batch_size,
        grad_accum=grad_accum,
        epochs=epochs,
        ratio=train_cfg["warmup_ratio"],
    )
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(output_dir / "checkpoints"),
            num_train_epochs=epochs,
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            warmup_steps=warmup_steps,
            lr_scheduler_type=train_cfg["lr_scheduler_type"],
            logging_steps=train_cfg["logging_steps"],
            # One eval and one checkpoint per epoch, all kept: whether epoch 2 or 3 helped is
            # then answered by the checkpoints rather than guessed. The final adapter is the
            # best epoch by *synthetic* val loss - a legitimate selection signal; the test
            # sets are never consulted.
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=None,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            # Adapter weights only - the optimizer state would add ~1.5GB per checkpoint on
            # the 14B, all of it shipped back in model.tar.gz for nothing.
            save_only_model=True,
            run_name=args.run_name,
            bf16=on_cuda,
            gradient_checkpointing=on_cuda,
            report_to="wandb" if use_wandb else "none",
            seed=config["seed"],
            # Our collator emits input_ids/labels/attention_mask; the default column
            # pruning would strip them, since the dataset is plain dicts.
            remove_unused_columns=False,
        ),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=PadCollator(tokenizer.pad_token_id),
    )

    gen_cfg = config["generation_eval"]
    if gen_cfg["enabled"] and not args.no_generation_eval:
        trainer.add_callback(
            GenerationEval(
                trainer=trainer,
                model=model,
                tokenizer=tokenizer,
                examples=val_raw[: gen_cfg["num_samples"]],
                max_new_tokens=gen_cfg["max_new_tokens"],
            )
        )

    trainer.train()
    print(
        f"\nbest checkpoint: {trainer.state.best_model_checkpoint} "
        f"(eval_loss {trainer.state.best_metric})"
    )

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    save_prompt_contract(output_dir, tokenizer)
    print(f"adapter + tokenizer + prompt contract saved to {output_dir}")

    eval_files = resolve_eval_files(args)
    if eval_files:
        pred_cfg = config["prediction"]
        print(f"\ngenerating predictions for {[p.name for p in eval_files]}")
        write_predictions(
            model,
            tokenizer,
            eval_files,
            output_dir,
            batch_size=pred_cfg["batch_size"],
            max_new_tokens=pred_cfg["max_new_tokens"],
            limit=max(4, args.max_examples // 10) if args.max_examples else None,
        )


if __name__ == "__main__":
    main()
