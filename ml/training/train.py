"""LoRA fine-tune for ACOS quad extraction.

One script for every environment. Data and output paths come from SageMaker's SM_CHANNEL_*
and SM_MODEL_DIR when present, and fall back to local paths otherwise, so the same file runs
on a laptop, a rented GPU, or a SageMaker training job with no branching.

Local smoke test on Apple silicon (verified end to end):
    python ml/training/train.py --model-id Qwen/Qwen2.5-0.5B-Instruct \\
        --max-examples 50 --epochs 1 --batch-size 1 --no-wandb

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

    set_seed(config["seed"])
    train_path, val_path, output_dir = resolve_paths(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_wandb = config["wandb"]["enabled"] and not args.no_wandb
    if use_wandb:
        os.environ.setdefault("WANDB_PROJECT", config["wandb"]["project"])

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
        grad_accum=train_cfg["gradient_accumulation_steps"],
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
            gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
            warmup_steps=warmup_steps,
            lr_scheduler_type=train_cfg["lr_scheduler_type"],
            logging_steps=train_cfg["logging_steps"],
            eval_strategy="steps",
            eval_steps=train_cfg["eval_steps"],
            save_steps=train_cfg["save_steps"],
            save_total_limit=2,
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

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    save_prompt_contract(output_dir, tokenizer)
    print(f"\nadapter + tokenizer + prompt contract saved to {output_dir}")


if __name__ == "__main__":
    main()
