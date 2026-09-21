"""LoRA fine-tune for ACOS quad extraction.

One script for every environment. Data and output paths come from SageMaker's SM_CHANNEL_*
and SM_MODEL_DIR when present, and fall back to local paths otherwise, so the same file runs
on a laptop, a rented GPU, or a SageMaker training job with no branching.

Local smoke test on Apple silicon (verified end to end):
    python ml/training/train.py --model-id Qwen/Qwen2.5-0.5B-Instruct \\
        --max-examples 50 --epochs 1 --batch-size 1 --no-wandb

`--batch-size 1` is not optional on MPS. Qwen's 151,936-token vocabulary makes the logits
tensor (batch x seq x vocab) the memory bottleneck rather than the weights, and batch 4 at
seq 1024 OOMs a 0.5B model on a 32GB Mac.

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

from prompt_format import SYSTEM_PROMPT, build_target, render_prompt

CONFIG_PATH = Path(__file__).parent / "config.yaml"
CANARY_TEXT = "個叉燒好正，不過個waiter好慢。"


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


def encode(
    example: dict[str, Any], tokenizer: Any, max_seq_len: int
) -> dict[str, list[int]] | None:
    """Tokenize one example, masking the prompt so loss is computed on the JSON target only.

    Over-length examples are dropped rather than truncated: a truncated target is malformed
    JSON, and training on it teaches the model to emit malformed JSON.
    """
    prompt_ids = tokenizer(
        render_prompt(tokenizer, example["text"]), add_special_tokens=False
    ).input_ids
    target_ids = tokenizer(
        build_target(example["quads"]), add_special_tokens=False
    ).input_ids + [tokenizer.eos_token_id]

    if len(prompt_ids) + len(target_ids) > max_seq_len:
        return None

    return {
        "input_ids": prompt_ids + target_ids,
        "labels": [-100] * len(prompt_ids) + target_ids,
    }


def build_dataset(
    examples: list[dict[str, Any]], tokenizer: Any, max_seq_len: int, name: str
) -> list[dict[str, list[int]]]:
    encoded = [encode(e, tokenizer, max_seq_len) for e in examples]
    kept = [e for e in encoded if e is not None]
    dropped = len(encoded) - len(kept)
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
    """Measure what loss cannot: does the output parse, and are the spans really copied?"""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        examples: list[dict[str, Any]],
        max_new_tokens: int,
    ) -> None:
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

        parsed = 0
        spans_total = 0
        spans_verbatim = 0
        predicted = 0

        for example in self.examples:
            try:
                quads = json.loads(self._generate(example["text"]))
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(quads, list):
                continue
            parsed += 1
            predicted += len(quads)
            for quad in quads:
                if not isinstance(quad, dict):
                    continue
                for field in ("term", "opinion"):
                    value = quad.get(field)
                    if not isinstance(value, str):
                        continue
                    spans_total += 1
                    if value == "NULL" or value in example["text"]:
                        spans_verbatim += 1

        if metrics is not None:
            metrics["eval_json_parse_rate"] = parsed / len(self.examples)
            metrics["eval_span_verbatim_rate"] = (
                spans_verbatim / spans_total if spans_total else 0.0
            )
            metrics["eval_quads_per_review"] = predicted / parsed if parsed else 0.0

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
        # bitsandbytes is CUDA-only, so there is no 4-bit path on Mac or CPU.
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
    """Persist the exact rendered prompt so inference can assert it matches training.

    Qwen3's template injects an empty <think></think> block when thinking is disabled. If
    serving renders the prompt even slightly differently, quality drops in a way that is
    easy to misread as a bad adapter.
    """
    contract = {
        "system_prompt": SYSTEM_PROMPT,
        "rendered_example": render_prompt(tokenizer, CANARY_TEXT),
        "canary_text": CANARY_TEXT,
    }
    (output_dir / "prompt_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8"
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
