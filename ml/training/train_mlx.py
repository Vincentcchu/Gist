"""QLoRA fine-tune for ACOS quad extraction on Apple silicon, via MLX.

Sibling of train.py, which is the PyTorch/PEFT path for CUDA (SageMaker, rented GPUs). Both read
config.yaml and tokenize through prompt_format.encode_example, so they train on byte-identical
token sequences and their results stay comparable.

Why this uses mlx-lm's trainer directly instead of the `mlx_lm.lora` CLI: four of the CLI's
behaviours would silently diverge from the PyTorch run (verified against mlx-lm 0.31.3 source):

1. Its ChatDataset renders the chat template without `enable_thinking=False`. On Qwen3-8B that
   leaves an empty <think></think> block inside the trained region, so the model learns to emit
   a sequence it never sees at inference. We tokenize ourselves instead.
2. Its batcher truncates over-length sequences with only a warning, cutting JSON targets
   mid-object. encode_example drops them instead.
3. LoRA goes on only the last 16 layers by default; PEFT adapts every layer.
4. LoRA `scale` is the multiplier itself (default 20.0), where PEFT uses alpha / rank. Our
   config's 64 / 32 = 2.0 - left at default the adapter update would be 10x larger.

The base model must already be quantized to 4-bit (mlx_lm.convert, see BUILD_GUIDE Phase 5):
training LoRA on top of quantized weights is what makes this QLoRA.

Smoke test:
    python ml/training/train_mlx.py --model-path ml/models/qwen3-0.6b-4bit \\
        --max-examples 50 --epochs 1 --no-wandb --output-dir ml/outputs/smoke-mlx

Full runs:
    python ml/training/train_mlx.py --output-dir ml/outputs/qwen3-8b
    python ml/training/train_mlx.py --model-path ml/models/qwen3-4b-instruct-2507-4bit \\
        --output-dir ml/outputs/qwen3-4b
"""

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import yaml
from mlx_lm import generate, load
from mlx_lm.tuner.callbacks import TrainingCallback
from mlx_lm.tuner.trainer import TrainingArgs, train
from mlx_lm.tuner.utils import linear_to_lora_layers, print_trainable_parameters

from prompt_format import (
    encode_example,
    generation_metrics,
    prompt_contract,
    render_prompt,
)

CONFIG_PATH = Path(__file__).parent / "config.yaml"
REPO_ROOT = Path(__file__).resolve().parents[2]


def portable_model_path(model_path: str) -> str:
    """Record local model paths relative to the repo root, whatever directory we ran from.

    A path stored as typed ('../models/x') breaks as soon as anything loads the adapter from a
    different working directory - mlx-lm then treats it as a Hugging Face repo id. Hub ids are
    returned unchanged.
    """
    path = Path(model_path)
    if not path.exists():
        return model_path
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_dataset(
    examples: list[dict[str, Any]], tokenizer: Any, max_seq_len: int, name: str
) -> list[tuple[list[int], int]]:
    """Pre-tokenize into (token_ids, prompt_length) tuples.

    mlx-lm's batcher accepts any dataset whose items are these tuples, and masks the loss to
    tokens from prompt_length onward - the MLX equivalent of train.py's -100 labels.
    """
    encoded = [encode_example(e, tokenizer, max_seq_len) for e in examples]
    kept = [e for e in encoded if e is not None]
    print(
        f"  {name}: {len(kept)} examples ({len(encoded) - len(kept)} dropped as over-length)"
    )
    return kept


def gradient_accumulation(effective_batch: int, micro_batch: int) -> int:
    """Derive accumulation so the optimizer always sees training.effective_batch_size.

    Both trainers derive it the same way, so their runs stay comparable whatever micro-batch
    each one's memory allows.
    """
    if effective_batch % micro_batch:
        raise ValueError(
            f"micro-batch {micro_batch} doesn't divide effective batch {effective_batch}"
        )
    return effective_batch // micro_batch


def lora_keys(model: nn.Module, target_modules: list[str]) -> set[str]:
    """Map config's short names (q_proj) to MLX's in-layer paths (self_attn.q_proj).

    Fails loudly if any target is missing, rather than silently adapting fewer modules than
    the PyTorch run does.
    """
    keys = {
        name
        for name, module in model.layers[0].named_modules()
        if name.split(".")[-1] in target_modules
        and isinstance(module, (nn.Linear, nn.QuantizedLinear))
    }
    found = {key.split(".")[-1] for key in keys}
    missing = set(target_modules) - found
    if missing:
        raise ValueError(f"target modules not found in model: {sorted(missing)}")
    return keys


def apply_lora(model: nn.Module, config: dict[str, Any]) -> dict[str, Any]:
    """Freeze the base model and attach LoRA. Returns the parameters needed to reload it."""
    lora = config["lora"]
    parameters = {
        "rank": lora["r"],
        "scale": lora["alpha"] / lora["r"],
        "dropout": lora["dropout"],
        "keys": sorted(lora_keys(model, lora["target_modules"])),
    }
    model.freeze()
    linear_to_lora_layers(model, config["mlx"]["num_layers"], parameters)
    print_trainable_parameters(model)
    return parameters


def schedule_steps(
    num_examples: int, batch_size: int, grad_accum: int, epochs: float
) -> tuple[int, int]:
    """Return (iters, optimizer_steps).

    mlx-lm's `iters` counts micro-batches; the optimizer (and so the LR schedule) only advances
    every grad_accum iters. Everything the schedule sees is in optimizer steps.
    """
    iters = math.ceil(epochs * num_examples / batch_size)
    return iters, max(1, iters // grad_accum)


def build_lr_schedule(
    learning_rate: float, total_steps: int, warmup_ratio: float
) -> Any:
    """Linear warmup then cosine decay to zero - the same shape as the PyTorch Trainer's."""
    warmup = max(1, int(total_steps * warmup_ratio))
    return optim.join_schedules(
        [
            optim.linear_schedule(0.0, learning_rate, warmup),
            optim.cosine_decay(learning_rate, max(1, total_steps - warmup)),
        ],
        [warmup],
    )


class ReportingCallback(TrainingCallback):
    """Log losses, and at every validation report run generation-eval on a val sample.

    The generation metrics are the same three train.py logs (from
    prompt_format.generation_metrics), so the two frameworks' dashboards line up.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        examples: list[dict[str, Any]],
        max_new_tokens: int,
        grad_accum: int,
        wandb_run: Any,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.examples = examples
        self.max_new_tokens = max_new_tokens
        self.grad_accum = grad_accum
        self.wandb_run = wandb_run

    def _log(self, iteration: int, values: dict[str, float]) -> None:
        if self.wandb_run is not None:
            self.wandb_run.log(
                {**values, "optimizer_step": iteration // self.grad_accum},
                step=iteration,
            )

    def _generate(self, text: str) -> str:
        prompt = self.tokenizer.encode(
            render_prompt(self.tokenizer, text), add_special_tokens=False
        )
        return generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=self.max_new_tokens,
            verbose=False,
        )

    def on_train_loss_report(self, train_info: dict) -> None:
        self._log(
            train_info["iteration"],
            {
                "train_loss": train_info["train_loss"],
                "learning_rate": train_info["learning_rate"],
                "tokens_per_second": train_info["tokens_per_second"],
                "peak_memory_gb": train_info["peak_memory"],
            },
        )

    def on_val_loss_report(self, val_info: dict) -> None:
        values = {"val_loss": val_info["val_loss"]}
        if self.examples:
            was_training = self.model.training
            self.model.eval()
            results = [(self._generate(e["text"]), e["text"]) for e in self.examples]
            if was_training:
                self.model.train()
            metrics = generation_metrics(results)
            values.update({f"eval_{name}": value for name, value in metrics.items()})
            print(
                "  generation-eval: "
                + "  ".join(f"{name} {value:.3f}" for name, value in metrics.items()),
                flush=True,
            )
        self._log(val_info["iteration"], values)


def save_adapter_config(
    output_dir: Path,
    model_path: str,
    lora_parameters: dict[str, Any],
    config: dict[str, Any],
    run: dict[str, Any],
) -> None:
    """Write adapter_config.json in the shape `mlx_lm.load(adapter_path=...)` reads back.

    fine_tune_type, num_layers and lora_parameters are what mlx-lm needs to rebuild the LoRA
    layers; the rest is a record of how the adapter was trained.
    """
    record = {
        "fine_tune_type": "lora",
        "model": model_path,
        "num_layers": config["mlx"]["num_layers"],
        "lora_parameters": lora_parameters,
        "max_seq_len": config["max_seq_len"],
        "seed": config["seed"],
        **run,
    }
    (output_dir / "adapter_config.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--data-dir", type=Path, default=Path("ml/data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("ml/outputs/mlx-run"))
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument(
        "--max-iters",
        type=int,
        default=None,
        help="stop after N iters regardless of epochs - for throughput measurement",
    )
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--no-generation-eval", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())

    mlx_cfg = config["mlx"]
    shared = config["training"]
    model_path = args.model_path or mlx_cfg["model_path"]
    epochs = args.epochs if args.epochs is not None else shared["epochs"]
    learning_rate = (
        args.learning_rate
        if args.learning_rate is not None
        else float(shared["learning_rate"])
    )
    batch_size = mlx_cfg["batch_size"]
    grad_accum = gradient_accumulation(shared["effective_batch_size"], batch_size)

    set_seed(config["seed"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"model: {model_path}\nout:   {args.output_dir}")

    model, tokenizer = load(model_path)

    train_raw = read_jsonl(args.data_dir / "train.jsonl")
    val_raw = read_jsonl(args.data_dir / "val.jsonl")
    if args.max_examples:
        train_raw = train_raw[: args.max_examples]
        val_raw = val_raw[: max(4, args.max_examples // 10)]

    print("tokenizing:")
    train_set = build_dataset(train_raw, tokenizer, config["max_seq_len"], "train")
    val_set = build_dataset(val_raw, tokenizer, config["max_seq_len"], "val")

    iters, optimizer_steps = schedule_steps(
        len(train_set), batch_size, grad_accum, epochs
    )
    print(f"schedule: {iters} iters = {optimizer_steps} optimizer steps")
    if args.max_iters and args.max_iters < iters:
        # The LR schedule stays sized for the full run, so a capped run sees the same early
        # learning rates the real run would - which is what a throughput measurement wants.
        iters = args.max_iters
        print(f"  capped at {iters} iters by --max-iters (LR schedule unchanged)")

    lora_parameters = apply_lora(model, config)
    run = {
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "iters": iters,
        "train_examples": len(train_set),
    }
    save_adapter_config(
        args.output_dir, portable_model_path(model_path), lora_parameters, config, run
    )
    (args.output_dir / "prompt_contract.json").write_text(
        json.dumps(prompt_contract(tokenizer), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    wandb_run = None
    if config["wandb"]["enabled"] and not args.no_wandb:
        import wandb

        wandb_run = wandb.init(
            project=config["wandb"]["project"],
            name=args.output_dir.name,
            config={"framework": "mlx", "model": model_path, **run, **lora_parameters},
        )

    gen_cfg = config["generation_eval"]
    eval_examples = (
        val_raw[: gen_cfg["num_samples"]]
        if gen_cfg["enabled"] and not args.no_generation_eval
        else []
    )
    callback = ReportingCallback(
        model=model,
        tokenizer=tokenizer,
        examples=eval_examples,
        max_new_tokens=gen_cfg["max_new_tokens"],
        grad_accum=grad_accum,
        wandb_run=wandb_run,
    )

    # mlx-lm counts iters (micro-batches). Logging is configured in optimizer steps; eval and
    # checkpoints happen once per epoch, matching train.py.
    iters_per_epoch = math.ceil(len(train_set) / batch_size)
    training_args = TrainingArgs(
        batch_size=batch_size,
        iters=iters,
        val_batches=min(mlx_cfg["val_batches"], len(val_set)),
        steps_per_report=shared["logging_steps"] * grad_accum,
        steps_per_eval=iters_per_epoch,
        steps_per_save=iters_per_epoch,
        max_seq_length=config["max_seq_len"],
        adapter_file=str(args.output_dir / "adapters.safetensors"),
        grad_checkpoint=mlx_cfg["grad_checkpoint"],
        grad_accumulation_steps=grad_accum,
    )
    optimizer = optim.AdamW(
        learning_rate=build_lr_schedule(
            learning_rate, optimizer_steps, shared["warmup_ratio"]
        )
    )

    train(
        model=model,
        optimizer=optimizer,
        train_dataset=train_set,
        val_dataset=val_set,
        args=training_args,
        training_callback=callback,
    )

    print(f"\npeak memory: {mx.get_peak_memory() / 1e9:.2f} GB")
    print(f"adapter + config + prompt contract saved to {args.output_dir}")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
