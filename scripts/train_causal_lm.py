from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from deid_pipeline.storage.env import apply_cache_env_defaults
from deid_pipeline.storage.layout import StorageLayout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a causal language model (dev-only).")
    parser.add_argument("--model-dir", required=True, help="Local model/tokenizer directory.")
    parser.add_argument("--output-dir", required=True, help="Output directory for the trained model.")
    parser.add_argument("--input-jsonl", required=True, help="Path to a JSONL corpus with a `text` field.")
    parser.add_argument("--max-examples", type=int, default=0, help="Optional cap on loaded examples.")
    parser.add_argument("--block-size", type=int, default=256, help="Token block size for training.")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Optional cap on training steps (overrides epochs when >0).",
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size.")
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Accumulate gradients to increase effective batch size.",
    )
    parser.add_argument(
        "--precision",
        choices=["auto", "fp16", "bf16", "none"],
        default="auto",
        help="Mixed precision mode (auto prefers bf16 then fp16 on CUDA).",
    )
    parser.add_argument(
        "--tf32",
        action="store_true",
        help="Enable TF32 matmul on supported NVIDIA GPUs (training speed).",
    )
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay.")
    parser.add_argument("--warmup-ratio", type=float, default=0.0, help="Warmup ratio (0..1).")
    parser.add_argument("--seed", type=int, default=0, help="Training seed.")
    parser.add_argument(
        "--preprocess-batch-size",
        type=int,
        default=512,
        help="Batch size used when tokenizing via datasets.map(batched=True).",
    )
    parser.add_argument(
        "--preprocess-num-proc",
        type=int,
        default=0,
        help="Number of processes for preprocessing (0 = single process).",
    )
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Allow network access (required to download models).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layout = StorageLayout.from_project_root(Path(__file__).resolve().parents[1])
    apply_cache_env_defaults(layout=layout)

    model_dir = str(Path(args.model_dir).expanduser().resolve())
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    input_jsonl = Path(args.input_jsonl).expanduser().resolve()
    if not input_jsonl.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_jsonl}")

    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("Training requires the `datasets` package. Install with: pip install datasets") from exc

    from transformers import (  # type: ignore
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )
    import torch  # type: ignore

    allow_network = bool(args.allow_network)

    ds = load_dataset("json", data_files=str(input_jsonl), split="train")
    if int(args.max_examples) > 0:
        ds = ds.select(range(min(int(args.max_examples), len(ds))))

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, local_files_only=not allow_network)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
        return tokenizer([str(t or "") for t in batch["text"]], truncation=False)

    tokenized = ds.map(
        tokenize_batch,
        batched=True,
        batch_size=int(args.preprocess_batch_size),
        num_proc=int(args.preprocess_num_proc) if int(args.preprocess_num_proc) > 0 else None,
        remove_columns=ds.column_names,
    )

    block_size = int(args.block_size)
    if hasattr(tokenizer, "model_max_length") and int(getattr(tokenizer, "model_max_length", 0) or 0) > 0:
        block_size = min(block_size, int(tokenizer.model_max_length))

    def group_texts(examples: Dict[str, List[List[int]]]) -> Dict[str, List[List[int]]]:
        concatenated: Dict[str, List[int]] = {}
        for key, value in examples.items():
            if not value:
                concatenated[key] = []
                continue
            concatenated[key] = sum(value, [])

        total_length = len(concatenated.get("input_ids", []))
        if total_length == 0:
            return {"input_ids": [], "attention_mask": [], "labels": []}

        total_length = (total_length // block_size) * block_size
        result: Dict[str, List[List[int]]] = {}
        for key, tokens in concatenated.items():
            if not tokens:
                result[key] = []
                continue
            result[key] = [tokens[i : i + block_size] for i in range(0, total_length, block_size)]

        result["labels"] = list(result.get("input_ids", []))
        return result

    lm_ds = tokenized.map(group_texts, batched=True, batch_size=1000)

    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=not allow_network)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    precision = str(args.precision).lower().strip()
    use_fp16 = False
    use_bf16 = False
    if precision == "auto":
        if torch.cuda.is_available():
            if hasattr(torch.cuda, "is_bf16_supported") and bool(torch.cuda.is_bf16_supported()):
                use_bf16 = True
            else:
                use_fp16 = True
    elif precision == "fp16":
        use_fp16 = True
    elif precision == "bf16":
        use_bf16 = True
    elif precision == "none":
        pass
    else:  # pragma: no cover
        raise ValueError(f"Unsupported precision: {precision}")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=float(args.epochs),
        max_steps=int(args.max_steps) if int(args.max_steps) > 0 else -1,
        per_device_train_batch_size=int(args.batch_size),
        gradient_accumulation_steps=int(args.gradient_accumulation_steps),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        warmup_ratio=float(args.warmup_ratio),
        fp16=bool(use_fp16),
        bf16=bool(use_bf16),
        tf32=bool(args.tf32) and torch.cuda.is_available(),
        seed=int(args.seed),
        data_seed=int(args.seed),
        logging_steps=50,
        save_strategy="no",
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_ds,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Wrote trained LM: {output_dir}")


if __name__ == "__main__":
    main()

