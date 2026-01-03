from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Set

from deid_pipeline.storage.env import apply_cache_env_defaults
from deid_pipeline.storage.layout import StorageLayout
from deid_pipeline.training.canonical import canonicalize_entities, canonicalize_entity_type
from deid_pipeline.training.datasets import (
    HuggingFaceTokenNERAdapter,
    adapter_ai4privacy_pii_masking_300k,
    adapter_nemotron_pii,
    iter_token_examples_to_span_examples,
)
from deid_pipeline.training.synthetic import generate_synthetic_span_examples
from deid_pipeline.training.io import iter_span_examples_jsonl
from deid_pipeline.training.tokenization import align_entities_to_tokens, build_bio_label_list


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a token-classification NER model (dev-only).")
    parser.add_argument("--model-dir", required=True, help="Local model/tokenizer directory.")
    parser.add_argument("--output-dir", required=True, help="Output directory for the trained model.")
    parser.add_argument("--language", choices=["zh", "en"], default="zh", help="Language for synthetic data.")
    parser.add_argument(
        "--input-jsonl",
        default="",
        help="Optional path to a prepared JSONL dataset (text + entities).",
    )
    parser.add_argument(
        "--dataset",
        default="synthetic",
        help=(
            "Training dataset. Supported: synthetic, tner/wikiann, levow/msra_ner, "
            "hltcoe/weibo_ner, ai4privacy/pii-masking-300k, nvidia/Nemotron-PII"
        ),
    )
    parser.add_argument("--dataset-config", default="", help="Optional dataset config name (e.g. 'zh').")
    parser.add_argument("--split", default="train", help="Dataset split to load (default: train).")
    parser.add_argument("--max-examples", type=int, default=0, help="Optional cap on loaded examples.")
    parser.add_argument("--num-examples", type=int, default=500, help="Synthetic training examples.")
    parser.add_argument("--max-length", type=int, default=256, help="Tokenizer max sequence length.")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Optional cap on training steps (overrides epochs when >0).",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Per-device batch size.")
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
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Accumulate gradients to increase effective batch size.",
    )
    parser.add_argument(
        "--preprocess-batch-size",
        type=int,
        default=256,
        help="Batch size used when tokenizing via datasets.map(batched=True).",
    )
    parser.add_argument(
        "--preprocess-num-proc",
        type=int,
        default=0,
        help="Number of processes for preprocessing (0 = single process).",
    )
    parser.add_argument(
        "--no-canonicalize-types",
        dest="canonicalize_types",
        action="store_false",
        help="Disable canonical entity type mapping (not recommended for mixed datasets).",
    )
    parser.set_defaults(canonicalize_types=True)
    parser.add_argument("--seed", type=int, default=0, help="Training seed.")
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Allow network access (required to download models/datasets).",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow execution of custom dataset loading code from the dataset repository (HF datasets).",
    )
    return parser.parse_args()


def _scan_entity_types_from_jsonl(
    path: Path,
    *,
    max_examples: int = 0,
    canonicalize: bool = True,
) -> Set[str]:
    types: Set[str] = set()
    for idx, example in enumerate(iter_span_examples_jsonl(path), start=1):
        for ent in example.entities:
            raw = ent.get("type")
            if not raw:
                continue
            mapped = canonicalize_entity_type(str(raw)) if canonicalize else str(raw)
            if mapped:
                types.add(str(mapped))
        if max_examples > 0 and idx >= int(max_examples):
            break
    return types


def main() -> None:
    args = parse_args()
    layout = StorageLayout.from_project_root(Path(__file__).resolve().parents[1])
    apply_cache_env_defaults(layout=layout)

    model_dir = str(Path(args.model_dir).expanduser().resolve())
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import Dataset, load_dataset  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "Training requires the `datasets` package. Install with: pip install datasets"
        ) from exc
    from transformers import (  # type: ignore
        AutoConfig,
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        Trainer,
        TrainingArguments,
    )
    import torch  # type: ignore

    dataset_name = str(args.dataset or "synthetic").strip()
    allow_network = bool(args.allow_network)
    ds: Any = None
    examples: Any = None
    entity_types: Set[str] = set()
    canonicalize = bool(getattr(args, "canonicalize_types", True))

    if str(args.input_jsonl).strip():
        input_jsonl = Path(args.input_jsonl).expanduser().resolve()
        if not input_jsonl.exists():
            raise FileNotFoundError(f"Input JSONL not found: {input_jsonl}")

        entity_types = _scan_entity_types_from_jsonl(
            input_jsonl,
            max_examples=int(args.max_examples),
            canonicalize=canonicalize,
        )
        if not entity_types:
            raise RuntimeError("No entity types found in the training examples.")

        ds = load_dataset("json", data_files=str(input_jsonl), split="train")
        if int(args.max_examples) > 0:
            ds = ds.select(range(min(int(args.max_examples), len(ds))))
    elif dataset_name == "synthetic":
        examples = generate_synthetic_span_examples(
            num_examples=int(args.num_examples),
            seed=0,
            language=args.language,
        )
    elif dataset_name in {"tner/wikiann", "levow/msra_ner", "hltcoe/weibo_ner"}:
        cfg = str(args.dataset_config).strip() or (
            args.language
            if dataset_name == "tner/wikiann"
            else "msra_ner"
            if dataset_name == "levow/msra_ner"
            else "default"
            if dataset_name == "hltcoe/weibo_ner"
            else None
        )
        adapter = HuggingFaceTokenNERAdapter(dataset_name, config_name=cfg)
        token_examples = adapter.iter_load(
            split=str(args.split),
            allow_network=allow_network,
            trust_remote_code=bool(args.trust_remote_code),
        )
        span_examples = iter_token_examples_to_span_examples(
            token_examples,
            separator="" if args.language == "zh" else " ",
            language=args.language,
            source=dataset_name,
        )
        if int(args.max_examples) > 0:
            span_examples = list(span_examples)[: int(args.max_examples)]
        else:
            span_examples = list(span_examples)
        examples = [{"text": ex.text, "entities": list(ex.entities)} for ex in span_examples]
    elif dataset_name in {"ai4privacy/pii-masking-300k", "nvidia/Nemotron-PII"}:
        adapter = (
            adapter_ai4privacy_pii_masking_300k(language=args.language)
            if dataset_name == "ai4privacy/pii-masking-300k"
            else adapter_nemotron_pii(language=args.language)
        )
        span_examples = adapter.iter_span_examples(
            split=str(args.split),
            allow_network=allow_network,
            trust_remote_code=bool(args.trust_remote_code),
        )
        if int(args.max_examples) > 0:
            span_examples = list(span_examples)[: int(args.max_examples)]
        else:
            span_examples = list(span_examples)
        examples = [{"text": ex.text, "entities": list(ex.entities)} for ex in span_examples]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if examples is not None:
        entity_types = set()
        for row in examples:
            for ent in row.get("entities", []) or []:
                raw = ent.get("type")
                if not raw:
                    continue
                mapped = canonicalize_entity_type(raw) if canonicalize else str(raw)
                if mapped:
                    entity_types.add(str(mapped))
        if not entity_types:
            raise RuntimeError("No entity types found in the training examples.")

    label_list = build_bio_label_list(entity_types)
    label_to_id = {label: idx for idx, label in enumerate(label_list)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        use_fast=True,
        local_files_only=not allow_network,
    )

    def tokenize_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
        texts = [str(t or "") for t in batch["text"]]
        entities_list = batch.get("entities", []) or [[] for _ in texts]

        enc = tokenizer(
            texts,
            truncation=True,
            max_length=int(args.max_length),
            return_offsets_mapping=True,
        )
        offsets_batch = enc.pop("offset_mapping")

        labels: List[List[int]] = []
        for ents, offsets in zip(entities_list, offsets_batch):
            ents_norm = canonicalize_entities(ents) if canonicalize else list(ents)
            labels.append(
                align_entities_to_tokens(
                    entities=ents_norm,
                    offset_mapping=[tuple(x) for x in offsets],
                    label_to_id=label_to_id,
                )
            )

        enc["labels"] = labels
        return enc

    if ds is None:
        ds = Dataset.from_list(examples)

    tokenized = ds.map(
        tokenize_batch,
        batched=True,
        batch_size=int(args.preprocess_batch_size),
        num_proc=int(args.preprocess_num_proc) if int(args.preprocess_num_proc) > 0 else None,
        remove_columns=ds.column_names,
    )

    cfg = AutoConfig.from_pretrained(model_dir, local_files_only=not allow_network)
    cfg.num_labels = len(label_list)
    cfg.id2label = {int(i): str(l) for i, l in id_to_label.items()}
    cfg.label2id = {str(l): int(i) for i, l in id_to_label.items()}

    try:
        model = AutoModelForTokenClassification.from_pretrained(
            model_dir,
            config=cfg,
            local_files_only=not allow_network,
        )
    except Exception:
        model = AutoModelForTokenClassification.from_config(cfg)

    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

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

    use_tf32 = bool(args.tf32) and torch.cuda.is_available()

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=float(args.epochs),
        max_steps=int(args.max_steps) if int(args.max_steps) > 0 else -1,
        per_device_train_batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        warmup_ratio=float(args.warmup_ratio),
        gradient_accumulation_steps=int(args.gradient_accumulation_steps),
        fp16=bool(use_fp16),
        bf16=bool(use_bf16),
        tf32=bool(use_tf32),
        seed=int(args.seed),
        data_seed=int(args.seed),
        logging_steps=50,
        save_strategy="no",
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print(f"Wrote trained model: {output_dir}")


if __name__ == "__main__":
    main()
