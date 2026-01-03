from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from deid_pipeline.storage.env import apply_cache_env_defaults
from deid_pipeline.storage.layout import StorageLayout, dataset_slug
from deid_pipeline.training.mix import SpanJsonlSource, write_mixed_dataset


def parse_dataset_spec(spec: str) -> Tuple[str, int]:
    """Parse `<dataset_name>[:max_examples]`."""

    raw = (spec or "").strip()
    if not raw:
        raise ValueError("Empty dataset spec.")

    if ":" not in raw:
        return raw, 0

    name, max_raw = raw.rsplit(":", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Invalid dataset spec: {spec}")

    try:
        max_examples = int(max_raw.strip())
    except Exception as exc:
        raise ValueError(f"Invalid max_examples in dataset spec: {spec}") from exc

    return name, int(max_examples)


def _load_config_file(path: str | Path) -> Dict[str, Any]:
    cfg_path = Path(path).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    raw = cfg_path.read_text(encoding="utf-8")
    suffix = cfg_path.suffix.lower()

    data: Any
    if suffix in {".yaml", ".yml"}:
        import yaml  # type: ignore

        data = yaml.safe_load(raw)
    elif suffix == ".json":
        data = json.loads(raw)
    else:
        # Try YAML first, then JSON.
        try:
            import yaml  # type: ignore

            data = yaml.safe_load(raw)
        except Exception:
            data = json.loads(raw)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping at the top level: {cfg_path}")
    return dict(data)


def _as_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)):
        return bool(int(value))
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _config_to_dataset_specs(value: Any) -> List[str]:
    if value is None:
        return []

    if isinstance(value, (list, tuple)):
        specs: List[str] = []
        for item in value:
            if isinstance(item, str):
                specs.append(item)
            elif isinstance(item, dict):
                name = str(item.get("name") or item.get("dataset") or "").strip()
                if not name:
                    raise ValueError(f"Invalid dataset entry (missing name): {item}")
                max_examples = int(item.get("max_examples") or item.get("max") or 0)
                specs.append(f"{name}:{max_examples}" if max_examples > 0 else name)
            else:
                raise ValueError(f"Invalid dataset entry: {item!r}")
        return specs

    if isinstance(value, str):
        return [value]

    raise ValueError(f"Invalid datasets config value: {value!r}")


def _config_to_provider_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [p.strip() for p in value.split(",") if p.strip()]
    if isinstance(value, (list, tuple)):
        return [str(p).strip() for p in value if str(p).strip()]
    raise ValueError(f"Invalid ONNX providers value: {value!r}")


def build_effective_config(args: argparse.Namespace) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {
        "run_name": "multi-zh-ner-demo",
        "language": "zh",
        "datasets": [
            "levow/msra_ner:5000",
            "tner/wikiann:2000",
            "hltcoe/weibo_ner:1350",
        ],
        "split": "train",
        "seed": 0,
        "shuffle": True,
        "base_model_dir": "",
        "epochs": 1,
        "max_steps": 0,
        "batch_size": 8,
        "max_length": 256,
        "precision": "auto",
        "tf32": False,
        "learning_rate": 5e-5,
        "weight_decay": 0.0,
        "warmup_ratio": 0.0,
        "gradient_accumulation_steps": 1,
        "preprocess_batch_size": 256,
        "preprocess_num_proc": 0,
        "canonicalize_types": True,
        "allow_network": False,
        "trust_remote_code": False,
        "force_prepare": False,
        "quantize": False,
        "opset": 17,
        "onnx_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "benchmark_chars": 10_000,
        "benchmark_runs": 20,
        "benchmark_warmup": 2,
        "json_out": "",
    }

    cfg = dict(defaults)

    config_path = str(getattr(args, "config", "") or "").strip()
    if config_path:
        raw = _load_config_file(config_path)

        mix = raw.get("mix") if isinstance(raw.get("mix"), dict) else {}
        training = raw.get("training") if isinstance(raw.get("training"), dict) else {}
        export = raw.get("export") if isinstance(raw.get("export"), dict) else {}
        benchmark = raw.get("benchmark") if isinstance(raw.get("benchmark"), dict) else {}
        outputs = raw.get("outputs") if isinstance(raw.get("outputs"), dict) else {}
        onnx = raw.get("onnx") if isinstance(raw.get("onnx"), dict) else {}

        if raw.get("run_name") is not None:
            cfg["run_name"] = str(raw.get("run_name"))
        if raw.get("language") is not None:
            cfg["language"] = str(raw.get("language"))
        if raw.get("split") is not None:
            cfg["split"] = str(raw.get("split"))

        datasets = _config_to_dataset_specs(raw.get("datasets"))
        if datasets:
            cfg["datasets"] = datasets

        if raw.get("seed") is not None:
            cfg["seed"] = int(raw.get("seed"))
        if mix.get("seed") is not None:
            cfg["seed"] = int(mix.get("seed"))
        if mix.get("shuffle") is not None:
            cfg["shuffle"] = _as_bool(mix.get("shuffle"), default=True)

        if raw.get("base_model_dir") is not None:
            cfg["base_model_dir"] = str(raw.get("base_model_dir"))

        if training.get("epochs") is not None:
            cfg["epochs"] = int(training.get("epochs"))
        if training.get("max_steps") is not None:
            cfg["max_steps"] = int(training.get("max_steps"))
        if training.get("batch_size") is not None:
            cfg["batch_size"] = int(training.get("batch_size"))
        if training.get("max_length") is not None:
            cfg["max_length"] = int(training.get("max_length"))
        if training.get("precision") is not None:
            cfg["precision"] = str(training.get("precision"))
        if training.get("tf32") is not None:
            cfg["tf32"] = _as_bool(training.get("tf32"), default=False)
        if training.get("learning_rate") is not None:
            cfg["learning_rate"] = float(training.get("learning_rate"))
        if training.get("weight_decay") is not None:
            cfg["weight_decay"] = float(training.get("weight_decay"))
        if training.get("warmup_ratio") is not None:
            cfg["warmup_ratio"] = float(training.get("warmup_ratio"))
        if training.get("gradient_accumulation_steps") is not None:
            cfg["gradient_accumulation_steps"] = int(training.get("gradient_accumulation_steps"))
        if training.get("preprocess_batch_size") is not None:
            cfg["preprocess_batch_size"] = int(training.get("preprocess_batch_size"))
        if training.get("preprocess_num_proc") is not None:
            cfg["preprocess_num_proc"] = int(training.get("preprocess_num_proc"))
        if training.get("canonicalize_types") is not None:
            cfg["canonicalize_types"] = _as_bool(training.get("canonicalize_types"), default=True)

        if raw.get("allow_network") is not None:
            cfg["allow_network"] = _as_bool(raw.get("allow_network"), default=False)
        if raw.get("trust_remote_code") is not None:
            cfg["trust_remote_code"] = _as_bool(raw.get("trust_remote_code"), default=False)
        if raw.get("force_prepare") is not None:
            cfg["force_prepare"] = _as_bool(raw.get("force_prepare"), default=False)

        if export.get("opset") is not None:
            cfg["opset"] = int(export.get("opset"))
        if raw.get("quantize") is not None:
            cfg["quantize"] = _as_bool(raw.get("quantize"), default=False)

        providers = _config_to_provider_list(raw.get("onnx_providers"))
        if providers:
            cfg["onnx_providers"] = providers
        providers = _config_to_provider_list(onnx.get("providers"))
        if providers:
            cfg["onnx_providers"] = providers

        if benchmark.get("chars") is not None:
            cfg["benchmark_chars"] = int(benchmark.get("chars"))
        if benchmark.get("runs") is not None:
            cfg["benchmark_runs"] = int(benchmark.get("runs"))
        if benchmark.get("warmup") is not None:
            cfg["benchmark_warmup"] = int(benchmark.get("warmup"))

        if outputs.get("json_out") is not None:
            cfg["json_out"] = str(outputs.get("json_out"))

    # CLI overrides (only keys explicitly provided are present due to SUPPRESS defaults).
    cli = vars(args)
    if "run_name" in cli:
        cfg["run_name"] = str(cli["run_name"])
    if "language" in cli:
        cfg["language"] = str(cli["language"])
    if "datasets" in cli and cli["datasets"]:
        cfg["datasets"] = list(cli["datasets"])
    if "split" in cli:
        cfg["split"] = str(cli["split"])
    if "seed" in cli:
        cfg["seed"] = int(cli["seed"])
    if "shuffle" in cli:
        cfg["shuffle"] = _as_bool(cli["shuffle"], default=True)
    if "base_model_dir" in cli:
        cfg["base_model_dir"] = str(cli["base_model_dir"])

    if "epochs" in cli:
        cfg["epochs"] = int(cli["epochs"])
    if "max_steps" in cli:
        cfg["max_steps"] = int(cli["max_steps"])
    if "batch_size" in cli:
        cfg["batch_size"] = int(cli["batch_size"])
    if "max_length" in cli:
        cfg["max_length"] = int(cli["max_length"])
    if "precision" in cli:
        cfg["precision"] = str(cli["precision"])
    if "tf32" in cli:
        cfg["tf32"] = _as_bool(cli["tf32"], default=False)
    if "learning_rate" in cli:
        cfg["learning_rate"] = float(cli["learning_rate"])
    if "weight_decay" in cli:
        cfg["weight_decay"] = float(cli["weight_decay"])
    if "warmup_ratio" in cli:
        cfg["warmup_ratio"] = float(cli["warmup_ratio"])
    if "gradient_accumulation_steps" in cli:
        cfg["gradient_accumulation_steps"] = int(cli["gradient_accumulation_steps"])
    if "preprocess_batch_size" in cli:
        cfg["preprocess_batch_size"] = int(cli["preprocess_batch_size"])
    if "preprocess_num_proc" in cli:
        cfg["preprocess_num_proc"] = int(cli["preprocess_num_proc"])
    if "canonicalize_types" in cli:
        cfg["canonicalize_types"] = _as_bool(cli["canonicalize_types"], default=True)

    if "allow_network" in cli:
        cfg["allow_network"] = _as_bool(cli["allow_network"], default=False)
    if "trust_remote_code" in cli:
        cfg["trust_remote_code"] = _as_bool(cli["trust_remote_code"], default=False)
    if "force_prepare" in cli:
        cfg["force_prepare"] = _as_bool(cli["force_prepare"], default=False)
    if "quantize" in cli:
        cfg["quantize"] = _as_bool(cli["quantize"], default=False)

    if "opset" in cli:
        cfg["opset"] = int(cli["opset"])
    if "onnx_providers" in cli:
        cfg["onnx_providers"] = _config_to_provider_list(cli["onnx_providers"])
    if "benchmark_chars" in cli:
        cfg["benchmark_chars"] = int(cli["benchmark_chars"])
    if "benchmark_runs" in cli:
        cfg["benchmark_runs"] = int(cli["benchmark_runs"])
    if "benchmark_warmup" in cli:
        cfg["benchmark_warmup"] = int(cli["benchmark_warmup"])
    if "json_out" in cli:
        cfg["json_out"] = str(cli["json_out"])

    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a repeatable multi-dataset NER training pipeline (dev-only)."
    )
    parser.add_argument(
        "--run-name",
        dest="run_name",
        default=argparse.SUPPRESS,
        help="Stable run name used for output directories.",
    )
    parser.add_argument(
        "--config",
        default="",
        help="Optional YAML/JSON config file (see configs/training/*.yaml).",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Print the effective config as JSON and exit.",
    )
    parser.add_argument(
        "--language",
        choices=["zh", "en"],
        default=argparse.SUPPRESS,
        help="Language tag.",
    )
    parser.add_argument(
        "--dataset",
        dest="datasets",
        action="append",
        default=argparse.SUPPRESS,
        help="Repeatable: `<dataset_name>[:max_examples]` (e.g. `levow/msra_ner:5000`).",
    )
    parser.add_argument("--split", default=argparse.SUPPRESS, help="Dataset split (default: train).")
    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS, help="Shuffle seed for mixed dataset.")
    parser.add_argument(
        "--shuffle",
        dest="shuffle",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Shuffle the mixed dataset (default).",
    )
    parser.add_argument(
        "--no-shuffle",
        dest="shuffle",
        action="store_false",
        default=argparse.SUPPRESS,
        help="Do not shuffle the mixed dataset.",
    )
    parser.add_argument(
        "--base-model-dir",
        dest="base_model_dir",
        default=argparse.SUPPRESS,
        help="Local base model/tokenizer directory.",
    )
    parser.add_argument("--epochs", type=int, default=argparse.SUPPRESS, help="Training epochs.")
    parser.add_argument(
        "--max-steps",
        dest="max_steps",
        type=int,
        default=argparse.SUPPRESS,
        help="Optional cap on training steps (overrides epochs when >0).",
    )
    parser.add_argument("--batch-size", type=int, default=argparse.SUPPRESS, help="Per-device batch size.")
    parser.add_argument("--max-length", type=int, default=argparse.SUPPRESS, help="Tokenizer max sequence length.")
    parser.add_argument(
        "--precision",
        choices=["auto", "fp16", "bf16", "none"],
        default=argparse.SUPPRESS,
        help="Mixed precision mode (auto prefers bf16 then fp16 on CUDA).",
    )
    parser.add_argument(
        "--tf32",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Enable TF32 matmul on supported NVIDIA GPUs (training speed).",
    )
    parser.add_argument(
        "--learning-rate",
        dest="learning_rate",
        type=float,
        default=argparse.SUPPRESS,
        help="Learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        dest="weight_decay",
        type=float,
        default=argparse.SUPPRESS,
        help="Weight decay.",
    )
    parser.add_argument(
        "--warmup-ratio",
        dest="warmup_ratio",
        type=float,
        default=argparse.SUPPRESS,
        help="Warmup ratio (0..1).",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        dest="gradient_accumulation_steps",
        type=int,
        default=argparse.SUPPRESS,
        help="Accumulate gradients to increase effective batch size.",
    )
    parser.add_argument(
        "--preprocess-batch-size",
        dest="preprocess_batch_size",
        type=int,
        default=argparse.SUPPRESS,
        help="Tokenizer preprocessing batch size.",
    )
    parser.add_argument(
        "--preprocess-num-proc",
        dest="preprocess_num_proc",
        type=int,
        default=argparse.SUPPRESS,
        help="Tokenizer preprocessing processes (0 = single process).",
    )
    parser.add_argument(
        "--no-canonicalize-types",
        dest="canonicalize_types",
        action="store_false",
        default=argparse.SUPPRESS,
        help="Disable canonical entity type mapping during training.",
    )
    parser.add_argument(
        "--allow-network",
        dest="allow_network",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Allow network access (required to download models/datasets).",
    )
    parser.add_argument(
        "--no-allow-network",
        dest="allow_network",
        action="store_false",
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--trust-remote-code",
        dest="trust_remote_code",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Allow execution of custom dataset loading code from the dataset repository (HF datasets).",
    )
    parser.add_argument(
        "--no-trust-remote-code",
        dest="trust_remote_code",
        action="store_false",
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--force-prepare",
        dest="force_prepare",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Re-generate prepared datasets even if JSONL files already exist.",
    )
    parser.add_argument(
        "--no-force-prepare",
        dest="force_prepare",
        action="store_false",
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--quantize",
        dest="quantize",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Quantize exported ONNX weights to INT8.",
    )
    parser.add_argument(
        "--no-quantize",
        dest="quantize",
        action="store_false",
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--opset", type=int, default=argparse.SUPPRESS, help="ONNX opset version.")
    parser.add_argument(
        "--onnx-providers",
        dest="onnx_providers",
        default=argparse.SUPPRESS,
        help="Comma-separated ONNX Runtime provider list (e.g. CUDAExecutionProvider,CPUExecutionProvider).",
    )
    parser.add_argument(
        "--benchmark-chars",
        dest="benchmark_chars",
        type=int,
        default=argparse.SUPPRESS,
        help="ONNX benchmark text length.",
    )
    parser.add_argument(
        "--benchmark-runs",
        dest="benchmark_runs",
        type=int,
        default=argparse.SUPPRESS,
        help="ONNX benchmark runs.",
    )
    parser.add_argument(
        "--benchmark-warmup",
        dest="benchmark_warmup",
        type=int,
        default=argparse.SUPPRESS,
        help="ONNX benchmark warmup runs.",
    )
    parser.add_argument("--json-out", dest="json_out", default=argparse.SUPPRESS, help="Optional JSON report output path.")
    return parser.parse_args()


def _python_env(repo_root: Path, *, allow_network: bool) -> Dict[str, str]:
    env = dict(os.environ)

    # Make `src/` importable for subprocess calls.
    py_path = str(repo_root / "src")
    env["PYTHONPATH"] = py_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    if not allow_network:
        env.setdefault("HF_HUB_OFFLINE", "1")

    return env


def run_step(cmd: Sequence[str], *, env: Mapping[str, str], cwd: Path) -> None:
    subprocess.check_call(list(cmd), cwd=str(cwd), env=dict(env))


def run_step_capture(cmd: Sequence[str], *, env: Mapping[str, str], cwd: Path) -> str:
    out = subprocess.check_output(list(cmd), cwd=str(cwd), env=dict(env))
    return out.decode("utf-8", errors="replace")


def _read_prepared_count(out_dir: Path) -> int | None:
    manifest_path = Path(out_dir) / "manifest.json"
    if manifest_path.exists():
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            return int(data.get("count", 0))
        except Exception:
            return None
    return None


def _read_prepared_manifest(out_dir: Path) -> Dict[str, Any] | None:
    manifest_path = Path(out_dir) / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return dict(data) if isinstance(data, dict) else None


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    layout = StorageLayout.from_project_root(repo_root)
    apply_cache_env_defaults(layout=layout)

    cfg = build_effective_config(args)
    if bool(getattr(args, "print_config", False)):
        print(json.dumps(cfg, ensure_ascii=False, indent=2))
        return

    run_slug = dataset_slug(str(cfg["run_name"]))
    split = str(cfg["split"])
    lang = str(cfg["language"])

    base_model_dir = (
        Path(cfg["base_model_dir"]).expanduser().resolve()
        if str(cfg["base_model_dir"]).strip()
        else (
            layout.edge_deid_models_home / ("bert-ner-zh" if lang == "zh" else "bert-ner-en")
        )
    )

    allow_network = bool(cfg["allow_network"])
    trust_remote_code = bool(cfg["trust_remote_code"])
    env = _python_env(repo_root, allow_network=allow_network)

    if not base_model_dir.exists():
        if not allow_network:
            raise FileNotFoundError(
                f"Base model directory not found: {base_model_dir}. "
                "Either download it first or re-run with --allow-network."
            )

        if lang == "zh":
            run_step(
                [
                    sys.executable,
                    str(repo_root / "scripts" / "download_models.py"),
                    "--allow-network",
                    "--only",
                    "ner_zh",
                ],
                env=env,
                cwd=repo_root,
            )
        else:
            raise RuntimeError(
                f"Automatic download is not configured for language={lang}. "
                f"Download your base model to: {base_model_dir}."
            )

    dataset_specs = list(cfg["datasets"])
    parsed_specs = [parse_dataset_spec(s) for s in dataset_specs]

    prepared_sources: List[SpanJsonlSource] = []
    for dataset_name, max_examples in parsed_specs:
        slug = dataset_slug(dataset_name)
        out_dir = layout.edge_deid_datasets_home / "processed" / slug / split
        jsonl_path = out_dir / "dataset.jsonl"
        manifest = _read_prepared_manifest(out_dir)
        manifest_count = int(manifest.get("count", 0)) if manifest is not None else None

        needs_prepare = bool(cfg["force_prepare"]) or not jsonl_path.exists()
        if (not needs_prepare) and int(max_examples) > 0 and manifest is not None:
            requested_max = int(manifest.get("requested_max_examples", 0) or 0)
            # `max_examples` is an upper bound, not a requirement. Only regenerate when the dataset
            # was previously prepared with a smaller cap.
            if requested_max > 0 and requested_max < int(max_examples):
                needs_prepare = True
            # Legacy manifests (missing `requested_max_examples`): attempt to regenerate when
            # network is available (or for synthetic) and the current file is smaller than desired.
            elif requested_max == 0 and manifest_count is not None and int(manifest_count) < int(max_examples):
                needs_prepare = allow_network or dataset_name == "synthetic"

        if needs_prepare:
            if (not allow_network) and dataset_name != "synthetic":
                if bool(cfg["force_prepare"]):
                    raise RuntimeError(
                        f"--force-prepare requested but network is disabled for dataset={dataset_name}. "
                        "Re-run with --allow-network."
                    )
                if not jsonl_path.exists():
                    raise FileNotFoundError(
                        f"Prepared dataset not found: {jsonl_path}. Re-run with --allow-network to download it."
                    )
                # Offline: keep the existing prepared dataset even if smaller than the desired cap.
                needs_prepare = False

            if needs_prepare:
                prepare_cmd = [
                    sys.executable,
                    str(repo_root / "scripts" / "prepare_dataset.py"),
                    "--dataset",
                    dataset_name,
                    "--language",
                    lang,
                    "--split",
                    split,
                    "--max-examples",
                    str(int(max_examples) if int(max_examples) > 0 else 0),
                ]
                if allow_network:
                    prepare_cmd.append("--allow-network")
                if trust_remote_code:
                    prepare_cmd.append("--trust-remote-code")
                run_step(prepare_cmd, env=env, cwd=repo_root)

        prepared_sources.append(
            SpanJsonlSource(
                name=dataset_name,
                jsonl_path=jsonl_path,
                max_examples=int(max_examples) if int(max_examples) > 0 else 0,
            )
        )

    mixed_out_dir = layout.edge_deid_datasets_home / "processed" / run_slug / split
    mixed_meta = write_mixed_dataset(
        output_dir=mixed_out_dir,
        dataset_name=str(cfg["run_name"]),
        split=split,
        sources=prepared_sources,
        shuffle=bool(cfg["shuffle"]),
        seed=int(cfg["seed"]),
    )

    training_dir = layout.edge_deid_training_runs_home / run_slug
    training_cmd = [
        sys.executable,
        str(repo_root / "scripts" / "train_token_classifier.py"),
        "--model-dir",
        str(base_model_dir),
        "--output-dir",
        str(training_dir),
        "--language",
        lang,
        "--input-jsonl",
        str(mixed_out_dir / "dataset.jsonl"),
        "--max-length",
        str(int(cfg["max_length"])),
        "--epochs",
        str(int(cfg["epochs"])),
        "--max-steps",
        str(int(cfg["max_steps"])),
        "--batch-size",
        str(int(cfg["batch_size"])),
        "--precision",
        str(cfg["precision"]),
        "--learning-rate",
        str(float(cfg["learning_rate"])),
        "--weight-decay",
        str(float(cfg["weight_decay"])),
        "--warmup-ratio",
        str(float(cfg["warmup_ratio"])),
        "--gradient-accumulation-steps",
        str(int(cfg["gradient_accumulation_steps"])),
        "--preprocess-batch-size",
        str(int(cfg["preprocess_batch_size"])),
        "--preprocess-num-proc",
        str(int(cfg["preprocess_num_proc"])),
        "--seed",
        str(int(cfg["seed"])),
    ]
    if not bool(cfg.get("canonicalize_types", True)):
        training_cmd.append("--no-canonicalize-types")
    if bool(cfg["tf32"]):
        training_cmd.append("--tf32")
    run_step(training_cmd, env=env, cwd=repo_root)

    onnx_out_dir = layout.edge_deid_models_home / run_slug
    export_cmd = [
        sys.executable,
        str(repo_root / "scripts" / "export_token_classifier_onnx.py"),
        "--model-dir",
        str(training_dir),
        "--output-dir",
        str(onnx_out_dir),
        "--file-name",
        "model.onnx",
        "--opset",
        str(int(cfg["opset"])),
    ]
    run_step(export_cmd, env=env, cwd=repo_root)

    validate_cmd = [
        sys.executable,
        str(repo_root / "scripts" / "validate_onnx_token_classifier.py"),
        "--onnx-model",
        str(onnx_out_dir / "model.onnx"),
        "--tokenizer-dir",
        str(onnx_out_dir),
        "--providers",
        ",".join(list(cfg["onnx_providers"])),
        "--pytorch-model-dir",
        str(training_dir),
    ]
    run_step(validate_cmd, env=env, cwd=repo_root)

    quantized_path = ""
    if bool(cfg["quantize"]):
        quant_cmd = [
            sys.executable,
            str(repo_root / "scripts" / "quantize_onnx_model.py"),
            "--input",
            str(onnx_out_dir / "model.onnx"),
            "--output",
            str(onnx_out_dir / "model.int8.onnx"),
        ]
        run_step(quant_cmd, env=env, cwd=repo_root)
        quantized_path = str(onnx_out_dir / "model.int8.onnx")

    bench_env = dict(env)
    bench_env["USE_STUB"] = "false"
    bench_cmd = [
        sys.executable,
        str(repo_root / "scripts" / "benchmark_onnx_ner.py"),
        "--onnx-model",
        quantized_path or str(onnx_out_dir / "model.onnx"),
        "--tokenizer-dir",
        str(onnx_out_dir),
        "--providers",
        ",".join(list(cfg["onnx_providers"])),
        "--runs",
        str(int(cfg["benchmark_runs"])),
        "--warmup",
        str(int(cfg["benchmark_warmup"])),
        "--chars",
        str(int(cfg["benchmark_chars"])),
    ]
    bench_json = run_step_capture(bench_cmd, env=bench_env, cwd=repo_root).strip()

    report: Dict[str, Any] = {
        "run_name": str(cfg["run_name"]),
        "run_slug": run_slug,
        "language": lang,
        "split": split,
        "created_unix_s": time.time(),
        "base_model_dir": str(base_model_dir),
        "config": dict(cfg),
        "prepared_sources": [
            {"name": s.name, "jsonl_path": str(Path(s.jsonl_path).expanduser().resolve())}
            for s in prepared_sources
        ],
        "mixed_dataset": mixed_meta,
        "training_output_dir": str(training_dir),
        "onnx_output_dir": str(onnx_out_dir),
        "onnx_model": str(onnx_out_dir / "model.onnx"),
        "onnx_model_int8": quantized_path,
        "benchmark_onnx_ner": json.loads(bench_json) if bench_json else {},
    }

    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if cfg["json_out"]:
        out_path = Path(cfg["json_out"]).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
