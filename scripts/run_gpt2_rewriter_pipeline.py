from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from deid_pipeline.storage.env import apply_cache_env_defaults
from deid_pipeline.storage.layout import StorageLayout, dataset_slug


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
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        return [value]
    raise ValueError(f"Invalid datasets value: {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a repeatable GPT-2 rewriter fine-tuning pipeline (dev-only).")
    parser.add_argument("--config", required=True, help="YAML/JSON config file under configs/training/.")
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the effective config as JSON and exit.",
    )
    return parser.parse_args()


def build_effective_config(args: argparse.Namespace) -> Dict[str, Any]:
    raw = _load_config_file(args.config)
    corpus = raw.get("corpus") if isinstance(raw.get("corpus"), dict) else {}
    rewriter = raw.get("rewriter") if isinstance(raw.get("rewriter"), dict) else {}
    training = raw.get("training") if isinstance(raw.get("training"), dict) else {}
    outputs = raw.get("outputs") if isinstance(raw.get("outputs"), dict) else {}

    cfg: Dict[str, Any] = {
        "run_name": str(raw.get("run_name") or "gpt2-zh-rewriter-smoke"),
        "language": str(raw.get("language") or "zh"),
        "split": str(raw.get("split") or "train"),
        # Optional: reuse an existing placeholder corpus instead of re-building it.
        "placeholder_input_jsonl": str(raw.get("placeholder_input_jsonl") or ""),
        "datasets": _config_to_dataset_specs(raw.get("datasets")) or [
            "levow/msra_ner:2000",
            "tner/wikiann:2000",
            "hltcoe/weibo_ner:1000",
            "synthetic:2000",
        ],
        "base_model_dir": str(raw.get("base_model_dir") or ""),
        "allow_network": _as_bool(raw.get("allow_network"), default=False),
        "trust_remote_code": _as_bool(raw.get("trust_remote_code"), default=False),
        "force_prepare": _as_bool(raw.get("force_prepare"), default=False),
        # Placeholder corpus settings.
        "min_chars": int(corpus.get("min_chars", 20)),
        "filter_cjk": _as_bool(corpus.get("filter_cjk"), default=True),
        "canonicalize_placeholders": _as_bool(corpus.get("canonicalize_placeholders"), default=True),
        # Rewriter corpus settings.
        "rewriter_min_chars": int(rewriter.get("min_chars", 40)),
        "rewriter_filter_cjk": _as_bool(rewriter.get("filter_cjk"), default=True),
        "rewriter_max_examples": int(rewriter.get("max_examples", 0)),
        "rewriter_seed": int(rewriter.get("seed", 0)),
        "noise_punct_prob": float(rewriter.get("noise_punct_prob", 0.35)),
        "noise_space_prob": float(rewriter.get("noise_space_prob", 0.25)),
        "noise_dup_prob": float(rewriter.get("noise_dup_prob", 0.08)),
        # Training settings.
        "epochs": int(training.get("epochs", 1)),
        "max_steps": int(training.get("max_steps", 0)),
        "batch_size": int(training.get("batch_size", 2)),
        "gradient_accumulation_steps": int(training.get("gradient_accumulation_steps", 1)),
        "block_size": int(training.get("block_size", 256)),
        "precision": str(training.get("precision", "auto")),
        "tf32": _as_bool(training.get("tf32"), default=False),
        "learning_rate": float(training.get("learning_rate", 5e-5)),
        "weight_decay": float(training.get("weight_decay", 0.0)),
        "warmup_ratio": float(training.get("warmup_ratio", 0.0)),
        "preprocess_batch_size": int(training.get("preprocess_batch_size", 512)),
        "preprocess_num_proc": int(training.get("preprocess_num_proc", 0)),
        "json_out": str(outputs.get("json_out") or ""),
    }
    return cfg


def _python_env(repo_root: Path, *, allow_network: bool) -> Dict[str, str]:
    env = dict(os.environ)
    py_path = str(repo_root / "src")
    env["PYTHONPATH"] = py_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    if not allow_network:
        env.setdefault("HF_HUB_OFFLINE", "1")
    return env


def run_step(cmd: Sequence[str], *, env: Mapping[str, str], cwd: Path) -> None:
    subprocess.check_call(list(cmd), cwd=str(cwd), env=dict(env))


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    layout = StorageLayout.from_project_root(repo_root)
    apply_cache_env_defaults(layout=layout)

    cfg = build_effective_config(args)
    if bool(args.print_config):
        print(json.dumps(cfg, ensure_ascii=False, indent=2))
        return

    run_slug = dataset_slug(str(cfg["run_name"]))
    split = str(cfg["split"])
    lang = str(cfg["language"])

    allow_network = bool(cfg["allow_network"])
    trust_remote_code = bool(cfg["trust_remote_code"])
    env = _python_env(repo_root, allow_network=allow_network)

    base_model_dir = (
        Path(cfg["base_model_dir"]).expanduser().resolve()
        if str(cfg["base_model_dir"]).strip()
        else (layout.models_home / "llm" / "gpt2-zh-base")
    )

    if not (base_model_dir / "config.json").exists():
        if not allow_network:
            raise FileNotFoundError(
                f"Base model directory not found: {base_model_dir}. "
                "Download it first or re-run with allow_network=true."
            )
        run_step(
            [
                sys.executable,
                str(repo_root / "scripts" / "download_models.py"),
                "--allow-network",
                "--only",
                "gpt2_zh_base",
            ],
            env=env,
            cwd=repo_root,
        )

    placeholder_input_jsonl = str(cfg.get("placeholder_input_jsonl") or "").strip()
    placeholder_corpus_path: Path
    placeholder_dir = layout.edge_deid_datasets_home / "processed" / run_slug / split / "placeholders"
    placeholder_dir.mkdir(parents=True, exist_ok=True)

    if placeholder_input_jsonl:
        placeholder_corpus_path = Path(placeholder_input_jsonl).expanduser().resolve()
        if not placeholder_corpus_path.exists():
            raise FileNotFoundError(f"placeholder_input_jsonl not found: {placeholder_corpus_path}")
    else:
        placeholder_corpus_path = placeholder_dir / "corpus.jsonl"
        if bool(cfg["force_prepare"]) or not placeholder_corpus_path.exists():
            prepare_cmd = [
                sys.executable,
                str(repo_root / "scripts" / "prepare_lm_corpus.py"),
                "--run-name",
                str(cfg["run_name"]),
                "--language",
                lang,
                "--split",
                split,
                "--min-chars",
                str(int(cfg["min_chars"])),
                "--out-dir",
                str(placeholder_dir),
            ]
            for spec in list(cfg["datasets"]):
                prepare_cmd.extend(["--dataset", str(spec)])
            if bool(cfg["filter_cjk"]):
                prepare_cmd.append("--filter-cjk")
            else:
                prepare_cmd.append("--no-filter-cjk")
            if bool(cfg["canonicalize_placeholders"]):
                prepare_cmd.append("--canonicalize-placeholders")
            else:
                prepare_cmd.append("--no-canonicalize-placeholders")
            if allow_network:
                prepare_cmd.append("--allow-network")
            if trust_remote_code:
                prepare_cmd.append("--trust-remote-code")
            run_step(prepare_cmd, env=env, cwd=repo_root)

    rewriter_dir = layout.edge_deid_datasets_home / "processed" / run_slug / split / "rewriter"
    rewriter_dir.mkdir(parents=True, exist_ok=True)
    rewriter_corpus_path = rewriter_dir / "corpus.jsonl"
    manifest_path = rewriter_dir / "manifest.json"

    if bool(cfg["force_prepare"]) or not rewriter_corpus_path.exists():
        prepare_cmd = [
            sys.executable,
            str(repo_root / "scripts" / "prepare_rewriter_corpus.py"),
            "--input-jsonl",
            str(placeholder_corpus_path),
            "--output-jsonl",
            str(rewriter_corpus_path),
            "--language",
            lang,
            "--max-examples",
            str(int(cfg["rewriter_max_examples"])),
            "--min-chars",
            str(int(cfg["rewriter_min_chars"])),
            "--seed",
            str(int(cfg["rewriter_seed"])),
            "--noise-punct-prob",
            str(float(cfg["noise_punct_prob"])),
            "--noise-space-prob",
            str(float(cfg["noise_space_prob"])),
            "--noise-dup-prob",
            str(float(cfg["noise_dup_prob"])),
            "--manifest-out",
            str(manifest_path),
        ]
        if bool(cfg["rewriter_filter_cjk"]):
            prepare_cmd.append("--filter-cjk")
        run_step(prepare_cmd, env=env, cwd=repo_root)

    training_dir = layout.edge_deid_training_runs_home / run_slug
    train_cmd = [
        sys.executable,
        str(repo_root / "scripts" / "train_causal_lm.py"),
        "--model-dir",
        str(base_model_dir),
        "--output-dir",
        str(training_dir),
        "--input-jsonl",
        str(rewriter_corpus_path),
        "--block-size",
        str(int(cfg["block_size"])),
        "--epochs",
        str(int(cfg["epochs"])),
        "--max-steps",
        str(int(cfg["max_steps"])),
        "--batch-size",
        str(int(cfg["batch_size"])),
        "--gradient-accumulation-steps",
        str(int(cfg["gradient_accumulation_steps"])),
        "--precision",
        str(cfg["precision"]),
        "--learning-rate",
        str(float(cfg["learning_rate"])),
        "--weight-decay",
        str(float(cfg["weight_decay"])),
        "--warmup-ratio",
        str(float(cfg["warmup_ratio"])),
        "--preprocess-batch-size",
        str(int(cfg["preprocess_batch_size"])),
        "--preprocess-num-proc",
        str(int(cfg["preprocess_num_proc"])),
        "--seed",
        "0",
    ]
    if bool(cfg["tf32"]):
        train_cmd.append("--tf32")
    if allow_network:
        train_cmd.append("--allow-network")
    run_step(train_cmd, env=env, cwd=repo_root)

    models_dir = layout.models_home / "llm" / "edge_deid" / run_slug
    models_dir.parent.mkdir(parents=True, exist_ok=True)
    if models_dir.exists():
        shutil.rmtree(models_dir)
    shutil.copytree(training_dir, models_dir)

    report: Dict[str, Any] = {
        "run_name": str(cfg["run_name"]),
        "run_slug": run_slug,
        "language": lang,
        "split": split,
        "created_unix_s": time.time(),
        "base_model_dir": str(base_model_dir),
        "config": dict(cfg),
        "placeholder_corpus_jsonl": str(placeholder_corpus_path),
        "rewriter_corpus_jsonl": str(rewriter_corpus_path),
        "rewriter_manifest_json": str(manifest_path),
        "training_output_dir": str(training_dir),
        "models_output_dir": str(models_dir),
    }

    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if cfg.get("json_out"):
        out_path = Path(str(cfg["json_out"])).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()

