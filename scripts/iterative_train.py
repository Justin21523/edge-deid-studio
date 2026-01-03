from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a 1-hour iterative GPT-2 DeID fine-tuning loop (dev-only).")
    parser.add_argument("--config", required=True, help="YAML/JSON config file.")
    parser.add_argument("--print-config", action="store_true", help="Print effective config as JSON and exit.")
    return parser.parse_args()


def build_effective_config(args: argparse.Namespace) -> Dict[str, Any]:
    raw = _load_config_file(args.config)
    data = raw.get("data") if isinstance(raw.get("data"), dict) else {}
    prompt = raw.get("prompt") if isinstance(raw.get("prompt"), dict) else {}
    training = raw.get("training") if isinstance(raw.get("training"), dict) else {}
    iterative = raw.get("iterative") if isinstance(raw.get("iterative"), dict) else {}
    generation = raw.get("generation") if isinstance(raw.get("generation"), dict) else {}
    evaluation = raw.get("evaluation") if isinstance(raw.get("evaluation"), dict) else {}

    cfg: Dict[str, Any] = {
        "run_name": str(raw.get("run_name") or "deid-gpt2-zh-iterative"),
        "language": str(raw.get("language") or "zh"),
        "train_jsonl": str(data.get("train_jsonl") or ""),
        "valid_jsonl": str(data.get("valid_jsonl") or ""),
        "max_train_examples": int(data.get("max_train_examples", 0)),
        "max_valid_examples": int(data.get("max_valid_examples", 0)),
        "regex_rules": str(data.get("regex_rules") or ""),
        "prompt_template": str(prompt.get("template_path") or "configs/prompts/deid_zh_v1.txt"),
        "banned_phrases": str(evaluation.get("banned_phrases") or "configs/eval/deid_format_zh.yaml"),
        "init_model_dir": str(training.get("init_model_dir") or ""),
        "steps_per_round": int(iterative.get("steps_per_round", training.get("steps_per_round", 1000))),
        "time_budget_s": int(iterative.get("time_budget_s", 3600)),
        "seed": int(iterative.get("seed", training.get("seed", 0))),
        "keep_last_k": int(iterative.get("keep_last_k", 2)),
        "keep_top_k": int(iterative.get("keep_top_k", 2)),
        "epochs": int(training.get("epochs", 1)),
        "batch_size": int(training.get("batch_size", 2)),
        "gradient_accumulation_steps": int(training.get("gradient_accumulation_steps", 8)),
        "block_size": int(training.get("block_size", 256)),
        "precision": str(training.get("precision", "auto")),
        "tf32": _as_bool(training.get("tf32"), default=True),
        "learning_rate": float(training.get("learning_rate", 3e-5)),
        "weight_decay": float(training.get("weight_decay", 0.01)),
        "warmup_ratio": float(training.get("warmup_ratio", 0.05)),
        "preprocess_batch_size": int(training.get("preprocess_batch_size", 512)),
        "preprocess_num_proc": int(training.get("preprocess_num_proc", 0)),
        "split_paragraphs": _as_bool(data.get("split_paragraphs"), default=True),
        "min_chars": int(data.get("min_chars", 20)),
        "max_prompt_tokens": int(generation.get("max_prompt_tokens", 768)),
        "max_new_tokens": int(generation.get("max_new_tokens", 256)),
        "do_sample": _as_bool(generation.get("do_sample"), default=False),
        "temperature": float(generation.get("temperature", 1.0)),
        "top_p": float(generation.get("top_p", 0.9)),
        "fluency_model_dir": str(evaluation.get("fluency_model_dir") or ""),
        "allow_network": _as_bool(raw.get("allow_network"), default=False),
        "trust_remote_code": _as_bool(raw.get("trust_remote_code"), default=False),
    }
    return cfg


def _python_env(repo_root: Path, *, allow_network: bool) -> Dict[str, str]:
    env = dict(os.environ)
    py_path = str(repo_root / "src")
    env["PYTHONPATH"] = py_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    if not allow_network:
        env.setdefault("HF_HUB_OFFLINE", "1")
    return env


def _git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_root), text=True).strip()
    except Exception:
        return ""


def run_step(cmd: Sequence[str], *, env: Mapping[str, str], cwd: Path) -> None:
    subprocess.check_call(list(cmd), cwd=str(cwd), env=dict(env))


def _composite_score(metrics: Dict[str, Any]) -> float:
    pii_recall = float(metrics.get("pii_removal_recall", 1.0) or 0.0)
    fmt = float(metrics.get("format_compliance_rate", 1.0) or 0.0)
    tcons = float(metrics.get("type_consistency", 1.0) or 0.0)
    over = float(metrics.get("over_rewrite_rate", 0.0) or 0.0)
    rep = float(metrics.get("repetition_3gram_rate", 0.0) or 0.0)
    leak_rate = float(metrics.get("pii_leak_rate", 0.0) or 0.0)

    score = (
        0.45 * pii_recall
        + 0.25 * fmt
        + 0.20 * tcons
        + 0.05 * (1.0 - min(1.0, over))
        + 0.05 * (1.0 - min(1.0, rep))
    )
    score -= min(1.0, leak_rate * 5.0)
    return float(score)


def _suggest_next_hparams(h: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Rule-based hyperparameter tweaks (deterministic)."""

    out = dict(h)
    leak_rate = float(metrics.get("pii_leak_rate", 0.0) or 0.0)
    over = float(metrics.get("over_rewrite_rate", 0.0) or 0.0)
    fmt = float(metrics.get("format_compliance_rate", 1.0) or 0.0)

    # Primary goal: eliminate leakage first.
    if leak_rate > 0.0:
        out["learning_rate"] = max(1e-6, float(out["learning_rate"]) * 0.7)
        out["warmup_ratio"] = min(0.10, float(out["warmup_ratio"]) + 0.01)
        return out

    # If the model over-rewrites, reduce LR and slightly increase weight decay.
    if over > 0.35:
        out["learning_rate"] = max(1e-6, float(out["learning_rate"]) * 0.85)
        out["weight_decay"] = min(0.10, float(out["weight_decay"]) + 0.01)
        return out

    # If format compliance is low, keep training stable (avoid sampling changes here).
    if fmt < 0.90:
        out["learning_rate"] = max(1e-6, float(out["learning_rate"]) * 0.9)
        out["warmup_ratio"] = min(0.10, float(out["warmup_ratio"]) + 0.005)
        return out

    # Otherwise, small cyclical LR nudges to explore without randomness.
    lr = float(out["learning_rate"])
    out["learning_rate"] = lr * 1.05 if lr < 5e-5 else lr * 0.9
    return out


def _write_summary_row(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def _prune_round_dirs(rounds_dir: Path, *, keep_dirs: Sequence[Path]) -> None:
    keep = {p.resolve() for p in keep_dirs}
    if not rounds_dir.exists():
        return
    for child in sorted(rounds_dir.iterdir()):
        if not child.is_dir():
            continue
        if child.resolve() in keep:
            continue
        shutil.rmtree(child, ignore_errors=True)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    layout = StorageLayout.from_project_root(repo_root)
    apply_cache_env_defaults(layout=layout)

    cfg = build_effective_config(args)
    if bool(args.print_config):
        print(json.dumps(cfg, ensure_ascii=False, indent=2))
        return

    if not str(cfg["train_jsonl"]).strip() or not str(cfg["valid_jsonl"]).strip():
        raise ValueError("Config must set data.train_jsonl and data.valid_jsonl.")

    run_slug = dataset_slug(str(cfg["run_name"]))
    allow_network = bool(cfg["allow_network"])
    trust_remote_code = bool(cfg["trust_remote_code"])
    env = _python_env(repo_root, allow_network=allow_network)

    init_model_dir = (
        Path(str(cfg["init_model_dir"])).expanduser().resolve()
        if str(cfg["init_model_dir"]).strip()
        else (layout.models_home / "llm" / "edge_deid" / "gpt2-zh-rewriter-large")
    )
    if not (init_model_dir / "config.json").exists():
        raise FileNotFoundError(f"init_model_dir not found: {init_model_dir}")

    # Prepare corpus once (prompt+target).
    corpus_dir = layout.edge_deid_datasets_home / "processed" / run_slug / "train"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = corpus_dir / "corpus.jsonl"
    corpus_manifest = corpus_dir / "manifest.json"

    run_step(
        [
            sys.executable,
            str(repo_root / "scripts" / "prepare_deid_pairs_corpus.py"),
            "--input-jsonl",
            str(Path(cfg["train_jsonl"]).expanduser().resolve()),
            "--output-jsonl",
            str(corpus_path),
            "--prompt-template",
            str(Path(cfg["prompt_template"]).expanduser().resolve()),
            "--max-examples",
            str(int(cfg["max_train_examples"])),
            "--min-chars",
            str(int(cfg["min_chars"])),
            "--manifest-out",
            str(corpus_manifest),
        ]
        + (["--split-paragraphs"] if bool(cfg["split_paragraphs"]) else []),
        env=env,
        cwd=repo_root,
    )

    # Output layout.
    rounds_dir = layout.edge_deid_training_runs_home / run_slug / "rounds"
    logs_root = layout.training_logs_home / "edge_deid" / run_slug
    logs_root.mkdir(parents=True, exist_ok=True)

    run_manifest: Dict[str, Any] = {
        "run_name": str(cfg["run_name"]),
        "run_slug": run_slug,
        "created_unix_s": time.time(),
        "git_commit": _git_commit(repo_root),
        "config": dict(cfg),
        "init_model_dir": str(init_model_dir),
        "corpus_jsonl": str(corpus_path),
        "corpus_manifest": str(corpus_manifest),
    }
    (logs_root / "run_manifest.json").write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_csv = logs_root / "summary.csv"
    best_json = logs_root / "best.json"

    best_score = -1e9
    best_dir: Path = init_model_dir
    best_metrics: Dict[str, Any] = {}
    best_hparams: Dict[str, Any] = {}

    hparams: Dict[str, Any] = {
        "learning_rate": float(cfg["learning_rate"]),
        "weight_decay": float(cfg["weight_decay"]),
        "warmup_ratio": float(cfg["warmup_ratio"]),
    }

    start = time.time()
    round_idx = 0
    round_scores: List[Tuple[float, Path]] = []

    while (time.time() - start) < float(cfg["time_budget_s"]):
        round_idx += 1
        round_name = f"round_{round_idx:03d}"
        round_dir = rounds_dir / round_name
        round_dir.mkdir(parents=True, exist_ok=True)

        # --- Train ---
        train_cmd = [
            sys.executable,
            str(repo_root / "scripts" / "train_causal_lm.py"),
            "--model-dir",
            str(best_dir),
            "--output-dir",
            str(round_dir),
            "--input-jsonl",
            str(corpus_path),
            "--block-size",
            str(int(cfg["block_size"])),
            "--epochs",
            str(int(cfg["epochs"])),
            "--max-steps",
            str(int(cfg["steps_per_round"])),
            "--batch-size",
            str(int(cfg["batch_size"])),
            "--gradient-accumulation-steps",
            str(int(cfg["gradient_accumulation_steps"])),
            "--precision",
            str(cfg["precision"]),
            "--learning-rate",
            str(float(hparams["learning_rate"])),
            "--weight-decay",
            str(float(hparams["weight_decay"])),
            "--warmup-ratio",
            str(float(hparams["warmup_ratio"])),
            "--preprocess-batch-size",
            str(int(cfg["preprocess_batch_size"])),
            "--preprocess-num-proc",
            str(int(cfg["preprocess_num_proc"])),
            "--seed",
            str(int(cfg["seed"])),
        ]
        if bool(cfg["tf32"]):
            train_cmd.append("--tf32")
        if allow_network:
            train_cmd.append("--allow-network")

        train_t0 = time.time()
        run_step(train_cmd, env=env, cwd=repo_root)
        train_s = float(time.time() - train_t0)

        # --- Eval ---
        eval_out_dir = logs_root / round_name
        eval_out_dir.mkdir(parents=True, exist_ok=True)
        eval_cmd = [
            sys.executable,
            str(repo_root / "scripts" / "evaluate_deid.py"),
            "--data-jsonl",
            str(Path(cfg["valid_jsonl"]).expanduser().resolve()),
            "--model-dir",
            str(round_dir),
            "--prompt-template",
            str(Path(cfg["prompt_template"]).expanduser().resolve()),
            "--banned-phrases",
            str(Path(cfg["banned_phrases"]).expanduser().resolve()),
            "--output-dir",
            str(eval_out_dir),
            "--max-examples",
            str(int(cfg["max_valid_examples"])),
            "--max-prompt-tokens",
            str(int(cfg["max_prompt_tokens"])),
            "--max-new-tokens",
            str(int(cfg["max_new_tokens"])),
            "--seed",
            str(int(cfg["seed"])),
            "--temperature",
            str(float(cfg["temperature"])),
            "--top-p",
            str(float(cfg["top_p"])),
        ]
        if bool(cfg["do_sample"]):
            eval_cmd.append("--do-sample")
        if str(cfg["regex_rules"]).strip():
            eval_cmd.extend(["--regex-rules", str(Path(cfg["regex_rules"]).expanduser().resolve())])
        if str(cfg["fluency_model_dir"]).strip():
            eval_cmd.extend(["--fluency-model-dir", str(Path(cfg["fluency_model_dir"]).expanduser().resolve())])
        if allow_network:
            eval_cmd.append("--allow-network")

        eval_t0 = time.time()
        run_step(eval_cmd, env=env, cwd=repo_root)
        eval_s = float(time.time() - eval_t0)

        metrics_path = eval_out_dir / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        score = _composite_score(metrics)
        round_scores.append((float(score), round_dir))

        if float(score) > float(best_score):
            best_score = float(score)
            best_dir = round_dir
            best_metrics = dict(metrics)
            best_hparams = dict(hparams)

        # Append summary row.
        summary_row = {
            "round": round_idx,
            "round_name": round_name,
            "checkpoint_dir": str(round_dir),
            "score": float(score),
            "train_seconds": float(train_s),
            "eval_seconds": float(eval_s),
            "learning_rate": float(hparams["learning_rate"]),
            "weight_decay": float(hparams["weight_decay"]),
            "warmup_ratio": float(hparams["warmup_ratio"]),
            "pii_removal_recall": float(metrics.get("pii_removal_recall", 0.0) or 0.0),
            "pii_leak_rate": float(metrics.get("pii_leak_rate", 0.0) or 0.0),
            "type_consistency": float(metrics.get("type_consistency", 0.0) or 0.0),
            "format_compliance_rate": float(metrics.get("format_compliance_rate", 0.0) or 0.0),
            "over_rewrite_rate": float(metrics.get("over_rewrite_rate", 0.0) or 0.0),
            "repetition_3gram_rate": float(metrics.get("repetition_3gram_rate", 0.0) or 0.0),
        }
        _write_summary_row(summary_csv, summary_row)

        # Save current best pointer.
        best_payload = {
            "best_score": float(best_score),
            "best_checkpoint_dir": str(best_dir),
            "best_hparams": dict(best_hparams),
            "best_metrics": dict(best_metrics),
            "updated_unix_s": time.time(),
        }
        best_json.write_text(json.dumps(best_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        # Prune checkpoints (keep last-K and top-K).
        last_keep = {round_dir}
        if int(cfg["keep_last_k"]) > 1:
            recent = sorted(rounds_dir.glob("round_*"))[-int(cfg["keep_last_k"]) :]
            last_keep = {p.resolve() for p in recent if p.is_dir()}

        top_keep: List[Path] = []
        for s, p in sorted(round_scores, key=lambda x: x[0], reverse=True)[: int(cfg["keep_top_k"])]:
            top_keep.append(p)

        keep_dirs = sorted({*last_keep, *(p.resolve() for p in top_keep)}, key=lambda p: str(p))
        _prune_round_dirs(rounds_dir, keep_dirs=keep_dirs)

        # Next round hyperparams.
        hparams = _suggest_next_hparams(hparams, metrics)

    # Copy best model to the canonical models location for downstream use.
    final_models_dir = layout.models_home / "llm" / "edge_deid" / run_slug
    final_models_dir.parent.mkdir(parents=True, exist_ok=True)
    if final_models_dir.exists():
        shutil.rmtree(final_models_dir)
    shutil.copytree(best_dir, final_models_dir)

    print(json.dumps({"best_checkpoint_dir": str(best_dir), "best_models_dir": str(final_models_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

