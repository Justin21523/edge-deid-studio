#!/usr/bin/env python
"""Export a token-classification model to ONNX using `optimum-cli` (dev-only).

This helper lives under `notebooks/` and follows the AI_WAREHOUSE 3.0 storage layout.
If `--model` is an HF repo id, network access is required and must be explicitly enabled.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from deid_pipeline.storage.env import apply_cache_env_defaults
from deid_pipeline.storage.layout import StorageLayout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a token-classification model to ONNX via optimum-cli (dev-only)."
    )
    parser.add_argument(
        "--model",
        default="ckiplab/bert-base-chinese-ner",
        help="Local model directory or HF repo id (requires --allow-network).",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory for exported ONNX artifacts.",
    )
    parser.add_argument(
        "--quantized-output-dir",
        default="",
        help="Output directory for quantized artifacts (INT8 weights).",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Allow network access (required when --model is an HF repo id).",
    )
    parser.add_argument(
        "--project-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="Repository root used to resolve the AI_WAREHOUSE layout fallback.",
    )
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print(">>> " + " ".join(cmd))
    subprocess.check_call(cmd)


def main() -> None:
    args = parse_args()
    layout = StorageLayout.from_project_root(args.project_root)
    apply_cache_env_defaults(layout=layout)

    model_arg = str(args.model).strip()
    model_path = Path(model_arg).expanduser()
    is_local = model_path.exists()

    if not args.allow_network:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        if not is_local and "/" in model_arg:
            raise RuntimeError(
                "Network access is disabled. Provide a local path via --model or re-run with --allow-network."
            )
    else:
        os.environ.pop("HF_HUB_OFFLINE", None)

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (layout.edge_deid_models_home / "ckiplab--bert-base-chinese-ner")
    )
    quantized_dir = (
        Path(args.quantized_output_dir).expanduser().resolve()
        if args.quantized_output_dir
        else output_dir.with_name(output_dir.name + ".int8")
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    quantized_dir.parent.mkdir(parents=True, exist_ok=True)

    run(
        [
            "optimum-cli",
            "export",
            "onnx",
            "--model",
            model_arg,
            "--task",
            "token-classification",
            "--opset",
            str(int(args.opset)),
            str(output_dir),
        ]
    )
    run(
        [
            "optimum-cli",
            "onnxruntime",
            "quantize",
            "--model",
            str(output_dir),
            "--weight-only",
            "--per-channel",
            "--dtype",
            "qint8",
            "--output",
            str(quantized_dir),
        ]
    )
    print(f"Wrote ONNX artifacts: {output_dir}")
    print(f"Wrote quantized artifacts: {quantized_dir}")


if __name__ == "__main__":
    main()
