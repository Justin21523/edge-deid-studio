from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from deid_pipeline.storage.env import apply_cache_env_defaults
from deid_pipeline.storage.layout import StorageLayout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download/export local models (dev-only; requires internet).")
    parser.add_argument(
        "--project-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Repository root used to resolve the AI_WAREHOUSE layout fallback.",
    )
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Allow network access (required to download models).",
    )
    parser.add_argument(
        "--only",
        default="",
        help="Optional comma-separated model keys to download (gpt2_base, ner_zh, ner_zh_onnx).",
    )
    return parser.parse_args()


def download_models() -> None:
    args = parse_args()
    layout = StorageLayout.from_project_root(args.project_root)
    apply_cache_env_defaults(layout=layout)

    if not bool(args.allow_network):
        raise RuntimeError("Network access is disabled. Re-run with --allow-network to download models.")

    from huggingface_hub import snapshot_download  # type: ignore
    from transformers import (  # type: ignore
        AutoModelForCausalLM,
        AutoModelForTokenClassification,
        AutoTokenizer,
    )

    models = {
        "gpt2_base": ("gpt2", layout.models_home / "llm" / "gpt2"),
        "gpt2_zh_base": ("IDEA-CCNL/Wenzhong-GPT2-110M", layout.models_home / "llm" / "gpt2-zh-base"),
        "ner_zh": ("ckiplab/bert-base-chinese-ner", layout.edge_deid_models_home / "bert-ner-zh"),
        "ner_zh_onnx": ("ckiplab/bert-base-chinese-ner", layout.edge_deid_models_home / "bert-ner-zh.onnx"),
    }

    only = [s.strip() for s in str(getattr(args, "only", "")).split(",") if s.strip()]
    if only:
        unknown = sorted(set(only) - set(models))
        if unknown:
            raise ValueError(f"Unknown model keys: {unknown}. Supported: {sorted(models)}")
        models = {k: v for k, v in models.items() if k in set(only)}

    for name, (repo_id, target) in models.items():
        target_path = Path(target).expanduser().resolve()

        if name == "gpt2_base":
            if not (target_path / "pytorch_model.bin").exists():
                print(f">> Downloading GPT-2 via Transformers -> {target_path}")
                target_path.mkdir(parents=True, exist_ok=True)
                tokenizer = AutoTokenizer.from_pretrained(repo_id)
                model = AutoModelForCausalLM.from_pretrained(repo_id)
                tokenizer.save_pretrained(target_path)
                model.save_pretrained(target_path)
            else:
                print(f"GPT-2 already exists at {target_path}, skipping.")

        elif name == "gpt2_zh_base":
            if not (target_path / "config.json").exists():
                print(f">> Downloading GPT-2 zh snapshot: {repo_id} -> {target_path}")
                target_path.mkdir(parents=True, exist_ok=True)
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=target_path,
                    local_dir_use_symlinks=False,
                )
            else:
                print(f"{name} already exists at {target_path}, skipping.")

        elif name == "ner_zh":
            if not target_path.exists():
                print(f">> Downloading NER model snapshot: {repo_id} -> {target_path}")
                target_path.mkdir(parents=True, exist_ok=True)
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=target_path,
                    local_dir_use_symlinks=False,
                )
            else:
                print(f"{name} already exists at {target_path}, skipping.")

        elif name == "ner_zh_onnx":
            pt_dir = layout.edge_deid_models_home / "bert-ner-zh"
            onnx_file = target_path
            if not onnx_file.exists():
                print(f">> Converting PyTorch model to ONNX: {pt_dir} -> {onnx_file}")
                _ = AutoTokenizer.from_pretrained(str(pt_dir), local_files_only=True)
                _ = AutoModelForTokenClassification.from_pretrained(str(pt_dir), local_files_only=True)

                export_dir = onnx_file.parent / "bert-ner-zh-onnx"
                export_dir.mkdir(parents=True, exist_ok=True)
                subprocess.check_call(
                    [
                        "optimum-cli",
                        "export",
                        "onnx",
                        "--model",
                        str(pt_dir),
                        "--task",
                        "token-classification",
                        "--opset",
                        "17",
                        str(export_dir),
                    ]
                )

                candidate = export_dir / "model.onnx"
                if not candidate.exists():
                    onnx_candidates = sorted(export_dir.glob("*.onnx"))
                    if len(onnx_candidates) == 1:
                        candidate = onnx_candidates[0]
                    else:
                        raise RuntimeError(f"Unexpected ONNX outputs: {onnx_candidates}")

                onnx_file.parent.mkdir(parents=True, exist_ok=True)
                candidate.replace(onnx_file)
            else:
                print(f"ONNX model already exists at {onnx_file}, skipping.")

        else:
            print(f"Unknown model key: {name}, skipping.")


if __name__ == "__main__":
    download_models()
