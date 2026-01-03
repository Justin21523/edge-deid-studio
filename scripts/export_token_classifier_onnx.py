from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from deid_pipeline.storage.env import apply_cache_env_defaults
from deid_pipeline.storage.layout import StorageLayout
from deid_pipeline.training.artifacts import copy_support_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a token-classification model to ONNX (dev-only).")
    parser.add_argument("--model-dir", required=True, help="Path to a trained model directory.")
    parser.add_argument("--output-dir", required=True, help="Output directory for ONNX artifacts.")
    parser.add_argument("--file-name", default="model.onnx", help="ONNX file name.")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    parser.add_argument(
        "--no-copy-support-files",
        action="store_true",
        help="Do not copy tokenizer/config files into the output directory.",
    )
    return parser.parse_args()


def _try_export_with_optimum_cli(*, model_dir: Path, output_dir: Path, opset: int) -> bool:
    """Attempt ONNX export via Optimum CLI; return True on success."""

    try:
        subprocess.check_call(
            [
                "optimum-cli",
                "export",
                "onnx",
                "--model",
                str(model_dir),
                "--task",
                "token-classification",
                "--opset",
                str(int(opset)),
                str(output_dir),
            ]
        )
        return True
    except FileNotFoundError:
        return False
    except subprocess.CalledProcessError:
        return False


def _export_with_torch_onnx(
    *,
    model_dir: Path,
    output_path: Path,
    opset: int,
) -> None:
    """Export to ONNX using `torch.onnx.export` (Optimum-free fallback)."""

    import inspect

    import torch  # type: ignore
    from transformers import AutoModelForTokenClassification  # type: ignore

    model = AutoModelForTokenClassification.from_pretrained(str(model_dir), local_files_only=True)
    model.eval()

    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    try:
        params = set(inspect.signature(model.forward).parameters.keys())
    except Exception:
        params = {"input_ids", "attention_mask", "token_type_ids"}

    input_names = ["input_ids"]
    if "attention_mask" in params:
        input_names.append("attention_mask")
    if "token_type_ids" in params:
        input_names.append("token_type_ids")

    batch = 2
    seq = 16
    input_ids = torch.zeros((batch, seq), dtype=torch.long)
    attention_mask = torch.ones((batch, seq), dtype=torch.long)
    token_type_ids = torch.zeros((batch, seq), dtype=torch.long)

    example_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }
    args_tuple = tuple(example_inputs[name] for name in input_names)

    dynamic_axes = {name: {0: "batch", 1: "sequence"} for name in input_names}
    dynamic_axes["logits"] = {0: "batch", 1: "sequence"}

    class _Wrapper(torch.nn.Module):
        def __init__(self, inner: torch.nn.Module, names: list[str]) -> None:
            super().__init__()
            self.inner = inner
            self.names = names

        def forward(self, *args):  # type: ignore[override]
            kwargs = {name: value for name, value in zip(self.names, args)}
            outputs = self.inner(**kwargs)
            return outputs.logits

    wrapper = _Wrapper(model, input_names)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        args_tuple,
        str(output_path),
        input_names=list(input_names),
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=int(opset),
        dynamo=False,
        external_data=False,
        do_constant_folding=True,
    )


def main() -> None:
    args = parse_args()
    layout = StorageLayout.from_project_root(Path(__file__).resolve().parents[1])
    apply_cache_env_defaults(layout=layout)

    model_dir = Path(args.model_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import onnx  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise ImportError("ONNX export requires `onnx`. Install with: pip install onnx") from exc

    used_optimum = _try_export_with_optimum_cli(model_dir=model_dir, output_dir=output_dir, opset=int(args.opset))

    desired = output_dir / str(args.file_name)
    if used_optimum and not desired.exists():
        candidates = sorted(output_dir.glob("*.onnx"))
        if len(candidates) == 1:
            candidates[0].rename(desired)
        elif candidates:
            raise RuntimeError(
                f"Unable to determine which ONNX file to rename (candidates={candidates}). "
                "Use the default file name or rename manually."
            )
    elif not used_optimum:
        _export_with_torch_onnx(model_dir=model_dir, output_path=desired, opset=int(args.opset))

    if not bool(args.no_copy_support_files):
        copied = copy_support_files(model_dir, output_dir, overwrite=True)
        if copied:
            print(f"Copied support files: {len(copied)}")

    print(f"Wrote ONNX model: {output_dir / args.file_name}")


if __name__ == "__main__":
    main()
