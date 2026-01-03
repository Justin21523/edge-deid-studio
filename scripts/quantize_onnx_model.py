from __future__ import annotations

import argparse
from pathlib import Path

from deid_pipeline.storage.env import apply_cache_env_defaults
from deid_pipeline.storage.layout import StorageLayout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dynamic INT8 quantization for an ONNX model.")
    parser.add_argument("--input", required=True, help="Path to input ONNX model.")
    parser.add_argument("--output", required=True, help="Path to output quantized ONNX model.")
    parser.add_argument(
        "--weight-type",
        choices=["qint8", "quint8"],
        default="qint8",
        help="Quantization weight type.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layout = StorageLayout.from_project_root(Path(__file__).resolve().parents[1])
    apply_cache_env_defaults(layout=layout)

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input model not found: {input_path}")

    try:
        import onnx  # noqa: F401
        from onnxruntime.quantization import QuantType, quantize_dynamic  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "ONNX quantization requires `onnx` and `onnxruntime` with quantization support. "
            "Install with: pip install onnx onnxruntime"
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    weight_type = QuantType.QInt8 if args.weight_type == "qint8" else QuantType.QUInt8

    quantize_dynamic(str(input_path), str(output_path), weight_type=weight_type)
    print(f"Wrote quantized model: {output_path}")


if __name__ == "__main__":
    main()
