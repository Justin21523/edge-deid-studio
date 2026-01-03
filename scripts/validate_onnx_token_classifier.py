from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np

from deid_pipeline.storage.env import apply_cache_env_defaults
from deid_pipeline.storage.layout import StorageLayout
from deid_pipeline.runtime.onnx import select_onnx_providers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate an ONNX token-classification model (dev-only).")
    parser.add_argument("--onnx-model", required=True, help="Path to ONNX model.")
    parser.add_argument("--tokenizer-dir", required=True, help="Path to local tokenizer/config directory.")
    parser.add_argument(
        "--providers",
        default="CUDAExecutionProvider,CPUExecutionProvider",
        help="Comma-separated provider list (filtered to available providers).",
    )
    parser.add_argument("--text", default="ID A123456789 and phone 0912345678.", help="Input text.")
    parser.add_argument("--max-length", type=int, default=256, help="Tokenizer max length.")
    parser.add_argument(
        "--pytorch-model-dir",
        default="",
        help="Optional PyTorch model directory to compare argmax predictions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layout = StorageLayout.from_project_root(Path(__file__).resolve().parents[1])
    apply_cache_env_defaults(layout=layout)

    onnx_path = Path(args.onnx_model).expanduser().resolve()
    tok_dir = Path(args.tokenizer_dir).expanduser().resolve()

    import onnxruntime as ort  # type: ignore
    from transformers import AutoModelForTokenClassification, AutoTokenizer  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(str(tok_dir), use_fast=True, local_files_only=True)
    encoding = tokenizer(
        args.text,
        return_tensors="np",
        truncation=True,
        max_length=int(args.max_length),
        padding="max_length",
    )

    requested = tuple(p.strip() for p in str(args.providers).split(",") if p.strip())
    providers = select_onnx_providers(requested)
    sess = ort.InferenceSession(str(onnx_path), providers=list(providers))
    input_names = {i.name for i in sess.get_inputs()}
    output_name = sess.get_outputs()[0].name
    ort_inputs: Dict[str, np.ndarray] = {k: encoding[k] for k in input_names if k in encoding}
    onnx_logits = sess.run([output_name], ort_inputs)[0]

    print(f"ONNX logits shape: {onnx_logits.shape}")
    print(f"ONNX providers used: {sess.get_providers()}")

    if args.pytorch_model_dir:
        import torch  # type: ignore

        model = AutoModelForTokenClassification.from_pretrained(
            str(Path(args.pytorch_model_dir).expanduser().resolve()),
            local_files_only=True,
        )
        model.eval()

        pt_inputs = {k: torch.from_numpy(v) for k, v in encoding.items() if k in {"input_ids", "attention_mask", "token_type_ids"}}
        with torch.no_grad():
            pt_logits = model(**pt_inputs).logits.detach().cpu().numpy()

        same = (pt_logits.argmax(axis=-1) == onnx_logits.argmax(axis=-1)).mean()
        print(f"Argmax agreement: {same * 100:.2f}%")


if __name__ == "__main__":
    main()
