from __future__ import annotations

from deid_pipeline.runtime.onnx import select_onnx_providers


def test_select_onnx_providers_filters_unavailable():
    providers = select_onnx_providers(["CUDAExecutionProvider", "CPUExecutionProvider"])
    assert "CPUExecutionProvider" in providers
    assert "CUDAExecutionProvider" not in providers
