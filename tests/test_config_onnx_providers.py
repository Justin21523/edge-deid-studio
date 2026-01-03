from __future__ import annotations

import importlib
import os

import deid_pipeline.config as config_mod


def test_config_onnx_providers_prefers_cuda_when_use_gpu(monkeypatch) -> None:
    monkeypatch.setenv("USE_GPU", "true")
    monkeypatch.delenv("ONNX_PROVIDERS", raising=False)

    reloaded = importlib.reload(config_mod)
    assert reloaded.Config.USE_GPU is True
    assert reloaded.Config.ONNX_PROVIDERS[0] == "CUDAExecutionProvider"

    # Restore the original module state for other tests.
    monkeypatch.delenv("USE_GPU", raising=False)
    importlib.reload(reloaded)


def test_config_onnx_providers_env_override(monkeypatch) -> None:
    monkeypatch.setenv("ONNX_PROVIDERS", "CPUExecutionProvider")

    reloaded = importlib.reload(config_mod)
    assert reloaded.Config.ONNX_PROVIDERS == ["CPUExecutionProvider"]

    # Restore the original module state for other tests.
    monkeypatch.delenv("ONNX_PROVIDERS", raising=False)
    importlib.reload(reloaded)

