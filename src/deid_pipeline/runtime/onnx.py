from __future__ import annotations

from typing import Iterable, Tuple


def select_onnx_providers(preferred: Iterable[str] | None = None) -> Tuple[str, ...]:
    """Return ONNX Runtime providers filtered to the locally available set."""

    import onnxruntime as ort  # type: ignore

    available = set(ort.get_available_providers())
    preferred_list = list(preferred) if preferred is not None else []

    selected = [prov for prov in preferred_list if prov in available]
    if not selected:
        selected = ["CPUExecutionProvider"] if "CPUExecutionProvider" in available else sorted(available)
    return tuple(selected)


def create_session_options(
    *,
    intra_op_num_threads: int | None = None,
    inter_op_num_threads: int | None = None,
) -> "onnxruntime.SessionOptions":
    """Create session options tuned for local inference."""

    import onnxruntime as ort  # type: ignore

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    if intra_op_num_threads is not None:
        sess_options.intra_op_num_threads = int(intra_op_num_threads)
    if inter_op_num_threads is not None:
        sess_options.inter_op_num_threads = int(inter_op_num_threads)

    return sess_options
