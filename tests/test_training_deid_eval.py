from __future__ import annotations

from deid_pipeline.training.deid_eval import DeidExample, detect_pii_blocks, evaluate_prediction


def test_deid_eval_detects_leakage_from_markup() -> None:
    ex = DeidExample(
        example_id="1",
        input_text="Call [PHONE]0912-345-678[/PHONE].",
        target_text="Call 0918-222-333.",
    )
    pii_blocks = detect_pii_blocks(ex.input_text, prefer_markup=True)
    report = evaluate_prediction(ex, "Call 0912-345-678.", pii_blocks=pii_blocks, banned_phrases=[])
    assert report.pii_total == 1
    assert report.pii_leak_count == 1
    assert report.pii_removal_recall == 0.0


def test_deid_eval_format_compliance_flags_prompt_artifacts() -> None:
    ex = DeidExample(example_id="1", input_text="Hi", target_text="Hello")
    report = evaluate_prediction(ex, "OUTPUT: Hello", pii_blocks=[], banned_phrases=[])
    assert report.format_compliant is False


def test_deid_eval_type_consistency_counts_regex_matches() -> None:
    ex = DeidExample(
        example_id="1",
        input_text="Email [EMAIL]bob@example.com[/EMAIL].",
        target_text="Email user123@example.com.",
    )
    pii_blocks = detect_pii_blocks(ex.input_text, prefer_markup=True)
    report = evaluate_prediction(ex, "Email user123@example.com.", pii_blocks=pii_blocks, banned_phrases=[])
    assert report.type_consistency == 1.0

