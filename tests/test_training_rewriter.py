from __future__ import annotations

import random

from deid_pipeline.training.rewriter import (
    build_prompt,
    fill_placeholders_with_fake_values,
    hash16,
    make_noisy,
)


class DummyProvider:
    def generate_deterministic(self, entity_type: str, original: str, *, context_hash: str) -> str:
        return f"<<{entity_type}:{original}:{context_hash}>>"


def test_hash16_is_deterministic() -> None:
    assert hash16("abc") == hash16("abc")
    assert len(hash16("abc")) == 16


def test_fill_placeholders_with_fake_values_canonicalizes_types() -> None:
    provider = DummyProvider()
    text = "Hi <LASTNAME_1>, email [EMAIL_ADDRESS]."
    ctx = hash16(text)
    filled = fill_placeholders_with_fake_values(text, provider, context_hash=ctx)
    assert "<LASTNAME" not in filled
    assert "[EMAIL_ADDRESS" not in filled
    assert "<<NAME:LASTNAME:0:" in filled
    assert "<<EMAIL:EMAIL_ADDRESS:1:" in filled


def test_make_noisy_preserves_replacement_tokens() -> None:
    rng = random.Random(0)
    clean = "王小明的統編是ABC123，請確認。"
    noisy = make_noisy(clean, rng)
    assert "ABC123" in noisy


def test_build_prompt_contains_input_output() -> None:
    prompt = build_prompt("noisy", "clean", language="zh")
    assert "INPUT:" in prompt
    assert "OUTPUT:" in prompt
    assert "noisy" in prompt
    assert "clean" in prompt

