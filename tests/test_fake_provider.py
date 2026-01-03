from __future__ import annotations

import re

import pytest

from deid_pipeline.config import Config
from deid_pipeline.pii.utils.fake_provider import FakeProvider


def test_fake_provider_generates_tw_patterns_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Config, "FAKER_LOCALE", "zh_TW")
    provider = FakeProvider()

    tw_id = provider.generate_deterministic("ID", "A123456789", context_hash="doc-1")
    assert re.match(r"^[A-Z][0-9]{9}$", tw_id)
    assert tw_id[1] in {"1", "2"}

    phone = provider.generate_deterministic("PHONE", "0912345678", context_hash="doc-1")
    assert re.match(r"^09[0-9]{8}$", phone)

    ubn = provider.generate_deterministic("UNIFIED_BUSINESS_NO", "12345678", context_hash="doc-1")
    assert re.match(r"^[0-9]{8}$", ubn)

    # Determinism for the same input triple.
    assert tw_id == provider.generate_deterministic("ID", "A123456789", context_hash="doc-1")


def test_fake_provider_fallback_without_faker_is_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Config, "FAKER_LOCALE", "zh_TW")
    monkeypatch.setattr(FakeProvider, "_try_init_faker", lambda self: None)

    provider = FakeProvider()
    tw_id = provider.generate_deterministic("ID", "A123456789", context_hash="doc-1")
    assert re.match(r"^[A-Z][0-9]{9}$", tw_id)

    phone = provider.generate_deterministic("PHONE", "0912345678", context_hash="doc-1")
    assert re.match(r"^09[0-9]{8}$", phone)

    assert tw_id == provider.generate_deterministic("ID", "A123456789", context_hash="doc-1")


def test_fake_provider_uses_ssn_like_format_for_en_locale(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Config, "FAKER_LOCALE", "en_US")
    monkeypatch.setattr(FakeProvider, "_try_init_faker", lambda self: None)

    provider = FakeProvider()
    ssn = provider.generate_deterministic("ID", "123-45-6789", context_hash="doc-1")
    assert re.match(r"^[0-9]{3}-[0-9]{2}-[0-9]{4}$", ssn)

    phone = provider.generate_deterministic("PHONE", "555-111-2222", context_hash="doc-1")
    assert re.match(r"^555-[0-9]{3}-[0-9]{4}$", phone)
