from __future__ import annotations

from deid_pipeline.training.canonical import canonicalize_entities, canonicalize_entity_type


def test_canonicalize_entity_type_maps_common_variants() -> None:
    assert canonicalize_entity_type("LASTNAME") == "NAME"
    assert canonicalize_entity_type("given_name") == "NAME"
    assert canonicalize_entity_type("user_name") == "USERNAME"
    assert canonicalize_entity_type("email_address") == "EMAIL"
    assert canonicalize_entity_type("phone_number") == "PHONE"
    assert canonicalize_entity_type("unified_business_no") == "UNIFIED_BUSINESS_NO"


def test_canonicalize_entity_type_strips_indices_and_long_placeholders() -> None:
    assert canonicalize_entity_type("PATIENT_LAST_NAME_1") == "NAME"
    assert canonicalize_entity_type("PHONE_2") == "PHONE"


def test_canonicalize_entity_type_falls_back_to_pii() -> None:
    assert canonicalize_entity_type("UNKNOWN_CUSTOM_FIELD") == "PII"


def test_canonicalize_entities_updates_entity_dicts() -> None:
    entities = [
        {"type": "LASTNAME", "span": (0, 3), "text": "Lin"},
        {"type": "EMAIL_ADDRESS", "span": (4, 10), "text": "a@b.c"},
    ]
    out = canonicalize_entities(entities)
    assert [e["type"] for e in out] == ["NAME", "EMAIL"]

