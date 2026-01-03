from __future__ import annotations

import re
from typing import Iterable, List, Optional, Sequence

from ..core.contracts import Entity


_TRAILING_INDEX_RE = re.compile(r"(?:[_-]?\d+)+$")
_NON_ALPHA_RE = re.compile(r"[^A-Z_]+")


CANONICAL_ENTITY_TYPES: tuple[str, ...] = (
    "NAME",
    "USERNAME",
    "EMAIL",
    "PHONE",
    "ID",
    "PASSPORT",
    "MEDICAL_ID",
    "UNIFIED_BUSINESS_NO",
    "ADDRESS",
    "ORGANIZATION",
    "DATE",
    "TIME",
    "IP_ADDRESS",
    "URL",
    "CREDIT_CARD",
    "BANK_ACCOUNT",
    "LICENSE_PLATE",
    "AGE",
    "PII",
)


EXACT_TYPE_MAP: dict[str, str] = {
    # Person/name variants
    "PER": "NAME",
    "PERSON": "NAME",
    "NAME": "NAME",
    "FULLNAME": "NAME",
    "FULL_NAME": "NAME",
    "FIRSTNAME": "NAME",
    "FIRST_NAME": "NAME",
    "GIVENNAME": "NAME",
    "GIVEN_NAME": "NAME",
    "LASTNAME": "NAME",
    "LAST_NAME": "NAME",
    "SURNAME": "NAME",
    "MIDDLE_NAME": "NAME",
    "PATIENT_NAME": "NAME",
    "DOCTOR_NAME": "NAME",
    # Accounts
    "USERNAME": "USERNAME",
    "USER_NAME": "USERNAME",
    "HANDLE": "USERNAME",
    # Contact
    "EMAIL": "EMAIL",
    "EMAIL_ADDRESS": "EMAIL",
    "PHONE": "PHONE",
    "PHONE_NUMBER": "PHONE",
    "MOBILE": "PHONE",
    "TEL": "PHONE",
    "FAX": "PHONE",
    # IDs
    "ID": "ID",
    "SSN": "ID",
    "NATIONAL_ID": "ID",
    "DRIVER_LICENSE": "ID",
    "PASSPORT": "PASSPORT",
    "MEDICAL_ID": "MEDICAL_ID",
    "UNIFIED_BUSINESS_NO": "UNIFIED_BUSINESS_NO",
    # Location/org
    "ADDRESS": "ADDRESS",
    "LOCATION": "ADDRESS",
    "LOC": "ADDRESS",
    "GPE": "ADDRESS",
    "CITY": "ADDRESS",
    "STATE": "ADDRESS",
    "ZIP": "ADDRESS",
    "POSTAL_CODE": "ADDRESS",
    "ORG": "ORGANIZATION",
    "ORGANIZATION": "ORGANIZATION",
    "COMPANY": "ORGANIZATION",
    "HOSPITAL": "ORGANIZATION",
    # Time-like
    "DATE": "DATE",
    "DOB": "DATE",
    "DATE_OF_BIRTH": "DATE",
    "BIRTHDATE": "DATE",
    "TIME": "TIME",
    "DATETIME": "TIME",
    # Network
    "IP": "IP_ADDRESS",
    "IP_ADDRESS": "IP_ADDRESS",
    "URL": "URL",
    "WEBSITE": "URL",
    # Financial
    "CREDIT_CARD": "CREDIT_CARD",
    "CREDITCARD": "CREDIT_CARD",
    "BANK_ACCOUNT": "BANK_ACCOUNT",
    "ACCOUNT_NUMBER": "BANK_ACCOUNT",
    "IBAN": "BANK_ACCOUNT",
    # Vehicles / misc
    "LICENSE_PLATE": "LICENSE_PLATE",
    "PLATE": "LICENSE_PLATE",
    "AGE": "AGE",
}


def normalize_raw_entity_type(raw: str) -> str:
    """Normalize a raw entity type token for downstream mapping."""

    value = (raw or "").strip().upper().replace("-", "_").replace(" ", "_")
    value = _TRAILING_INDEX_RE.sub("", value)
    value = _NON_ALPHA_RE.sub("", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value


def canonicalize_entity_type(
    raw: str | None,
    *,
    unknown_label: str = "PII",
) -> Optional[str]:
    """Map a raw entity type into a compact canonical set.

    This is used to keep the token-classification label space stable across datasets.
    """

    if raw is None:
        return None

    key = normalize_raw_entity_type(str(raw))
    if not key:
        return None

    if key in EXACT_TYPE_MAP:
        return EXACT_TYPE_MAP[key]

    # Heuristic fallbacks (handles long placeholder names like PATIENT_LAST_NAME).
    if "EMAIL" in key or "MAIL" in key:
        return "EMAIL"
    if "PHONE" in key or key.startswith("TEL") or "MOBILE" in key:
        return "PHONE"
    if "PASSPORT" in key:
        return "PASSPORT"
    if "MED" in key and "ID" in key:
        return "MEDICAL_ID"
    if key.endswith("_ID") or key.endswith("ID") or "SSN" in key:
        return "ID"
    if "NAME" in key and "USER" not in key:
        return "NAME"
    if "USER" in key or "HANDLE" in key:
        return "USERNAME"
    if "ADDRESS" in key or "CITY" in key or "STATE" in key or "ZIP" in key or "POSTAL" in key:
        return "ADDRESS"
    if "ORG" in key or "COMPANY" in key or "HOSPITAL" in key:
        return "ORGANIZATION"
    if "DATE" in key or "BIRTH" in key or key == "DOB":
        return "DATE"
    if "TIME" in key:
        return "TIME"
    if "IP" in key:
        return "IP_ADDRESS"
    if "URL" in key or "WEB" in key:
        return "URL"
    if "CREDIT" in key or "CARD" in key:
        return "CREDIT_CARD"
    if "BANK" in key or "ACCOUNT" in key or "IBAN" in key:
        return "BANK_ACCOUNT"
    if "PLATE" in key or "LICENSE" in key:
        return "LICENSE_PLATE"
    if "AGE" in key:
        return "AGE"

    unknown = normalize_raw_entity_type(unknown_label)
    return unknown or None


def canonicalize_entities(
    entities: Sequence[Entity] | Iterable[Entity],
    *,
    unknown_label: str = "PII",
) -> List[Entity]:
    """Canonicalize entity `type` fields and drop entities with invalid types."""

    out: List[Entity] = []
    for ent in entities:
        ent_type = canonicalize_entity_type(ent.get("type"), unknown_label=unknown_label)
        if not ent_type:
            continue
        normalized = dict(ent)
        normalized["type"] = str(ent_type)
        out.append(normalized)
    return out

