from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple

from ..core.contracts import Entity


def _random_tw_id(rng: random.Random) -> str:
    letter = rng.choice("ABCDEFGHJKLMNPQRSTUVXYWZ")
    digits = "".join(str(rng.randint(0, 9)) for _ in range(9))
    return f"{letter}{digits}"


def _random_tw_phone(rng: random.Random) -> str:
    digits = "".join(str(rng.randint(0, 9)) for _ in range(8))
    return f"09{digits}"


def _random_email(rng: random.Random, idx: int) -> str:
    user = f"user{idx}{rng.randint(0, 9999):04d}"
    return f"{user}@example.com"


def generate_synthetic_span_examples(
    *,
    num_examples: int = 100,
    seed: int = 0,
    language: str = "zh",
) -> List[Dict[str, Any]]:
    """Generate synthetic NER examples with gold spans (offline, deterministic).

    The output schema is designed for CI:
    - `text`: full string
    - `entities`: list of entities with `type`, `span`, and `text`
    """

    rng = random.Random(int(seed))

    if language == "zh":
        names = ["\u738b\u5c0f\u660e", "\u9673\u6021\u541b", "\u6797\u5fd7\u660e", "\u5f35\u96c5\u5a77"]
        addresses = [
            "\u53f0\u5317\u5e02\u4fe1\u7fa9\u8def1\u865f",
            "\u65b0\u5317\u5e02\u4e2d\u5c71\u8def10\u865f",
            "\u53f0\u4e2d\u5e02\u6c11\u751f\u8def99\u865f",
        ]
        templates: List[Tuple[str, str]] = [
            ("\u75c5\u60a3\u59d3\u540d\uff1a", "\u3002\n"),
            ("\u806f\u7d61\u65b9\u5f0f\uff1a", "\u3002\n"),
            ("\u8eab\u5206\u8b49\uff1a", "\u3002\n"),
        ]
    else:
        names = ["John Smith", "Alice Chen", "Michael Brown", "Emily Davis"]
        addresses = ["1 Main St", "10 Broadway Ave", "99 Market Rd"]
        templates = [
            ("Patient name: ", ".\n"),
            ("Contact: ", ".\n"),
            ("ID: ", ".\n"),
        ]

    examples: List[Dict[str, Any]] = []

    for idx in range(int(num_examples)):
        parts: List[str] = []
        entities: List[Entity] = []
        cursor = 0

        def add_text(s: str) -> None:
            nonlocal cursor
            parts.append(s)
            cursor += len(s)

        def add_entity(entity_type: str, value: str) -> None:
            nonlocal cursor
            start = cursor
            add_text(value)
            end = cursor
            entities.append(
                {
                    "type": entity_type,
                    "span": (start, end),
                    "text": value,
                    "confidence": 1.0,
                    "score": 1.0,
                    "source": "synthetic",
                    "language": language,
                }
            )

        name = rng.choice(names)
        phone = _random_tw_phone(rng) if language == "zh" else f"+1-555-{rng.randint(100, 999)}-{rng.randint(1000, 9999)}"
        email = _random_email(rng, idx)
        address = rng.choice(addresses)
        ident = _random_tw_id(rng) if language == "zh" else f"{rng.randint(100, 999)}-{rng.randint(10, 99)}-{rng.randint(1000, 9999)}"

        # Name
        add_text(templates[0][0])
        add_entity("NAME", name)
        add_text(templates[0][1])

        # Contact (phone + email)
        add_text(templates[1][0])
        add_entity("PHONE", phone)
        add_text(", ")
        add_entity("EMAIL", email)
        add_text(templates[1][1])

        # Address
        add_text("Address: " if language != "zh" else "\u5730\u5740\uff1a")
        add_entity("ADDRESS", address)
        add_text(".\n")

        # ID
        add_text(templates[2][0])
        add_entity("ID", ident)
        add_text(templates[2][1])

        text = "".join(parts)
        examples.append({"text": text, "entities": entities})

    return examples
