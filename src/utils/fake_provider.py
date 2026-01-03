from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Optional

from deid_pipeline.config import Config
from deid_pipeline.pii.utils import logger


class GPT2Provider:
    """Deprecated compatibility fake-data provider (legacy UI/scripts).

    Important:
    - This provider never downloads models.
    - If a local GPT-2 directory is not present, generation falls back to Faker.
    - Prefer using `deid_pipeline.pii.utils.fake_provider.FakeProvider` for new code.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        locale: str = "zh_TW",
        enable_gpt2: bool = True,
    ):
        self.locale = locale
        self.cache: Dict[str, str] = {}

        resolved = Path(model_path) if model_path is not None else Config.GPT2_MODEL_PATH
        self.model_path = resolved.expanduser().resolve()

        self._enabled = bool(enable_gpt2) and self.model_path.exists()
        self._tokenizer = None
        self._model = None
        self._faker = self._try_init_faker(locale)

        if self._enabled:
            self._try_load_model()

    @staticmethod
    def _try_init_faker(locale: str):
        try:
            from faker import Faker  # type: ignore

            return Faker(locale)
        except Exception as exc:
            logger.warning("faker is unavailable; falling back to placeholders: %s", exc)
            return None

    def _try_load_model(self) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

            self._tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path), local_files_only=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path), local_files_only=True
            )
            self._model.eval()
            logger.info("Loaded local GPT-2 model: %s", self.model_path)
        except Exception as exc:
            logger.warning(
                "Failed to load local GPT-2 model; disabling GPT-2 provider: %s", exc
            )
            self._enabled = False
            self._tokenizer = None
            self._model = None

    def generate(self, entity_type: str, original: str) -> str:
        """Return a fake value with per-process caching."""

        cache_key = f"{entity_type}:{original}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        value: str
        if self._enabled and self._model is not None and self._tokenizer is not None:
            try:
                value = self._gpt2_generate(entity_type, original)
            except Exception as exc:
                logger.warning("GPT-2 generation failed; falling back: %s", exc)
                value = self.fallback_fake(entity_type, original)
        else:
            value = self.fallback_fake(entity_type, original)

        self.cache[cache_key] = value
        return value

    def _gpt2_generate(self, entity_type: str, original: str) -> str:
        import torch  # type: ignore

        prompt = (
            f"Replace the following {entity_type} value with a fictional value in {self.locale}: "
            f"'{original}'.\nReplacement:"
        )
        inputs = self._tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = self._model.generate(
                inputs.input_ids,
                max_length=int(inputs.input_ids.shape[1]) + 20,
                num_return_sequences=1,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        generated = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "Replacement:" in generated:
            return generated.split("Replacement:", 1)[-1].strip()
        return generated.replace(prompt, "").strip()

    def fallback_fake(self, entity_type: str, original: str) -> str:
        """Faker-based (or placeholder) fake value generation."""

        faker = self._faker
        if faker is None:
            digest = hashlib.sha256(f"{entity_type}:{original}".encode("utf-8")).hexdigest()[:8]
            return f"<{entity_type}:{digest}>"

        if entity_type == "NAME":
            return faker.name()
        if entity_type in {"ID", "TW_ID"}:
            # Legacy behavior: plausible format but not checksum-valid.
            first_letter = chr(ord("A") + faker.random_int(0, 25))
            gender_code = str(faker.random_int(1, 2))
            seq_code = str(faker.random_int(0, 999999)).zfill(6)
            check_sum = faker.random_int(0, 9)
            return f"{first_letter}{gender_code}{seq_code}{check_sum}"
        if entity_type == "PHONE":
            return faker.phone_number()
        if entity_type == "EMAIL":
            return faker.email()
        if entity_type == "ADDRESS":
            return faker.address()

        return faker.text(max_nb_chars=12)


class FakerProvider:
    """Deprecated Faker-only provider (legacy UI/scripts).

    Prefer using `deid_pipeline.pii.utils.fake_provider.FakeProvider` for new code.
    """

    def __init__(self, locale: str = "zh_TW"):
        from faker import Faker  # type: ignore

        self.faker = Faker(locale)
        self.cache: Dict[str, str] = {}

    def _generate(self, entity_type: str) -> str:
        match entity_type:
            case "NAME":
                return self.faker.name()
            case "ID" | "TW_ID":
                return self.faker.ssn(min_age=18, max_age=60)
            case "PHONE":
                return self.faker.phone_number()
            case "EMAIL":
                return self.faker.email()
            case _:
                return self.faker.word()

    def fake(self, entity_type: str, original: str) -> str:
        cache_key = f"{entity_type}:{original}"
        if cache_key not in self.cache:
            self.cache[cache_key] = self._generate(entity_type)
        return self.cache[cache_key]

