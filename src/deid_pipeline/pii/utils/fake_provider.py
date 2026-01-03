from __future__ import annotations

import hashlib
import random
from typing import Dict, Optional

from ...config import Config
from . import logger


class FakeProvider:
    """Generate replacement values for detected PII.

    This is a legacy implementation that will be replaced by a deterministic,
    locale-aware provider with a global cache in a later milestone.

    Design constraints:
    - Avoid importing optional dependencies (faker/transformers/torch) at import-time.
    - Never perform network calls at runtime; generation must be local-only.
    """

    def __init__(self):
        self.config = Config()
        self.cache: Dict[str, str] = {}

        self._faker = self._try_init_faker()
        self._gpt2_enabled = (
            getattr(self.config, "USE_GPT2_FAKE_PROVIDER", False)
            and self.config.GPT2_MODEL_PATH.exists()
        )
        self._gpt2_tokenizer = None
        self._gpt2_model = None

        if self._gpt2_enabled:
            self._try_init_gpt2()

    def generate(self, entity_type: str, original: str) -> str:
        """Generate a replacement value with per-process caching."""

        cache_key = f"{entity_type}:{original}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        value = self._generate_impl(entity_type, original)

        if len(self.cache) >= self.config.FAKER_CACHE_SIZE:
            self.cache.pop(next(iter(self.cache)))
        self.cache[cache_key] = value
        return value

    def generate_deterministic(self, entity_type: str, original: str, *, context_hash: str) -> str:
        """Generate a deterministic replacement value for a given context.

        The output must be stable across runs for the same input triple:
        (entity_type, original, context_hash).
        """

        cache_key = f"{entity_type}:{original}:{context_hash}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        value = self._generate_deterministic_impl(entity_type, original, context_hash=context_hash)

        if len(self.cache) >= self.config.FAKER_CACHE_SIZE:
            self.cache.pop(next(iter(self.cache)))
        self.cache[cache_key] = value
        return value

    def _generate_impl(self, entity_type: str, original: str) -> str:
        if self._gpt2_enabled and self._gpt2_model is not None and self._gpt2_tokenizer is not None:
            try:
                return self._gpt2_generate(entity_type, original)
            except Exception as exc:
                logger.warning("GPT-2 generation failed; falling back: %s", exc)

        if self._faker is not None:
            return self._faker_generate(entity_type)

        # Last-resort deterministic placeholder (keeps tests runnable without faker).
        digest = hashlib.sha256(f"{entity_type}:{original}".encode("utf-8")).hexdigest()[:8]
        return f"<{entity_type}:{digest}>"

    def _generate_deterministic_impl(self, entity_type: str, original: str, *, context_hash: str) -> str:
        stable_key = f"{entity_type}:{original}:{context_hash}"
        seed = int(hashlib.sha256(stable_key.encode("utf-8")).hexdigest()[:8], 16)

        if self._gpt2_enabled and self._gpt2_model is not None and self._gpt2_tokenizer is not None:
            try:
                return self._gpt2_generate(entity_type, original)
            except Exception as exc:
                logger.warning("GPT-2 generation failed; falling back: %s", exc)

        if self._faker is not None:
            try:
                self._faker.seed_instance(seed)
            except Exception:
                # If the faker version does not support seed_instance, fall back to placeholder.
                return self._placeholder(entity_type, stable_key)

            return self._faker_generate(entity_type)

        return self._fallback_generate(entity_type, stable_key, seed=seed)

    def _fallback_generate(self, entity_type: str, stable_key: str, *, seed: int) -> str:
        """Deterministic fallback when Faker is unavailable (offline/local-only)."""

        rng = random.Random(int(seed))
        locale = str(getattr(self.config, "FAKER_LOCALE", "en_US") or "en_US").lower()
        is_tw = locale.replace("-", "_") in {"zh_tw", "zh_hant_tw"} or "tw" in locale

        if entity_type in {"ID", "TW_ID"}:
            if is_tw:
                letter = rng.choice("ABCDEFGHJKLMNPQRSTUVXYWZ")
                gender = rng.choice(["1", "2"])
                mid = "".join(str(rng.randint(0, 9)) for _ in range(7))
                checksum = str(rng.randint(0, 9))
                return f"{letter}{gender}{mid}{checksum}"

            return f"{rng.randint(0, 999):03d}-{rng.randint(0, 99):02d}-{rng.randint(0, 9999):04d}"

        if entity_type == "PHONE":
            if is_tw:
                return "09" + "".join(str(rng.randint(0, 9)) for _ in range(8))
            return f"555-{rng.randint(100, 999):03d}-{rng.randint(0, 9999):04d}"

        if entity_type == "EMAIL":
            return f"user{rng.randint(0, 999999):06d}@example.com"

        if entity_type == "UNIFIED_BUSINESS_NO":
            return f"{rng.randint(0, 99999999):08d}"

        if entity_type == "PASSPORT":
            prefix = rng.choice(["P", "PA", "PB"])
            digits = "".join(str(rng.randint(0, 9)) for _ in range(7))
            return f"{prefix}{digits}"

        if entity_type == "MEDICAL_ID":
            digits = "".join(str(rng.randint(0, 9)) for _ in range(7))
            return f"M{digits}"

        if entity_type == "CONTRACT_NO":
            return f"CN-{rng.randint(0, 999999):06d}"

        if entity_type == "ORGANIZATION":
            return f"Example Organization {rng.randint(1, 9999)}"

        if entity_type == "NAME":
            if is_tw:
                names = [
                    "\u738b\u5c0f\u660e",
                    "\u9673\u6021\u541b",
                    "\u6797\u5fd7\u660e",
                    "\u5f35\u96c5\u5a77",
                ]
                return rng.choice(names)
            return rng.choice(["John Smith", "Alice Chen", "Michael Brown", "Emily Davis"])

        if entity_type == "ADDRESS":
            if is_tw:
                addresses = [
                    "\u53f0\u5317\u5e02\u4fe1\u7fa9\u8def1\u865f",
                    "\u65b0\u5317\u5e02\u4e2d\u5c71\u8def10\u865f",
                    "\u53f0\u4e2d\u5e02\u6c11\u751f\u8def99\u865f",
                ]
                return rng.choice(addresses)
            return f"{rng.randint(1, 999)} Main Street"

        return self._placeholder(entity_type, stable_key)

    @staticmethod
    def _placeholder(entity_type: str, stable_key: str) -> str:
        digest = hashlib.sha256(stable_key.encode("utf-8")).hexdigest()[:8]
        return f"<{entity_type}:{digest}>"

    def _try_init_faker(self):
        try:
            from faker import Faker  # type: ignore

            return Faker(self.config.FAKER_LOCALE)
        except Exception as exc:
            logger.warning("faker is unavailable; using placeholder replacements: %s", exc)
            return None

    def _try_init_gpt2(self) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

            logger.info("Loading local GPT-2 model from %s", self.config.GPT2_MODEL_PATH)
            self._gpt2_tokenizer = AutoTokenizer.from_pretrained(
                str(self.config.GPT2_MODEL_PATH), local_files_only=True
            )
            self._gpt2_model = AutoModelForCausalLM.from_pretrained(
                str(self.config.GPT2_MODEL_PATH), local_files_only=True
            )
            self._gpt2_model.eval()
        except Exception as exc:
            logger.warning("Failed to load local GPT-2; disabling GPT-2 provider: %s", exc)
            self._gpt2_enabled = False
            self._gpt2_tokenizer = None
            self._gpt2_model = None

    def _gpt2_generate(self, entity_type: str, original: str) -> str:
        import torch  # type: ignore

        prompt = (
            f"Replace the following {entity_type} value with a fictional value that fits the context: "
            f"'{original}'.\nReplacement:"
        )
        inputs = self._gpt2_tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = self._gpt2_model.generate(
                inputs.input_ids,
                max_length=int(inputs.input_ids.shape[1]) + 20,
                num_return_sequences=1,
                do_sample=False,
                pad_token_id=self._gpt2_tokenizer.eos_token_id,
            )

        generated = self._gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "Replacement:" in generated:
            return generated.split("Replacement:", 1)[-1].strip()
        return generated.replace(prompt, "").strip()

    def _faker_generate(self, entity_type: str) -> str:
        faker = self._faker
        if faker is None:
            raise RuntimeError("faker is not initialized.")

        locale = str(getattr(self.config, "FAKER_LOCALE", "en_US") or "en_US").lower()
        is_tw = locale.replace("-", "_") in {"zh_tw", "zh_hant_tw"} or "tw" in locale

        if entity_type == "NAME":
            return faker.name()
        if entity_type == "ADDRESS":
            return faker.address()
        if entity_type == "PHONE":
            if is_tw:
                # Match the default Taiwan mobile regex: 09xx-xxx-xxx (dashes optional).
                digits = "".join(str(faker.random_int(0, 9)) for _ in range(8))
                return f"09{digits}"
            return f"555-{faker.random_int(100, 999):03d}-{faker.random_int(0, 9999):04d}"
        if entity_type == "EMAIL":
            return faker.email()
        if entity_type in {"ID", "TW_ID"}:
            if is_tw:
                first_letter = chr(ord("A") + faker.random_int(0, 25))
                gender_code = str(faker.random_int(1, 2))
                seq_code = str(faker.random_int(0, 9_999_999)).zfill(7)
                check_sum = str(faker.random_int(0, 9))
                return f"{first_letter}{gender_code}{seq_code}{check_sum}"

            # SSN-like format for English datasets: 123-45-6789
            return f"{faker.random_int(0, 999):03d}-{faker.random_int(0, 99):02d}-{faker.random_int(0, 9999):04d}"
        if entity_type == "UNIFIED_BUSINESS_NO":
            return f"{faker.random_int(0, 99_999_999):08d}"
        if entity_type == "PASSPORT":
            return faker.bothify(text="??#######") if faker.random_int(0, 1) else faker.bothify(text="?#######")
        if entity_type == "MEDICAL_ID":
            return f"M{faker.random_number(digits=7, fix_len=True)}"
        if entity_type == "CONTRACT_NO":
            return f"CN-{faker.random_number(digits=6)}"
        if entity_type == "ORGANIZATION":
            return faker.company()
        return faker.text(max_nb_chars=20)
