from faker import Faker
import os, logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ..base import Entity
from ..config import Config

logger = logging.getLogger(__name__)

class GPT2Provider:
    def __init__(self, model_path=None):
        self.cache = {}
        self.model = None
        self.tokenizer = None

        if model_path is None:
            model_path = Config.GPT2_MODEL_PATH

        if os.path.exists(model_path):
            self.load_model(model_path)

    def load_model(self, model_path):
        """載入 GPT-2 模型"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
            self.model.eval()
            logger.info(f"已載入 GPT-2 模型: {model_path}")
        except Exception as e:
            logger.error(f"載入 GPT-2 模型失敗: {e}")
            self.model = None

    def generate(self, entity_type: str, original: str) -> str:
        """生成假資料"""
        cache_key = f"{entity_type}:{original}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        if self.model is None:
            return self.fallback_fake(entity_type)

        prompt = f"請把以下敏感資訊「{original}」用中文{entity_type}替換："
        try:
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
            outputs = self.model.generate(
                input_ids,
                max_length=len(input_ids[0]) + 20,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            fake_text = generated.split("：")[-1].strip()
            self.cache[cache_key] = fake_text
            return fake_text
        except Exception as e:
            logger.error(f"GPT-2 生成失敗: {e}")
            return self.fallback_fake(entity_type)

    def fallback_fake(self, entity_type: str) -> str:
        from faker import Faker
        fake = Faker("zh_TW")
        if entity_type == "NAME":
            return fake.name()
        elif entity_type == "ADDRESS":
            return fake.address()
        elif entity_type == "PHONE":
            return fake.phone_number()
        elif entity_type == "EMAIL":
            return fake.email()
        else:
            return fake.text(max_nb_chars=10)

class FakerProvider:
    def __init__(self):
        self.faker = Faker("zh_TW")
        self.cache = {}

    def _generate(self, etype):
        match etype:
            case "NAME":   return self.faker.name()
            case "ID":     return self.faker.ssn(min_age=18, max_age=60)
            case "PHONE":  return self.faker.phone_number()
            case "EMAIL":  return self.faker.email()
            case _ :       return self.faker.word()

    def fake(self, etype:str, original:str):
        key = f"{etype}:{original}"
        if key not in self.cache:
            self.cache[key] = self._generate(etype)
        return self.cache[key]
