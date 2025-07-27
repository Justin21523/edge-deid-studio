# src/deid_pipeline/pii/utils/fake_provider.py
import os
from typing import Dict
import torch
from faker import Faker
from transformers import AutoModelForCausalLM, AutoTokenizer
from ...config import Config
from . import logger

class FakeProvider:
    def __init__(self):
        self.config = Config()
        self.cache: Dict[str, str] = {}
        self.gpt2_provider = None
        self.faker = Faker(self.config.FAKER_LOCALE)

        # 嘗試初始化GPT-2
        self._init_gpt2()

    def _init_gpt2(self):
        """初始化GPT-2模型"""
        if not os.path.exists(self.config.GPT2_MODEL_PATH):
            logger.warning(f"GPT-2 model path not found: {self.config.GPT2_MODEL_PATH}")
            return
        try:
            logger.info("Loading GPT-2 model...")
            self.gpt2_tokenizer = AutoTokenizer.from_pretrained(self.config.GPT2_MODEL_PATH)
            self.gpt2_model = AutoModelForCausalLM.from_pretrained(self.config.GPT2_MODEL_PATH)
            self.gpt2_model.eval()
            logger.info("GPT-2 model loaded successfully")
            self.gpt2_provider = True
        except Exception as e:
            logger.error(f"Failed to load GPT-2: {e}")
            self.gpt2_provider = False

    def generate(self, entity_type: str, original: str) -> str:
        """生成假資料，保持一致性"""
        cache_key = f"{entity_type}:{original}"
        # 檢查快取
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # 根據類型生成假資料
        fake_value = self._generate_fake(entity_type, original)
        # enforce cache size
        if len(self.cache) >= self.config.FAKER_CACHE_SIZE:
            # pop 最舊一筆
            self.cache.pop(next(iter(self.cache)))
        # 存入快取
        self.cache[cache_key] = fake_value
        return fake_value

    def _generate_fake(self, entity_type: str, original: str) -> str:
        """實際生成假資料的邏輯"""
        # 嘗試使用GPT-2生成
        if self.gpt2_provider:
            try:
                return self._gpt2_generate(entity_type, original)
            except Exception as e:
                logger.error(f"GPT-2生成失敗，使用Faker回退: {str(e)}")

        # 使用Faker作為備用
        return self._faker_generate(entity_type)

    def _gpt2_generate(self, entity_type: str, original: str) -> str:
        """使用GPT-2生成假資料"""
        prompt = f"請將以下{entity_type}『{original}』替換為符合上下文的中文虛構內容:"

        # 編碼輸入
        inputs = self.gpt2_tokenizer(prompt, return_tensors="pt")

        # 生成文本
        with torch.no_grad():
            outputs = self.gpt2_model.generate(
                inputs.input_ids,
                max_length=len(inputs.input_ids[0]) + 20,  # 增加20個token
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.gpt2_tokenizer.eos_token_id
            )

        # 解碼輸出
        generated = self.gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 提取生成的假資料
        if ":" in generated:
            return generated.split(":")[-1].strip()
        elif "：" in generated:
            return generated.split("：")[-1].strip()

        # 回退處理
        return generated.replace(prompt, "").strip()

    def _faker_generate(self, entity_type: str) -> str:
        """使用Faker生成假資料"""
        if entity_type == "NAME":
            return self.faker.name()
        elif entity_type == "ADDRESS":
            return self.faker.address()
        elif entity_type == "PHONE":
            return self.faker.phone_number()
        elif entity_type == "EMAIL":
            return self.faker.email()
        elif entity_type == "TW_ID":
            # 台灣身分證格式
            first_letter = chr(ord('A') + self.faker.random_int(0, 25))
            gender_code = str(self.faker.random_int(1, 2))
            seq_code = str(self.faker.random_int(0, 999999)).zfill(6)
            check_sum = self.faker.random_int(0, 9)
            return f"{first_letter}{gender_code}{seq_code}{check_sum}"
        elif entity_type == "UNIFIED_BUSINESS_NO":
            # 台灣統一編號
            return str(self.faker.random_int(10000000, 99999999))
        elif entity_type == "PASSPORT":
            return self.faker.bothify(text="\?#??????")  # 例如 A1234567
        elif entity_type == "MEDICAL_ID":
            return f"M{self.faker.random_number(digits=7, fix_len=True)}"
        elif entity_type == "CONTRACT_NO":
            return f"CN-{self.faker.random_number(digits=6)}"
        elif entity_type == "ORGANIZATION":
            return self.faker.company()
        else:
            return self.faker.text(max_nb_chars=20)
