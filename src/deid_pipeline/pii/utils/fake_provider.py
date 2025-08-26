# src/deid_pipeline/pii/utils/fake_provider.py
"""
EdgeDeID Studio - 假資料生成提供器
使用 GPT-2 和 Faker 生成本地化假資料，支援中英文
"""

import re
import random
import logging
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from pathlib import Path

# 第三方庫
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
from faker import Faker
from faker.providers import BaseProvider

logger = logging.getLogger(__name__)

@dataclass
class GenerationConfig:
    """生成配置"""
    max_length: int = 50
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    do_sample: bool = True
    num_return_sequences: int = 3

class ChineseNameProvider(BaseProvider):
    """中文姓名提供器"""

    def __init__(self, generator):
        super().__init__(generator)
        # 常見中文姓氏
        self.surnames = [
            '王', '李', '張', '劉', '陳', '楊', '黃', '趙', '吳', '周',
            '徐', '孫', '馬', '朱', '胡', '林', '郭', '何', '高', '羅',
            '鄭', '梁', '謝', '宋', '唐', '許', '韓', '馮', '鄧', '曹'
        ]

        # 常見中文名字字符
        self.name_chars = [
            '偉', '芳', '娜', '敏', '靜', '麗', '強', '磊', '軍', '洋',
            '勇', '艷', '傑', '娟', '濤', '明', '超', '秀', '霞', '平',
            '剛', '桂', '英', '華', '玉', '萍', '紅', '娜', '輝', '鵬',
            '雲', '帆', '雪', '梅', '琳', '佳', '慧', '婷', '雯', '蘭'
        ]

    def chinese_name(self):
        """生成中文姓名"""
        surname = random.choice(self.surnames)
        name_length = random.choice([1, 2])  # 1-2個字的名字
        name = ''.join(random.choices(self.name_chars, k=name_length))
        return surname + name

class TaiwanAddressProvider(BaseProvider):
    """台灣地址提供器"""

    def __init__(self, generator):
        super().__init__(generator)
        self.cities = [
            '台北市', '新北市', '桃園市', '台中市', '台南市', '高雄市',
            '基隆市', '新竹市', '嘉義市', '新竹縣', '苗栗縣', '彰化縣',
            '南投縣', '雲林縣', '嘉義縣', '屏東縣', '宜蘭縣', '花蓮縣',
            '台東縣', '澎湖縣', '金門縣', '連江縣'
        ]

        self.districts = {
            '台北市': ['中正區', '大同區', '中山區', '松山區', '大安區', '萬華區', '信義區', '士林區', '北投區', '內湖區', '南港區', '文山區'],
            '新北市': ['板橋區', '三重區', '中和區', '永和區', '新莊區', '新店區', '樹林區', '鶯歌區', '三峽區', '淡水區'],
            '台中市': ['中區', '東區', '南區', '西區', '北區', '北屯區', '西屯區', '南屯區', '太平區', '大里區'],
            '高雄市': ['新興區', '前金區', '苓雅區', '鹽埕區', '鼓山區', '旗津區', '前鎮區', '三民區', '楠梓區', '小港區']
        }

        self.roads = ['中山路', '中正路', '民生路', '民權路', '復興路', '和平路', '信義路', '仁愛路', '忠孝路', '光復路']

    def taiwan_address(self):
        """生成台灣地址"""
        city = random.choice(self.cities)
        districts = self.districts.get(city, ['中正區', '信義區', '大安區'])
        district = random.choice(districts)
        road = random.choice(self.roads)
        number = random.randint(1, 999)
        return f"{city}{district}{road}{number}號"

class FakeProvider:
    """
    假資料生成提供器

    結合 GPT-2 語言模型和 Faker 庫，生成高品質的本地化假資料
    支援中英文，並維持與原始資料的格式一致性
    """

    def __init__(self,
                 gpt2_model_path: Optional[str] = None,
                 chinese_model_path: Optional[str] = None,
                 use_gpu: bool = False,
                 generation_config: Optional[GenerationConfig] = None):
        """
        初始化假資料提供器

        Args:
            gpt2_model_path: GPT-2 英文模型路徑
            chinese_model_path: GPT-2 中文模型路徑
            use_gpu: 是否使用 GPU
            generation_config: 生成配置
        """
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        self.generation_config = generation_config or GenerationConfig()

        # 初始化模型
        self.gpt2_model = None
        self.gpt2_tokenizer = None
        self.chinese_model = None
        self.chinese_tokenizer = None

        # 載入模型
        self._load_models(gpt2_model_path, chinese_model_path)

        # 初始化 Faker
        self.faker_en = Faker('en_US')
        self.faker_zh = Faker('zh_TW')

        # 添加自定義提供器
        self.faker_zh.add_provider(ChineseNameProvider)
        self.faker_zh.add_provider(TaiwanAddressProvider)

        # 生成快取
        self._generation_cache = {}

        # 預定義模板
        self._init_templates()

        logger.info(f"FakeProvider 初始化完成，使用設備: {self.device}")

    def _load_models(self, gpt2_path: Optional[str], chinese_path: Optional[str]):
        """載入 GPT-2 模型"""
        try:
            # 載入英文 GPT-2
            if gpt2_path and Path(gpt2_path).exists():
                self.gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_path).to(self.device)
                self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_path)
                self.gpt2_tokenizer.pad_token = self.gpt2_tokenizer.eos_token
                logger.info(f"已載入英文 GPT-2 模型: {gpt2_path}")
            else:
                # 使用預設模型
                self.gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2').to(self.device)
                self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
                self.gpt2_tokenizer.pad_token = self.gpt2_tokenizer.eos_token
                logger.info("已載入預設英文 GPT-2 模型")

            # 載入中文 GPT-2
            if chinese_path and Path(chinese_path).exists():
                self.chinese_model = GPT2LMHeadModel.from_pretrained(chinese_path).to(self.device)
                self.chinese_tokenizer = GPT2Tokenizer.from_pretrained(chinese_path)
                logger.info(f"已載入中文 GPT-2 模型: {chinese_path}")
            else:
                logger.warning("未找到中文 GPT-2 模型，將使用 Faker 生成中文資料")

        except Exception as e:
            logger.error(f"載入 GPT-2 模型失敗: {e}")
            self.gpt2_model = None
            self.gpt2_tokenizer = None

    def _init_templates(self):
        """初始化生成模板"""
        self.templates = {
            'person_name_en': [
                "My name is",
                "Hi, I'm",
                "This is",
                "Meet"
            ],
            'person_name_zh': [
                "我的名字是",
                "我叫",
                "這是",
                "他叫"
            ],
            'company_en': [
                "The company",
                "Our organization",
                "This business",
                "The firm"
            ],
            'company_zh': [
                "這家公司",
                "我們公司",
                "這個機構",
                "這家企業"
            ],
            'address_en': [
                "I live at",
                "The address is",
                "Located at",
                "Visit us at"
            ],
            'address_zh': [
                "地址在",
                "住在",
                "位於",
                "地點是"
            ]
        }

    def _detect_language(self, text: str) -> str:
        """偵測文字語言"""
        # 簡單的中文檢測
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        total_chars = len(re.findall(r'[a-zA-Z\u4e00-\u9fff]', text))

        if total_chars == 0:
            return 'en'

        chinese_ratio = chinese_chars / total_chars
        return 'zh' if chinese_ratio > 0.3 else 'en'

    def _generate_with_gpt2(self,
                           prompt: str,
                           language: str = 'en',
                           max_attempts: int = 3) -> Optional[str]:
        """使用 GPT-2 生成文字"""
        try:
            # 選擇模型
            if language == 'zh' and self.chinese_model is not None:
                model = self.chinese_model
                tokenizer = self.chinese_tokenizer
            elif self.gpt2_model is not None:
                model = self.gpt2_model
                tokenizer = self.gpt2_tokenizer
            else:
                return None

            # 編碼輸入
            inputs = tokenizer.encode(prompt, return_tensors='pt').to(self.device)

            # 生成文字
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=inputs.shape[1] + self.generation_config.max_length,
                    temperature=self.generation_config.temperature,
                    top_k=self.generation_config.top_k,
                    top_p=self.generation_config.top_p,
                    do_sample=self.generation_config.do_sample,
                    num_return_sequences=self.generation_config.num_return_sequences,
                    pad_token_id=tokenizer.eos_token_id,
                    attention_mask=torch.ones_like(inputs)
                )

            # 解碼結果
            results = []
            for output in outputs:
                generated_text = tokenizer.decode(output, skip_special_tokens=True)
                # 移除提示詞
                if generated_text.startswith(prompt):
                    generated_text = generated_text[len(prompt):].strip()
                if generated_text:
                    results.append(generated_text)

            # 選擇最佳結果
            if results:
                return random.choice(results)

        except Exception as e:
            logger.error(f"GPT-2 生成失敗: {e}")

        return None

    def generate_fake_name(self,
                          original: str,
                          preserve_format: bool = True) -> str:
        """生成假姓名"""
        language = self._detect_language(original)

        # 嘗試使用 GPT-2 生成
        if self.gpt2_model is not None:
            templates = self.templates.get(f'person_name_{language}', self.templates['person_name_en'])
            prompt = random.choice(templates) + " "

            generated = self._generate_with_gpt2(prompt, language)
            if generated:
                # 提取姓名部分
                name_match = re.search(r'^([A-Za-z\u4e00-\u9fff\s]+)', generated)
                if name_match:
                    fake_name = name_match.group(1).strip()
                    if len(fake_name) > 1:
                        return self._format_name(fake_name, original, preserve_format)

        # 降級到 Faker
        if language == 'zh':
            return self.faker_zh.chinese_name()
        else:
            return self.faker_en.name()

    def generate_fake_phone(self,
                           original: str,
                           preserve_format: bool = True) -> str:
        """生成假電話號碼"""
        # 保持原始格式
        if preserve_format:
            # 提取格式模式
            digit_pattern = re.sub(r'\d', 'X', original)

            # 生成新號碼
            fake_digits = ''.join([str(random.randint(0, 9)) for _ in range(original.count(digit) for digit in '0123456789')])

            result = original
            digit_index = 0
            for i, char in enumerate(original):
                if char.isdigit() and digit_index < len(fake_digits):
                    result = result[:i] + fake_digits[digit_index] + result[i+1:]
                    digit_index += 1

            return result
        else:
            # 台灣手機號碼格式
            if self._detect_language(original) == 'zh':
                return f"09{random.randint(10000000, 99999999)}"
            else:
                return self.faker_en.phone_number()

    def generate_fake_email(self,
                           original: str,
                           preserve_format: bool = True) -> str:
        """生成假電子郵件"""
        if preserve_format:
            # 保持域名，只替換用戶名
            parts = original.split('@')
            if len(parts) == 2:
                username, domain = parts

                # 生成假用戶名
                if self._detect_language(username) == 'zh':
                    fake_username = self.faker_zh.user_name()
                else:
                    fake_username = self.faker_en.user_name()

                return f"{fake_username}@{domain}"

        # 完全生成新郵件
        language = self._detect_language(original)
        if language == 'zh':
            return self.faker_zh.email()
        else:
            return self.faker_en.email()

    def generate_fake_id_card(self,
                             original: str,
                             preserve_format: bool = True) -> str:
        """生成假身分證號"""
        # 台灣身分證格式: A123456789
        if re.match(r'^[A-Z]\d{9}, original'):
            letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            fake_letter = random.choice(letters)
            fake_digits = ''.join([str(random.randint(0, 9)) for _ in range(9)])
            return fake_letter + fake_digits

        # 其他格式保持長度
        if preserve_format:
            result = ''
            for char in original:
                if char.isalpha():
                    result += random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
                elif char.isdigit():
                    result += str(random.randint(0, 9))
                else:
                    result += char
            return result
        else:
            return self.faker_en.ssn()

    def generate_fake_address(self,
                             original: str,
                             preserve_format: bool = True) -> str:
        """生成假地址"""
        language = self._detect_language(original)

        # 嘗試使用 GPT-2 生成
        if self.gpt2_model is not None:
            templates = self.templates.get(f'address_{language}', self.templates['address_en'])
            prompt = random.choice(templates) + " "

            generated = self._generate_with_gpt2(prompt, language)
            if generated:
                # 提取地址部分
                address_lines = generated.split('\n')[0].strip()
                if len(address_lines) > 5:
                    return address_lines

        # 降級到 Faker
        if language == 'zh':
            return self.faker_zh.taiwan_address()
        else:
            return self.faker_en.address().replace('\n', ' ')

    def generate_fake_company(self,
                             original: str,
                             preserve_format: bool = True) -> str:
        """生成假公司名"""
        language = self._detect_language(original)

        # 嘗試使用 GPT-2 生成
        if self.gpt2_model is not None:
            templates = self.templates.get(f'company_{language}', self.templates['company_en'])
            prompt = random.choice(templates) + " "

            generated = self._generate_with_gpt2(prompt, language)
            if generated:
                # 提取公司名部分
                company_match = re.search(r'^([A-Za-z\u4e00-\u9fff\s&.,\-]+)', generated)
                if company_match:
                    company_name = company_match.group(1).strip()
                    if len(company_name) > 2:
                        return company_name

        # 降級到 Faker
        if language == 'zh':
            return self.faker_zh.company()
        else:
            return self.faker_en.company()

    def generate_fake_date(self,
                          original: str,
                          preserve_format: bool = True) -> str:
        """生成假日期"""
        # 嘗試解析原始日期格式
        date_patterns = [
            (r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})', '%Y-%m-%d'),
            (r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})', '%m-%d-%Y'),
            (r'(\d{4})年(\d{1,2})月(\d{1,2})日', '%Y年%m月%d日'),
        ]

        if preserve_format:
            for pattern, format_str in date_patterns:
                match = re.match(pattern, original)
                if match:
                    # 生成隨機日期
                    fake_date = self.faker_en.date_between(start_date='-10y', end_date='today')

                    # 根據原始格式返回
                    if '年' in original:
                        return fake_date.strftime('%Y年%m月%d日')
                    elif original.count('/') > 0:
                        return fake_date.strftime('%Y/%m/%d')
                    else:
                        return fake_date.strftime('%Y-%m-%d')

        # 預設格式
        return self.faker_en.date()

    def generate_fake_text(self,
                          original: str,
                          preserve_format: bool = True) -> str:
        """生成通用假文字"""
        language = self._detect_language(original)

        # 使用 GPT-2 生成相似長度的文字
        if self.gpt2_model is not None:
            # 使用原文前幾個字作為提示
            prompt_length = min(10, len(original) // 2)
            prompt = original[:prompt_length]

            generated = self._generate_with_gpt2(prompt, language)
            if generated:
                # 截取相似長度
                target_length = len(original)
                if len(generated) > target_length:
                    return generated[:target_length]
                else:
                    return generated

        # 降級到隨機文字
        if preserve_format:
            return self._generate_similar_text(original)
        else:
            return self.faker_en.text(max_nb_chars=len(original))

    def _format_name(self, fake_name: str, original: str, preserve_format: bool) -> str:
        """格式化姓名"""
        if not preserve_format:
            return fake_name

        # 保持大小寫格式
        if original.isupper():
            return fake_name.upper()
        elif original.islower():
            return fake_name.lower()
        elif original.istitle():
            return fake_name.title()

        return fake_name

    def _generate_similar_text(self, original: str) -> str:
        """生成相似文字"""
        result = []
        for char in original:
            if char.isalpha():
                if char.isupper():
                    result.append(random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
                else:
                    result.append(random.choice('abcdefghijklmnopqrstuvwxyz'))
            elif char.isdigit():
                result.append(str(random.randint(0, 9)))
            else:
                result.append(char)

        return ''.join(result)

    def clear_cache(self):
        """清空生成快取"""
        self._generation_cache.clear()
        logger.info("假資料生成快取已清空")

    def get_cache_stats(self) -> Dict[str, int]:
        """獲取快取統計"""
        return {
            'cache_size': len(self._generation_cache),
            'cache_hits': getattr(self, '_cache_hits', 0),
            'cache_misses': getattr(self, '_cache_misses', 0)
        }
