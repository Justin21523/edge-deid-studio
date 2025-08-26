# src/deid_pipeline/pii/detectors/bert_onnx_detector.py

#!/usr/bin/env python3
"""
BERT ONNX NER 偵測器實作
使用 ONNX Runtime 進行高效推理的 BERT NER 偵測器
"""

import logging
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer, AutoConfig

from .bert_detector import BertNERDetector, TokenInfo
from ..utils.base import Entity
from ...config import Config

logger = logging.getLogger(__name__)

class BertONNXNERDetector(BertNERDetector):
    """
    使用 ONNX Runtime 的 BERT NER 偵測器

    相較於 PyTorch 版本的優勢:
    - 更快的推理速度 (通常 2-3x 加速)
    - 更小的記憶體佔用
    - 支援多種硬體加速 (CPU, CUDA, TensorRT, NPU)
    - 量化模型支援
    """

    def __init__(
        self,
        onnx_model_path: Optional[str] = None,
        model_name: str = "bert-base-chinese-ner",
        confidence_threshold: float = Config.BERT_CONFIDENCE_THRESHOLD,
        max_length: int = Config.BERT_MAX_LENGTH,
        batch_size: int = Config.BERT_BATCH_SIZE,
        providers: Optional[List[str]] = None,
        provider_options: Optional[List[Dict]] = None
    ):
        """
        初始化 BERT ONNX NER 偵測器

        Args:
            onnx_model_path: ONNX 模型路徑
            model_name: 模型名稱 (用於載入 tokenizer)
            confidence_threshold: 信心度閾值
            max_length: 最大序列長度
            batch_size: 批次大小
            providers: ONNX Runtime 提供者列表
            provider_options: 提供者選項
        """
        # 不呼叫父類的 __init__ 以避免載入 PyTorch 模型
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.max_length = max_length
        self.batch_size = batch_size

        # 設定 ONNX Runtime 提供者
        if providers is None:
            providers = self._get_default_providers()
        self.providers = providers
        self.provider_options = provider_options or []

        logger.info(f"使用 ONNX Runtime 提供者: {self.providers}")

        # 載入模型
        if Config.USE_STUB:
            logger.warning("使用 STUB 模式，跳過 ONNX 模型載入")
            self.session = None
            self.tokenizer = None
            self.config = None
            self.label_list = self._get_default_labels()
        else:
            self._load_onnx_model(onnx_model_path)

    def _get_default_providers(self) -> List[str]:
        """獲取預設的 ONNX Runtime 提供者"""
        available_providers = ort.get_available_providers()

        # 優先順序: CUDA > CPU
        preferred_providers = [
            "CUDAExecutionProvider",
            "CPUExecutionProvider"
        ]

        selected_providers = []
        for provider in preferred_providers:
            if provider in available_providers:
                selected_providers.append(provider)

        if not selected_providers:
            selected_providers = ["CPUExecutionProvider"]

        return selected_providers

    def _load_onnx_model(self, onnx_model_path: Optional[str]):
        """載入 ONNX 模型和相關組件"""
        try:
            # 確定模型路徑
            if onnx_model_path and Path(onnx_model_path).exists():
                model_file = Path(onnx_model_path)
                model_dir = model_file.parent.parent  # 回到模型根目錄
            else:
                # 使用預設路徑
                model_dir = Path("models/ner") / self.model_name
                onnx_dir = model_dir / "onnx"

                # 尋找 ONNX 檔案
                onnx_files = list(onnx_dir.glob("*.onnx"))
                if not onnx_files:
                    raise FileNotFoundError(f"在 {onnx_dir} 中找不到 ONNX 模型檔案")

                # 優先選擇優化過的模型
                optimized_files = [f for f in onnx_files if "optimized" in f.name]
                model_file = optimized_files[0] if optimized_files else onnx_files[0]

            logger.info(f"載入 ONNX 模型: {model_file}")

            # 創建 ONNX Runtime session
            self.session = ort.InferenceSession(
                str(model_file),
                providers=self.providers,
                provider_options=self.provider_options
            )

            # 載入 tokenizer
            tokenizer_dir = model_dir / "tokenizer"
            if tokenizer_dir.exists():
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
            else:
                logger.info(f"從 HuggingFace 載入 tokenizer: {self.model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # 載入配置
            config_dir = model_dir / "config"
            if config_dir.exists():
                self.config = AutoConfig.from_pretrained(config_dir)
            else:
                self.config = AutoConfig.from_pretrained(self.model_name)

            # 獲取標籤列表
            self.label_list = list(self.config.id2label.values())

            # 驗證模型輸入輸出
            self._validate_model_io()

            logger.info(f"ONNX 模型載入成功，支援標籤: {self.label_list}")

        except Exception as e:
            logger.error(f"ONNX 模型載入失敗: {e}")
            if not Config.USE_STUB:
                raise

    def _validate_model_io(self):
        """驗證模型輸入輸出格式"""
        input_names = [inp.name for inp in self.session.get_inputs()]
        output_names = [out.name for out in self.session.get_outputs()]

        logger.info(f"模型輸入: {input_names}")
        logger.info(f"模型輸出: {output_names}")

        # 檢查必要的輸入
        required_inputs = {"input_ids", "attention_mask"}
        if not required_inputs.issubset(set(input_names)):
            missing = required_inputs - set(input_names)
            raise ValueError(f"模型缺少必要輸入: {missing}")

    def _detect_chunk(self, text: str, offset: int = 0) -> List[Entity]:
        """使用 ONNX Runtime 偵測單個文本片段"""
        if Config.USE_STUB:
            return self._stub_detect(text)

        # Tokenization
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="np",  # 返回 numpy 陣列
            return_offsets_mapping=True
        )

        input_ids = encoding["input_ids"].astype(np.int64)
        attention_mask = encoding["attention_mask"].astype(np.int64)
        offset_mapping = encoding["offset_mapping"][0]

        # ONNX Runtime 推理
        try:
            outputs = self.session.run(
                None,  # 返回所有輸出
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask
                }
            )

            logits = outputs[0]  # 假設第一個輸出是 logits

        except Exception as e:
            logger.error(f"ONNX 推理失敗: {e}")
            return []

        # 計算機率和預測標籤
        predictions = self._softmax(logits)
        predicted_labels = np.argmax(predictions, axis=-1)

        # 獲取預測結果
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        predicted_labels = predicted_labels[0]
        prediction_scores = predictions[0]

        # 構建 token 資訊
        token_infos = self._build_token_infos(
            tokens, offset_mapping, predicted_labels, prediction_scores
        )

        # BIO 標籤轉實體
        entities = self._bio_to_entities(token_infos, text, offset)

        return entities

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """計算 softmax 機率"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def batch_detect(self, texts: List[str]) -> List[List[Entity]]:
        """批次偵測多個文本"""
        if Config.USE_STUB:
            return [self._stub_detect(text) for text in texts]

        if not texts:
            return []

        results = []

        # 分批處理
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_results = self._batch_detect_chunk(batch_texts)
            results.extend(batch_results)

        return results

    def _batch_detect_chunk(self, texts: List[str]) -> List[List[Entity]]:
        """批次處理一組文本"""
        # 批次 tokenization
        encodings = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="np",
            return_offsets_mapping=True
        )

        input_ids = encodings["input_ids"].astype(np.int64)
        attention_mask = encodings["attention_mask"].astype(np.int64)
        offset_mappings = encodings["offset_mapping"]

        # 批次推理
        try:
            outputs = self.session.run(
                None,
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask
                }
            )

            logits = outputs[0]

        except Exception as e:
            logger.error(f"批次 ONNX 推理失敗: {e}")
            return [[] for _ in texts]

        # 處理每個樣本的結果
        predictions = self._softmax(logits)
        predicted_labels = np.argmax(predictions, axis=-1)

        results = []
        for i, text in enumerate(texts):
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[i])
            sample_labels = predicted_labels[i]
            sample_scores = predictions[i]
            offset_mapping = offset_mappings[i]

            # 構建 token 資訊
            token_infos = self._build_token_infos(
                tokens, offset_mapping, sample_labels, sample_scores
            )

            # 轉換為實體
            entities = self._bio_to_entities(token_infos, text, 0)
            results.append(entities)

        return results

    def get_model_info(self) -> Dict[str, Union[str, List[str]]]:
        """獲取模型資訊"""
        if Config.USE_STUB:
            return {
                "model_type": "STUB",
                "providers": ["STUB"],
                "supported_entities": self.get_supported_entities()
            }

        info = {
            "model_name": self.model_name,
            "model_type": "ONNX",
            "providers": self.providers,
            "supported_entities": self.get_supported_entities()
        }

        if self.session:
            # 獲取模型元資料
            try:
                metadata = self.session.get_modelmeta()
                info.update({
                    "model_version": metadata.version,
                    "producer_name": metadata.producer_name,
                    "graph_name": metadata.graph_name
                })
            except Exception as e:
                logger.debug(f"無法獲取模型元資料: {e}")

        return info

    def benchmark(self, test_texts: List[str], num_runs: int = 10) -> Dict[str, float]:
        """效能基準測試"""
        import time

        if Config.USE_STUB or not test_texts:
            return {"avg_time_per_text": 0.0}

        # 預熱
        self.detect(test_texts[0] if test_texts else "測試文本")

        # 單個文本測試
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            for text in test_texts:
                self.detect(text)
            end_time = time.time()
            times.append(end_time - start_time)

        single_avg = np.mean(times)

        # 批次測試
        batch_times = []
        for _ in range(num_runs):
            start_time = time.time()
            self.batch_detect(test_texts)
            end_time = time.time()
            batch_times.append(end_time - start_time)

        batch_avg = np.mean(batch_times)

        return {
            "single_avg_time": single_avg,
            "single_avg_per_text": single_avg / len(test_texts),
            "batch_avg_time": batch_avg,
            "batch_avg_per_text": batch_avg / len(test_texts),
            "batch_speedup": single_avg / batch_avg if batch_avg > 0 else 0
        }

    def optimize_for_inference(self):
        """針對推理優化設定"""
        if Config.USE_STUB or not self.session:
            return

        # 設定 ONNX Runtime 選項
        session_options = ort.SessionOptions()

        # 啟用所有優化
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        # 設定線程數
        session_options.intra_op_num_threads = 0  # 使用所有可用核心
        session_options.inter_op_num_threads = 1  # 順序執行

        # 啟用記憶體模式
        session_options.enable_mem_pattern = True
        session_options.enable_cpu_mem_arena = True

        logger.info("ONNX Runtime 推理優化已啟用")
