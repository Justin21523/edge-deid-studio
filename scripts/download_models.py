# scripts/download_models.py

"""
模型下載和 ONNX 轉換腳本
支援從 Hugging Face 下載 BERT NER 模型並轉換為 ONNX 格式
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import requests
from tqdm import tqdm
import torch
import onnx
import onnxruntime as ort
from onnxruntime.tools import optimizer
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoConfig,
    GPT2LMHeadModel,
    GPT2Tokenizer
)
from huggingface_hub import hf_hub_download
from torch.onnx import export as torch_onnx_export


# 設定日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelDownloader:
    """模型下載和轉換管理器"""

    def __init__(self, base_model_dir: str = "models"):
        self.base_model_dir = Path(base_model_dir)
        self.base_model_dir.mkdir(exist_ok=True)

        # 預設模型配置
        self.models_config = {
            "ner": {
                "model_name": "dbmdz/bert-large-cased-finetuned-conll03-english",
                "local_path": self.base_model_dir / "ner",
                "files": ["config.json", "pytorch_model.bin", "tokenizer_config.json", "vocab.txt"]
            },
            "ner_chinese": {
                "model_name": "ckiplab/bert-base-chinese-ner",
                "local_path": self.base_model_dir / "ner_chinese",
                "files": ["config.json", "pytorch_model.bin", "tokenizer_config.json", "vocab.txt"]
            },
            "gpt2": {
                "model_name": "gpt2",
                "local_path": self.base_model_dir / "gpt2",
                "files": ["config.json", "pytorch_model.bin", "tokenizer_config.json", "vocab.json", "merges.txt"]
            },
            "gpt2_chinese": {
                "model_name": "uer/gpt2-chinese-cluecorpussmall",
                "local_path": self.base_model_dir / "gpt2_chinese",
                "files": ["config.json", "pytorch_model.bin", "tokenizer_config.json", "vocab.json"]
            },
            "layout": {
                "model_name": "microsoft/layoutlmv3-base",
                "local_path": self.base_model_dir / "layout",
                "files": ["config.json", "pytorch_model.bin", "tokenizer_config.json"]
            }
        }

        # ONNX 模型下載配置
        self.onnx_models = {
            "bert_ner_onnx": {
                "local_path": self.base_model_dir / "ner" / "model.onnx",
                "url": "https://huggingface.co/optimum/bert-base-NER/resolve/main/model.onnx"
            },
            "layout_onnx": {
                "local_path": self.base_model_dir / "layout" / "model.onnx",
                "url": "https://huggingface.co/microsoft/layoutlmv3-base-onnx/resolve/main/model.onnx"
            }
        }

    def download_model(self, model_key: str, force_download: bool = False) -> Dict[str, Path]:
        """
        下載指定模型

        Args:
            model_key: 模型配置鍵值
            force_download: 強制重新下載

        Returns:
            包含模型路徑的字典
        """
        if model_key not in self.model_configs:
            raise ValueError(f"未知模型: {model_key}")

        config = self.model_configs[model_key]
        model_name = config["model_name"]

        # 設定本地路徑
        local_model_dir = self.base_model_dir / "ner" / model_key
        local_model_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"下載模型: {model_name}")

        if not force_download and self._check_model_exists(local_model_dir):
            logger.info(f"模型已存在: {local_model_dir}")
        else:
            # 下載 tokenizer
            logger.info("下載 tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=local_model_dir / "tokenizer"
            )
            tokenizer.save_pretrained(local_model_dir / "tokenizer")

            # 下載模型
            logger.info("下載模型...")
            model = AutoModelForTokenClassification.from_pretrained(
                model_name,
                cache_dir=local_model_dir / "pytorch_model"
            )
            model.save_pretrained(local_model_dir / "pytorch_model")

            # 下載配置
            model_config = AutoConfig.from_pretrained(model_name)
            model_config.save_pretrained(local_model_dir / "config")

        return {
            "model_dir": local_model_dir,
            "tokenizer_dir": local_model_dir / "tokenizer",
            "pytorch_model_dir": local_model_dir / "pytorch_model",
            "config_dir": local_model_dir / "config"
        }

    def download_file(self, url: str, local_path: Path) -> bool:
        """下載單個文件"""
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)

            response = requests.get(url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            with open(local_path, 'wb') as f, tqdm(
                desc=local_path.name,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

            logger.info(f"已下載: {local_path}")
            return True

        except Exception as e:
            logger.error(f"下載失敗 {url}: {e}")
            return False

    def download_hf_model(self, model_name: str, local_path: Path, files: List[str]) -> bool:
        """從 Hugging Face 下載模型"""
        try:
            local_path.mkdir(parents=True, exist_ok=True)

            logger.info(f"開始下載 {model_name} 到 {local_path}")

            for file_name in files:
                try:
                    file_path = hf_hub_download(
                        repo_id=model_name,
                        filename=file_name,
                        cache_dir=str(local_path.parent),
                        local_dir=str(local_path),
                        local_dir_use_symlinks=False
                    )
                    logger.info(f"已下載: {file_name}")
                except Exception as e:
                    logger.warning(f"跳過文件 {file_name}: {e}")

            # 驗證關鍵文件
            config_file = local_path / "config.json"
            if not config_file.exists():
                logger.error(f"缺少關鍵配置文件: {config_file}")
                return False

            logger.info(f"模型 {model_name} 下載完成")
            return True

        except Exception as e:
            logger.error(f"下載模型 {model_name} 失敗: {e}")
            return False

    def convert_to_onnx(self, model_key: str, optimize: bool = True) -> Path:
        """
        將 PyTorch 模型轉換為 ONNX 格式

        Args:
            model_key: 模型配置鍵值
            optimize: 是否優化 ONNX 模型

        Returns:
            ONNX 模型路徑
        """
        if model_key not in self.model_configs:
            raise ValueError(f"未知模型: {model_key}")

        config = self.model_configs[model_key]
        model_paths = self.download_model(model_key)

        # 載入模型和 tokenizer
        logger.info(f"載入模型進行 ONNX 轉換: {model_key}")

        tokenizer = AutoTokenizer.from_pretrained(
            model_paths["tokenizer_dir"]
        )
        model = AutoModelForTokenClassification.from_pretrained(
            model_paths["pytorch_model_dir"]
        )
        model.eval()

        # 準備示例輸入
        max_length = config["max_length"]
        dummy_text = "這是一個測試文本用於ONNX轉換。" if config["language"] == "zh" else "This is a test text for ONNX conversion."

        inputs = tokenizer(
            dummy_text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # ONNX 輸出路徑
        onnx_model_dir = model_paths["model_dir"] / "onnx"
        onnx_model_dir.mkdir(exist_ok=True)
        onnx_path = onnx_model_dir / f"{model_key}.onnx"

        # 執行轉換
        logger.info(f"轉換到 ONNX: {onnx_path}")

        with torch.no_grad():
            torch_onnx_export(
                model,
                (inputs["input_ids"], inputs["attention_mask"]),
                str(onnx_path),
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=["input_ids", "attention_mask"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch_size", 1: "sequence_length"},
                    "attention_mask": {0: "batch_size", 1: "sequence_length"},
                    "logits": {0: "batch_size", 1: "sequence_length"}
                }
            )

        # 驗證 ONNX 模型
        self._validate_onnx_model(onnx_path, inputs)

        # 優化模型
        if optimize:
            optimized_path = self._optimize_onnx_model(onnx_path)
            return optimized_path

        return onnx_path


    def _check_model_exists(self, model_dir: Path) -> bool:
        """檢查模型是否已存在"""
        required_dirs = ["tokenizer", "pytorch_model", "config"]
        return all((model_dir / dir_name).exists() for dir_name in required_dirs)

    def _validate_onnx_model(self, onnx_path: Path, sample_inputs: Dict[str, torch.Tensor]):
        """驗證 ONNX 模型"""
        logger.info("驗證 ONNX 模型...")

        # 檢查模型結構
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)

        # 測試推理
        session = ort.InferenceSession(str(onnx_path))

        ort_inputs = {
            "input_ids": sample_inputs["input_ids"].numpy(),
            "attention_mask": sample_inputs["attention_mask"].numpy()
        }

        outputs = session.run(None, ort_inputs)
        logger.info(f"ONNX 模型輸出 shape: {outputs[0].shape}")
        logger.info("ONNX 模型驗證成功!")

    def _optimize_onnx_model(self, onnx_path: Path) -> Path:
        """優化 ONNX 模型"""
        logger.info("優化 ONNX 模型...")

        optimized_path = onnx_path.with_suffix(".optimized.onnx")

        # 基本優化
        optimizer.optimize_model(
            str(onnx_path),
            str(optimized_path),
            file_type="onnx"
        )

        logger.info(f"優化完成: {optimized_path}")
        return optimized_path


    def verify_models(self) -> Dict[str, bool]:
        """驗證所有模型是否正確下載"""
        results = {}

        for model_key, config in self.models_config.items():
            local_path = config["local_path"]
            config_file = local_path / "config.json"

            if config_file.exists():
                try:
                    # 嘗試載入模型驗證
                    if "ner" in model_key:
                        AutoTokenizer.from_pretrained(str(local_path))
                        results[model_key] = True
                    elif "gpt2" in model_key:
                        GPT2Tokenizer.from_pretrained(str(local_path))
                        results[model_key] = True
                    else:
                        results[model_key] = True

                    logger.info(f"✓ {model_key} 驗證通過")

                except Exception as e:
                    logger.error(f"✗ {model_key} 驗證失敗: {e}")
                    results[model_key] = False
            else:
                logger.error(f"✗ {model_key} 不存在: {config_file}")
                results[model_key] = False

        return results

    def download_all(self) -> bool:
        """下載所有必要的模型"""
        logger.info("開始下載 EdgeDeID Studio 所需模型...")

        success_count = 0
        total_count = len(self.models_config)

        # 下載 Hugging Face 模型
        for model_key, config in self.models_config.items():
            if self.download_hf_model(
                config["model_name"],
                config["local_path"],
                config["files"]
            ):
                success_count += 1

        # 下載 ONNX 模型
        for onnx_key, config in self.onnx_models.items():
            if not config["local_path"].exists():
                if self.download_file(config["url"], config["local_path"]):
                    logger.info(f"ONNX 模型下載完成: {onnx_key}")

        # 轉換 ONNX 模型
        self.convert_to_onnx("bert_ner")

        # 驗證結果
        verification_results = self.verify_models()
        verified_count = sum(verification_results.values())

        logger.info(f"模型下載完成: {success_count}/{total_count} 成功")
        logger.info(f"模型驗證完成: {verified_count}/{total_count} 通過")

        if verified_count == total_count:
            logger.info("🎉 所有模型下載並驗證成功！")
            return True
        else:
            logger.warning("⚠️  部分模型下載或驗證失敗，請檢查網絡連接和磁盤空間")
            return False

def main():
    """主函數"""
    downloader = ModelDownloader()

    # 檢查現有模型
    logger.info("檢查現有模型...")
    existing_results = downloader.verify_models()
    missing_models = [k for k, v in existing_results.items() if not v]

    if not missing_models:
        logger.info("✓ 所有模型已存在且驗證通過")
        return True

    logger.info(f"需要下載的模型: {missing_models}")

    # 下載缺少的模型
    success = downloader.download_all()

    if success:
        logger.info("模型下載流程完成，可以開始使用 EdgeDeID Studio")
    else:
        logger.error("模型下載流程中發生錯誤，請檢查並重試")
        sys.exit(1)

    return success

if __name__ == "__main__":
    main()


