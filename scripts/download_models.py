# scripts/download_models.py

from pathlib import Path
from huggingface_hub import snapshot_download

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForTokenClassification,
)
from optimum.onnxruntime import ORTModelForTokenClassification

# 要下載／匯出的模型清單
# key: 內部識別用；value: (HF repo id, 本地儲存路徑)
MODELS = {
    "gpt2_base": ("gpt2", "models/gpt2"),
    "ner_zh": ("ckiplab/bert-base-chinese-ner", "models/ner/bert-ner-zh"),
    "ner_zh_onnx": ("ckiplab/bert-base-chinese-ner", "edge_models/bert-ner-zh.onnx"),
}


def download_models():
    for name, (repo_id, target) in MODELS.items():
        target_path = Path(target)

        # 1) 下載並儲存 GPT-2 Transformer 模型
        if name == "gpt2_base":
            if not (target_path / "pytorch_model.bin").exists():
                print(f">> Downloading GPT-2 via Transformers → {target_path}")
                target_path.mkdir(parents=True, exist_ok=True)
                tokenizer = AutoTokenizer.from_pretrained(repo_id)
                model = AutoModelForCausalLM.from_pretrained(repo_id)
                tokenizer.save_pretrained(target_path)
                model.save_pretrained(target_path)
            else:
                print(f"GPT-2 already exists at {target_path}, skipping.")

        # 2) HF Hub snapshot 下載中文 NER 模型
        elif name == "ner_zh":
            if not target_path.exists():
                print(f">> Downloading NER model snapshot: {repo_id} → {target_path}")
                target_path.mkdir(parents=True, exist_ok=True)
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=target_path,
                    local_dir_use_symlinks=False,
                )
            else:
                print(f"{name} already exists at {target_path}, skipping.")

        # 3) 從已下載的 PyTorch NER 模型匯出 ONNX
        elif name == "ner_zh_onnx":
            pt_dir = Path("models/ner/bert-ner-zh")
            onnx_file = target_path
            if not onnx_file.exists():
                print(f">> Converting PyTorch model to ONNX: {pt_dir} → {onnx_file}")
                # 載入 PyTorch 版 tokenizer & model
                tokenizer = AutoTokenizer.from_pretrained(pt_dir)
                model = AutoModelForTokenClassification.from_pretrained(pt_dir)
                # 呼叫 Optimum 直接匯出 ONNX
                ORTModelForTokenClassification.from_pretrained(
                    pt_dir,
                    export=True,
                    from_transformers=True,
                    export_dir=onnx_file.parent,
                    file_name=onnx_file.name,
                )
            else:
                print(f"ONNX model already exists at {onnx_file}, skipping.")

        else:
            print(f"Unknown model key: {name}, skipping.")


if __name__ == "__main__":
    download_models()
