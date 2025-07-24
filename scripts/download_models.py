# scripts/download_models.py

from pathlib import Path
from huggingface_hub import snapshot_download

# 對 Transformer LM 使用 save_pretrained
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {
    "ner_zh":     ("ckiplab/bert-base-chinese-ner", "models/ner/bert-ner-zh"),
    "gpt2_base":  ("gpt2",                         "models/gpt2"),
}

if __name__ == "__main__":
    for name, (repo_id, target) in MODELS.items():
        target_path = Path(target)
        if name == "gpt2_base":
            # 用 transformers 下載並儲存 GPT-2
            if not (target_path / "pytorch_model.bin").exists():
                print(f"→ Downloading GPT-2 via Transformers → {target_path}")
                target_path.mkdir(parents=True, exist_ok=True)
                tokenizer = AutoTokenizer.from_pretrained(repo_id)
                model     = AutoModelForCausalLM.from_pretrained(repo_id)
                tokenizer.save_pretrained(target_path)
                model.save_pretrained(target_path)
            else:
                print(f"GPT-2 already exists at {target_path}, skipping.")
        else:
            # 用 HF Hub snapshot 下載 NER 模型
            if not target_path.exists():
                print(f"→ snapshot_download {repo_id} → {target_path}")
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=target_path,
                    local_dir_use_symlinks=False
                )
            else:
                print(f"{name} already exists at {target_path}, skipping.")
