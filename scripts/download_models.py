from huggingface_hub import snapshot_download
from pathlib import Path

# 定義要下載的模型 repo id 與存放路徑
MODELS = {
    "ner_zh":      ("ckiplab/bert-base-chinese-ner",   "models/ner/bert-ner-zh"),
    "gpt2_base":   ("gpt2",                            "models/gpt2"),
    # 如果還有其他模型可在此加入
}

if __name__ == "__main__":
    for name, (repo_id, target) in MODELS.items():
        target_path = Path(target)
        if not target_path.exists():
            print(f"Downloading {repo_id} → {target_path}")
            snapshot_download(repo_id=repo_id, local_dir=target_path, local_dir_use_symlinks=False)
        else:
            print(f"{name} already exists at {target_path}, skipping.")
