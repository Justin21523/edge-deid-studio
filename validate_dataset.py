import os
import json

def validate_dataset(dataset_dir):
    """驗證生成的資料集完整性"""
    # 檢查元數據文件
    metadata_path = os.path.join(dataset_dir, "dataset_metadata.json")
    if not os.path.exists(metadata_path):
        print("錯誤: 找不到元數據文件")
        return False

    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception as e:
        print(f"元數據文件解析錯誤: {e}")
        return False

    # 檢查每個項目
    valid_count = 0
    for item in metadata:
        if validate_dataset_item(item, dataset_dir):
            valid_count += 1

    # 檢查摘要報告
    summary_path = os.path.join(dataset_dir, "dataset_summary.json")
    if not os.path.exists(summary_path):
        print("警告: 找不到摘要報告文件")

    print(f"\n驗證完成! 總項目: {len(metadata)}, 有效項目: {valid_count}")
    return valid_count == len(metadata)

def validate_dataset_item(item, dataset_dir):
    """驗證單個資料項目"""
    item_id = item["id"]
    doc_type = item["metadata"]["type"]

    # 檢查基本文件是否存在
    required_files = [
        os.path.join(dataset_dir, "pdf", f"{doc_type}_doc_{item_id}.pdf"),
        os.path.join(dataset_dir, "word", f"{doc_type}_doc_{item_id}.docx"),
        os.path.join(dataset_dir, "scanned", f"{doc_type}_doc_{item_id}.png"),
        os.path.join(dataset_dir, doc_type, f"{doc_type}_doc_{item_id}.txt")
    ]

    # 財務類別需額外檢查Excel和PPT
    if doc_type == "financial":
        required_files.append(os.path.join(dataset_dir, "excel", f"financial_data_{item_id}.xlsx"))
        required_files.append(os.path.join(dataset_dir, "ppt", f"financial_report_{item_id}.pptx"))

    # 驗證所有文件是否存在
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(os.path.basename(file_path))

    if missing_files:
        print(f"項目 {item_id} ({doc_type}) 缺少文件: {', '.join(missing_files)}")
        return False

    return True

if __name__ == "__main__":
    DATASET_DIR = "generated_dataset"
    print(f"開始驗證資料集: {DATASET_DIR}")

    if validate_dataset(DATASET_DIR):
        print("資料集驗證成功!")
    else:
        print("資料集驗證發現問題!")
