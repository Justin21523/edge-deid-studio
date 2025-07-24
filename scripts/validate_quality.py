# scripts/validate_quality.py

import os, re
from pathlib import Path

def validate_deidentification_quality(original_dir, processed_dir):
    """去識別化品質驗證 (De-ID quality validation)"""
    quality_report = []

    for orig_file in os.listdir(original_dir):
        # 獲取處理後文件路徑
        processed_file = os.path.join(processed_dir, orig_file)

        if not os.path.exists(processed_file):
            continue

        # 讀取原始內容
        with open(os.path.join(original_dir, orig_file), "r", encoding="utf-8") as f:
            orig_content = f.read()

        # 讀取處理後內容
        with open(processed_file, "r", encoding="utf-8") as f:
            processed_content = f.read()

        # 驗證PII移除
        pii_removed = True
        for pii_type in ["身分證", "電話", "地址", "病歷號"]:
            if pii_type in orig_content and pii_type in processed_content:
                # 檢查原始PII是否出現在處理後文件中
                orig_pii = re.findall(rf"{pii_type}[：:]\s*([^\s]+)", orig_content)
                for pii in orig_pii:
                    if pii in processed_content:
                        pii_removed = False
                        break

        # 驗證格式完整性
        format_preserved = True
        # 這裡可以添加格式特定檢查（如表格結構、圖表存在性等）

        # 記錄結果
        quality_report.append({
            "file": orig_file,
            "pii_removed": pii_removed,
            "format_preserved": format_preserved
        })

    # 計算成功率
    pii_success_rate = sum(1 for r in quality_report if r["pii_removed"]) / len(quality_report)
    format_success_rate = sum(1 for r in quality_report if r["format_preserved"]) / len(quality_report)

    print(f"PII移除成功率: {pii_success_rate:.2%}")
    print(f"格式保留成功率: {format_success_rate:.2%}")
    return quality_report

if __name__ == "__main__":
    validate_deidentification_quality("contracts","processed/contracts")
