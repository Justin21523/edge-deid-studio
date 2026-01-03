from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List


PII_PATTERNS: Dict[str, str] = {
    "ID": r"[A-Z]\d{9}",
    "PHONE": r"09\d{2}-?\d{3}-?\d{3}",
    "EMAIL": r"[\w\.-]+@[\w\.-]+\.[A-Za-z]{2,}",
}


def validate_deidentification_quality(original_dir: str, processed_dir: str) -> List[Dict[str, Any]]:
    """Validate that PII strings from the original files are not present after de-identification.

    Notes:
    - This script is intended for plain-text outputs (txt/csv/json). Binary formats (pdf/docx/xlsx/pptx)
      require format-aware validation and are out of scope for this lightweight check.
    """

    original_path = Path(original_dir).expanduser().resolve()
    processed_path = Path(processed_dir).expanduser().resolve()

    if not original_path.exists():
        raise FileNotFoundError(f"Original directory not found: {original_path}")
    if not processed_path.exists():
        raise FileNotFoundError(f"Processed directory not found: {processed_path}")

    report: List[Dict[str, Any]] = []

    for filename in os.listdir(original_path):
        orig_file = original_path / filename
        proc_file = processed_path / filename
        if not orig_file.is_file() or not proc_file.exists():
            continue

        orig_content = orig_file.read_text(encoding="utf-8", errors="replace")
        proc_content = proc_file.read_text(encoding="utf-8", errors="replace")

        leaked: Dict[str, List[str]] = {}
        for pii_type, pattern in PII_PATTERNS.items():
            values = set(re.findall(pattern, orig_content))
            leaked_values = [v for v in values if v and v in proc_content]
            if leaked_values:
                leaked[pii_type] = leaked_values

        pii_removed = not leaked

        report.append(
            {
                "file": filename,
                "pii_removed": pii_removed,
                "leaked": leaked,
            }
        )

    if not report:
        print("No comparable files found for validation.")
        return report

    pii_success_rate = sum(1 for r in report if r["pii_removed"]) / len(report)
    print(f"PII removal success rate: {pii_success_rate:.2%}")
    return report


if __name__ == "__main__":
    validate_deidentification_quality("contracts", "processed/contracts")

