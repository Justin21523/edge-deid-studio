# scripts/run_automated_pipeline.py

import os, time, json
from deid_pipeline import DeidPipeline

def run_automated_test_pipeline(dataset_dir):
    """自動化測試管線 (Automated test pipeline)"""
    pipeline = DeidPipeline(language="zh")
    results = []
    for root, _, files in os.walk(dataset_dir):
        for fn in files:
            path = os.path.join(root, fn)
            start = time.time()
            res = pipeline.process(path)
            elapsed = time.time() - start
            results.append({
                "file": path,
                "format": fn.split('.')[-1],
                "pii_count": len(res.entities),
                "processing_time": elapsed
            })
    out = "pipeline_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {out}")

if __name__ == "__main__":
    run_automated_test_pipeline("advanced_test_dataset")
