# scripts/benchmark_formats.py

import os, time
from deid_pipeline import DeidPipeline

def benchmark_formats(dataset_dir, formats=["pdf", "docx", "xlsx", "png"]):
    """不同格式效能基準測試"""
    pipeline = DeidPipeline(language="zh")
    results = {}
    for fmt in formats:
        # 收集該格式所有文件
        fmt_files = [f for f in os.listdir(dataset_dir) if f.endswith(fmt)]

        # 測試處理時間
        processing_times = []
        for file in fmt_files[:10]:  # 每種格式測試10個文件
            start_time = time.time()
            pipeline.process(os.path.join(dataset_dir, file))
            processing_times.append(time.time() - start_time)

        # 記錄結果
        results[fmt] = {
            "avg_time": sum(processing_times) / len(processing_times),
            "min_time": min(processing_times),
            "max_time": max(processing_times)
        }

    # 輸出結果
    print("格式處理效能報告:")
    for fmt, data in results.items():
        print(f"{fmt.upper()}格式: 平均 {data['avg_time']:.2f}秒")
