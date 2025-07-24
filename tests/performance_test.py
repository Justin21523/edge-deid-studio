import timeit
import statistics
from deid_pipeline.pii.detectors import regex_detector, bert_detector, composite
from test_data_factory import TestDataFactory

def benchmark_detector(detector, text_length=10000):
    """效能基準測試"""
    data_factory = TestDataFactory()
    text, _ = data_factory.generate_test_document(pii_count=text_length//200)

    # 測試函數
    def detect_wrapper():
        return detector.detect(text)

    # 執行時間測試
    times = timeit.repeat(detect_wrapper, number=1, repeat=5)

    return {
        "min": min(times),
        "max": max(times),
        "avg": statistics.mean(times),
        "text_length": len(text)
    }

def run_performance_suite():
    """執行完整效能測試"""
    results = {}

    # 測試不同偵測器
    detectors = {
        "Regex": regex_detector.RegexDetector(config_path="configs/regex_zh.yaml"),
        "BERT": bert_detector.BertNERDetector(model_dir="models/ner/zh_tw"),
        "Composite": composite.CompositeDetector()
    }

    for name, detector in detectors.items():
        print(f"測試 {name} 偵測器效能...")
        results[name] = benchmark_detector(detector)

    # 測試不同文本長度
    text_lengths = [1000, 5000, 10000, 20000]
    length_results = {}

    for length in text_lengths:
        print(f"測試 {length} 字元文本效能...")
        length_results[length] = benchmark_detector(
            composite.CompositeDetector(),
            text_length=length
        )

    # 輸出結果
    print("\n偵測器效能比較:")
    for name, data in results.items():
        print(f"{name}: 平均 {data['avg']:.4f}秒 (長度 {data['text_length']}字元)")

    print("\n文本長度影響:")
    for length, data in length_results.items():
        print(f"{length}字元: 平均 {data['avg']:.4f}秒")
