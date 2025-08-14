# run_recognizers_check.py

from presidio_detector import analyzer
from presidio_analyzer import AnalyzerEngine

print("=== Recognizers in registry ===")
for r in analyzer.registry.recognizers:
    print(f"{r.name}: {r.supported_entities}")


analyzer = AnalyzerEngine()
for r in analyzer.registry.recognizers:
    # 每個 recognizer 的名字與它支援的實體類型
    print(f"{r.name}: {r.supported_entities}")
    # 如果是以 PatternRecognizer 實作的 recognizer，就可以看到它的 regex
    if hasattr(r, "patterns"):
        for p in r.patterns:
            print("  •", p.name, "→", p.regex)
