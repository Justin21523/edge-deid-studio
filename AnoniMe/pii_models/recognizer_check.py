# run_recognizers_check.py

from presidio_detector import analyzer

print("=== Recognizers in registry ===")
for r in analyzer.registry.recognizers:
    print(f"{r.name}: {r.supported_entities}")
