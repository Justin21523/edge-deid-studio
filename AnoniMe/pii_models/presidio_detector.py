# pii_models/presidio_detector.py

from presidio_analyzer import AnalyzerEngine, RecognizerRegistry, PatternRecognizer, Pattern
from presidio_analyzer.nlp_engine import NlpEngineProvider
from pii_models.custom_recognizer_plus import register_custom_entities

# 1) 定義 spaCy 多語模型
nlp_config = {
    "nlp_engine_name": "spacy",
    "models": [
        {"lang_code": "en", "model_name": "en_core_web_sm"},
        {"lang_code": "zh", "model_name": "zh_core_web_sm"},
    ],
}

# 2) 用 Provider 建立 NlpEngine
provider = NlpEngineProvider(nlp_configuration=nlp_config)
nlp_engine = provider.create_engine()

# 3) 用這個引擎去初始化 Analyzer
analyzer = AnalyzerEngine(
    nlp_engine=nlp_engine,
    supported_languages=["en", "zh"]
)

# 4) 註冊自訂實體
register_custom_entities(analyzer)

# 沒有自訂的 detect_pii 函數，直接使用 analyzer 的 analyze 方法
# def detect_pii(text):
#     # 你要偵測的 PII 類型
#     entities = ["PHONE_NUMBER", "EMAIL_ADDRESS", "PERSON", "LOCATION"]
#     results = []
#     # 先用英文跑一次
#     results += analyzer.analyze(text=text, entities=entities, language="en")
#     # 再用中文跑一次
#     results += analyzer.analyze(text=text, entities=entities, language="zh")
#     # 最後依起始位置排序，去重（如果需要的話）
#     results = sorted(results, key=lambda x: x.start)
#     return results

def detect_pii(
    text: str,
    language: str = "auto",
    score_threshold: float = 0.5,
):
    # Always pass a real language code here:
    results = analyzer.analyze(
        text=text,
        entities=None,
        language=language,
    )
    # NEW!!!
    # 用 dict 包裝每個 result，加上 raw_txt
    filtered = []
    for r in results:
        if r.score >= score_threshold:
            filtered.append({
                "entity_type": r.entity_type,
                "start": r.start,
                "end": r.end,
                "score": r.score,
                "raw_txt": text[r.start:r.end]
            })
    print(f"偵測到的 PII 實體：{len(filtered)} 個")
    print("篩選後的實體：", filtered)
    # return [r for r in results if r.score >= score_threshold]
    return filtered

if __name__ == "__main__":
    print("=== Registered recognizers ===")
    for r in analyzer.registry.recognizers:
        print(f"{r.name} → supports: {r.supported_entities}; langs: {r.supported_language}")

