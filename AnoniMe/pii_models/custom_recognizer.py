
from presidio_analyzer import Pattern, PatternRecognizer
from presidio_analyzer import AnalyzerEngine

def register_custom_entities(analyzer: AnalyzerEngine):
    # 1) Define your regex as a Pattern
    tw_id_pattern = Pattern(
        name="tw_id_pattern",
        regex=r"\b[A-Z][12]\d{8}\b",
        score=0.85
    )
    tw_ubn_pattern = Pattern(
        name="tw_ubn_pattern",
        regex=r"\b\d{8}\b",
        score=0.80
    )
    # tw_phone_pattern = Pattern(
        
    # )

    # 2) Wrap them in PatternRecognizers
    # 為 zh 和 en 各建立一個 recognizer
    for lang in ("zh", "en"):
        tw_id_recognizer = PatternRecognizer(
            supported_entity="TW_ID_NUMBER",
            patterns=[tw_id_pattern],
            supported_language=lang,    # 單一語言
            context=["身分證","身份證"]
        )
        tw_ubn_recognizer = PatternRecognizer(
            supported_entity="UNIFIED_BUSINESS_NO",
            patterns=[tw_ubn_pattern],
            supported_language=lang,
            context=["統編", "統一編號"]
        )

        # 3) Register with your AnalyzerEngine
        analyzer.registry.add_recognizer(tw_id_recognizer)
        analyzer.registry.add_recognizer(tw_ubn_recognizer)
