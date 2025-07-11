def deid(text: str) -> str:
    # 這裡接 ner_inference、mask_pii、pseudonymize
    return text.replace("敏感訊息", "[REDACTED]")
