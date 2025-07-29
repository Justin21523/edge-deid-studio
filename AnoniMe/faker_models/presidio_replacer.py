from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
import random, string

anonymizer = AnonymizerEngine()

def fake_tw_id():
    # 亂數產生 A/B 開頭，第二碼 1/2，後面配 8 位數
    letters = random.choice(string.ascii_uppercase)
    gender = random.choice("12")
    nums   = "".join(random.choices(string.digits, k=8))
    return f"{letters}{gender}{nums}"

def fake_ubn():
    # 8 位數統編
    return "".join(random.choices(string.digits, k=8))

def replace_pii(text, analyzer_results):
    operators = {}
    for res in analyzer_results:
        et = res.entity_type

        if et == "EMAIL_ADDRESS":
            operators[et] = OperatorConfig("replace", {"new_value": "user@example.com"})

        # 新增：台灣身分證
        elif et == "TW_ID_NUMBER":
            operators[et] = OperatorConfig("replace", {"new_value": fake_tw_id()})

        # 新增：統一編號
        elif et == "UNIFIED_BUSINESS_NO":
            operators[et] = OperatorConfig("replace", {"new_value": fake_ubn()})

        else:
            # 其他一律遮蔽
            length = res.end - res.start
            operators[et] = OperatorConfig("mask", {
                "masking_char": "★", "chars_to_mask": length, "from_end": False
            })

    return anonymizer.anonymize(
        text=text,
        analyzer_results=analyzer_results,
        operators=operators
    ).text
