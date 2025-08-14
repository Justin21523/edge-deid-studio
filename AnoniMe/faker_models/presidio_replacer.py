from presidio_anonymizer import AnonymizerEngine
from presidio_analyzer import RecognizerResult
from presidio_anonymizer.entities import OperatorConfig
import random, string
from faker import Faker

anonymizer = AnonymizerEngine()
fake = Faker()

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
    # 將 analyzer_results 轉換為 RecognizerResult 物件
    recognizer_results = [
        RecognizerResult(
            entity_type=res["entity_type"],
            start=res["start"],
            end=res["end"],
            score=res["score"]
        )
        for res in analyzer_results
    ]
    operators = {}
    for res in recognizer_results:
        et = res.entity_type

        if et == "EMAIL_ADDRESS":
            operators[et] = OperatorConfig("replace", {"new_value": "user@example.com"})

        # 新增：台灣身分證
        elif et == "TW_ID_NUMBER":
            operators[et] = OperatorConfig("replace", {"new_value": fake_tw_id()})

        # 新增：統一編號
        elif et == "UNIFIED_BUSINESS_NO":
            operators[et] = OperatorConfig("replace", {"new_value": fake_ubn()})
        
        elif et == "CREDIT_CARD":
            operators[et] = OperatorConfig("replace", {"new_value": fake.credit_card_number()})
            
        elif et == "PERSON":
            operators[et] = OperatorConfig("replace", {"new_value": fake.name()})
        
        elif et == "LOCATION":
            operators[et] = OperatorConfig("replace", {"new_value": fake.address()})
        
        elif et == "IP_ADDRESS":
            operators[et] = OperatorConfig("replace", {"new_value": fake.ipv4()})
        
        elif et == "URL":
            operators[et] = OperatorConfig("replace", {"new_value": fake.url()})
            
        elif et == "MAC_ADDRESS":
            operators[et] = OperatorConfig("replace", {"new_value": fake.mac_address()})

        # TW
        elif et == "TW_PHONE_NUMBER":
            operators[et] = OperatorConfig("replace", {"new_value": f"09{''.join([str(random.randint(0,9)) for _ in range(8)])}"})
            
        elif et == "TW_ID_NUMBER":
            operators[et] = OperatorConfig("replace", {"new_value": f"{random.choice(string.ascii_uppercase)}{random.choice(['1','2'])}{''.join(str(random.randint(0,9)) for _ in range(8))}"})
        
        elif et == "UNIFIED_BUSINESS_NO":
            operators[et] = OperatorConfig("replace", {"new_value": f"{''.join([str(random.randint(0,9)) for _ in range(8)])}"})

        elif et == "TW_HEALTH_INSURANCE":
            operators[et] = OperatorConfig("replace", {"new_value": f"0000{''.join([str(random.randint(0,9)) for _ in range(6)])}"})
        
        elif et == "TW_PASSPORT_NUMBER":
            operators[et] = OperatorConfig("replace", {"new_value": f"3{''.join([str(random.randint(0,9)) for _ in range(7)])}"})
        
        else:
            # # 其他一律遮蔽 ★
            # length = res.end - res.start
            # operators[et] = OperatorConfig("mask", {
            #     "masking_char": "★", "chars_to_mask": length, "from_end": False
            # })
            # 如果沒有特定的替換邏輯，保留原字串 raw text
            operators[et] = OperatorConfig("replace", {"new_value": text[res.start:res.end]})

    # 傳遞 recognizer_results 而不是原始的 analyzer_results
    return anonymizer.anonymize(
        text=text,
        analyzer_results=recognizer_results,
        operators=operators
    ).text
