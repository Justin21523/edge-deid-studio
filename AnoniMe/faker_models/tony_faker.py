import torch
# from transformers import GPT2LMHeadModel, GPT2Tokenizer
import random
import string
from faker import Faker

class SimpleGPT2TagGenerator:
    def generate_with_faker_by_tag_list(self, tag_list):
        """根據標籤 list 產生對應的假資料"""
        fake = self.fake
        result = []
        cache = {}
        tag_map = {
            # Global 
            "CREDIT_CARD": lambda: fake.credit_card_number(),
            "PERSON": lambda: fake.name(),
            "CRYPTO": lambda: fake.sha256(),
            "PHONE_NUMBER": lambda: fake.phone_number(),
            "DATE_TIME": lambda: fake.date_time().isoformat(),
            "EMAIL_ADDRESS": lambda: fake.email(),
            "URL": lambda: fake.url(),
            "IBAN_CODE": lambda: fake.iban(),
            "IP_ADDRESS": lambda: fake.ipv4(),
            "LOCATION": lambda: fake.address(),
            # US
            "US_BANK_NUMBER": lambda: fake.bban(),
            "US_DRIVER_LICENSE": lambda: fake.license_plate(),
            "US_ITIN": lambda: "".join([str(random.randint(0,9)) for _ in range(9)]),  # US ITIN 格式，9 bits羅馬數字
            "US_PASSPORT": lambda: fake.passport_number(), # 新版 one alphabet + eight numbers
            "US_SSN": lambda: fake.ssn(),
            # UK
            "UK_NHS": lambda: f"{''.join([str(random.randint(0,9)) for _ in range(3)])} {''.join([str(random.randint(0,9)) for _ in range(3)])} {''.join([str(random.randint(0,9)) for _ in range(4)])}",  # 9bits, 3 3 4 格式
            "UK_NINO": lambda: f"{''.join(random.choice(string.ascii_uppercase) for _ in range(2))} {''.join([str(random.randint(0,9)) for _ in range(6)])} {str(random.choice(string.ascii_uppercase))}",  # UK NINO 格式
            # TW
            "TW_HEALTH_INSURANCE": lambda: f"0000{''.join([str(random.randint(0,9)) for _ in range(6)])}",  # 10碼
            "TW_ID_NUMBER": lambda: f"{random.choice(string.ascii_uppercase)}{random.choice(['1','2'])}{''.join(str(random.randint(0,9)) for _ in range(8))}",
            "UNIFIED_BUSINESS_NO": lambda: f"{''.join([str(random.randint(0,9)) for _ in range(8)])}",  # 8碼
            "TW_PHONE_NUMBER": lambda: f"09{''.join([str(random.randint(0,9)) for _ in range(8)])}",  # 台灣手機
            "TW_PASSPORT_NUMBER": lambda: f"3{''.join([str(random.randint(0,9)) for _ in range(7)])}",  # 台灣護照號碼，8碼
            # 自訂
            "MAC_ADDRESS": lambda: fake.mac_address(),
            # 忘記規則
            # "MEDICAL_LICENSE": lambda: f"ML{random.randint(100000,999999)}",
        }
        for item in tag_list:
            tag = item["entity_type"]
            func = tag_map.get(tag)
            if func:
                fake_value = func()
            else:
                fake_value = f"未支援標籤: {tag}"
            result.append({
                "entity_type": tag,
                "raw_txt": item["raw_txt"],  # 假設 raw_txt 是原始文字
                "fake_value": fake_value,
                "start": item["start"],
                "end": item["end"],
                "score": item["score"]
            })
        return result
    
    def __init__(self, model_name='gpt2'):
        """初始化，使用預訓練的GPT-2模型"""
        print("載入預訓練的GPT-2模型...")
        # self.tokenizer = GPT2Tokenizer.from_pretrained(model_name) # encode text
        # self.model = GPT2LMHeadModel.from_pretrained(model_name) #
        
        # # 設定pad token
        # if self.tokenizer.pad_token is None:
        #     self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.fake = Faker(['zh_TW', 'en_US'])
        print("模型載入完成！")
    
    def generate_with_prompt_engineering(self, tag_template):
        """使用提示工程的方式生成資料"""
        results = {}
        
        # 不同的提示模板
        
        for tag in tag_template:
            # prompts = f"According to {tag}, generate a real and reasonable piece of data without any explanation or labels."
            prompts = "name."

            input_ids = self.tokenizer.encode(prompts, return_tensors='pt')  # 切詞，編譯成 tensor格式
            # attention_mask = torch.ones_like(input_ids)  # 全部為1，因為沒有padding
            
            # 生成
            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    # attention_mask=attention_mask,
                    max_length=input_ids.shape[1] + 10,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=2
                )
            
            # 解碼輸出
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            print(f"生成的文本: {generated_text}")
            # 提取生成的部分（移除原始提示）
            # generated_part = generated_text[len(prompt):].strip()
            if tag not in results:
                results[tag] = generated_text
        
        return results
    
    
def test_all_methods(pii_list):
    print("=== GPT-2 預訓練模型標籤資料生成測試 ===\n")
    
    # 初始化生成器
    generator = SimpleGPT2TagGenerator()
    
    # 測試 faker 產生 PII 標籤資料
    tag_list = [
        # Global
        "CREDIT_CARD", "PERSON", "CRYPTO", "PHONE_NUMBER", "DATE_TIME", "EMAIL_ADDRESS", "URL", "IBAN_CODE", "IP_ADDRESS", "LOCATION",
        # US
        "US_BANK_NUMBER", "US_DRIVER_LICENSE", "US_ITIN", "US_PASSPORT", "US_SSN",
        # UK
        "UK_NHS", "UK_NINO",
        # TW
        "TW_HEALTH_INSURANCE", "TW_ID_NUMBER", "UNIFIED_BUSINESS_NO", "TW_PHONE_NUMBER", "TW_PASSPORT_NUMBER",
        # 自訂
        "MAC_ADDRESS", "MEDICAL_LICENSE"
    ]
    print("方法1: Faker 產生 PII 標籤資料")
    print("-" * 30)
    faker_results = generator.generate_with_faker_by_tag_list(pii_list)
    print("生成結果:", faker_results)
    return faker_results
    # for tag, value in faker_results.items():
    #     print(f"{tag}: {value}")

    # 方法2: GPT-2 生成（原本邏輯）
    # tag_template = ["<NAME>", "<PHONE_NUMBER>", "<EMAIL_ADDRESS>"]
    # print("\n方法2: GPT-2 生成")
    # print("-" * 30)
    # results1 = generator.generate_with_prompt_engineering(tag_template)
    # print("生成結果:", results1)

def keep_highest_score_per_raw_txt(fake_results_list):
    best = {}
    for item in fake_results_list:
        raw_txt = item["raw_txt"]
        if raw_txt not in best or item["score"] > best[raw_txt]["score"]:
            best[raw_txt] = item
    # 回傳只保留最高分的結果
    print("保留最高分的結果:", list(best.values()))
    return list(best.values())

# if __name__ == "__main__":
#     print("請確保已安裝以下套件:")
#     print("pip install transformers torch faker")
#     print("="*50)
    
