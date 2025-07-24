from faker import Faker
import random

class TestDataFactory:
    def __init__(self, locale='zh_TW'):
        self.fake = Faker(locale)

    def generate_tw_id(self):
        """生成台灣身分證測試資料"""
        first_letter = random.choice('ABCDEFGHJKLMNPQRSTUVXYWZIO')
        numbers = ''.join(str(random.randint(0, 9)) for _ in range(8))
        return f"{first_letter}{numbers}"

    def generate_medical_record(self):
        """生成醫療病歷號測試資料"""
        patterns = [
            f"AM-{self.fake.random_number(digits=6, fix_len=True)}",
            f"{random.choice(['北','中','南'])}醫-{self.fake.random_number(digits=5)}",
            f"病歷號：{self.fake.random_number(digits=8)}"
        ]
        return random.choice(patterns)

    def generate_test_document(self, pii_count=10):
        """生成包含各類PII的測試文件"""
        content = self.fake.paragraph(nb_sentences=20)

        # 插入各種PII類型
        pii_types = [
            ("TW_ID", self.generate_tw_id),
            ("PHONE", lambda: self.fake.phone_number()),
            ("MEDICAL_RECORD", self.generate_medical_record),
            ("NAME", lambda: self.fake.name()),
            ("ADDRESS", lambda: self.fake.address())
        ]

        insertion_points = sorted(random.sample(range(len(content)), pii_count))
        for i, point in enumerate(insertion_points):
            pii_type, generator = random.choice(pii_types)
            pii_value = generator()
            content = content[:point] + f" {pii_value} " + content[point:]

        return content, pii_types
