import random
import string
from datetime import datetime, timedelta
from .config import *

class PIIGenerator:
    """繁體中文PII資料生成器"""

    @staticmethod
    def generate_tw_id():
        """生成台灣身分證字號 (符合官方規則)"""
        # 區域碼 (A-Z 除去 I, O)
        area_codes = "ABCDEFGHJKLMNPQRSTUVXYWZ"
        first_char = random.choice(area_codes)

        # 性別碼 (1:男, 2:女)
        gender_code = random.choice(['1', '2'])

        # 隨機7位數字
        random_digits = ''.join(str(random.randint(0, 9)) for _ in range(7))

        # 組合前9碼
        partial_id = first_char + gender_code + random_digits

        # 計算檢查碼
        # 轉換字母為數字 (A=10, B=11, ...)
        first_char_value = ord(first_char) - 55 if ord(first_char) > 74 else ord(first_char) - 64
        weights = [1, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        total = first_char_value * weights[0] + int(gender_code) * weights[1]

        for i, digit in enumerate(random_digits):
            total += int(digit) * weights[i+2]

        check_digit = (10 - (total % 10)) % 10

        return f"{first_char}{gender_code}{random_digits}{check_digit}"

    @staticmethod
    def generate_tw_phone():
        """生成台灣手機號碼"""
        prefix = "09"
        middle = random.randint(10, 99)  # 10-99
        end = random.randint(100000, 999999)  # 100000-999999

        # 隨機選擇格式：0912-345-678 或 0912345678
        if random.random() > 0.5:
            return f"{prefix}{middle:02d}-{end//1000:03d}-{end%1000:03d}"
        else:
            return f"{prefix}{middle:02d}{end}"

    @staticmethod
    def generate_tw_address():
        """生成台灣地址"""
        # 選擇地區
        region = random.choice(list(TAIWAN_LOCATIONS.keys()))
        city = random.choice(TAIWAN_LOCATIONS[region])

        # 生成街道
        street_type = random.choice(["路", "街", "大道"])
        street_name = random.choice(STREET_NAMES)
        lane = f"{random.randint(1, 100)}巷" if random.random() > 0.7 else ""
        alley = f"{random.randint(1, 20)}弄" if lane and random.random() > 0.5 else ""
        number = f"{random.randint(1, 100)}號"

        # 可選的樓層資訊
        floor = ""
        if random.random() > 0.5:
            floor_num = random.randint(1, 25)
            floor = f"{floor_num}樓"
            if random.random() > 0.7:
                floor += f"之{random.randint(1, 5)}"

        # 組合地址
        address = f"{city}{street_name}{street_type}"
        if lane:
            address += f"{lane}"
        if alley:
            address += f"{alley}"
        address += f"{number}"
        if floor:
            address += f"{floor}"

        return address

    @staticmethod
    def generate_tw_name():
        """生成繁體中文姓名"""
        surname = random.choice(SURNAMES)
        given_name = random.choice(GIVEN_NAMES)

        # 30%機率有雙名
        if random.random() > 0.7:
            second_given = random.choice(GIVEN_NAMES)
            while second_given == given_name:
                second_given = random.choice(GIVEN_NAMES)
            given_name += second_given

        return f"{surname}{given_name}"

    @staticmethod
    def generate_medical_record():
        """生成醫療病歷號碼"""
        hospital = random.choice(HOSPITALS)
        hospital_code = ''.join(c for c in hospital if c.isalpha())[:3].upper()

        # 不同格式的病歷號
        formats = [
            lambda: f"{hospital_code}-{random.randint(100000, 999999)}",  # TWH-123456
            lambda: f"{random.randint(10000000, 99999999)}",              # 12345678
            lambda: f"MR{random.choice(['A','B','C'])}{random.randint(10000, 99999)}",  # MRA12345
            lambda: f"病歷號：{random.randint(1000000000, 9999999999)}"   # 病歷號：1234567890
        ]

        return random.choice(formats)()

    @staticmethod
    def generate_credit_card():
        """生成信用卡號 (模擬格式)"""
        # 主要信用卡發卡機構識別碼
        issuers = ['4', '5', '34', '37', '6']
        prefix = random.choice(issuers)

        # 生成卡號
        length = 16 if len(prefix) == 1 else 15
        digits = ''.join(str(random.randint(0, 9)) for _ in range(length - len(prefix)))

        return prefix + digits

    @staticmethod
    def generate_date_of_birth(min_age=18, max_age=90):
        """生成出生日期"""
        current_year = datetime.now().year
        birth_year = current_year - random.randint(min_age, max_age)
        birth_month = random.randint(1, 12)

        # 處理不同月份的天數
        if birth_month == 2:
            max_day = 29 if (birth_year % 4 == 0 and birth_year % 100 != 0) or (birth_year % 400 == 0) else 28
        elif birth_month in [4, 6, 9, 11]:
            max_day = 30
        else:
            max_day = 31

        birth_day = random.randint(1, max_day)

        # 隨機選擇格式
        formats = [
            lambda y, m, d: f"{y}年{m}月{d}日",
            lambda y, m, d: f"{y}-{m:02d}-{d:02d}",
            lambda y, m, d: f"{d}/{m}/{y % 100:02d}"
        ]

        return random.choice(formats)(birth_year, birth_month, birth_day)

    @staticmethod
    def generate_email(name=None):
        """生成電子郵件"""
        if not name:
            name = PIIGenerator.generate_tw_name()

        # 移除名字中的空格
        name = name.replace(" ", "")

        # 常見郵件服務商
        domains = [
            "gmail.com", "yahoo.com.tw", "hotmail.com", "outlook.com",
            "msn.com", "pchome.com.tw", "hinet.net"
        ]

        # 郵件前綴格式
        formats = [
            lambda n: f"{n}",
            lambda n: f"{n}{random.randint(1, 99)}",
            lambda n: f"{n[0]}{n[1:]}{random.randint(10, 99)}",
            lambda n: f"{n}.{random.randint(1970, 2023)}"
        ]

        prefix = random.choice(formats)(name)
        domain = random.choice(domains)

        return f"{prefix}@{domain}".lower()

    @staticmethod
    def generate_passport():
        """生成護照號碼 (模擬台灣護照格式)"""
        return f"{random.choice('ABCDEFGH')}{random.randint(1000000, 9999999)}"

    @staticmethod
    def generate_license_plate():
        """生成車牌號碼 (台灣格式)"""
        # 台灣車牌格式: ABC-123 或 123-ABC
        if random.random() > 0.5:
            letters = ''.join(random.choice(string.ascii_uppercase) for _ in range(3))
            numbers = ''.join(str(random.randint(0, 9)) for _ in range(3))
            return f"{letters}-{numbers}"
        else:
            numbers = ''.join(str(random.randint(0, 9)) for _ in range(3))
            letters = ''.join(random.choice(string.ascii_uppercase) for _ in range(3))
            return f"{numbers}-{letters}"

    @staticmethod
    def generate_health_insurance():
        """生成健保卡號 (模擬格式)"""
        return f"{random.randint(10000000000, 99999999999)}"

    @staticmethod
    def generate_random_pii():
        """隨機生成一種PII類型"""
        pii_types = [
            ("TW_ID", PIIGenerator.generate_tw_id),
            ("PHONE", PIIGenerator.generate_tw_phone),
            ("ADDRESS", PIIGenerator.generate_tw_address),
            ("NAME", PIIGenerator.generate_tw_name),
            ("MEDICAL_RECORD", PIIGenerator.generate_medical_record),
            ("DATE_OF_BIRTH", PIIGenerator.generate_date_of_birth),
            ("EMAIL", PIIGenerator.generate_email),
            ("CREDIT_CARD", PIIGenerator.generate_credit_card),
            ("PASSPORT", PIIGenerator.generate_passport),
            ("LICENSE_PLATE", PIIGenerator.generate_license_plate),
            ("HEALTH_INSURANCE", PIIGenerator.generate_health_insurance)
        ]

        return random.choice(pii_types)
