from __future__ import annotations

import random
import string
from datetime import datetime, timedelta

from .config import *


class PIIGenerator:
    """Traditional Chinese (zh_TW) synthetic PII generator."""

    @staticmethod
    def generate_tw_id():
        """Generate a Taiwan national ID number (rule-based)."""
        # Area code (A-Z excluding I, O)
        area_codes = "ABCDEFGHJKLMNPQRSTUVXYWZ"
        first_char = random.choice(area_codes)

        # Gender code (1: male, 2: female)
        gender_code = random.choice(['1', '2'])

        # Random 7 digits
        random_digits = ''.join(str(random.randint(0, 9)) for _ in range(7))

        # Assemble first 9 characters
        partial_id = first_char + gender_code + random_digits

        # Compute check digit
        # Convert letter to number (A=10, B=11, ...)
        first_char_value = ord(first_char) - 55 if ord(first_char) > 74 else ord(first_char) - 64
        weights = [1, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        total = first_char_value * weights[0] + int(gender_code) * weights[1]

        for i, digit in enumerate(random_digits):
            total += int(digit) * weights[i+2]

        check_digit = (10 - (total % 10)) % 10

        return f"{first_char}{gender_code}{random_digits}{check_digit}"

    @staticmethod
    def generate_tw_phone():
        """Generate a Taiwan mobile phone number."""
        prefix = "09"
        middle = random.randint(10, 99)  # 10-99
        end = random.randint(100000, 999999)  # 100000-999999

        # Randomly choose a format: 0912-345-678 or 0912345678
        if random.random() > 0.5:
            return f"{prefix}{middle:02d}-{end//1000:03d}-{end%1000:03d}"
        else:
            return f"{prefix}{middle:02d}{end}"

    @staticmethod
    def generate_tw_address():
        """Generate a Taiwan address (synthetic)."""
        # Choose region
        region = random.choice(list(TAIWAN_LOCATIONS.keys()))
        city = random.choice(TAIWAN_LOCATIONS[region])

        # Generate street
        street_type = random.choice(["路", "街", "大道"])
        street_name = random.choice(STREET_NAMES)
        lane = f"{random.randint(1, 100)}巷" if random.random() > 0.7 else ""
        alley = f"{random.randint(1, 20)}弄" if lane and random.random() > 0.5 else ""
        number = f"{random.randint(1, 100)}號"

        # Optional floor info
        floor = ""
        if random.random() > 0.5:
            floor_num = random.randint(1, 25)
            floor = f"{floor_num}樓"
            if random.random() > 0.7:
                floor += f"之{random.randint(1, 5)}"

        # Assemble address
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
        """Generate a Traditional Chinese name."""
        surname = random.choice(SURNAMES)
        given_name = random.choice(GIVEN_NAMES)

        # 30% chance of a 2-character given name
        if random.random() > 0.7:
            second_given = random.choice(GIVEN_NAMES)
            while second_given == given_name:
                second_given = random.choice(GIVEN_NAMES)
            given_name += second_given

        return f"{surname}{given_name}"

    @staticmethod
    def generate_medical_record():
        """Generate a synthetic medical record identifier."""
        hospital = random.choice(HOSPITALS)
        hospital_code = ''.join(c for c in hospital if c.isalpha())[:3].upper()

        # Multiple output formats
        formats = [
            lambda: f"{hospital_code}-{random.randint(100000, 999999)}",  # TWH-123456
            lambda: f"{random.randint(10000000, 99999999)}",              # 12345678
            lambda: f"MR{random.choice(['A','B','C'])}{random.randint(10000, 99999)}",  # MRA12345
            lambda: f"病歷號：{random.randint(1000000000, 9999999999)}"   # Example: label + numeric id
        ]

        return random.choice(formats)()

    @staticmethod
    def generate_credit_card():
        """Generate a synthetic credit card number (format-only, not Luhn-valid)."""
        # Major issuer prefixes
        issuers = ['4', '5', '34', '37', '6']
        prefix = random.choice(issuers)

        # Generate digits
        length = 16 if len(prefix) == 1 else 15
        digits = ''.join(str(random.randint(0, 9)) for _ in range(length - len(prefix)))

        return prefix + digits

    @staticmethod
    def generate_date_of_birth(min_age=18, max_age=90):
        """Generate a date of birth."""
        current_year = datetime.now().year
        birth_year = current_year - random.randint(min_age, max_age)
        birth_month = random.randint(1, 12)

        # Handle different month lengths
        if birth_month == 2:
            max_day = 29 if (birth_year % 4 == 0 and birth_year % 100 != 0) or (birth_year % 400 == 0) else 28
        elif birth_month in [4, 6, 9, 11]:
            max_day = 30
        else:
            max_day = 31

        birth_day = random.randint(1, max_day)

        # Randomly choose an output format
        formats = [
            lambda y, m, d: f"{y}年{m}月{d}日",
            lambda y, m, d: f"{y}-{m:02d}-{d:02d}",
            lambda y, m, d: f"{d}/{m}/{y % 100:02d}"
        ]

        return random.choice(formats)(birth_year, birth_month, birth_day)

    @staticmethod
    def generate_email(name=None):
        """Generate an email address."""
        if not name:
            name = PIIGenerator.generate_tw_name()

        # Remove whitespace from the name
        name = name.replace(" ", "")

        # Common email providers
        domains = [
            "gmail.com", "yahoo.com.tw", "hotmail.com", "outlook.com",
            "msn.com", "pchome.com.tw", "hinet.net"
        ]

        # Prefix formats
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
        """Generate a synthetic passport number (Taiwan-like format)."""
        return f"{random.choice('ABCDEFGH')}{random.randint(1000000, 9999999)}"

    @staticmethod
    def generate_license_plate():
        """Generate a Taiwan license plate number."""
        # Taiwan formats: ABC-123 or 123-ABC
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
        """Generate a synthetic health insurance number (format-only)."""
        return f"{random.randint(10000000000, 99999999999)}"

    @staticmethod
    def generate_random_pii():
        """Randomly select a PII type generator."""
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
