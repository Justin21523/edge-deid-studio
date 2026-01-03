import random

class TestDataFactory:
    __test__ = False  # pytest: this is a helper, not a test case

    def __init__(self, locale='zh_TW'):
        self.locale = locale

    def generate_tw_id(self):
        """Generate a synthetic Taiwan-like ID value (not checksum-valid)."""

        first_letter = random.choice("ABCDEFGHJKLMNPQRSTUVXYWZ")
        gender = str(random.randint(1, 2))
        body = "".join(str(random.randint(0, 9)) for _ in range(7))
        checksum = str(random.randint(0, 9))
        return f"{first_letter}{gender}{body}{checksum}"

    def generate_medical_record(self):
        """Generate a synthetic medical identifier (regex-friendly)."""

        prefix = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        digits = "".join(str(random.randint(0, 9)) for _ in range(random.choice([7, 8])))
        return f"{prefix}{digits}"

    def generate_test_document(self, pii_count=10):
        """Generate a synthetic text document with embedded PII-like values."""

        words = ["lorem", "ipsum", "dolor", "sit", "amet"]
        content = " ".join(random.choice(words) for _ in range(200))

        pii_generators = [
            ("ID", self.generate_tw_id),
            ("PHONE", lambda: f"09{random.randint(0, 99):02d}{random.randint(0, 9_999_999):07d}"),
            ("MEDICAL_ID", self.generate_medical_record),
            ("NAME", lambda: "John Doe"),
            ("ADDRESS", lambda: "123 Example Rd"),
            ("EMAIL", lambda: f"user{random.randint(0, 9999):04d}@example.com"),
        ]

        insertion_points = sorted(random.sample(range(len(content)), pii_count))
        inserted = []
        for i, point in enumerate(insertion_points):
            pii_type, generator = random.choice(pii_generators)
            pii_value = generator()
            inserted.append((pii_type, pii_value))
            content = content[:point] + f" {pii_value} " + content[point:]

        return content, inserted
