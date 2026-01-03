import unittest
import os

from deid_pipeline.config import Config
from deid_pipeline.pii.detectors import bert_detector, get_detector, regex_detector
from deid_pipeline.pii.utils.replacer import Replacer
from test_data_factory import TestDataFactory


class PIITestSuite(unittest.TestCase):
    def setUp(self):
        self.data_factory = TestDataFactory()
        self.regex_detector = regex_detector.RegexDetector(config_path="configs/regex_zh.yaml")
        self.bert_detector = bert_detector.BertNERDetector(model_dir=str(Config.NER_MODEL_PATH_ZH))
        self.composite_detector = get_detector("zh")
        self.replacer = Replacer()

    def test_regex_detection(self):
        """Test regex detector precision on simple cases."""
        test_cases = [
            ("\u8eab\u5206\u8b49 A123456789", "ID", "A123456789"),
            ("\u96fb\u8a71 0912-345-678", "PHONE", "0912-345-678"),
            ("\u75c5\u6b77\u865f M1234567", "MEDICAL_ID", "M1234567")
        ]

        for text, expected_type, expected_value in test_cases:
            entities = self.regex_detector.detect(text)
            self.assertGreaterEqual(len(entities), 1)
            self.assertEqual(entities[0]['type'], expected_type)
            self.assertEqual(text[entities[0]['span'][0]:entities[0]['span'][1]], expected_value)

    def test_bert_detection_recall(self):
        """Test BERT detector recall on synthetic documents."""
        if not os.path.exists(str(Config.NER_MODEL_PATH_ZH)):
            self.skipTest("Local NER model directory is missing.")

        # Generate 100 test samples
        recall_count = 0
        for _ in range(100):
            text, inserted = self.data_factory.generate_test_document(pii_count=5)
            entities = self.bert_detector.detect(text)

            # Ensure all injected PII types were detected
            detected_types = {e['type'] for e in entities}
            expected_types = {t[0] for t in inserted}

            if expected_types.issubset(detected_types):
                recall_count += 1

        recall_rate = recall_count / 100
        print(f"BERT recall rate: {recall_rate:.2%}")
        self.assertGreaterEqual(recall_rate, 0.90)  # Expect >= 90% recall

    def test_composite_detection_conflict(self):
        """Test composite detector can run and return stable spans."""
        text = "A123456789 0912-345-678"
        entities = self.composite_detector.detect(text)

        self.assertGreaterEqual(len(entities), 1)
        for ent in entities:
            self.assertIn("span", ent)
            self.assertIn("type", ent)

    def test_replacement_consistency(self):
        """Test replacement consistency for identical originals."""
        text = "\u75c5\u4ebaA: \u8eab\u5206\u8b49A123456789, \u96fb\u8a710912345678\u3002\u75c5\u4ebaB: \u8eab\u5206\u8b49A123456789"
        entities = self.composite_detector.detect(text)
        replaced_text, _ = self.replacer.replace(text, entities)

        # Extract all replacement values for the same original ID
        replaced_values = set()
        _replaced, events = self.replacer.replace(text, entities, context_hash="suite-doc")
        for ev in events:
            if ev.get("original") == "A123456789":
                replaced_values.add(ev.get("replacement"))

        # The same original should map to the same replacement
        self.assertEqual(len(replaced_values), 1)

    def test_contextual_replacement(self):
        """Test format-preserving replacement in a small context."""
        text = "\u75c5\u6b77\u865f: M1234567 \u8a3a\u65b7: \u611f\u5192"
        entities = self.composite_detector.detect(text)
        replaced_text, _ = self.replacer.replace(text, entities)

        # Ensure the record number pattern is preserved
        self.assertRegex(replaced_text, r"\u75c5\u6b77\u865f: [A-Z][0-9]{7,8}")
