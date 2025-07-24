import unittest
from deid_pipeline.pii.detectors import regex_detector, bert_detector, composite
from deid_pipeline.pii.utils.replacer import Replacer
from test_data_factory import TestDataFactory

class PIITestSuite(unittest.TestCase):
    def setUp(self):
        self.data_factory = TestDataFactory()
        self.regex_detector = regex_detector.RegexDetector(config_path="configs/regex_zh.yaml")
        self.bert_detector = bert_detector.BertNERDetector(model_dir="models/ner/zh_tw")
        self.composite_detector = composite.CompositeDetector()
        self.replacer = Replacer()

    def test_regex_detection(self):
        """測試正則表達式檢測準確率"""
        test_cases = [
            ("身分證 A123456789", "TW_ID", "A123456789"),
            ("電話 0912-345-678", "PHONE", "0912-345-678"),
            ("病歷號 AM-123456", "MEDICAL_RECORD", "AM-123456")
        ]

        for text, expected_type, expected_value in test_cases:
            entities = self.regex_detector.detect(text)
            self.assertGreaterEqual(len(entities), 1)
            self.assertEqual(entities[0]['type'], expected_type)
            self.assertEqual(text[entities[0]['span'][0]:entities[0]['span'][1]], expected_value)

    def test_bert_detection_recall(self):
        """測試BERT模型召回率"""
        # 生成100個測試樣本
        recall_count = 0
        for _ in range(100):
            text, pii_data = self.data_factory.generate_test_document(pii_count=5)
            entities = self.bert_detector.detect(text)

            # 檢查是否所有插入的PII都被檢測到
            detected_types = {e['type'] for e in entities}
            expected_types = {t[0] for t in pii_data}

            if expected_types.issubset(detected_types):
                recall_count += 1

        recall_rate = recall_count / 100
        print(f"BERT模型召回率: {recall_rate:.2%}")
        self.assertGreaterEqual(recall_rate, 0.90)  # 期望召回率90%以上

    def test_composite_detection_conflict(self):
        """測試複合檢測器衝突處理"""
        # 刻意製造重疊實體
        text = "A123456789 0912-345-678"  # 身分證和電話號碼重疊
        entities = self.composite_detector.detect(text)

        # 應只檢測到優先級更高的身分證
        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0]['type'], 'TW_ID')

    def test_replacement_consistency(self):
        """測試替換一致性"""
        text = "病人A: 身分證A123456789, 電話0912345678。病人B: 身分證A123456789"
        entities = self.composite_detector.detect(text)
        replaced_text, _ = self.replacer.replace(text, entities)

        # 提取所有替換後的值
        replaced_values = set()
        for entity in entities:
            if entity['text'] == "A123456789":
                replaced_values.add(entity['replaced_with'])

        # 相同原始值應被替換為相同假值
        self.assertEqual(len(replaced_values), 1)

    def test_contextual_replacement(self):
        """測試語境感知替換"""
        text = "病歷號: AM-123456 診斷: 感冒"
        entities = self.composite_detector.detect(text)
        replaced_text, _ = self.replacer.replace(text, entities)

        # 檢查病歷號格式是否保持
        self.assertRegex(replaced_text, r"病歷號: [A-Z]{2}-\d{6}")
