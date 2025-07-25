# tests/test_data_generator_integration.py

import unittest
from sensitive_data_generator import FileWriter
from deid_pipeline import DeidPipeline

class DeidentificationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """生成測試資料集"""
        cls.dataset = FileWriter.generate_dataset(
            output_dir="test_data",
            num_items=50,
            formats=["txt"]
        )

    def test_pii_detection(self):
        """測試PII偵測準確率 (PII detection accuracy)"""
        pipeline = DeidPipeline(language="zh")
        results = []
        for item in self.dataset:
            result = pipeline.process(item["document"])
            texts = [e["text"] for e in result.entities]
            expected = item["value"]
            detected = expected in texts
            # 至少要偵測到一個 entity
            self.assertTrue(len(result.entities) > 0, f"未偵測到任何PII: {item['id']}")
            # 主要PII要被偵測到
            self.assertTrue(detected, f"未偵測到 {expected} in {item['document'][:50]}...")
            results.append(detected)
        rate = sum(results) / len(results)
        print(f"PII偵測率: {rate:.2%}")
        self.assertGreaterEqual(rate, 0.95)

    def test_replacement_consistency(self):
        """測試替換一致性 (Replacement consistency)"""
        pipeline = DeidPipeline(language="zh")
        for item in self.dataset:
            result = pipeline.process(item["document"])
            seen = {}
            for ent in result.entities:
                orig, repl = ent["text"], ent["replaced_with"]
                if orig in seen:
                    self.assertEqual(seen[orig], repl, f"不一致替換 for {orig}")
                else:
                    seen[orig] = repl

if __name__ == "__main__":
    unittest.main()
