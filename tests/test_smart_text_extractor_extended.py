from src.deid_pipeline.parser.text_extractor import SmartTextExtractor
from src.deid_pipeline.parser.position_mapper import TextPositionMapper
import unittest
from pathlib import Path

class TestTextExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = SmartTextExtractor()
        self.test_dir = Path("tests/test_docs")

    def test_pdf_text_extraction(self):
        path = self.test_dir / "sample.pdf"
        text, layout = self.extractor.extract(str(path))
        self.assertGreater(len(text), 100)
        self.assertEqual(len(layout.pages), 3)
        self.assertGreater(len(layout.pages[0].blocks), 5)

    def test_pdf_table_extraction(self):
        path = self.test_dir / "with_table.pdf"
        _, layout = self.extractor.extract(str(path))
        self.assertGreater(len(layout.pages[0].tables), 0)
        table = layout.pages[0].tables[0]
        self.assertGreater(len(table.cells), 10)

    def test_docx_format_preservation(self):
        path = self.test_dir / "styled.docx"
        _, layout = self.extractor.extract(str(path))
        block = layout.pages[0].blocks[0]
        self.assertTrue(block.is_bold)
        self.assertEqual(block.size, 14)

    def test_ocr_fallback(self):
        path = self.test_dir / "scanned.pdf"
        text, _ = self.extractor.extract(str(path))
        self.assertIn("重要合約條款", text)

    def test_position_mapping(self):
        path = self.test_dir / "sample.pdf"
        _, layout = self.extractor.extract(str(path))
        mapper = TextPositionMapper(layout)

        # 測試特定文字位置
        positions = mapper.get_original_position(100, 120)
        self.assertGreater(len(positions), 0)
        self.assertTrue(all(0 <= p[0][0] <= 1 for p in positions))
