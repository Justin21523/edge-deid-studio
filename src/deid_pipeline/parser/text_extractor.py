# src/deid_pipeline/parser/text_extractor.py
import os
import re
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
import pdfplumber
from docx import Document
import pandas as pd
from bs4 import BeautifulSoup
from .ocr import OCRAdapter
from ..config import Config
from ..pii.utils import logger
from .layout import DocumentLayout, PageLayout, TextBlock, TableBlock, TableCell

class TextExtractor(ABC):
    """抽象文字提取器"""
    @abstractmethod
    def extract(self, file_path: str) -> DocumentLayout:
        pass

class TextPositionMapper:
    """座標映射工具"""
    def __init__(self, layout: DocumentLayout):
        self.layout = layout
        self.char_positions = self._build_char_index()

    def _build_char_index(self) -> List[Tuple]:
        """建立字符級位置索引"""
        index = []
        char_offset = 0

        for page in self.layout.pages:
            for block in page.blocks:
                words = re.split(r'(\s+)', block.text)
                word_x0, word_y0 = block.bbox[0], block.bbox[1]

                for word in words:
                    if not word.strip():
                        char_offset += len(word)
                        continue

                    # 簡化計算：平均分配字符寬度
                    char_width = (block.bbox[2] - block.bbox[0]) / len(word)

                    for i, char in enumerate(word):
                        char_bbox = (
                            word_x0 + i * char_width,
                            word_y0,
                            word_x0 + (i+1) * char_width,
                            block.bbox[3]
                        )
                        index.append((char_offset + i, char_bbox, page.page_num))

                    char_offset += len(word)
                    word_x0 += len(word) * char_width

        return index

    def get_original_position(self, start_idx: int, end_idx: int) -> List[Tuple]:
        """獲取原始文件中的位置"""
        positions = []
        current_bbox = None

        for idx in range(start_idx, end_idx):
            if idx >= len(self.char_positions):
                break

            char_idx, bbox, page_num = self.char_positions[idx]

            if current_bbox and self._is_continuous(current_bbox, bbox):
                # 合併連續bbox
                current_bbox = (
                    current_bbox[0], current_bbox[1],
                    bbox[2], current_bbox[3]
                )
            else:
                if current_bbox:
                    positions.append((current_bbox, page_num))
                current_bbox = bbox

        if current_bbox:
            positions.append((current_bbox, page_num))

        return positions

    def _is_continuous(self, bbox1: Tuple, bbox2: Tuple, threshold=0.1) -> bool:
        """檢查兩個bbox是否連續"""
        # 簡化實現：檢查x軸連續性
        return abs(bbox1[2] - bbox2[0]) < threshold

# 格式解析引擎實作
#1. PDF 深度處理（使用 pdfplumber）
class PDFTextExtractor(TextExtractor):
    def extract(self, file_path: str) -> DocumentLayout:
        layout = DocumentLayout()

        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                page_layout = PageLayout(
                    page_num=page_num,
                    width=page.width,
                    height=page.height
                )

                # 提取文本
                for obj in page.extract_text(x_tolerance=2, y_tolerance=2, keep_blank_chars=True):
                    # 處理文本塊
                    text = obj['text']
                    bbox = (
                        obj['x0'] / page.width,
                        obj['top'] / page.height,
                        obj['x1'] / page.width,
                        obj['bottom'] / page.height
                    )
                    font = obj.get('fontname')
                    size = obj.get('size')

                    block = TextBlock(text, bbox, font, size)
                    page_layout.blocks.append(block)

                # 提取表格
                for table in page.extract_tables():
                    table_block = TableBlock()

                    for row_idx, row in enumerate(table):
                        for col_idx, cell in enumerate(row):
                            # 簡化實現 - 實際需處理合併單元格
                            cell_bbox = self._estimate_cell_bbox(table, row_idx, col_idx, page)
                            table_cell = TableCell(
                                text=cell,
                                bbox=cell_bbox,
                                row=row_idx,
                                col=col_idx
                            )
                            table_block.cells.append(table_cell)

                    if table_block.cells:
                        page_layout.tables.append(table_block)

                layout.pages.append(page_layout)

        return layout

    def _estimate_cell_bbox(self, table, row_idx, col_idx, page):
        """估算單元格位置 (簡化實現)"""
        # 實際應根據pdfplumber的表格結構計算
        return (0.1, 0.1, 0.2, 0.2)

# Office 文件支援

class DocxTextExtractor(TextExtractor):
    def extract(self, file_path: str) -> DocumentLayout:
        layout = DocumentLayout()
        page_layout = PageLayout(page_num=1, width=8.5*72, height=11*72)  # 假設A4尺寸

        doc = Document(file_path)

        # 提取段落
        y_offset = 0.1  # 從頁面頂部開始
        for para in doc.paragraphs:
            text = para.text
            if not text.strip():
                continue

            # 估算位置 (實際應根據樣式計算)
            bbox = (0.1, y_offset, 0.9, y_offset + 0.05)

            # 獲取樣式
            style = para.style
            font = style.font.name if style and style.font else None
            size = style.font.size.pt if style and style.font else None
            is_bold = style.font.bold if style and style.font else False

            block = TextBlock(text, bbox, font, size, is_bold=is_bold)
            page_layout.blocks.append(block)
            y_offset += 0.06  # 下移

        # 提取表格
        for table in doc.tables:
            table_block = TableBlock()

            for row_idx, row in enumerate(table.rows):
                for col_idx, cell in enumerate(row.cells):
                    cell_text = cell.text
                    # 簡化位置估算
                    cell_bbox = (
                        0.1 + col_idx * 0.2,
                        y_offset + row_idx * 0.1,
                        0.1 + (col_idx+1) * 0.2,
                        y_offset + (row_idx+1) * 0.1
                    )
                    table_cell = TableCell(
                        text=cell_text,
                        bbox=cell_bbox,
                        row=row_idx,
                        col=col_idx
                    )
                    table_block.cells.append(table_cell)

            if table_block.cells:
                page_layout.tables.append(table_block)
                y_offset += len(table.rows) * 0.1 + 0.1  # 下移

        layout.pages.append(page_layout)
        return layout

# HTML 結構保留
class HTMLTextExtractor(TextExtractor):
    def extract(self, file_path: str) -> DocumentLayout:
        layout = DocumentLayout()
        page_layout = PageLayout(page_num=1, width=800, height=600)  # 假設寬高

        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')

        # 移除不需要的元素
        for script in soup(['script', 'style']):
            script.extract()

        # 提取文本塊
        y_offset = 0.1
        for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'div']):
            text = element.get_text(strip=True, separator=' ')
            if not text:
                continue

            # 估算位置 (實際應通過CSS計算)
            bbox = (0.1, y_offset, 0.9, y_offset + 0.05)

            # 獲取樣式
            class_list = element.get('class', [])
            is_bold = 'bold' in class_list or element.name.startswith('h')

            block = TextBlock(text, bbox, is_bold=is_bold)
            page_layout.blocks.append(block)
            y_offset += 0.06

        layout.pages.append(page_layout)
        return layout

# 智慧回退機制與主提取類
class SmartTextExtractor:
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.ocr_adapter = OCRAdapter(engine=self.config.OCR_ENGINE)
        self.extractors = {
            '.pdf': PDFTextExtractor(),
            '.docx': DocxTextExtractor(),
            '.doc': DocxTextExtractor(),  # 需轉換處理
            '.html': HTMLTextExtractor(),
            '.htm': HTMLTextExtractor(),
            '.xlsx': self._extract_excel,
            '.xls': self._extract_excel,
        }

    def extract(self, file_path: str) -> Tuple[str, DocumentLayout]:
        """返回 (純文字, 結構化佈局)"""
        ext = os.path.splitext(file_path)[1].lower()

        try:
            if ext in self.extractors:
                # 原生解析
                layout = self.extractors[ext].extract(file_path)
                full_text = self._layout_to_text(layout)
                return full_text, layout
            elif ext in ('.jpg', '.jpeg', '.png', '.tiff', '.bmp'):
                # 純影像處理
                return self._process_image(file_path)
            else:
                # 純文字或未知格式
                return self._fallback_to_text(file_path)
        except Exception as e:
            self.config.logger.error(f"解析失敗: {str(e)}")
            # 回退到OCR
            return self._fallback_to_ocr(file_path)

    def _layout_to_text(self, layout: DocumentLayout) -> str:
        """將結構化佈局轉換為純文字"""
        text_parts = []
        for page in layout.pages:
            for block in page.blocks:
                text_parts.append(block.text)
            for table in page.tables:
                # 簡化表格轉換
                for row in self._table_rows(table):
                    text_parts.append("\t".join(row))
        return "\n".join(text_parts)

    def _table_rows(self, table: TableBlock) -> List[List[str]]:
        """將表格轉換為二維數組"""
        # 實際應根據行列索引重組
        max_row = max(cell.row for cell in table.cells) + 1
        max_col = max(cell.col for cell in table.cells) + 1

        grid = [['' for _ in range(max_col)] for _ in range(max_row)]

        for cell in table.cells:
            for r in range(cell.row, cell.row + cell.rowspan):
                for c in range(cell.col, cell.col + cell.colspan):
                    if r < max_row and c < max_col:
                        grid[r][c] = cell.text

        return grid

    def _process_image(self, file_path: str) -> Tuple[str, DocumentLayout]:
        """處理影像文件"""
        text, ocr_layout = self.ocr_adapter.recognize(file_path)
        return text, self._ocr_to_layout(ocr_layout)

    def _ocr_to_layout(self, ocr_result) -> DocumentLayout:
        """將OCR結果轉換為DocumentLayout"""
        layout = DocumentLayout()
        page = PageLayout(1, ocr_result['width'], ocr_result['height'])

        for word in ocr_result['words']:
            bbox = (
                word['bbox'][0] / page.width,
                word['bbox'][1] / page.height,
                word['bbox'][2] / page.width,
                word['bbox'][3] / page.height
            )
            block = TextBlock(word['text'], bbox)
            page.blocks.append(block)

        layout.pages.append(page)
        return layout

    def _fallback_to_text(self, file_path: str) -> Tuple[str, DocumentLayout]:
        """純文字回退"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            # 創建簡單佈局
            layout = DocumentLayout()
            page = PageLayout(1, 800, 600)
            block = TextBlock(text, (0.1, 0.1, 0.9, 0.9))
            page.blocks.append(block)
            layout.pages.append(page)

            return text, layout
        except:
            # 最終回退到OCR
            return self._fallback_to_ocr(file_path)

    def _fallback_to_ocr(self, file_path: str) -> Tuple[str, DocumentLayout]:
        """OCR回退策略"""
        self.config.logger.warning(f"使用OCR回退處理: {file_path}")

        if file_path.lower().endswith('.pdf'):
            # PDF OCR處理
            return self._process_pdf_with_ocr(file_path)
        else:
            # 其他格式當作影像處理
            return self._process_image(file_path)

    def _process_pdf_with_ocr(self, file_path: str) -> Tuple[str, DocumentLayout]:
        """PDF OCR處理"""
        full_text = []
        full_layout = DocumentLayout()

        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                # 轉換為影像
                img = page.to_image(resolution=300).original

                # OCR識別
                text, ocr_layout = self.ocr_adapter.recognize(img)
                full_text.append(text)

                # 轉換為統一佈局
                page_layout = self._ocr_to_layout(ocr_layout)
                page_layout.page_num = page_num
                full_layout.pages.append(page_layout)

        return "\n".join(full_text), full_layout

    def _extract_excel(self, file_path: str) -> DocumentLayout:
        """Excel文件提取"""
        layout = DocumentLayout()
        page = PageLayout(1, 800, 600)

        try:
            df = pd.read_excel(file_path, sheet_name=None)
            y_offset = 0.1

            for sheet_name, data in df.items():
                # 添加工作表標題
                title_block = TextBlock(
                    text=f"工作表: {sheet_name}",
                    bbox=(0.1, y_offset, 0.9, y_offset + 0.05),
                    is_bold=True
                )
                page.blocks.append(title_block)
                y_offset += 0.06

                # 添加表格數據
                for row_idx, row in data.iterrows():
                    row_text = "\t".join(str(x) for x in row.values)
                    row_block = TextBlock(
                        text=row_text,
                        bbox=(0.1, y_offset, 0.9, y_offset + 0.05)
                    )
                    page.blocks.append(row_block)
                    y_offset += 0.05

                y_offset += 0.1  # 工作表間距

            layout.pages.append(page)
            return layout
        except Exception as e:
            self.config.logger.error(f"Excel解析失敗: {str(e)}")
            raise
