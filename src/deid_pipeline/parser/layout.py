from typing import List, Tuple, Dict

class DocumentLayout:
    """統一的文件結構表示"""
    def __init__(self):
        self.pages: List[PageLayout] = []
        self.metadata: Dict = {}

class PageLayout:
    """單頁文件結構"""
    def __init__(self, page_num: int, width: float, height: float):
        self.page_num = page_num
        self.width = width
        self.height = height
        self.blocks: List[TextBlock] = []
        self.tables: List[TableBlock] = []

class TextBlock:
    """文字區塊結構"""
    __slots__ = ('text', 'bbox', 'font', 'size', 'color', 'is_bold', 'is_italic')
    def __init__(self, text: str, bbox: Tuple[float, float, float, float],
                 font: str = None, size: float = None, color: str = None,
                 is_bold: bool = False, is_italic: bool = False):
        self.text = text
        self.bbox = bbox  # (x0, y0, x1, y1) 正規化座標
        self.font = font
        self.size = size
        self.color = color
        self.is_bold = is_bold
        self.is_italic = is_italic

class TableBlock:
    """表格區塊結構"""
    def __init__(self):
        self.cells: List[TableCell] = []
        self.bbox = None

class TableCell:
    """表格單元格結構"""
    __slots__ = ('text', 'bbox', 'row', 'col', 'rowspan', 'colspan')
    def __init__(self, text: str, bbox: Tuple, row: int, col: int,
                 rowspan: int = 1, colspan: int = 1):
        self.text = text
        self.bbox = bbox
        self.row = row
        self.col = col
        self.rowspan = rowspan
        self.colspan = colspan
