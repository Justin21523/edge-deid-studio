import re
from typing import List, Tuple
from .layout import DocumentLayout

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
