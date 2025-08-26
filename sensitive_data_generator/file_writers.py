import os
import csv
import json
import random
from fpdf import FPDF
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from .formatters import DataFormatter
from .generators import PIIGenerator

class FileWriter:
    """檔案輸出工具"""

    @staticmethod
    def write_text_file(content, output_dir, filename=None):
        """寫入文字檔案"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"generated_document_{timestamp}.txt"

        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        return filepath

    @staticmethod
    def write_pdf_file(content, output_dir, filename=None):
        """寫入PDF檔案"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"generated_document_{timestamp}.pdf"

        filepath = os.path.join(output_dir, filename)

        # 建立PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.add_font("NotoSansTC", "", "fonts/NotoSansTC-Regular.ttf", uni=True)
        pdf.set_font("NotoSansTC", size=12)

        # 添加內容 (處理多行文本)
        pdf.multi_cell(0, 10, content)

        pdf.output(filepath)
        return filepath

    @staticmethod
    def write_image_file(content, output_dir, filename=None):
        """寫入圖片檔案 (模擬掃描文件)"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"generated_document_{timestamp}.png"

        filepath = os.path.join(output_dir, filename)

        # 建立圖片
        img = Image.new('RGB', (800, 1200), color=(255, 255, 255))
        d = ImageDraw.Draw(img)

        # 使用繁體中文字型
        base_dir = os.path.dirname(__file__)
        try:
            font_path = os.path.join(base_dir, "fonts", "NotoSansTC-Regular.ttf")
            font = ImageFont.truetype(font_path, 16)
        except:
            font = ImageFont.load_default()

        # 添加內容
        y_position = 50
        for line in content.split('\n'):
            d.text((50, y_position), line, fill=(0, 0, 0), font=font)
            y_position += 30
            if y_position > 1100:
                break

        # 添加一些雜訊模擬真實文件
        for _ in range(50):
            x = random.randint(0, 800)
            y = random.randint(0, 1200)
            r = random.randint(1, 3)
            d.ellipse([x, y, x+r, y+r], fill=(200, 200, 200))

        img.save(filepath)
        return filepath

    @staticmethod
    def write_csv_file(data, output_dir, filename=None):
        """寫入CSV檔案"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"generated_data_{timestamp}.csv"

        filepath = os.path.join(output_dir, filename)

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["類型", "值", "上下文"])
            for row in data:
                writer.writerow(row)

        return filepath

    @staticmethod
    def write_json_file(data, output_dir, filename=None):
        """寫入JSON檔案"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"generated_data_{timestamp}.json"

        filepath = os.path.join(output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return filepath

    @staticmethod
    def generate_dataset(output_dir, num_items=100, formats=["txt", "pdf", "image", "csv", "json"]):
        """生成完整測試資料集"""
        results = []

        # 確保輸出目錄存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 生成資料
        for i in range(num_items):
            # 隨機選擇PII類型
            pii_type, generator = PIIGenerator.generate_random_pii()
            pii_value = generator()

            # 生成上下文
            context = DataFormatter.generate_paragraph(1, 3, 0.1)

            # 生成完整文件
            document = DataFormatter.generate_random_document()

            # 儲存資料
            data_record = {
                "id": i+1,
                "type": pii_type,
                "value": pii_value,
                "context": context,
                "document": document,
                "files": []
            }

            # 輸出多種格式
            if "txt" in formats:
                txt_file = FileWriter.write_text_file(document, os.path.join(output_dir, "text"))
                data_record["files"].append({"format": "txt", "path": txt_file})

            if "pdf" in formats:
                pdf_file = FileWriter.write_pdf_file(document, os.path.join(output_dir, "pdf"))
                data_record["files"].append({"format": "pdf", "path": pdf_file})

            if "image" in formats:
                img_file = FileWriter.write_image_file(document, os.path.join(output_dir, "images"))
                data_record["files"].append({"format": "image", "path": img_file})

            results.append(data_record)

        # 輸出結構化資料
        if "csv" in formats:
            csv_data = []
            for item in results:
                csv_data.append([item["type"], item["value"], item["context"]])
            FileWriter.write_csv_file(csv_data, os.path.join(output_dir, "structured"))

        if "json" in formats:
            json_data = []
            for item in results:
                json_data.append({
                    "type": item["type"],
                    "value": item["value"],
                    "context": item["context"]
                })
            FileWriter.write_json_file(json_data, os.path.join(output_dir, "structured"))

        # 儲存完整元資料
        FileWriter.write_json_file(results, output_dir, "metadata_full.json")

        return results
