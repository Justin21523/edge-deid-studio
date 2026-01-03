from __future__ import annotations

import random
from datetime import datetime, timedelta

from .generators import PIIGenerator
from . import HOSPITALS, MEDICAL_SPECIALTIES


class DataFormatter:
    """Formatting utilities for synthetic document generation."""

    @staticmethod
    def generate_paragraph(min_sentences=3, max_sentences=8, pii_density=0.3):
        """Generate a paragraph containing synthetic PII placeholders."""
        # Traditional Chinese sentence templates (zh_TW locale content)
        sentence_templates = [
            "根據最新報告顯示，{PII} 的情況需要進一步關注。",
            "在 {DATE} 的會議中，我們討論了關於 {NAME} 的提案。",
            "請聯絡 {NAME}，電話號碼是 {PHONE}，地址是 {ADDRESS}。",
            "病患 {NAME}，病歷號碼 {MEDICAL_RECORD}，將於下週進行複診。",
            "信用卡號 {CREDIT_CARD} 將於本月到期，請更新付款資訊。",
            "您的身份證字號 {TW_ID} 需要進行驗證。",
            "寄送地址：{ADDRESS}，收件人：{NAME}。",
            "請於 {DATE} 攜帶身分證 {TW_ID} 至本機構辦理手續。",
            "電子郵件 {EMAIL} 已收到您的諮詢，將盡快回覆。",
            "護照號碼 {PASSPORT} 已通過審核，可至櫃台領取。"
        ]

        # Build paragraph.
        paragraph = ""
        num_sentences = random.randint(min_sentences, max_sentences)

        for _ in range(num_sentences):
            template = random.choice(sentence_templates)

            # Replace placeholders with generated values.
            while True:
                pii_count = template.count("{")
                if pii_count == 0 or random.random() > pii_density:
                    break

                # Randomly choose a PII type to replace.
                pii_type, generator = PIIGenerator.generate_random_pii()
                template = template.replace("{" + pii_type + "}", generator(), 1)

            paragraph += template

        return paragraph

    @staticmethod
    def generate_medical_record():
        """Generate a full medical record-like document."""
        # Basic info
        name = PIIGenerator.generate_tw_name()
        gender = random.choice(["男", "女"])
        dob = PIIGenerator.generate_date_of_birth()
        id_num = PIIGenerator.generate_tw_id()
        phone = PIIGenerator.generate_tw_phone()
        address = PIIGenerator.generate_tw_address()
        record_num = PIIGenerator.generate_medical_record()

        # Visit info
        visit_date = (datetime.now() - timedelta(days=random.randint(1, 365))).strftime("%Y-%m-%d")
        hospital = random.choice(HOSPITALS)
        department = random.choice(MEDICAL_SPECIALTIES)
        doctor = "Dr. " + PIIGenerator.generate_tw_name()

        # Diagnosis and prescription
        diagnoses = ["感冒", "流感", "高血壓", "糖尿病", "氣喘", "胃炎", "關節炎", "偏頭痛"]
        treatments = ["藥物治療", "物理治療", "手術", "追蹤觀察", "飲食控制"]
        medications = ["抗生素", "止痛藥", "降血壓藥", "胰島素", "消炎藥"]

        diagnosis = random.choice(diagnoses)
        treatment = random.choice(treatments)
        medication = random.choice(medications)

        # Assemble into a medical record.
        record = f"""
        ====== 醫療記錄 ======
        病歷號: {record_num}
        日期: {visit_date}
        醫院: {hospital} - {department}
        醫師: {doctor}

        --- 病患資訊 ---
        姓名: {name}
        性別: {gender}
        出生日期: {dob}
        身分證字號: {id_num}
        電話: {phone}
        地址: {address}

        --- 診斷資訊 ---
        主訴: {DataFormatter.generate_paragraph(1, 2, 0.1)}
        診斷: {diagnosis}
        處置: {treatment}
        處方: {medication}，每日{random.randint(1, 3)}次，每次{random.randint(1, 3)}顆

        --- 注意事項 ---
        {DataFormatter.generate_paragraph(1, 2, 0.2)}
        ====================
        """

        return record

    @staticmethod
    def generate_financial_document():
        """Generate a financial statement-like document."""
        # Client info
        client_name = PIIGenerator.generate_tw_name()
        client_id = PIIGenerator.generate_tw_id()
        client_address = PIIGenerator.generate_tw_address()
        client_phone = PIIGenerator.generate_tw_phone()
        client_email = PIIGenerator.generate_email(client_name)

        # Account info
        account_number = ''.join(str(random.randint(0, 9)) for _ in range(14))
        credit_card = PIIGenerator.generate_credit_card()

        # Transactions
        transactions = []
        for _ in range(random.randint(3, 10)):
            date = (datetime.now() - timedelta(days=random.randint(1, 30))).strftime("%Y-%m-%d")
            merchant = random.choice(["百貨公司", "超市", "餐廳", "加油站", "線上購物", "電信繳費"])
            amount = round(random.uniform(100, 10000), 2)
            transactions.append(f"{date} | {merchant} | NT${amount:,.2f}")

        # Assemble into a document.
        document = f"""
        ====== 帳戶對帳單 ======
        客戶姓名: {client_name}
        身份證字號: {client_id}
        聯絡地址: {client_address}
        聯絡電話: {client_phone}
        電子郵件: {client_email}

        帳戶號碼: {account_number}
        信用卡號: {credit_card}

        --- 近期交易記錄 ---
        {chr(10).join(transactions)}

        總結餘: NT${round(random.uniform(-5000, 50000), 2):,.2f}
        =====================
        """

        return document

    @staticmethod
    def generate_random_document():
        """Randomly generate one of the supported synthetic documents."""
        doc_types = [
            DataFormatter.generate_medical_record,
            DataFormatter.generate_financial_document,
            lambda: DataFormatter.generate_paragraph(10, 20, 0.4)
        ]

        return random.choice(doc_types)()
