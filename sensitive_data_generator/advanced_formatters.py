# sensitive_data_generator/advanced_formatters.py

import random
from datetime import datetime, timedelta
from .generators import PIIGenerator
from .config import HOSPITALS

class AdvancedDataFormatter:
    """進階資料格式生成器"""

    @staticmethod
    def generate_contract_document():
        """生成合約文件（含敏感資料）"""
        parties = {
            "甲方": PIIGenerator.generate_tw_name(),
            "乙方": PIIGenerator.generate_tw_name(),
            "甲方身分證": PIIGenerator.generate_tw_id(),
            "乙方身分證": PIIGenerator.generate_tw_id(),
            "甲方地址": PIIGenerator.generate_tw_address(),
            "乙方地址": PIIGenerator.generate_tw_address(),
            "簽約日期": (datetime.now() - timedelta(days=random.randint(1, 365))\
                        .strftime("%Y年%m月%d日"))
        }

        contract = f"""
                        合 約 書

        立合約書人：
        甲方：{parties['甲方']}（身分證字號：{parties['甲方身分證']}）
        住址：{parties['甲方地址']}

        乙方：{parties['乙方']}（身分證字號：{parties['乙方身分證']}）
        住址：{parties['乙方地址']}

        茲因雙方同意訂立本合約，共同遵守下列條款：

        第一條 合約目的
        甲方同意委託乙方進行專案開發，乙方同意接受委託。

        第二條 合約期間
        本合約自簽訂之日起生效，有效期間為一年，至{parties['簽約日期']}止。

        第三條 報酬及支付方式
        甲方應支付乙方總報酬新台幣{random.randint(100000, 500000):,}元整。
        付款方式：簽約時支付30%，期中支付40%，驗收完成支付30%。

        第四條 保密條款
        雙方同意對本合約內容及執行過程中獲知之他方營業秘密負保密義務。

        第五條 違約處理
        任一方違反本合約條款時，應賠償他方因此所受之損害。

        第六條 管轄法院
        因本合約涉訟時，雙方同意以台灣台北地方法院為第一審管轄法院。

        立合約書人：

        甲方：___________________
        （簽名或蓋章）

        乙方：___________________
        （簽名或蓋章）

        中華民國 {parties['簽約日期']}
        """

        return contract

    @staticmethod
    def generate_medical_report():
        """生成詳細醫療報告（含圖表引用）"""
        patient = {
            "name": PIIGenerator.generate_tw_name(),
            "id": PIIGenerator.generate_tw_id(),
            "dob": PIIGenerator.generate_date_of_birth(min_age=18, max_age=90),
            "phone": PIIGenerator.generate_tw_phone(),
            "address": PIIGenerator.generate_tw_address(),
            "record_num": PIIGenerator.generate_medical_record()
        }

        # 醫療數據
        test_results = {
            "blood_pressure": f"{random.randint(110, 140)}/{random.randint(70, 90)} mmHg",
            "heart_rate": f"{random.randint(60, 100)} bpm",
            "glucose": f"{random.randint(70, 200)} mg/dL",
            "cholesterol": f"{random.randint(150, 250)} mg/dL"
        }

        report = f"""
        ==============================
        {random.choice(HOSPITALS)} 醫療報告
        ==============================

        病患資訊:
        姓名: {patient['name']}
        病歷號: {patient['record_num']}
        出生日期: {patient['dob']}
        聯絡電話: {patient['phone']}
        住址: {patient['address']}

        就診日期: {(datetime.now() - timedelta(days=random.randint(1, 30)).strftime('%Y-%m-%d'))}
        主治醫師: {PIIGenerator.generate_tw_name()} 醫師

        臨床診斷:
        - {random.choice(['上呼吸道感染', '高血壓', '第二型糖尿病', '退化性關節炎'])}
        - {random.choice(['輕度貧血', '高血脂症', '胃食道逆流'])}

        檢驗結果:
        1. 血壓: {test_results['blood_pressure']}
        2. 心率: {test_results['heart_rate']}
        3. 血糖: {test_results['glucose']}
        4. 膽固醇: {test_results['cholesterol']}

        影像檢查:
        - {random.choice(['胸部X光: 無明顯異常', '腹部超音波: 輕度脂肪肝', '心電圖: 竇性心律'])}

        處方:
        1. {random.choice(['Amoxicillin 500mg', 'Lisinopril 10mg', 'Metformin 500mg'])}
           每日{random.randint(1, 3)}次，每次{random.randint(1, 2)}顆
        2. {random.choice(['維生素D補充劑', '益生菌', '止痛藥'])}
           必要時服用

        醫囑:
        - {random.choice(['建議定期追蹤血壓', '控制飲食與體重', '適度運動'])}
        - 下次回診日期: {(datetime.now() + timedelta(days=random.randint(14, 60)).strftime('%Y-%m-%d'))}

        [請參閱附件圖表分析]
        ==============================
        """

        return report

    @staticmethod
    def generate_financial_statement():
        """生成財務報表（含複雜表格）"""
        client = {
            "name": PIIGenerator.generate_tw_name(),
            "id": PIIGenerator.generate_tw_id(),
            "account": ''.join(str(random.randint(0, 9)) for _ in range(12)),
            "credit_card": PIIGenerator.generate_credit_card()
        }

        # 生成交易記錄
        transactions = []
        for _ in range(10):
            date = (datetime.now() - timedelta(days=random.randint(1, 30))\
                .strftime("%Y-%m-%d"))
            merchant = random.choice(["百貨公司", "超市", "餐廳", "加油站", "線上購物", "電信繳費"])
            amount = round(random.uniform(100, 10000), 2)
            transactions.append({
                "date": date,
                "description": merchant,
                "amount": amount
            })

        # 生成報表
        statement = f"""
        客戶財務報表

        客戶資訊:
        姓名: {client['name']}
        身分證字號: {client['id']}
        帳戶號碼: {client['account']}
        信用卡號: {client['credit_card']}

        交易記錄:
        日期         | 描述         | 金額 (NT$)
        ------------|--------------|-----------
        """

        for t in transactions:
            statement += f"{t['date']} | {t['description']} | {t['amount']:,.2f}\n"

        statement += f"""
        總支出: NT$ {sum(t['amount'] for t in transactions):,.2f}
        帳戶餘額: NT$ {random.uniform(10000, 500000):,.2f}

        圖表分析:
        [請參閱附件支出分類圖]
        """

        return statement
