# BitoGuard — 智慧合規風險雷達

AML（反洗錢）自動化合規平台，為 BitoPro（幣託）AWS Hackathon 競賽專案。

## 核心功能

- 透過 XGBoost + SHAP 機器學習偵測人頭戶、洗錢路徑及機房集體詐騙
- PySpark BFS 圖分析（Glue）追蹤 User-to-IP 多跳關聯
- Amazon Bedrock（Claude）生成人類可讀的結構化風險診斷書
- 事件驅動架構：S3 → Lambda → Bedrock → DynamoDB

## 風險分級

| 機率 | 等級 | 行動 |
|---|---|---|
| ≥ 0.90 | EXTREME | 制式凍結通知，24h 內提交 STR |
| 0.75–0.90 | HIGH | 深度診斷，緊急 24h 電訪 |
| 0.65–0.75 | BOUNDARY | 邊界案例分析，3 個工作日核實 |
| < 0.65 | — | Rule-Based，不呼叫 LLM |

## 核心使用者

- 合規主管：閱讀 `compliance_summary` 白話摘要，決策凍結 / 電訪 / 列管
- 風控工程師：調整模型超參數、特徵工程、SHAP 分析
- DevOps / MLOps：維護 SageMaker Pipeline、Glue Job、Lambda 部署
