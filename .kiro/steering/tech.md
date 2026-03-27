# BitoGuard — 技術棧與開發規範

## AWS 服務（競賽核准清單）

| 服務 | 用途 | 限制 |
|---|---|---|
| Amazon S3 | 原始資料 / 特徵 / 模型儲存 | PublicAccessBlock 必須全部啟用 |
| AWS Glue | PySpark BFS 圖分析 | Region: us-east-1 / us-west-2 |
| Amazon Athena | SQL 特徵查詢 | Engine v3（Trino 395） |
| Amazon SageMaker | XGBoost 訓練 / HPO / 推論 | Instance: ml.m5.xlarge（無 GPU） |
| Amazon Bedrock | Claude 診斷書生成 | 嚴格 < 1 RPS，僅白名單模型 |
| AWS Lambda | 事件驅動診斷 | 512 MB / 900s timeout / arm64 |
| Amazon DynamoDB | 診斷書持久化 | PAY_PER_REQUEST / SSE / TTL 90d |
| Amazon SQS | Lambda DLQ | SqsManagedSseEnabled: true |
| Amazon CloudWatch | 指標 / 告警 / Dashboard | — |
| AWS SAM | IaC 部署 | template.yaml |

## Bedrock 模型白名單

```python
MODEL_SONNET = "anthropic.claude-3-5-sonnet-20241022-v2:0"   # BOUNDARY / HIGH
MODEL_HAIKU  = "anthropic.claude-3-haiku-20240307-v1:0"      # EXTREME（制式報告）
```

嚴禁啟用其他模型。

## Python 依賴套件

```
boto3 / botocore    # AWS SDK
sagemaker           # SageMaker SDK
xgboost             # 模型訓練
shap                # 可解釋性
pandas / numpy      # 資料處理
pyarrow             # Parquet 格式
tenacity            # Retry / Backoff
requests            # BitoPro API 呼叫
scikit-learn        # StratifiedKFold / metrics
```

## 常用指令

```bash
# 端到端管線（本地模式）
python main_pipeline.py

# 端到端管線（AWS 完整模式）
python main_pipeline.py --aws-mode

# 從特定階段繼續（跳過已完成步驟）
python main_pipeline.py --start-from validate

# 合規檢核（部署前必須全數 PASS）
python check_compliance.py

# 合規檢核（含即時 S3 查詢）
python check_compliance.py --live-s3 --bucket <bucket-name>

# SAM 部署
sam build
sam deploy --region us-east-1
```

## 技術規範

### Region 鎖定
所有腳本必須透過 `_validate_region()` 驗證，僅允許 `us-east-1` 或 `us-west-2`。

### 速率限制
Bedrock 呼叫必須使用 `_BedrockRateLimiter`（min_interval=1.1s）。
ThrottlingException：指數退避，base=2s，max=60s，最多 5 次重試。

### PII 防護（三層）
- Layer 1: 欄位層過濾（`filter_pii_context()`）
- Layer 2: 數值特徵白名單（`filter_pii_contributions()`）
- Layer 3: 正則掃描（`scan_prompt_for_pii()`，臨送 Bedrock 前）

### 憑證管理
- 禁止任何形式的 Hardcode（Access Key / Secret / Password / Token）
- 所有憑證必須透過環境變數注入
- `ACCOUNT_ID`、`your-hackathon-bucket` 為佔位符，部署前必須替換

### Instance Type 規範
- 訓練 / 推論 / Batch Transform：`ml.m5.xlarge`（4 vCPU / 16 GB）
- 禁止 p3 / g5 / p4 等 GPU 系列

## 合規自動化

執行 `python check_compliance.py` 驗證：
- C-1：硬編碼憑證掃描（4 條正則規則）
- C-2：S3 PublicAccessBlock 靜態 + 即時掃描
- C-3：PII 過濾器三層功能驗證

所有 3 項必須 PASS 才可部署。

## 環境變數

| 變數名稱 | 用途 | 預設值 |
|---|---|---|
| `AWS_DEFAULT_REGION` | 所有 boto3 client | `us-east-1` |
| `S3_BUCKET` | 資料儲存桶名稱 | `your-hackathon-bucket` |
| `SAGEMAKER_ROLE_ARN` | SageMaker / Glue 執行角色 | 必須設定 |
| `GLUE_ROLE_ARN` | Glue Crawler 執行角色 | 必須設定 |
| `BEDROCK_REGION` | Bedrock API Region | `us-east-1` |
| `DYNAMO_TABLE` | DynamoDB 診斷書資料表 | `bito-diagnoses` |
| `SNS_ALERT_ARN` | SNS 告警 Topic ARN | `""（選用）` |
| `RISK_THRESHOLD` | 診斷書觸發門檻 | `0.65` |
