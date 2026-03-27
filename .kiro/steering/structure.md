# BitoGuard — 專案結構

```
C:\AWS\
├── template.yaml                   # SAM IaC（Lambda + DynamoDB + S3 + SQS）
├── athena_graph_hops.sql           # Athena BFS 圖分析查詢
│
├── 資料層
│   ├── bito_api_ingester.py        # BitoPro API 爬取 + S3 Parquet 上傳（Checkpoint）
│   ├── bito_data_manager.py        # API 資料表定義（欄位映射 / 精度 / 時間正規化）
│   └── ingest_to_s3.py             # 資料注入總流程 + Glue Catalog 建立
│
├── 特徵工程層
│   ├── glue_graph_hops.py          # Glue PySpark BFS（超級節點加鹽 / Union-Find）
│   └── feature_store.py            # SageMaker Feature Store CRUD
│
├── 模型訓練層
│   ├── train_xgboost_script.py     # SageMaker 訓練 Entry Point（對抗增強 / 代價敏感）
│   ├── train_sagemaker.py          # HPO Tuner + Model Registry + Batch Transform
│   └── download_model.py           # S3 模型下載 + 解壓縮
│
├── 評估與視覺化層
│   ├── validation_report.py        # Threshold Sweep + 技術報告生成
│   └── visualize.py                # PR 曲線 / SHAP Beeswarm / Feature Importance
│
├── 診斷推論層
│   ├── xai_bedrock.py              # SHAP + Bedrock 風險診斷書生成（1 RPS 速率限制）
│   └── lambda_diagnosis.py         # Lambda Handler（S3 → Bedrock → DynamoDB）
│
├── 事件回應層
│   └── incident_response_workflow.py  # 事件回應 + Feedback Loop + 增量重訓
│
└── 整合與合規層
    ├── main_pipeline.py            # CLI 總入口（8 階段端到端管線）
    └── check_compliance.py         # 合規自動化檢核（C-1 / C-2 / C-3）
```

## 管線階段

`main_pipeline.py` 支援 `--start-from <stage>` 斷點續跑：

| 階段 | 名稱 | 說明 |
|---|---|---|
| Stage A | aws_ingest | API → S3 → Glue Crawler → Glue Job（AWS 模式） |
| Stage 1 | feature | 資料載入 + 特徵工程（快取至 feature_cache.parquet） |
| Stage 2 | train | 5-Fold CV + SageMaker XGBoost 訓練 |
| Stage 3 | download | 從 S3 下載模型 |
| Stage 4 | validate | Threshold Sweep + 驗證報告 |
| Stage 5 | visualize | 生成四張圖表（plots/） |
| Stage 6 | xai | SHAP + Bedrock 風險診斷書 |
| Stage Z | aws_submit | submission.csv 格式校對（AWS 模式） |

## 資料流

```
BitoPro API → ingest_to_s3 → S3 Raw Parquet
                                  ↓
             glue_graph_hops   BFS Hop Features → S3 + Feature Store
                                  ↓
             train_sagemaker   XGBoost HPO → Model Registry
                                  ↓
             xai_bedrock       SHAP + Bedrock → StructuredDiagnosis
                                  ↓
             lambda_diagnosis  S3 CSV → DynamoDB 診斷書
                                  ↓
             incident_response DynamoDB → SNS 告警 → 增量重訓
```

## 輸出檔案

| 檔案 | 說明 |
|---|---|
| `submission.csv` | 競賽提交（user_id, status） |
| `submission_with_prob.csv` | 含預測機率 |
| `validation_report.json/txt` | 驗證報告 |
| `xai_reports.json` | 風險診斷書 |
| `plots/` | 四張圖表 |
| `pipeline_state.json` | 中間狀態（斷點續跑用） |
| `feature_cache.parquet` | 特徵快取 |
| `cv_report.json` | Cross Validation 報告 |
