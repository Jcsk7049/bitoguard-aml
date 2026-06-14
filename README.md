# 🛡️ BitoGuard AML

> 加密貨幣交易所反洗錢(AML)風險偵測系統 — 從資料擷取、特徵工程、圖譜分析、
> 模型訓練到可解釋性(XAI)與互動式儀表板的完整 pipeline。

針對 BitoPro 交易所資料,辨識具洗錢風險的高風險帳戶。核心為 **LightGBM + XGBoost**
集成模型,搭配交易圖譜(graph hops)特徵與 SHAP / Bedrock 的可解釋性診斷,
並以 **Streamlit** 儀表板呈現結果。

---

## ✨ 功能特色

- **端到端 Pipeline**:API 擷取 → 特徵工程 → 圖譜跳數 → 模型訓練 → 預測 → XAI 診斷
- **集成模型**:LightGBM (0.60) + XGBoost (0.40) 加權集成
- **圖譜特徵**:以黑名單鄰居跳數、IP 共用等關聯特徵強化偵測
- **可解釋性 (XAI)**:SHAP 特徵貢獻 + AWS Bedrock 生成自然語言診斷報告
- **互動式儀表板**:Streamlit 呈現模型表現、風險名單與個案診斷
- **雲端部署就緒**:支援本機、Streamlit Community Cloud 與 AWS(SageMaker / SAM)

## 📊 模型表現(OOF 交叉驗證)

| 指標 | 數值 |
|------|------|
| 模型 | LGB(0.60) + XGB(0.40) 集成 |
| AUC | **0.832** |
| F1 | 0.301 |
| Precision | 0.275 |
| Recall | 0.332 |
| Accuracy | 0.950 |
| 特徵數 | 32 |
| 預測名單 | 12,753 帳戶中標記 501 個高風險 |

**最重要特徵(前 5)**:`total_volume`、`age`、`swap_twd_volume`、
`min_retention_minutes`、`tx_per_day`。

---

## 🚀 快速開始(本機)

需求:Python 3.9+

```bash
# 1. 安裝相依套件
pip install -r requirements.txt

# 2. 啟動儀表板
streamlit run app.py
```

瀏覽器開啟 <http://localhost:8501> 即可。儀表板讀取 repo 內已附的結果檔
(`cv_report_lgb.json`、`submission_with_prob.csv`、`oof_predictions.csv`、
`feature_importance.csv`),即使沒有原始資料也能直接展示。

## ☁️ 部署到 Streamlit Community Cloud

1. 前往 <https://share.streamlit.io> 用 GitHub 登入。
2. **Create app** → 選 repo `Jcsk7049/bitoguard-aml`、分支,**Main file** 填 `app.py`。
3. (可選)Advanced settings 選 Python 3.11 → **Deploy**。

詳細說明見 [`DEPLOY_STREAMLIT_CLOUD.md`](DEPLOY_STREAMLIT_CLOUD.md)。
> 注意:Streamlit Community Cloud 免費版閒置會休眠。若需永不休眠 + 自有網域,
> 可改用自架 + Cloudflare Tunnel,見 [`DEPLOY_CLOUDFLARE_TUNNEL.md`](DEPLOY_CLOUDFLARE_TUNNEL.md)。

---

## 🔧 執行完整 Pipeline

將原始 CSV 放入 `./data/`,再執行:

```bash
# 從特徵工程開始跑完整流程(約 5–10 分鐘)
python main_pipeline.py --start-from feature --csv-dir ./data

# 只跑 XAI 診斷(已有模型時)
python main_pipeline.py --only xai --csv-dir ./data
```

輸出:
- `submission.csv` / `submission_with_prob.csv` — 風險帳戶名單與機率
- `oof_predictions.csv` — 交叉驗證 out-of-fold 預測
- `xai_reports.json` — 每個高風險帳戶的診斷說明

更多用法見 [`QUICK_START.md`](QUICK_START.md) 與 [`STEP_BY_STEP_GUIDE.md`](STEP_BY_STEP_GUIDE.md)。

---

## 📁 專案結構

```
bitoguard-aml/
├── app.py                      # Streamlit 儀表板(部署主程式)
├── main_pipeline.py            # Pipeline 總調度
│
├── 資料擷取
│   ├── bito_api_ingester.py    # 從 BitoPro API 擷取資料
│   ├── bito_data_manager.py    # 資料管理
│   └── ingest_to_s3.py         # 上傳 S3
│
├── 特徵工程 / 圖譜
│   ├── feature_store.py        # 特徵工程
│   ├── athena_graph_hops.sql   # Athena 圖譜跳數查詢
│   └── glue_graph_hops.py      # Glue 圖譜計算
│
├── 模型訓練
│   ├── lgb_pipeline.py         # LightGBM 訓練
│   ├── train_xgboost_script.py # XGBoost 訓練
│   └── train_sagemaker.py      # SageMaker 訓練
│
├── 可解釋性 / 報告
│   ├── xai_bedrock.py          # SHAP + Bedrock 診斷
│   ├── validation_report.py    # 驗證報告
│   ├── visualize.py            # 視覺化
│   └── generate_charts.py      # 圖表產生
│
├── 結果檔(已附)
│   ├── cv_report_lgb.json      # 交叉驗證指標
│   ├── submission_with_prob.csv
│   ├── oof_predictions.csv
│   └── feature_importance.csv
│
├── 部署
│   ├── requirements.txt
│   ├── template.yaml           # AWS SAM 範本
│   ├── Dockerfile              # 容器化
│   └── docker-compose.yml      # 自架 + Cloudflare Tunnel
│
└── 文件(START_HERE.md / QUICK_START.md / *_GUIDE.md ...)
```

## 🧱 技術棧

- **語言 / 框架**:Python、Streamlit
- **建模**:LightGBM、XGBoost、scikit-learn
- **資料**:pandas、numpy、pyarrow
- **視覺化**:Plotly、matplotlib
- **雲端**:AWS S3 / Athena / Glue / SageMaker / Bedrock(SAM 部署)

## 📚 延伸文件

- [`START_HERE.md`](START_HERE.md) — 新手入口
- [`COMPLETE_EXECUTION_GUIDE.md`](COMPLETE_EXECUTION_GUIDE.md) — 完整執行指南
- [`MODEL_PERFORMANCE_ANALYSIS.md`](MODEL_PERFORMANCE_ANALYSIS.md) — 模型表現分析
- [`HOW_TO_RUN.md`](HOW_TO_RUN.md) — 執行方式
