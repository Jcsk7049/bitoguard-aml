# BitoGuard 完整執行指南

## 系統概述

BitoGuard 是一個 AML（反洗錢）自動化合規平台，用於偵測人頭戶、洗錢路徑及機房集體詐騙。

**核心流程：**
```
資料載入 → 特徵工程 → 模型訓練 → 驗證評估 → 視覺化 → XAI診斷 → 提交
```

---

## 前置準備

### 1. 環境檢查

```bash
# 檢查 Python 版本（需要 3.8+）
python --version

# 檢查必要套件
python -c "import xgboost, pandas, numpy, shap; print('✓ 所有套件已安裝')"
```

### 2. 資料準備

確保 `./data/` 目錄下有以下 CSV 檔案：
```
./data/
├── user_info/dt=2026-03-26/part-00000.csv          # 用戶資訊
├── twd_transfer/dt=2026-03-26/part-00000.csv       # 台幣轉帳
├── usdt_swap/dt=2026-03-26/part-00000.csv          # USDT 交換
├── usdt_twd_trading/dt=2026-03-26/part-00000.csv   # USDT-TWD 交易
├── train_label/dt=2026-03-26/part-00000.csv        # 訓練標籤（黑名單）
└── predict_label/dt=2026-03-26/part-00000.csv      # 預測標籤（待評估用戶）
```

### 3. 環境變數設置（可選）

```bash
# 設置 AWS 區域
export AWS_DEFAULT_REGION=us-west-2

# 設置超參數（可選，CLI 會覆蓋）
export XGBOOST_MAX_DEPTH=8
export XGBOOST_ETA=0.05
export XGBOOST_NUM_ROUND=1000
export CV_FOLDS=5
```

---

## 完整執行流程

### 方案 A：完整端到端執行（推薦）

**適用場景：** 第一次運行或需要完整重新訓練

```bash
# 清除舊狀態
rm -f pipeline_state.json

# 運行完整管線（從特徵工程開始）
python main_pipeline.py \
  --start-from feature \
  --csv-dir ./data \
  --max-depth 8 \
  --eta 0.05 \
  --num-round 1000 \
  --cv-folds 5
```

**執行時間：** ~5-10 分鐘

**輸出檔案：**
- `submission.csv` - 競賽提交檔（user_id, status）
- `submission_with_prob.csv` - 含預測機率
- `xai_reports.json` - 風險診斷報告
- `cv_report.json` - 交叉驗證報告
- `feature_cache.parquet` - 特徵快取
- `model.json` - 訓練好的 XGBoost 模型
- `pipeline_state.json` - 管線狀態（用於斷點續跑）

---

### 方案 B：快速推論（已有模型）

**適用場景：** 模型已訓練，只需要生成新的預測

```bash
# 直接運行 XAI 階段
python main_pipeline.py \
  --start-from xai \
  --csv-dir ./data
```

**執行時間：** ~3-5 秒

**前置條件：**
- `feature_cache.parquet` 存在
- `model.json` 存在
- `pipeline_state.json` 存在

---

### 方案 C：分階段執行（調試用）

**適用場景：** 需要調試或修改特定階段

#### Stage 1：特徵工程
```bash
python main_pipeline.py \
  --only feature \
  --csv-dir ./data
```
**輸出：** `feature_cache.parquet`

#### Stage 2：模型訓練
```bash
python main_pipeline.py \
  --only train \
  --csv-dir ./data \
  --max-depth 8 \
  --eta 0.05 \
  --num-round 1000
```
**輸出：** `model.json`, `cv_report.json`

#### Stage 4：驗證報告
```bash
python main_pipeline.py \
  --only validate \
  --csv-dir ./data
```
**輸出：** `validation_report.json`

#### Stage 5：視覺化
```bash
python main_pipeline.py \
  --only visualize \
  --csv-dir ./data
```
**輸出：** `plots/` 目錄（PR 曲線、SHAP 圖等）

#### Stage 6：XAI 診斷 + 提交
```bash
python main_pipeline.py \
  --only xai \
  --csv-dir ./data
```
**輸出：** `submission.csv`, `xai_reports.json`

---

### 方案 D：從特定階段繼續（斷點續跑）

**適用場景：** 管線中途中斷，需要從某個階段重新開始

```bash
# 從訓練階段開始（跳過特徵工程）
python main_pipeline.py \
  --start-from train \
  --csv-dir ./data

# 從驗證階段開始
python main_pipeline.py \
  --start-from validate \
  --csv-dir ./data

# 從 XAI 階段開始
python main_pipeline.py \
  --start-from xai \
  --csv-dir ./data
```

---

## 超參數調整

### 預設超參數
```
max_depth: 6          # 樹的最大深度
eta: 0.1              # 學習率
gamma: 0.0            # 最小損失減少
num_round: 500        # 迭代次數
cv_folds: 5           # 交叉驗證折數
```

### 改進檢測率的調整
```bash
# 更敏感的模型（捕捉更多風險用戶）
python main_pipeline.py \
  --start-from feature \
  --csv-dir ./data \
  --max-depth 8 \
  --eta 0.05 \
  --num-round 1000 \
  --cv-folds 5
```

### 降低誤報的調整
```bash
# 更保守的模型（減少誤報）
python main_pipeline.py \
  --start-from feature \
  --csv-dir ./data \
  --max-depth 4 \
  --eta 0.2 \
  --num-round 300 \
  --cv-folds 5
```

---

## 執行結果驗證

### 1. 檢查提交檔案格式

```bash
# 查看 submission.csv 前 5 行
head -5 submission.csv

# 統計黑名單用戶
python -c "
import pandas as pd
df = pd.read_csv('submission.csv')
flagged = df[df['status'] == 1]
print(f'總用戶: {len(df):,}')
print(f'黑名單: {len(flagged):,}')
print(f'檢測率: {len(flagged)/len(df)*100:.2f}%')
"
```

### 2. 檢查診斷報告

```bash
# 查看 xai_reports.json 結構
python -c "
import json
with open('xai_reports.json') as f:
    reports = json.load(f)
print(f'診斷報告數: {len(reports)}')
for r in reports:
    print(f\"  - user_id {r['user_id']}: {r['probability']*100:.2f}% ({r['scoring_tier']})\")
"
```

### 3. 檢查模型性能

```bash
# 查看交叉驗證結果
python -c "
import json
with open('cv_report.json') as f:
    cv = json.load(f)
summary = cv['summary']
print(f\"Precision: {summary['precision_mean']:.4f} ± {summary['precision_std']:.4f}\")
print(f\"Recall: {summary['recall_mean']:.4f} ± {summary['recall_std']:.4f}\")
print(f\"F1 Score: {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}\")
"
```

---

## 常見問題排查

### 問題 1：找不到資料檔案

**症狀：** `ValueError: 無法載入任何資料表`

**解決方案：**
```bash
# 檢查資料目錄結構
ls -R ./data/

# 確保 CSV 檔案存在
find ./data -name "*.csv" | head -10
```

### 問題 2：模型性能低（Recall < 1%）

**症狀：** 黑名單用戶檢測率很低

**解決方案：**
1. 增加 `max_depth`（從 6 → 8）
2. 降低 `eta`（從 0.1 → 0.05）
3. 增加 `num_round`（從 500 → 1000）

```bash
python main_pipeline.py --reset --start-from feature \
  --csv-dir ./data \
  --max-depth 8 --eta 0.05 --num-round 1000
```

### 問題 3：AWS 憑證錯誤

**症狀：** `NoCredentialsError: Unable to locate credentials`

**解決方案：**
- 在本地環境，跳過 `download` 階段
- 直接使用 `--start-from xai` 或 `--only xai`

### 問題 4：記憶體不足

**症狀：** `MemoryError` 或進程被殺死

**解決方案：**
1. 減少 `cv_folds`（從 5 → 3）
2. 減少 `num_round`（從 1000 → 500）
3. 增加系統記憶體或使用更小的資料集

---

## 完整執行腳本

### 一鍵執行（推薦）

```bash
#!/bin/bash
set -e

echo "=========================================="
echo "BitoGuard 完整管線執行"
echo "=========================================="

# 清除舊狀態
rm -f pipeline_state.json

# 執行完整管線
python main_pipeline.py \
  --start-from feature \
  --csv-dir ./data \
  --max-depth 8 \
  --eta 0.05 \
  --num-round 1000 \
  --cv-folds 5

# 驗證輸出
echo ""
echo "=========================================="
echo "執行結果驗證"
echo "=========================================="
python verify_output.py

echo ""
echo "✅ 完整管線執行成功！"
echo "準備提交的檔案："
echo "  - submission.csv"
echo "  - submission_with_prob.csv"
echo "  - xai_reports.json"
```

保存為 `run_pipeline.sh`，然後執行：
```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

---

## 管線階段詳解

| 階段 | 名稱 | 輸入 | 輸出 | 時間 |
|------|------|------|------|------|
| 1 | 特徵工程 | CSV 檔案 | feature_cache.parquet | 3-5s |
| 2 | 模型訓練 | feature_cache | model.json, cv_report.json | 2-3m |
| 3 | 下載模型 | SageMaker | model.json | 1-2m |
| 4 | 驗證報告 | model.json | validation_report.json | 1-2m |
| 5 | 視覺化 | model.json | plots/ | 30-60s |
| 6 | XAI 診斷 | model.json | submission.csv, xai_reports.json | 3-5s |

---

## 最佳實踐

### 1. 首次運行
```bash
# 完整執行，不跳過任何階段
python main_pipeline.py --start-from feature --csv-dir ./data
```

### 2. 調試特定階段
```bash
# 只執行某個階段
python main_pipeline.py --only train --csv-dir ./data
```

### 3. 快速迭代
```bash
# 特徵已快取，只重新訓練
python main_pipeline.py --start-from train --csv-dir ./data
```

### 4. 生成最終提交
```bash
# 使用已訓練的模型生成提交檔
python main_pipeline.py --only xai --csv-dir ./data
```

### 5. 合規檢核（部署前）
```bash
# 檢查硬編碼憑證、S3 安全設置、PII 過濾
python check_compliance.py
```

---

## 輸出檔案說明

### submission.csv
```
user_id,status
967903,0
204939,0
876703,1
...
```
- `status=1`：黑名單用戶（需要進一步調查）
- `status=0`：正常用戶

### submission_with_prob.csv
```
user_id,probability,status
967903,0.0234,0
876703,0.6643,1
...
```
- `probability`：模型預測的風險機率（0-1）
- 用於調整決策閾值

### xai_reports.json
```json
[
  {
    "user_id": 876703,
    "probability": 0.6643,
    "scoring_tier": "BOUNDARY",
    "action": {
      "primary_action": "CALL_VERIFY",
      "steps": ["限制提領", "電話訪查", "補件要求"]
    },
    "shap_contributions": [...]
  }
]
```
- 詳細的風險診斷報告
- 包含 SHAP 特徵貢獻度分析
- 建議的風控行動

---

## 下一步

1. **提交競賽**
   - 上傳 `submission.csv` 到競賽平台

2. **優化模型**
   - 調整超參數改進性能
   - 增加特徵工程（啟用 Glue 圖分析）

3. **部署到 AWS**
   - 使用 SAM 部署 Lambda 函數
   - 設置 DynamoDB 診斷書存儲
   - 配置 SNS 告警

4. **監控和維護**
   - 定期重新訓練模型
   - 監控預測性能
   - 收集反饋進行增量學習

---

## 支援

如有問題，請檢查：
1. `pipeline.log` - 執行日誌
2. `cv_report.json` - 模型性能
3. `xai_reports.json` - 診斷詳情
