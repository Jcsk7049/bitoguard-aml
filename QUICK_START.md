# BitoGuard 快速開始指南

## 🚀 最快 5 分鐘上手

### 1️⃣ 檢查環境
```bash
python --version  # 需要 3.8+
python -c "import xgboost, pandas; print('✓')"
```

### 2️⃣ 準備資料
確保 `./data/` 目錄下有 CSV 檔案

### 3️⃣ 一鍵執行
```bash
python main_pipeline.py --start-from feature --csv-dir ./data
```

### 4️⃣ 查看結果
```bash
# 黑名單用戶
head -5 submission.csv

# 診斷報告
cat xai_reports.json | python -m json.tool
```

---

## 📋 常用命令

### 完整執行（推薦首次）
```bash
python main_pipeline.py --start-from feature --csv-dir ./data
```
⏱️ 時間：5-10 分鐘

### 快速推論（已有模型）
```bash
python main_pipeline.py --only xai --csv-dir ./data
```
⏱️ 時間：3-5 秒

### 調整超參數重新訓練
```bash
python main_pipeline.py --reset --start-from feature \
  --csv-dir ./data \
  --max-depth 8 --eta 0.05 --num-round 1000
```
⏱️ 時間：5-10 分鐘

### 只執行特定階段
```bash
# 特徵工程
python main_pipeline.py --only feature --csv-dir ./data

# 模型訓練
python main_pipeline.py --only train --csv-dir ./data

# XAI 診斷
python main_pipeline.py --only xai --csv-dir ./data
```

### 從中斷點繼續
```bash
# 從訓練階段開始（跳過特徵工程）
python main_pipeline.py --start-from train --csv-dir ./data

# 從 XAI 階段開始
python main_pipeline.py --start-from xai --csv-dir ./data
```

---

## 📊 輸出檔案

| 檔案 | 說明 | 用途 |
|------|------|------|
| `submission.csv` | 競賽提交檔 | 上傳到競賽平台 |
| `submission_with_prob.csv` | 含預測機率 | 分析和調整 |
| `xai_reports.json` | 風險診斷報告 | 人工審核 |
| `cv_report.json` | 模型性能 | 評估模型 |
| `model.json` | 訓練好的模型 | 推論用 |
| `feature_cache.parquet` | 特徵快取 | 加速重新訓練 |

---

## 🔍 驗證結果

### 檢查黑名單用戶
```bash
python -c "
import pandas as pd
df = pd.read_csv('submission.csv')
flagged = df[df['status'] == 1]
print(f'黑名單用戶: {len(flagged):,}')
print(flagged['user_id'].tolist())
"
```

### 檢查模型性能
```bash
python -c "
import json
with open('cv_report.json') as f:
    cv = json.load(f)
s = cv['summary']
print(f\"Precision: {s['precision_mean']:.2%}\")
print(f\"Recall: {s['recall_mean']:.2%}\")
print(f\"F1: {s['f1_mean']:.4f}\")
"
```

### 檢查診斷報告
```bash
python -c "
import json
with open('xai_reports.json') as f:
    reports = json.load(f)
for r in reports:
    print(f\"user_id {r['user_id']}: {r['probability']*100:.1f}% ({r['scoring_tier']})\")
"
```

---

## ⚙️ 超參數調整

### 提高檢測率（捕捉更多風險用戶）
```bash
--max-depth 8 --eta 0.05 --num-round 1000
```

### 降低誤報（更保守）
```bash
--max-depth 4 --eta 0.2 --num-round 300
```

### 平衡性能
```bash
--max-depth 6 --eta 0.1 --num-round 500
```

---

## 🐛 常見問題

### Q: 找不到資料檔案？
```bash
# 檢查資料目錄
ls -R ./data/
```

### Q: 模型性能低？
```bash
# 重新訓練，調整超參數
python main_pipeline.py --reset --start-from feature \
  --csv-dir ./data --max-depth 8 --eta 0.05 --num-round 1000
```

### Q: AWS 憑證錯誤？
```bash
# 跳過 download 階段，直接用 XAI
python main_pipeline.py --only xai --csv-dir ./data
```

### Q: 記憶體不足？
```bash
# 減少迭代次數
python main_pipeline.py --start-from feature \
  --csv-dir ./data --num-round 300 --cv-folds 3
```

---

## 📈 性能指標

| 指標 | 目標 | 當前 |
|------|------|------|
| Precision | > 10% | 8.69% |
| Recall | > 0.5% | 0.36% |
| F1 Score | > 0.01 | 0.0070 |
| 黑名單檢測 | > 1 | 2 |

---

## 🎯 下一步

1. ✅ 執行完整管線
2. ✅ 驗證輸出檔案
3. ✅ 上傳 `submission.csv` 到競賽平台
4. 📊 監控預測性能
5. 🔄 根據反饋調整模型

---

## 📚 詳細文檔

- `COMPLETE_EXECUTION_GUIDE.md` - 完整執行指南
- `MODEL_PERFORMANCE_ANALYSIS.md` - 性能分析
- `PIPELINE_FIX_SUMMARY.md` - 修復摘要
- `SAGEMAKER_FIX_GUIDE.md` - SageMaker 修復指南

---

## 💡 提示

- 首次執行使用 `--start-from feature`
- 後續快速推論使用 `--only xai`
- 調試時使用 `--only <stage>`
- 中斷後使用 `--start-from <stage>` 繼續
- 完全重新開始使用 `--reset`

---

**準備好了嗎？執行這個命令開始：**
```bash
python main_pipeline.py --start-from feature --csv-dir ./data
```
