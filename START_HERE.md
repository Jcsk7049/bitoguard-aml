# 🚀 從這裡開始

## 快速開始（3 個步驟）

### 步驟 1: 執行互動式示範

```bash
bash interactive_demo.sh
```

這會帶你逐步看到：
- 原始資料
- 特徵工程
- 模型訓練
- 預測結果
- SHAP 解釋

**時間：約 5 分鐘**

---

### 步驟 2: 查看完整報告

```bash
cat COMPLETE_DATA_FLOW_REPORT.md | less
```

或在編輯器中打開 `COMPLETE_DATA_FLOW_REPORT.md`

這份報告包含：
- 完整的資料流圖
- 每一步的詳細解釋
- 為什麼檢測率低
- 如何提高檢測率

---

### 步驟 3: 執行完整分析

```bash
python data_flow_analysis.py
```

這會生成完整的分析報告，包括：
- 原始資料統計
- 特徵工程過程
- 用戶詳細分析
- 模型訓練過程
- 預測結果分析
- XAI 診斷報告

**時間：約 30 秒**

---

## 📚 詳細指南

如果你想深入了解每一步，請查看：

```bash
cat STEP_BY_STEP_GUIDE.md | less
```

這份指南包含：
- 9 個詳細步驟
- 每一步的指令
- 預期輸出
- 常見問題解答

---

## 🎯 快速命令參考

### 查看原始資料

```bash
# 訓練標籤
head -20 data/train_label/dt=2026-03-26/part-00000.csv

# 台幣轉帳記錄
head -20 data/twd_transfer/dt=2026-03-26/part-00000.csv

# 統計資訊
python -c "
import pandas as pd
df = pd.read_csv('data/train_label/dt=2026-03-26/part-00000.csv')
print(f'總樣本: {len(df):,}')
print(f'人頭戶: {df[\"status\"].sum():,} ({df[\"status\"].mean()*100:.2f}%)')
"
```

### 查看特定用戶

```bash
# 用戶 876703 的交易記錄
python -c "
import pandas as pd
df = pd.read_csv('data/twd_transfer/dt=2026-03-26/part-00000.csv')
user = df[df['user_id'] == 876703]
print(f'交易次數: {len(user)}')
print(f'入金: {(user[\"kind\"] == 0).sum()}')
print(f'出金: {(user[\"kind\"] == 1).sum()}')
"
```

### 查看預測結果

```bash
# 統計
python -c "
import pandas as pd
sub = pd.read_csv('submission_with_prob.csv')
print(f'總用戶: {len(sub):,}')
print(f'檢測: {sub[\"status\"].sum()} ({sub[\"status\"].mean()*100:.2f}%)')
"

# 高風險用戶
python -c "
import pandas as pd
sub = pd.read_csv('submission_with_prob.csv')
high = sub[sub['probability'] > 0.5].sort_values('probability', ascending=False)
print(high[['user_id', 'probability', 'status']])
"
```

### 查看 SHAP 解釋

```bash
python -c "
import json
with open('xai_reports.json', encoding='utf-8') as f:
    reports = json.load(f)
for r in reports:
    print(f'用戶 {r[\"user_id\"]}: {r[\"probability\"]:.2%}')
    for c in r['shap_contributions']:
        print(f'  {c[\"feature_label\"]}: {c[\"contribution_pct\"]:.1f}%')
"
```

---

## 🔧 實驗

### 實驗 1: 降低閾值

編輯 `main_pipeline.py`，找到這一行（約在第 60 行）：

```python
RISK_THRESHOLD = 0.65
```

改為：

```python
RISK_THRESHOLD = 0.5
```

然後執行：

```bash
python main_pipeline.py --start-from validate --csv-dir ./data
```

查看結果：

```bash
python -c "
import pandas as pd
sub = pd.read_csv('submission.csv')
print(f'檢測數量: {sub[\"status\"].sum()}')
"
```

---

### 實驗 2: 調整超參數

編輯 `train_sagemaker.py`，找到 `HYPERPARAMS`（約在第 50 行）：

```python
HYPERPARAMS = {
    ...
    'scale_pos_weight': 30,    # 從 1.0 改為 30
    'max_depth': 6,             # 從 8 改為 6
    'eta': 0.1,                 # 從 0.05 改為 0.1
    ...
}
```

然後重新訓練：

```bash
python main_pipeline.py --start-from train --csv-dir ./data
```

---

## 📊 系統狀態檢查

隨時執行這個命令查看系統狀態：

```bash
python check_status.py
```

---

## 🆘 遇到問題？

### 問題 1: 找不到 CSV 檔案

```bash
# 檢查資料目錄
ls -la data/

# 應該看到這些目錄：
# - train_label/
# - predict_label/
# - twd_transfer/
# - usdt_twd_trading/
# - usdt_swap/
# - user_info/
```

### 問題 2: 模組未安裝

```bash
pip install pandas numpy xgboost scikit-learn pyarrow
```

### 問題 3: 編碼錯誤

如果看到 `UnicodeEncodeError`，使用：

```bash
export PYTHONIOENCODING=utf-8
python data_flow_analysis.py
```

---

## 📖 文件索引

- `START_HERE.md` ← 你在這裡
- `STEP_BY_STEP_GUIDE.md` - 詳細的逐步指南
- `COMPLETE_DATA_FLOW_REPORT.md` - 完整資料流報告
- `HOW_TO_RUN.md` - 執行指南
- `check_status.py` - 系統狀態檢查工具
- `data_flow_analysis.py` - 完整分析工具
- `interactive_demo.sh` - 互動式示範腳本

---

## 🎉 開始吧！

執行這個命令開始：

```bash
bash interactive_demo.sh
```

祝你探索愉快！ 🚀
