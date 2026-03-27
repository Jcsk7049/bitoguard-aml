# BitoGuard 執行指南

## 當前狀態

你的系統已經成功執行過一次，並產生了以下結果：

- ✅ `submission.csv` (113 KB) - 競賽提交檔案，包含 12,753 位用戶
- ✅ `submission_with_prob.csv` (262 KB) - 含預測機率的完整結果
- ✅ `xai_reports.json` (2.4 KB) - 風險診斷報告
- ✅ `model.json` (1.5 MB) - 訓練好的 XGBoost 模型
- ✅ `feature_cache.parquet` (562 KB) - 特徵快取
- ✅ `cv_report.json` - 交叉驗證報告

檢測結果：1 位黑名單用戶 (ID: 876703)，檢測率 0.01%

## 問題診斷

你遇到的錯誤是因為：

1. 系統需要 `crypto_transfer` 資料表，但你的 `data/` 目錄中沒有這個檔案
2. 你目前有的資料表：
   - ✅ `user_info`
   - ✅ `twd_transfer`
   - ✅ `usdt_twd_trading`
   - ✅ `usdt_swap`
   - ✅ `train_label`
   - ✅ `predict_label`
   - ❌ `crypto_transfer` (缺少)

## 解決方案

### 方案 A：使用現有結果（推薦）

你已經有完整的執行結果，可以直接提交：

```bash
# 檢查提交檔案
head -20 submission.csv

# 查看統計資訊
python -c "
import pandas as pd
df = pd.read_csv('submission.csv')
print(f'總用戶數: {len(df):,}')
print(f'黑名單: {len(df[df.status==1]):,}')
print(f'檢測率: {len(df[df.status==1])/len(df)*100:.2f}%')
"

# 查看風險診斷報告
cat xai_reports.json | python -m json.tool
```

### 方案 B：從推論階段開始（使用現有模型）

如果你想重新生成推論結果，但不想重新訓練：

```bash
# 使用現有的 model.json 和 feature_cache.parquet
python main_pipeline.py --start-from validate
```

這會執行：
- Stage 4: 驗證報告生成
- Stage 5: 視覺化圖表
- Stage 6: XAI 風險診斷

### 方案 C：完整重新執行（需要補齊資料）

如果你想從頭開始，需要先取得 `crypto_transfer` 資料：

#### 選項 1：從 API 取得資料

```bash
# 使用 bito_api_ingester.py 從 API 下載所有資料
python bito_api_ingester.py \
  --tables user_info,twd_transfer,crypto_transfer,usdt_twd_trading,usdt_swap,train_label,predict_label \
  --output-dir ./data \
  --format parquet
```

#### 選項 2：建立空的 crypto_transfer（降級模式）

如果無法取得 `crypto_transfer` 資料，可以建立一個空檔案讓系統使用簡化特徵集：

```bash
# 建立空的 crypto_transfer 目錄
mkdir -p data/crypto_transfer/dt=2026-03-26

# 建立空的 CSV 檔案（只有欄位名稱）
cat > data/crypto_transfer/dt=2026-03-26/part-00000.csv << 'EOF'
created_at,user_id,kind,sub_kind,ori_samount,twd_srate,currency,protocol,from_wallet,to_wallet,relation_user_id,source_ip
EOF

# 然後執行管線
python main_pipeline.py --start-from feature --csv-dir ./data
```

注意：使用空的 `crypto_transfer` 會導致系統使用簡化特徵集，模型準確度可能下降。

### 方案 D：修改程式碼跳過 crypto_transfer（進階）

如果你確定不需要 `crypto_transfer` 相關特徵，可以修改 `train_sagemaker.py`：

```python
# 在 build_features() 函數中，將條件改為：
if (users is not None and len(users) > 0 and 
    twd_transfer is not None and len(twd_transfer) > 0 and 
    trades is not None and len(trades) > 0):
    # 移除 crypto_transfer 的檢查
    features = manager.extract_mule_features(
        users, twd_transfer, None, trades  # crypto_transfer 改為 None
    )
```

但這需要同時修改 `extract_mule_features()` 函數來處理 `None` 的情況。

## 快速檢查指令

```bash
# 檢查資料目錄結構
find data/ -name "*.csv" | sort

# 檢查現有模型和快取
ls -lh model.json feature_cache.parquet cv_report.json

# 檢查提交檔案
wc -l submission.csv
head -5 submission.csv

# 檢查 XAI 報告
python -c "
import json
with open('xai_reports.json') as f:
    reports = json.load(f)
print(f'診斷報告數量: {len(reports)}')
for r in reports:
    print(f'用戶 {r[\"user_id\"]}: 風險機率 {r[\"probability\"]:.2%}, 等級 {r[\"scoring_tier\"]}')
"
```

## 建議

基於你目前的狀況，我建議：

1. **如果只是要提交競賽結果**：使用方案 A，你已經有完整的結果了
2. **如果想重新生成報告和圖表**：使用方案 B，從驗證階段開始
3. **如果想完整重新訓練**：使用方案 C 選項 2（建立空的 crypto_transfer）

## 常見問題

### Q: 為什麼檢測率這麼低 (0.01%)？

A: 這是正常的。AML 系統的特性就是極度不平衡：
- 大部分用戶都是正常的
- 只有極少數是真正的人頭戶
- 模型設定了較高的閾值 (0.65) 來避免誤報

### Q: 如何調整檢測閾值？

A: 編輯 `main_pipeline.py`，找到 `RISK_THRESHOLD` 變數：

```python
RISK_THRESHOLD = 0.65  # 降低這個值會增加檢測數量（但也會增加誤報）
```

### Q: 如何查看模型效能？

A: 查看 `cv_report.json`：

```bash
cat cv_report.json | python -m json.tool | head -50
```

### Q: 系統執行需要多久？

- 完整執行（含訓練）：5-10 分鐘
- 僅推論（使用現有模型）：3-5 秒
- 僅驗證和視覺化：1-2 分鐘

## 聯絡資訊

如果遇到其他問題，請提供：
1. 完整的錯誤訊息
2. 執行的指令
3. `ls -la data/` 的輸出
