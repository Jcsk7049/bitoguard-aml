# BitoGuard 逐步執行指南

跟著這個指南，你將親自執行整個系統，看到從原始資料到最終預測的每一步。

---

## 🎯 目標

你將學會：
1. 查看原始資料長什麼樣
2. 理解特徵工程如何轉換資料
3. 觀察模型訓練過程
4. 分析預測結果
5. 理解為什麼某些用戶被標記為高風險

---

## 📋 前置準備

### 檢查你的環境

在 SageMaker Notebook Terminal 中執行：

```bash
# 1. 確認你在正確的目錄
pwd
# 應該顯示: /home/ec2-user/SageMaker

# 2. 確認 Python 版本
python --version
# 應該是 Python 3.x

# 3. 確認必要套件已安裝
python -c "import pandas, numpy, xgboost, sklearn; print('All packages OK')"
```

---

## 步驟 1️⃣：查看原始資料

### 1.1 查看訓練標籤（真實答案）

```bash
# 查看前 20 行
head -20 data/train_label/dt=2026-03-26/part-00000.csv
```

**你會看到：**
```
user_id,status
930627,1    ← 這是人頭戶
995754,1    ← 這是人頭戶
196191,1    ← 這是人頭戶
...
```

**問題：總共有多少人頭戶？**

```bash
# 統計人頭戶數量
python -c "
import pandas as pd
df = pd.read_csv('data/train_label/dt=2026-03-26/part-00000.csv')
print(f'總樣本數: {len(df):,}')
print(f'人頭戶數: {df[\"status\"].sum():,}')
print(f'人頭戶比例: {df[\"status\"].mean() * 100:.2f}%')
"
```

**預期輸出：**
```
總樣本數: 51,017
人頭戶數: 1,640
人頭戶比例: 3.21%
```

---

### 1.2 查看台幣轉帳記錄

```bash
# 查看前 10 筆交易
head -10 data/twd_transfer/dt=2026-03-26/part-00000.csv
```

**你會看到：**
```
created_at,user_id,kind,ori_samount,source_ip,bank_code,status
2025-10-15 11:36:07+00:00,37,0,10000.0,,,
2025-10-29 18:13:15+00:00,216,0,1000.0,,,
```

**欄位說明：**
- `kind=0`: 入金（用戶存錢進來）
- `kind=1`: 出金（用戶提錢出去）
- `ori_samount`: 金額（台幣）

**問題：總共有多少筆交易？**

```bash
python -c "
import pandas as pd
df = pd.read_csv('data/twd_transfer/dt=2026-03-26/part-00000.csv')
print(f'總交易筆數: {len(df):,}')
print(f'入金筆數: {(df[\"kind\"] == 0).sum():,}')
print(f'出金筆數: {(df[\"kind\"] == 1).sum():,}')
print(f'涉及用戶數: {df[\"user_id\"].nunique():,}')
"
```

---

### 1.3 查看特定用戶的交易記錄

讓我們看看被標記為高風險的用戶 876703：

```bash
python -c "
import pandas as pd

# 載入交易記錄
df = pd.read_csv('data/twd_transfer/dt=2026-03-26/part-00000.csv')

# 篩選用戶 876703
user_df = df[df['user_id'] == 876703].copy()

# 轉換時間格式
user_df['created_at'] = pd.to_datetime(user_df['created_at'])
user_df['hour'] = user_df['created_at'].dt.hour

# 標記類型
user_df['type'] = user_df['kind'].map({0: '入金', 1: '出金'})

# 排序並顯示
user_df = user_df.sort_values('created_at', ascending=False)
print(f'用戶 876703 的交易記錄（最近 10 筆）:')
print('=' * 80)
for _, row in user_df.head(10).iterrows():
    time = row['created_at'].strftime('%Y-%m-%d %H:%M')
    type_ = row['type']
    amount = row['ori_samount']
    hour = row['hour']
    night = '🌙 深夜' if (hour >= 22 or hour <= 6) else ''
    print(f'{time}  {type_:4s}  {amount:>10,.0f} 元  {night}')

# 統計
print('\\n統計:')
print(f'總交易次數: {len(user_df)}')
print(f'入金次數: {(user_df[\"kind\"] == 0).sum()}')
print(f'出金次數: {(user_df[\"kind\"] == 1).sum()}')
night_count = ((user_df['hour'] >= 22) | (user_df['hour'] <= 6)).sum()
print(f'深夜交易: {night_count} 筆 ({night_count/len(user_df)*100:.1f}%)')
"
```

**你會看到什麼？**
- 這個用戶有 25 筆入金
- 0 筆出金（只進不出，可疑！）
- 32% 的交易發生在深夜（22:00-06:00）

---

## 步驟 2️⃣：特徵工程

現在讓我們看看系統如何從原始交易記錄提取特徵。

### 2.1 查看已經計算好的特徵

```bash
python -c "
import pandas as pd

# 載入特徵快取
features = pd.read_parquet('feature_cache.parquet')

print(f'總用戶數: {len(features):,}')
print(f'\\n特徵列表:')
for col in features.columns:
    if col != 'user_id':
        print(f'  - {col}')

print(f'\\n特徵統計:')
print(features[['twd_deposit_count', 'twd_withdraw_count', 'night_tx_ratio']].describe())
"
```

### 2.2 查看特定用戶的特徵

```bash
python -c "
import pandas as pd

features = pd.read_parquet('feature_cache.parquet')

# 查看用戶 876703
user_feat = features[features['user_id'] == 876703].iloc[0]

print('用戶 876703 的特徵:')
print('=' * 80)
print(f'台幣入金次數: {user_feat[\"twd_deposit_count\"]:.0f} 次')
print(f'台幣出金次數: {user_feat[\"twd_withdraw_count\"]:.0f} 次')
print(f'深夜交易比例: {user_feat[\"night_tx_ratio\"]:.2%}')

# 比較平均值
avg_deposit = features['twd_deposit_count'].mean()
avg_withdraw = features['twd_withdraw_count'].mean()
avg_night = features['night_tx_ratio'].mean()

print(f'\\n與平均值比較:')
print(f'入金次數: {user_feat[\"twd_deposit_count\"]:.0f} vs 平均 {avg_deposit:.2f} (高 {user_feat[\"twd_deposit_count\"]/avg_deposit:.1f}x)')
print(f'出金次數: {user_feat[\"twd_withdraw_count\"]:.0f} vs 平均 {avg_withdraw:.2f}')
print(f'深夜交易: {user_feat[\"night_tx_ratio\"]:.2%} vs 平均 {avg_night:.2%}')
"
```

**你會看到：**
- 用戶 876703 的入金次數是平均值的 10 倍！
- 出金次數是 0（正常用戶平均 0.52 次）
- 深夜交易比例高於平均

---

## 步驟 3️⃣：模型訓練

### 3.1 查看訓練配置

```bash
python -c "
import json

with open('cv_report.json') as f:
    cv = json.load(f)

print('模型訓練配置:')
print('=' * 80)
print(f'交叉驗證折數: {cv[\"cv_config\"][\"n_splits\"]}')
print(f'驗證策略: {cv[\"cv_config\"][\"strategy\"]}')

print(f'\\n超參數:')
for key, val in cv['hyperparams'].items():
    print(f'  {key}: {val}')
"
```

### 3.2 查看訓練效能

```bash
python -c "
import json
import numpy as np

with open('cv_report.json') as f:
    cv = json.load(f)

print('交叉驗證結果:')
print('=' * 80)
print(f'{'Fold':<6} {'Precision':<12} {'Recall':<12} {'F1':<12}')
print('-' * 80)

for fold in cv['folds']:
    print(f'{fold[\"fold\"]:<6} {fold[\"precision\"]:<12.4f} {fold[\"recall\"]:<12.4f} {fold[\"f1\"]:<12.4f}')

# 計算平均
avg_prec = np.mean([f['precision'] for f in cv['folds']])
avg_rec = np.mean([f['recall'] for f in cv['folds']])
avg_f1 = np.mean([f['f1'] for f in cv['folds']])

print('-' * 80)
print(f'{'平均':<6} {avg_prec:<12.4f} {avg_rec:<12.4f} {avg_f1:<12.4f}')

print(f'\\n💡 為什麼 F1 這麼低？')
print(f'   因為人頭戶只佔 3.21%，這是極度不平衡的資料集。')
print(f'   模型非常保守，寧可漏報也不誤報。')
"
```

---

## 步驟 4️⃣：模型預測

### 4.1 查看預測結果

```bash
python -c "
import pandas as pd

# 載入預測結果
sub = pd.read_csv('submission_with_prob.csv')

print('預測結果統計:')
print('=' * 80)
print(f'總用戶數: {len(sub):,}')
print(f'預測為人頭戶: {sub[\"status\"].sum():,}')
print(f'預測為正常: {(sub[\"status\"] == 0).sum():,}')
print(f'檢測率: {sub[\"status\"].mean() * 100:.2f}%')

print(f'\\n風險機率分布:')
bins = [0, 0.3, 0.5, 0.65, 0.75, 0.9, 1.0]
labels = ['極低 (0-0.3)', '低 (0.3-0.5)', '中 (0.5-0.65)', 
          '邊界 (0.65-0.75)', '高 (0.75-0.9)', '極高 (0.9-1.0)']
sub['risk_level'] = pd.cut(sub['probability'], bins=bins, labels=labels)
print(sub['risk_level'].value_counts().sort_index())

print(f'\\n高風險用戶（機率 > 0.5）:')
print('-' * 80)
high_risk = sub[sub['probability'] > 0.5].sort_values('probability', ascending=False)
print(f'{'User ID':<12} {'Probability':<15} {'Status':<10}')
print('-' * 80)
for _, row in high_risk.head(10).iterrows():
    status = '人頭戶' if row['status'] == 1 else '正常'
    print(f'{row[\"user_id\"]:<12} {row[\"probability\"]:<15.4f} {status:<10}')
"
```

---

## 步驟 5️⃣：可解釋 AI（SHAP 分析）

### 5.1 查看 XAI 診斷報告

```bash
python -c "
import json

with open('xai_reports.json', encoding='utf-8') as f:
    reports = json.load(f)

print(f'XAI 診斷報告數量: {len(reports)}')
print('=' * 80)

for i, report in enumerate(reports, 1):
    print(f'\\n【報告 {i}】用戶 {report[\"user_id\"]}')
    print('-' * 80)
    print(f'風險機率: {report[\"probability\"]:.2%}')
    print(f'風險等級: {report[\"scoring_tier\"]}')
    
    print(f'\\nSHAP 特徵貢獻度（為什麼被標記？）:')
    for contrib in report['shap_contributions']:
        feature = contrib['feature_label']
        value = contrib['feature_value']
        shap_val = contrib['shap_value']
        pct = contrib['contribution_pct']
        print(f'  • {feature}: {value}')
        print(f'    → SHAP 值: {shap_val:+.4f} ({pct:.1f}% 貢獻度)')
    
    print(f'\\n建議行動:')
    for step in report['action']['steps']:
        print(f'  • {step}')
"
```

---

## 步驟 6️⃣：完整透明化分析

現在執行完整的分析工具：

```bash
python data_flow_analysis.py
```

這會顯示：
1. 原始資料統計
2. 特徵工程過程
3. 用戶 876703 和 140471 的詳細分析
4. 模型訓練過程
5. 預測結果分析
6. XAI 診斷報告

---

## 步驟 7️⃣：實驗：調整閾值

讓我們試試降低閾值，看看會檢測到多少用戶。

### 7.1 查看不同閾值下的檢測數量

```bash
python -c "
import pandas as pd

sub = pd.read_csv('submission_with_prob.csv')

print('不同閾值下的檢測數量:')
print('=' * 80)
print(f'{'閾值':<10} {'檢測數量':<12} {'檢測率':<12}')
print('-' * 80)

for threshold in [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]:
    count = (sub['probability'] >= threshold).sum()
    rate = count / len(sub) * 100
    print(f'{threshold:<10.2f} {count:<12} {rate:<12.2f}%')

print(f'\\n💡 當前閾值: 0.65')
print(f'   如果降低到 0.5，會檢測到 {(sub[\"probability\"] >= 0.5).sum()} 個用戶')
"
```

### 7.2 查看閾值 0.5 下的用戶列表

```bash
python -c "
import pandas as pd

sub = pd.read_csv('submission_with_prob.csv')

# 閾值 0.5
threshold = 0.5
flagged = sub[sub['probability'] >= threshold].sort_values('probability', ascending=False)

print(f'閾值 {threshold} 下的高風險用戶:')
print('=' * 80)
print(f'{'User ID':<12} {'Probability':<15}')
print('-' * 80)
for _, row in flagged.iterrows():
    print(f'{row[\"user_id\"]:<12} {row[\"probability\"]:<15.4f}')
"
```

---

## 步驟 8️⃣：重新訓練（可選）

如果你想從頭開始重新訓練模型：

### 8.1 清除舊結果

```bash
# 備份現有結果
mkdir -p backup
cp submission.csv backup/
cp model.json backup/
cp feature_cache.parquet backup/

# 清除舊結果（可選）
# rm model.json feature_cache.parquet submission.csv
```

### 8.2 從特徵工程開始重新執行

```bash
python main_pipeline.py --start-from feature --csv-dir ./data
```

這會執行：
- Stage 1: 特徵工程（約 30 秒）
- Stage 2: 模型訓練（約 2-3 分鐘）
- Stage 3: 下載模型（跳過，使用本地模型）
- Stage 4: 驗證報告（約 10 秒）
- Stage 5: 視覺化（約 20 秒）
- Stage 6: XAI 診斷（約 5 秒）

總時間：約 3-5 分鐘

---

## 步驟 9️⃣：調整超參數（進階）

如果你想提高檢測率，可以調整超參數。

### 9.1 編輯訓練腳本

```bash
# 使用 nano 或 vi 編輯
nano train_sagemaker.py
```

找到 `HYPERPARAMS` 區塊（約在第 50 行），修改：

```python
HYPERPARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'num_round': 1000,
    'early_stopping_rounds': 30,
    'min_child_weight': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42,
    'scale_pos_weight': 30,    # 從 1.0 改為 30（增加人頭戶權重）
    'max_depth': 6,             # 從 8 改為 6（更簡單的模型）
    'eta': 0.1,                 # 從 0.05 改為 0.1（更激進的學習）
    'gamma': 0.0,
}
```

### 9.2 重新訓練

```bash
python main_pipeline.py --start-from train --csv-dir ./data
```

### 9.3 比較結果

```bash
python -c "
import pandas as pd

sub = pd.read_csv('submission_with_prob.csv')
print(f'新模型檢測數量: {sub[\"status\"].sum()}')
print(f'新模型檢測率: {sub[\"status\"].mean() * 100:.2f}%')
"
```

---

## 🎯 總結

完成以上步驟後，你應該能夠：

✅ 看到原始資料的樣子
✅ 理解特徵工程如何轉換資料
✅ 觀察模型訓練過程
✅ 分析預測結果
✅ 理解 SHAP 如何解釋預測
✅ 調整閾值和超參數

---

## 🆘 常見問題

### Q1: 為什麼檢測率這麼低？

A: 這是正常的！原因：
1. 訓練集人頭戶只佔 3.21%
2. 閾值設定保守（0.65）
3. 特徵數量有限（只有 3 個）
4. 缺少 crypto_transfer 資料

### Q2: 如何提高檢測率？

A: 三種方法：
1. **降低閾值**（最簡單）：編輯 `main_pipeline.py`，將 `RISK_THRESHOLD` 從 0.65 改為 0.5
2. **調整超參數**：增加 `scale_pos_weight` 到 30
3. **增加特徵**：補齊 crypto_transfer 資料

### Q3: 執行時遇到錯誤怎麼辦？

A: 常見錯誤：
- `找不到 CSV 檔案`: 確認 `data/` 目錄存在
- `模組未安裝`: 執行 `pip install pandas numpy xgboost scikit-learn`
- `記憶體不足`: 使用較小的資料集或增加 Instance 大小

### Q4: 如何查看更詳細的日誌？

A: 執行時加上 `--verbose` 參數：
```bash
python main_pipeline.py --start-from feature --csv-dir ./data --verbose
```

---

## 📚 延伸閱讀

- `COMPLETE_DATA_FLOW_REPORT.md` - 完整資料流報告
- `HOW_TO_RUN.md` - 執行指南
- `check_status.py` - 系統狀態檢查工具

---

## 🎉 恭喜！

你現在完全理解 BitoGuard 系統的運作方式了！

這不再是黑盒子，你可以看到：
- 每一筆原始交易
- 每一個提取的特徵
- 每一個模型決策
- 每一個 SHAP 貢獻度

開始執行吧！ 🚀
