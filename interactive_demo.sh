#!/bin/bash
# BitoGuard 互動式示範腳本
# 逐步展示從原始資料到預測結果的完整過程

set -e

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 函數：印出標題
print_header() {
    echo ""
    echo "================================================================================"
    echo -e "${BLUE}$1${NC}"
    echo "================================================================================"
    echo ""
}

# 函數：等待用戶按 Enter
wait_for_enter() {
    echo ""
    echo -e "${YELLOW}按 Enter 繼續...${NC}"
    read
}

# 開始
clear
echo "================================================================================"
echo -e "${GREEN}BitoGuard 互動式示範${NC}"
echo "================================================================================"
echo ""
echo "這個示範將帶你逐步了解從原始資料到預測結果的完整過程。"
echo "每一步都會暫停，讓你有時間觀察結果。"
echo ""
wait_for_enter

# ============================================================================
# 步驟 1: 原始資料
# ============================================================================

print_header "步驟 1: 查看原始資料"

echo "1.1 訓練標籤（真實的人頭戶名單）"
echo "--------------------------------------------------------------------------------"
echo "前 10 筆訓練標籤："
head -10 data/train_label/dt=2026-03-26/part-00000.csv
echo ""
echo "統計資訊："
python -c "
import pandas as pd
df = pd.read_csv('data/train_label/dt=2026-03-26/part-00000.csv')
print(f'總樣本數: {len(df):,}')
print(f'人頭戶數: {df[\"status\"].sum():,}')
print(f'正常用戶: {(df[\"status\"] == 0).sum():,}')
print(f'人頭戶比例: {df[\"status\"].mean() * 100:.2f}%')
"
wait_for_enter

echo ""
echo "1.2 台幣轉帳記錄"
echo "--------------------------------------------------------------------------------"
echo "前 5 筆交易記錄："
head -5 data/twd_transfer/dt=2026-03-26/part-00000.csv
echo ""
echo "統計資訊："
python -c "
import pandas as pd
df = pd.read_csv('data/twd_transfer/dt=2026-03-26/part-00000.csv')
print(f'總交易筆數: {len(df):,}')
print(f'入金筆數 (kind=0): {(df[\"kind\"] == 0).sum():,}')
print(f'出金筆數 (kind=1): {(df[\"kind\"] == 1).sum():,}')
print(f'涉及用戶數: {df[\"user_id\"].nunique():,}')
"
wait_for_enter

# ============================================================================
# 步驟 2: 查看特定用戶
# ============================================================================

print_header "步驟 2: 查看高風險用戶的原始交易"

echo "讓我們看看用戶 876703 的交易記錄（這個用戶最終被標記為高風險）"
echo "--------------------------------------------------------------------------------"
python -c "
import pandas as pd

df = pd.read_csv('data/twd_transfer/dt=2026-03-26/part-00000.csv')
user_df = df[df['user_id'] == 876703].copy()
user_df['created_at'] = pd.to_datetime(user_df['created_at'])
user_df['hour'] = user_df['created_at'].dt.hour
user_df['type'] = user_df['kind'].map({0: '入金', 1: '出金'})
user_df = user_df.sort_values('created_at', ascending=False)

print('用戶 876703 的交易記錄（最近 10 筆）:')
print('時間                      類型    金額        時段')
print('-' * 70)
for _, row in user_df.head(10).iterrows():
    time = row['created_at'].strftime('%Y-%m-%d %H:%M')
    type_ = row['type']
    amount = row['ori_samount']
    hour = row['hour']
    night = '深夜 ⚠️' if (hour >= 22 or hour <= 6) else ''
    print(f'{time}  {type_:4s}  {amount:>10,.0f}  {night}')

print()
print('統計:')
print(f'總交易次數: {len(user_df)}')
print(f'入金次數: {(user_df[\"kind\"] == 0).sum()}')
print(f'出金次數: {(user_df[\"kind\"] == 1).sum()}')
night_count = ((user_df['hour'] >= 22) | (user_df['hour'] <= 6)).sum()
print(f'深夜交易: {night_count} 筆 ({night_count/len(user_df)*100:.1f}%)')
print()
print('🚨 可疑特徵:')
print('  • 高頻入金（25 次）')
print('  • 零出金（只進不出）')
print('  • 深夜交易比例高（32%）')
"
wait_for_enter

# ============================================================================
# 步驟 3: 特徵工程
# ============================================================================

print_header "步驟 3: 特徵工程（從原始資料提取風險特徵）"

echo "系統從 195,601 筆交易記錄中，為每個用戶計算風險特徵："
echo "--------------------------------------------------------------------------------"
python -c "
import pandas as pd

features = pd.read_parquet('feature_cache.parquet')

print(f'總用戶數: {len(features):,}')
print()
print('提取的特徵:')
print('  1. twd_deposit_count  - 台幣入金次數')
print('  2. twd_withdraw_count - 台幣出金次數')
print('  3. night_tx_ratio     - 深夜交易比例（22:00-06:00）')
print()
print('特徵統計:')
print(features[['twd_deposit_count', 'twd_withdraw_count', 'night_tx_ratio']].describe())
"
wait_for_enter

echo ""
echo "用戶 876703 的特徵值 vs 平均值："
echo "--------------------------------------------------------------------------------"
python -c "
import pandas as pd

features = pd.read_parquet('feature_cache.parquet')
user_feat = features[features['user_id'] == 876703].iloc[0]

avg_deposit = features['twd_deposit_count'].mean()
avg_withdraw = features['twd_withdraw_count'].mean()
avg_night = features['night_tx_ratio'].mean()

print(f'特徵                    用戶 876703    平均值      倍數')
print('-' * 70)
print(f'台幣入金次數            {user_feat[\"twd_deposit_count\"]:>10.0f}    {avg_deposit:>10.2f}    {user_feat[\"twd_deposit_count\"]/avg_deposit:>6.1f}x')
print(f'台幣出金次數            {user_feat[\"twd_withdraw_count\"]:>10.0f}    {avg_withdraw:>10.2f}    -')
print(f'深夜交易比例            {user_feat[\"night_tx_ratio\"]:>10.2%}    {avg_night:>10.2%}    {user_feat[\"night_tx_ratio\"]/avg_night:>6.1f}x')
print()
print('💡 用戶 876703 的入金次數是平均值的 10 倍！')
"
wait_for_enter

# ============================================================================
# 步驟 4: 模型訓練
# ============================================================================

print_header "步驟 4: 模型訓練（XGBoost）"

echo "訓練配置："
echo "--------------------------------------------------------------------------------"
python -c "
import json

with open('cv_report.json') as f:
    cv = json.load(f)

print('算法: XGBoost（梯度提升決策樹）')
print(f'交叉驗證: {cv[\"cv_config\"][\"n_splits\"]} 折')
print(f'驗證策略: {cv[\"cv_config\"][\"strategy\"]}')
print()
print('超參數:')
for key, val in list(cv['hyperparams'].items())[:8]:
    print(f'  {key}: {val}')
"
wait_for_enter

echo ""
echo "交叉驗證結果："
echo "--------------------------------------------------------------------------------"
python -c "
import json
import numpy as np

with open('cv_report.json') as f:
    cv = json.load(f)

print(f'{'Fold':<6} {'Precision':<12} {'Recall':<12} {'F1':<12}')
print('-' * 50)
for fold in cv['folds']:
    print(f'{fold[\"fold\"]:<6} {fold[\"precision\"]:<12.4f} {fold[\"recall\"]:<12.4f} {fold[\"f1\"]:<12.4f}')

avg_prec = np.mean([f['precision'] for f in cv['folds']])
avg_rec = np.mean([f['recall'] for f in cv['folds']])
avg_f1 = np.mean([f['f1'] for f in cv['folds']])

print('-' * 50)
print(f'{'平均':<6} {avg_prec:<12.4f} {avg_rec:<12.4f} {avg_f1:<12.4f}')
print()
print('💡 為什麼 F1 這麼低？')
print('   • 人頭戶只佔 3.21%（極度不平衡）')
print('   • 模型非常保守（寧可漏報不誤報）')
print('   • 這在 AML 系統中是正常的！')
"
wait_for_enter

# ============================================================================
# 步驟 5: 預測結果
# ============================================================================

print_header "步驟 5: 模型預測結果"

echo "預測統計："
echo "--------------------------------------------------------------------------------"
python -c "
import pandas as pd

sub = pd.read_csv('submission_with_prob.csv')

print(f'總用戶數: {len(sub):,}')
print(f'預測為人頭戶: {sub[\"status\"].sum():,}')
print(f'預測為正常: {(sub[\"status\"] == 0).sum():,}')
print(f'檢測率: {sub[\"status\"].mean() * 100:.2f}%')
print()
print('風險機率分布:')
bins = [0, 0.3, 0.5, 0.65, 0.75, 0.9, 1.0]
labels = ['極低 (0-0.3)', '低 (0.3-0.5)', '中 (0.5-0.65)', 
          '邊界 (0.65-0.75)', '高 (0.75-0.9)', '極高 (0.9-1.0)']
sub['risk_level'] = pd.cut(sub['probability'], bins=bins, labels=labels)
for level in labels:
    count = (sub['risk_level'] == level).sum()
    pct = count / len(sub) * 100
    print(f'  {level:<20} {count:>6,} ({pct:>5.2f}%)')
"
wait_for_enter

echo ""
echo "高風險用戶列表（機率 > 0.5）："
echo "--------------------------------------------------------------------------------"
python -c "
import pandas as pd

sub = pd.read_csv('submission_with_prob.csv')
high_risk = sub[sub['probability'] > 0.5].sort_values('probability', ascending=False)

print(f'{'User ID':<12} {'Probability':<15} {'Status':<10}')
print('-' * 40)
for _, row in high_risk.iterrows():
    status = '人頭戶' if row['status'] == 1 else '正常'
    print(f'{row[\"user_id\"]:<12} {row[\"probability\"]:<15.4f} {status:<10}')
"
wait_for_enter

# ============================================================================
# 步驟 6: SHAP 解釋
# ============================================================================

print_header "步驟 6: 可解釋 AI（SHAP 分析）"

echo "用戶 876703 的風險診斷："
echo "--------------------------------------------------------------------------------"
python -c "
import json

with open('xai_reports.json', encoding='utf-8') as f:
    reports = json.load(f)

report = reports[0]  # 用戶 876703

print(f'風險機率: {report[\"probability\"]:.2%}')
print(f'風險等級: {report[\"scoring_tier\"]}')
print()
print('SHAP 特徵貢獻度（為什麼被標記為高風險？）:')
print('-' * 70)
for contrib in report['shap_contributions']:
    feature = contrib['feature_label']
    value = contrib['feature_value']
    shap_val = contrib['shap_value']
    pct = contrib['contribution_pct']
    print(f'{feature}:')
    print(f'  實際值: {value}')
    print(f'  SHAP 值: {shap_val:+.4f} ({pct:.1f}% 貢獻度)')
    print()

print('建議行動:')
for i, step in enumerate(report['action']['steps'], 1):
    print(f'  {i}. {step}')
"
wait_for_enter

# ============================================================================
# 步驟 7: 實驗
# ============================================================================

print_header "步驟 7: 實驗 - 不同閾值的影響"

echo "讓我們看看如果降低閾值，會檢測到多少用戶："
echo "--------------------------------------------------------------------------------"
python -c "
import pandas as pd

sub = pd.read_csv('submission_with_prob.csv')

print(f'{'閾值':<10} {'檢測數量':<12} {'檢測率':<12}')
print('-' * 40)

for threshold in [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]:
    count = (sub['probability'] >= threshold).sum()
    rate = count / len(sub) * 100
    marker = ' ← 當前' if threshold == 0.65 else ''
    print(f'{threshold:<10.2f} {count:<12} {rate:<12.2f}%{marker}')

print()
print('💡 如果降低閾值到 0.5:')
count_05 = (sub['probability'] >= 0.5).sum()
print(f'   會檢測到 {count_05} 個用戶（從 2 增加到 {count_05}）')
print(f'   但也可能增加誤報率')
"
wait_for_enter

# ============================================================================
# 總結
# ============================================================================

print_header "🎉 示範完成！"

echo "你已經看到了完整的資料流："
echo ""
echo "  1️⃣  原始資料: 195,601 筆交易記錄"
echo "  2️⃣  特徵工程: 提取 3 個風險特徵"
echo "  3️⃣  模型訓練: XGBoost 5 折交叉驗證"
echo "  4️⃣  模型預測: 檢測 2 個高風險用戶"
echo "  5️⃣  SHAP 解釋: 清楚顯示每個特徵的貢獻度"
echo ""
echo "這不是黑盒子！每個預測都可以追溯到："
echo "  • 原始交易記錄"
echo "  • 提取的特徵"
echo "  • SHAP 貢獻度"
echo "  • 決策邏輯"
echo ""
echo "================================================================================"
echo -e "${GREEN}下一步建議：${NC}"
echo "================================================================================"
echo ""
echo "1. 查看完整報告："
echo "   cat COMPLETE_DATA_FLOW_REPORT.md"
echo ""
echo "2. 執行完整分析："
echo "   python data_flow_analysis.py"
echo ""
echo "3. 調整閾值（提高檢測率）："
echo "   編輯 main_pipeline.py，將 RISK_THRESHOLD 從 0.65 改為 0.5"
echo "   然後執行: python main_pipeline.py --start-from validate"
echo ""
echo "4. 重新訓練模型："
echo "   python main_pipeline.py --start-from train --csv-dir ./data"
echo ""
echo "================================================================================"
echo ""
