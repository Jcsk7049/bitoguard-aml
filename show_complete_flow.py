import pandas as pd
import numpy as np
import json

print("=" * 80)
print("BitoGuard 完整資料流分析")
print("=" * 80)

print("\n[步驟 1] 原始資料")
print("-" * 80)
train_label = pd.read_csv('data/train_label/dt=2026-03-26/part-00000.csv')
print(f'訓練標籤: 總數={len(train_label):,}, status=1: {train_label["status"].sum():,} ({train_label["status"].mean()*100:.2f}%)')
print(f'前 5 個 status=1: {train_label[train_label["status"]==1].head()["user_id"].tolist()}')

twd = pd.read_csv('data/twd_transfer/dt=2026-03-26/part-00000.csv')
print(f'台幣轉帳: 總數={len(twd):,}, kind=0: {(twd["kind"]==0).sum():,}, kind=1: {(twd["kind"]==1).sum():,}, 唯一用戶={twd["user_id"].nunique():,}')

print("\n[步驟 2] 用戶 876703 原始交易")
print("-" * 80)
user_df = twd[twd['user_id'] == 876703].copy()
user_df['created_at'] = pd.to_datetime(user_df['created_at'])
user_df['hour'] = user_df['created_at'].dt.hour
user_df = user_df.sort_values('created_at', ascending=False)

print(f'{"時間":<20} {"kind":<6} {"金額":>12} {"hour":>6}')
for _, row in user_df.head(10).iterrows():
    print(f'{row["created_at"].strftime("%Y-%m-%d %H:%M"):<20} {row["kind"]:<6} {row["ori_samount"]:>12,.0f} {row["hour"]:>6}')

night = ((user_df['hour'] >= 22) | (user_df['hour'] <= 6)).sum()
print(f'統計: 總數={len(user_df)}, kind=0: {(user_df["kind"]==0).sum()}, kind=1: {(user_df["kind"]==1).sum()}, 深夜={night} ({night/len(user_df)*100:.1f}%)')

print("\n[步驟 3] 特徵工程")
print("-" * 80)
features = pd.read_parquet('feature_cache.parquet')
print(f'總用戶數: {len(features):,}')
print(f'特徵: {[c for c in features.columns if c != "user_id"]}')
print(f'\n特徵統計:')
print(features[['twd_deposit_count', 'twd_withdraw_count', 'night_tx_ratio']].describe().to_string())

user_feat = features[features['user_id'] == 876703].iloc[0]
avg_deposit = features['twd_deposit_count'].mean()
avg_withdraw = features['twd_withdraw_count'].mean()
avg_night = features['night_tx_ratio'].mean()

print(f'\n用戶 876703: deposit={user_feat["twd_deposit_count"]:.0f}, withdraw={user_feat["twd_withdraw_count"]:.0f}, night={user_feat["night_tx_ratio"]:.4f}')
print(f'平均值: deposit={avg_deposit:.2f}, withdraw={avg_withdraw:.2f}, night={avg_night:.4f}')
print(f'倍數: deposit={user_feat["twd_deposit_count"]/avg_deposit:.1f}x, night={user_feat["night_tx_ratio"]/avg_night:.1f}x')

print("\n[步驟 4] 模型訓練")
print("-" * 80)
with open('cv_report.json') as f:
    cv = json.load(f)

print(f'配置: n_splits={cv["cv_config"]["n_splits"]}, strategy={cv["cv_config"]["strategy"]}')
print(f'超參數: max_depth={cv["hyperparams"]["max_depth"]}, eta={cv["hyperparams"]["eta"]}, num_round={cv["hyperparams"]["num_round"]}')

print(f'\n{"Fold":<6} {"Precision":<12} {"Recall":<12} {"F1":<12}')
for fold in cv['folds']:
    print(f'{fold["fold"]:<6} {fold["precision"]:<12.4f} {fold["recall"]:<12.4f} {fold["f1"]:<12.4f}')

avg_prec = np.mean([f['precision'] for f in cv['folds']])
avg_rec = np.mean([f['recall'] for f in cv['folds']])
avg_f1 = np.mean([f['f1'] for f in cv['folds']])
print(f'{"平均":<6} {avg_prec:<12.4f} {avg_rec:<12.4f} {avg_f1:<12.4f}')

print("\n[步驟 5] 預測結果")
print("-" * 80)
sub = pd.read_csv('submission_with_prob.csv')
print(f'總數={len(sub):,}, status=1: {sub["status"].sum():,} ({sub["status"].mean()*100:.4f}%)')

bins = [0, 0.3, 0.5, 0.65, 0.75, 0.9, 1.0]
labels = ['[0,0.3)', '[0.3,0.5)', '[0.5,0.65)', '[0.65,0.75)', '[0.75,0.9)', '[0.9,1.0]']
sub['risk_level'] = pd.cut(sub['probability'], bins=bins, labels=labels)

print(f'\n機率分布:')
for level in labels:
    count = (sub['risk_level'] == level).sum()
    print(f'{level:<15} {count:>6,} ({count/len(sub)*100:>6.2f}%)')

print(f'\nprobability >= 0.5:')
high_risk = sub[sub['probability'] >= 0.5].sort_values('probability', ascending=False)
print(f'{"user_id":<12} {"probability":<15} {"status"}')
for _, row in high_risk.iterrows():
    print(f'{row["user_id"]:<12} {row["probability"]:<15.6f} {row["status"]}')

print("\n[步驟 6] SHAP 分析")
print("-" * 80)
with open('xai_reports.json', encoding='utf-8') as f:
    reports = json.load(f)

for report in reports:
    print(f'\n用戶 {report["user_id"]}: probability={report["probability"]:.6f}, tier={report["scoring_tier"]}')
    print(f'SHAP 貢獻:')
    for contrib in report['shap_contributions']:
        print(f'  {contrib["feature_label"]}: value={contrib["feature_value"]}, shap={contrib["shap_value"]:.6f}, pct={contrib["contribution_pct"]:.2f}%')
    print(f'行動: {", ".join(report["action"]["steps"][:2])}...')

print("\n[步驟 7] 閾值實驗")
print("-" * 80)
print(f'{"threshold":<12} {"count":<12} {"rate %"}')
for threshold in [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]:
    count = (sub['probability'] >= threshold).sum()
    rate = count / len(sub) * 100
    marker = ' <- current' if threshold == 0.65 else ''
    print(f'{threshold:<12.2f} {count:<12} {rate:<12.4f}{marker}')

print("\n" + "=" * 80)
print("完成")
print("=" * 80)
