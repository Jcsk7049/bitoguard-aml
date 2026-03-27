#!/usr/bin/env python3
"""
BitoGuard 資料流透明化分析工具

完整展示從原始資料 → 特徵工程 → 模型訓練 → 預測結果的全過程
讓你看到每一步的轉換，不再是黑盒子！
"""

import pandas as pd
import numpy as np
import json
import xgboost as xgb
from pathlib import Path

def print_section(title):
    """印出區塊標題"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def analyze_raw_data(csv_dir="./data"):
    """分析原始資料"""
    print_section("【步驟 1】原始資料檢視")
    
    # 1. 訓練標籤（真實答案）
    print("\n1.1 訓練標籤（真實的人頭戶名單）")
    print("-" * 80)
    train_label = pd.read_csv(f"{csv_dir}/train_label/dt=2026-03-26/part-00000.csv")
    print(f"總訓練樣本數: {len(train_label):,}")
    print(f"人頭戶數量: {train_label['status'].sum():,}")
    print(f"正常用戶數量: {(train_label['status'] == 0).sum():,}")
    print(f"人頭戶比例: {train_label['status'].mean() * 100:.2f}%")
    print(f"\n前 5 個人頭戶 ID:")
    print(train_label[train_label['status'] == 1].head()['user_id'].tolist())
    
    # 2. 用戶基本資料
    print("\n1.2 用戶基本資料（user_info）")
    print("-" * 80)
    users = pd.read_csv(f"{csv_dir}/user_info/dt=2026-03-26/part-00000.csv", dtype=str)
    print(f"總用戶數: {len(users):,}")
    print(f"欄位: {list(users.columns)}")
    print(f"\n範例資料（前 3 筆）:")
    print(users.head(3)[['user_id', 'sex', 'career', 'income_source', 'user_source']].to_string(index=False))
    
    # 3. 台幣轉帳記錄
    print("\n1.3 台幣轉帳記錄（twd_transfer）")
    print("-" * 80)
    twd = pd.read_csv(f"{csv_dir}/twd_transfer/dt=2026-03-26/part-00000.csv")
    print(f"總交易筆數: {len(twd):,}")
    print(f"入金筆數 (kind=0): {(twd['kind'] == 0).sum():,}")
    print(f"出金筆數 (kind=1): {(twd['kind'] == 1).sum():,}")
    print(f"涉及用戶數: {twd['user_id'].nunique():,}")
    print(f"\n範例資料（前 3 筆）:")
    print(twd.head(3)[['created_at', 'user_id', 'kind', 'ori_samount']].to_string(index=False))
    
    # 4. 交易記錄
    print("\n1.4 USDT/TWD 交易記錄（usdt_twd_trading）")
    print("-" * 80)
    trades = pd.read_csv(f"{csv_dir}/usdt_twd_trading/dt=2026-03-26/part-00000.csv")
    print(f"總交易筆數: {len(trades):,}")
    print(f"買單筆數 (is_buy=1): {(trades['is_buy'] == 1).sum():,}")
    print(f"賣單筆數 (is_buy=0): {(trades['is_buy'] == 0).sum():,}")
    print(f"涉及用戶數: {trades['user_id'].nunique():,}")
    
    return train_label, users, twd, trades

def analyze_feature_engineering(train_label, users, twd, trades):
    """分析特徵工程過程"""
    print_section("【步驟 2】特徵工程（從原始資料提取風險特徵）")
    
    # 載入已經計算好的特徵
    if Path("feature_cache.parquet").exists():
        features = pd.read_parquet("feature_cache.parquet")
        print(f"\n[OK] 載入特徵快取: {len(features):,} 個用戶")
        print(f"特徵數量: {len(features.columns) - 1} 個（扣除 user_id）")
        print(f"\n特徵列表:")
        for i, col in enumerate(features.columns):
            if col != 'user_id':
                print(f"  {i+1:2d}. {col}")
        
        # 展示特徵統計
        print("\n特徵統計摘要:")
        print("-" * 80)
        numeric_cols = features.select_dtypes(include='number').columns
        stats = features[numeric_cols].describe().T
        print(stats[['mean', 'std', 'min', 'max']].to_string())
        
        return features
    else:
        print("[ERROR] 特徵快取不存在，請先執行: python main_pipeline.py --start-from feature")
        return None

def analyze_specific_user(user_id, features, twd, trades):
    """分析特定用戶的詳細資料"""
    print_section(f"【步驟 3】用戶 {user_id} 的詳細分析")
    
    # 3.1 原始交易記錄
    print(f"\n3.1 用戶 {user_id} 的台幣轉帳記錄")
    print("-" * 80)
    user_twd = twd[twd['user_id'] == user_id].copy()
    if len(user_twd) > 0:
        user_twd['created_at'] = pd.to_datetime(user_twd['created_at'])
        user_twd['hour'] = user_twd['created_at'].dt.hour
        user_twd['kind_label'] = user_twd['kind'].map({0: '入金', 1: '出金'})
        print(f"總交易筆數: {len(user_twd)}")
        print(f"入金次數: {(user_twd['kind'] == 0).sum()}")
        print(f"出金次數: {(user_twd['kind'] == 1).sum()}")
        print(f"深夜交易 (22:00-06:00): {((user_twd['hour'] >= 22) | (user_twd['hour'] <= 6)).sum()} 筆")
        print(f"\n最近 10 筆交易:")
        display_cols = ['created_at', 'kind_label', 'ori_samount', 'hour']
        print(user_twd.sort_values('created_at', ascending=False).head(10)[display_cols].to_string(index=False))
    else:
        print(f"[WARN] 用戶 {user_id} 沒有台幣轉帳記錄")
    
    # 3.2 提取的特徵
    print(f"\n3.2 用戶 {user_id} 的風險特徵")
    print("-" * 80)
    if features is not None:
        user_features = features[features['user_id'] == user_id]
        if len(user_features) > 0:
            user_feat = user_features.iloc[0]
            print(f"台幣入金次數: {user_feat.get('twd_deposit_count', 0):.0f}")
            print(f"台幣出金次數: {user_feat.get('twd_withdraw_count', 0):.0f}")
            print(f"深夜交易比例: {user_feat.get('night_tx_ratio', 0):.2%}")
            
            # 顯示所有非零特徵
            print(f"\n所有非零特徵:")
            for col in user_features.columns:
                if col != 'user_id':
                    val = user_feat[col]
                    if val != 0:
                        print(f"  - {col}: {val}")
        else:
            print(f"[WARN] 用戶 {user_id} 沒有特徵資料")

def analyze_model_training():
    """分析模型訓練過程"""
    print_section("【步驟 4】模型訓練過程")
    
    # 載入交叉驗證報告
    if Path("cv_report.json").exists():
        with open("cv_report.json") as f:
            cv_report = json.load(f)
        
        print("\n4.1 訓練配置")
        print("-" * 80)
        print(f"交叉驗證折數: {cv_report['cv_config']['n_splits']}")
        print(f"驗證策略: {cv_report['cv_config']['strategy']}")
        
        print("\n4.2 模型超參數")
        print("-" * 80)
        for key, val in cv_report['hyperparams'].items():
            print(f"  {key}: {val}")
        
        print("\n4.3 各折驗證效能")
        print("-" * 80)
        print(f"{'Fold':<6} {'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
        print("-" * 80)
        for fold_data in cv_report['folds']:
            fold = fold_data['fold']
            thresh = fold_data['threshold']
            prec = fold_data['precision']
            rec = fold_data['recall']
            f1 = fold_data['f1']
            print(f"{fold:<6} {thresh:<12.2f} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f}")
        
        print("\n4.4 平均效能")
        print("-" * 80)
        if 'average' in cv_report:
            avg = cv_report['average']
            print(f"平均 Precision: {avg['precision']:.4f}")
            print(f"平均 Recall: {avg['recall']:.4f}")
            print(f"平均 F1: {avg['f1']:.4f}")
            print(f"最佳閾值: {avg['threshold']:.2f}")
        else:
            # 手動計算平均
            folds = cv_report['folds']
            avg_prec = np.mean([f['precision'] for f in folds])
            avg_rec = np.mean([f['recall'] for f in folds])
            avg_f1 = np.mean([f['f1'] for f in folds])
            print(f"平均 Precision: {avg_prec:.4f}")
            print(f"平均 Recall: {avg_rec:.4f}")
            print(f"平均 F1: {avg_f1:.4f}")
        
        return cv_report
    else:
        print("[ERROR] 交叉驗證報告不存在")
        return None

def analyze_model_predictions():
    """分析模型預測結果"""
    print_section("【步驟 5】模型預測結果分析")
    
    # 載入提交檔案
    if Path("submission_with_prob.csv").exists():
        submission = pd.read_csv("submission_with_prob.csv")
        
        print("\n5.1 預測統計")
        print("-" * 80)
        print(f"總預測用戶數: {len(submission):,}")
        print(f"預測為人頭戶: {submission['status'].sum():,}")
        print(f"預測為正常: {(submission['status'] == 0).sum():,}")
        print(f"檢測率: {submission['status'].mean() * 100:.2f}%")
        
        print("\n5.2 風險機率分布")
        print("-" * 80)
        bins = [0, 0.3, 0.5, 0.65, 0.75, 0.9, 1.0]
        labels = ['極低 (0-0.3)', '低 (0.3-0.5)', '中 (0.5-0.65)', 
                  '邊界 (0.65-0.75)', '高 (0.75-0.9)', '極高 (0.9-1.0)']
        submission['risk_level'] = pd.cut(submission['probability'], bins=bins, labels=labels)
        print(submission['risk_level'].value_counts().sort_index().to_string())
        
        print("\n5.3 高風險用戶列表（機率 > 0.5）")
        print("-" * 80)
        high_risk = submission[submission['probability'] > 0.5].sort_values('probability', ascending=False)
        print(f"{'User ID':<12} {'Probability':<15} {'Status':<10}")
        print("-" * 80)
        for _, row in high_risk.head(20).iterrows():
            print(f"{row['user_id']:<12} {row['probability']:<15.4f} {'人頭戶' if row['status'] == 1 else '正常':<10}")
        
        return submission
    else:
        print("[ERROR] 預測結果不存在")
        return None

def analyze_xai_diagnosis():
    """分析 XAI 診斷報告"""
    print_section("【步驟 6】可解釋 AI 診斷（為什麼被標記？）")
    
    if Path("xai_reports.json").exists():
        with open("xai_reports.json", encoding='utf-8') as f:
            reports = json.load(f)
        
        print(f"\n診斷報告數量: {len(reports)}")
        
        for i, report in enumerate(reports, 1):
            print(f"\n6.{i} 用戶 {report['user_id']} 的風險診斷")
            print("-" * 80)
            print(f"風險機率: {report['probability']:.2%}")
            print(f"風險等級: {report['scoring_tier']}")
            print(f"威脅模式: {report['threat_pattern_zh']}")
            
            print(f"\nSHAP 特徵貢獻度（為什麼被標記？）:")
            for contrib in report['shap_contributions']:
                feature = contrib['feature_label']
                value = contrib['feature_value']
                shap_val = contrib['shap_value']
                pct = contrib['contribution_pct']
                direction = contrib['direction']
                print(f"  • {feature}: {value}")
                print(f"    → SHAP 值: {shap_val:.4f} ({pct:.1f}% 貢獻度) - {direction}")
            
            print(f"\n建議行動:")
            for step in report['action']['steps']:
                print(f"  • {step}")
        
        return reports
    else:
        print("[ERROR] XAI 診斷報告不存在")
        return None

def compare_with_ground_truth(submission, train_label):
    """比較預測結果與真實標籤"""
    print_section("【步驟 7】模型效能驗證（預測 vs 真實）")
    
    # 合併預測和真實標籤
    merged = submission.merge(
        train_label[['user_id', 'status']].rename(columns={'status': 'true_status'}),
        on='user_id',
        how='inner'
    )
    
    print("\n7.1 混淆矩陣")
    print("-" * 80)
    from sklearn.metrics import confusion_matrix, classification_report
    
    cm = confusion_matrix(merged['true_status'], merged['status'])
    print(f"{'':>15} {'預測正常':>12} {'預測人頭戶':>12}")
    print(f"{'真實正常':<15} {cm[0,0]:>12,} {cm[0,1]:>12,}")
    print(f"{'真實人頭戶':<15} {cm[1,0]:>12,} {cm[1,1]:>12,}")
    
    print("\n7.2 詳細指標")
    print("-" * 80)
    print(classification_report(
        merged['true_status'], 
        merged['status'],
        target_names=['正常用戶', '人頭戶'],
        digits=4
    ))
    
    # 找出誤判案例
    print("\n7.3 誤判分析")
    print("-" * 80)
    
    # False Positives（誤報）
    fp = merged[(merged['true_status'] == 0) & (merged['status'] == 1)]
    print(f"誤報（False Positive）: {len(fp)} 個")
    if len(fp) > 0:
        print(f"誤報用戶 ID: {fp['user_id'].tolist()[:10]}")
    
    # False Negatives（漏報）
    fn = merged[(merged['true_status'] == 1) & (merged['status'] == 0)]
    print(f"漏報（False Negative）: {len(fn)} 個")
    if len(fn) > 0:
        print(f"漏報用戶 ID: {fn['user_id'].tolist()[:10]}")
        print(f"漏報用戶的風險機率分布:")
        print(fn['probability'].describe().to_string())

def main():
    """主程式"""
    print("=" * 80)
    print("  BitoGuard 資料流透明化分析")
    print("  從原始資料到最終預測的完整過程")
    print("=" * 80)
    
    # 步驟 1: 原始資料
    train_label, users, twd, trades = analyze_raw_data()
    
    # 步驟 2: 特徵工程
    features = analyze_feature_engineering(train_label, users, twd, trades)
    
    # 步驟 3: 分析高風險用戶
    if features is not None:
        # 分析被標記的用戶
        high_risk_users = [876703, 140471]
        for user_id in high_risk_users:
            analyze_specific_user(user_id, features, twd, trades)
    
    # 步驟 4: 模型訓練
    cv_report = analyze_model_training()
    
    # 步驟 5: 預測結果
    submission = analyze_model_predictions()
    
    # 步驟 6: XAI 診斷
    reports = analyze_xai_diagnosis()
    
    # 步驟 7: 效能驗證
    if submission is not None:
        compare_with_ground_truth(submission, train_label)
    
    print("\n" + "=" * 80)
    print("  分析完成！")
    print("=" * 80)
    print("\n💡 重點發現:")
    print("  1. 人頭戶特徵: 高頻入金 + 深夜交易 + 零出金")
    print("  2. 模型使用 XGBoost 決策樹，可完全解釋每個預測")
    print("  3. SHAP 值顯示每個特徵對預測的貢獻度")
    print("  4. 檢測率低是正常的（真實世界中人頭戶本來就很少）")
    print("\n📊 如何提高檢測率:")
    print("  1. 降低閾值: 編輯 main_pipeline.py 中的 RISK_THRESHOLD")
    print("  2. 調整超參數: 編輯 train_sagemaker.py 中的 HYPERPARAMS")
    print("  3. 增加特徵: 在 bito_data_manager.py 中添加新特徵")

if __name__ == "__main__":
    main()
