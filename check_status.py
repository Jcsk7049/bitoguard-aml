#!/usr/bin/env python3
"""
BitoGuard 系統狀態檢查工具

快速診斷系統狀態，並提供執行建議。
"""

import os
import json
import glob
from pathlib import Path

def check_file(path, description):
    """檢查檔案是否存在並顯示大小"""
    if os.path.exists(path):
        size = os.path.getsize(path)
        if size < 1024:
            size_str = f"{size} B"
        elif size < 1024 * 1024:
            size_str = f"{size / 1024:.1f} KB"
        else:
            size_str = f"{size / 1024 / 1024:.1f} MB"
        print(f"  ✅ {description}: {size_str}")
        return True
    else:
        print(f"  ❌ {description}: 不存在")
        return False

def check_data_tables():
    """檢查資料表是否存在"""
    tables = [
        "user_info",
        "twd_transfer", 
        "crypto_transfer",
        "usdt_twd_trading",
        "usdt_swap",
        "train_label",
        "predict_label"
    ]
    
    existing = []
    missing = []
    
    for table in tables:
        # 檢查兩種格式
        simple = f"data/{table}.csv"
        partition = f"data/{table}/dt=*/part-*.csv"
        
        if os.path.exists(simple):
            existing.append(table)
        elif glob.glob(partition):
            existing.append(table)
        else:
            missing.append(table)
    
    return existing, missing

def check_submission():
    """檢查提交檔案並統計"""
    if not os.path.exists("submission.csv"):
        return None
    
    try:
        import pandas as pd
        df = pd.read_csv("submission.csv")
        total = len(df)
        flagged = len(df[df["status"] == 1])
        rate = flagged / total * 100 if total > 0 else 0
        return {
            "total": total,
            "flagged": flagged,
            "rate": rate
        }
    except Exception as e:
        return {"error": str(e)}

def check_xai_reports():
    """檢查 XAI 報告"""
    if not os.path.exists("xai_reports.json"):
        return None
    
    try:
        with open("xai_reports.json") as f:
            reports = json.load(f)
        return reports
    except Exception as e:
        return {"error": str(e)}

def main():
    print("=" * 70)
    print("BitoGuard 系統狀態檢查")
    print("=" * 70)
    print()
    
    # 1. 檢查核心檔案
    print("【1】核心檔案狀態")
    print("-" * 70)
    has_model = check_file("model.json", "訓練模型")
    has_cache = check_file("feature_cache.parquet", "特徵快取")
    has_submission = check_file("submission.csv", "競賽提交檔")
    has_submission_prob = check_file("submission_with_prob.csv", "含機率提交檔")
    has_xai = check_file("xai_reports.json", "XAI 診斷報告")
    has_cv = check_file("cv_report.json", "交叉驗證報告")
    print()
    
    # 2. 檢查資料表
    print("【2】資料表狀態")
    print("-" * 70)
    existing, missing = check_data_tables()
    
    if existing:
        print(f"  ✅ 已存在 ({len(existing)}):")
        for table in existing:
            print(f"     - {table}")
    
    if missing:
        print(f"  ❌ 缺少 ({len(missing)}):")
        for table in missing:
            print(f"     - {table}")
    print()
    
    # 3. 檢查提交結果
    print("【3】提交結果統計")
    print("-" * 70)
    submission = check_submission()
    if submission:
        if "error" in submission:
            print(f"  ⚠️  讀取錯誤: {submission['error']}")
        else:
            print(f"  總用戶數: {submission['total']:,}")
            print(f"  黑名單用戶: {submission['flagged']:,}")
            print(f"  檢測率: {submission['rate']:.2f}%")
    else:
        print("  ❌ 提交檔案不存在")
    print()
    
    # 4. 檢查 XAI 報告
    print("【4】XAI 診斷報告")
    print("-" * 70)
    xai = check_xai_reports()
    if xai:
        if isinstance(xai, dict) and "error" in xai:
            print(f"  ⚠️  讀取錯誤: {xai['error']}")
        else:
            print(f"  診斷報告數量: {len(xai)}")
            for report in xai:
                user_id = report.get("user_id", "N/A")
                prob = report.get("probability", 0)
                tier = report.get("scoring_tier", "N/A")
                print(f"  - 用戶 {user_id}: 風險機率 {prob:.2%}, 等級 {tier}")
    else:
        print("  ❌ XAI 報告不存在")
    print()
    
    # 5. 執行建議
    print("【5】執行建議")
    print("=" * 70)
    
    if has_submission and has_model and has_xai:
        print("✅ 系統已完整執行，結果可直接用於競賽提交")
        print()
        print("建議操作：")
        print("  1. 查看提交檔案：head -20 submission.csv")
        print("  2. 查看診斷報告：cat xai_reports.json | python -m json.tool")
        print("  3. 如需重新生成報告：python main_pipeline.py --start-from validate")
        
    elif has_model and has_cache:
        print("✅ 模型和特徵已就緒，可從驗證階段開始")
        print()
        print("建議操作：")
        print("  python main_pipeline.py --start-from validate")
        
    elif "crypto_transfer" in missing:
        print("⚠️  缺少 crypto_transfer 資料表")
        print()
        print("解決方案：")
        print()
        print("【方案 A】建立空的 crypto_transfer（使用簡化特徵集）")
        print("  mkdir -p data/crypto_transfer/dt=2026-03-26")
        print("  echo 'created_at,user_id,kind,sub_kind,ori_samount,twd_srate,currency,protocol,from_wallet,to_wallet,relation_user_id,source_ip' > data/crypto_transfer/dt=2026-03-26/part-00000.csv")
        print("  python main_pipeline.py --start-from feature --csv-dir ./data")
        print()
        print("【方案 B】從 API 下載完整資料")
        print("  python bito_api_ingester.py --tables crypto_transfer --output-dir ./data")
        print("  python main_pipeline.py --start-from feature --csv-dir ./data")
        
    else:
        print("⚠️  系統尚未執行或資料不完整")
        print()
        print("建議操作：")
        print("  python main_pipeline.py --csv-dir ./data")
    
    print()
    print("=" * 70)
    print("詳細說明請參考：HOW_TO_RUN.md")
    print("=" * 70)

if __name__ == "__main__":
    main()
