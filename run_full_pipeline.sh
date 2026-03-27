#!/bin/bash

# BitoGuard 完整管線執行腳本
# 在 SageMaker Notebook Terminal 上運行

set -e  # 任何錯誤就停止

echo "=========================================="
echo "BitoGuard AML 管線 - 完整執行"
echo "=========================================="
echo ""

# 檢查環境
echo "[1/5] 檢查環境..."
python --version
python -c "import xgboost; print(f'XGBoost: {xgboost.__version__}')"
python -c "import pandas; print(f'Pandas: {pandas.__version__}')"
echo "✓ 環境檢查完成"
echo ""

# 清除舊狀態
echo "[2/5] 清除舊狀態..."
rm -f pipeline_state.json
echo "✓ 狀態已清除"
echo ""

# 運行完整管線（從 feature 開始）
echo "[3/5] 運行完整管線..."
echo "  - Stage 1: 特徵工程"
echo "  - Stage 2: 模型訓練（調整超參數）"
echo "  - Stage 3-5: 驗證、視覺化"
echo "  - Stage 6: XAI 診斷書 + 提交"
echo ""

python main_pipeline.py \
  --start-from feature \
  --csv-dir ./data \
  --max-depth 8 \
  --eta 0.05 \
  --num-round 1000 \
  --cv-folds 5

echo ""
echo "✓ 管線執行完成"
echo ""

# 檢查輸出
echo "[4/5] 檢查輸出檔案..."
ls -lh submission.csv submission_with_prob.csv xai_reports.json 2>/dev/null || echo "⚠ 部分檔案未生成"
echo ""

# 顯示提交統計
echo "[5/5] 提交統計..."
echo ""
echo "submission.csv 內容預覽："
head -5 submission.csv
echo "..."
echo ""
echo "黑名單統計："
python -c "
import pandas as pd
df = pd.read_csv('submission.csv')
flagged = df[df['status'] == 1]
print(f'總用戶數: {len(df):,}')
print(f'黑名單用戶: {len(flagged):,}')
print(f'檢測率: {len(flagged)/len(df)*100:.2f}%')
print()
print('黑名單用戶 ID:')
print(flagged['user_id'].tolist())
"
echo ""

echo "=========================================="
echo "✅ 完整管線執行成功！"
echo "=========================================="
echo ""
echo "輸出檔案："
echo "  - submission.csv (競賽提交)"
echo "  - submission_with_prob.csv (含機率)"
echo "  - xai_reports.json (診斷報告)"
echo ""
echo "下一步："
echo "  1. 檢查 submission.csv 格式"
echo "  2. 驗證黑名單用戶"
echo "  3. 提交競賽"
echo ""
