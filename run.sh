#!/bin/bash
# ============================================================
#  BitoGuard AML — 一鍵端到端執行腳本
#  適用：SageMaker Terminal  或  本機 Git Bash / WSL
#
#  用法：
#    bash run.sh           # 訓練 + 啟動 UI
#    bash run.sh --no-ui   # 只訓練，不啟動 Streamlit
# ============================================================
set -e

NO_UI=false
[[ "$1" == "--no-ui" ]] && NO_UI=true

# ── 0. 偵測環境路徑 ──────────────────────────────────────────
if [ -d "/home/ec2-user/SageMaker" ]; then
    WORKDIR="/home/ec2-user/SageMaker"
    echo "環境：SageMaker"
else
    WORKDIR="$(cd "$(dirname "$0")" && pwd)"
    echo "環境：本機 ($WORKDIR)"
fi
cd "$WORKDIR"

echo ""
echo "============================================================"
echo "  BitoGuard AML — 端到端執行"
echo "============================================================"
echo ""

# ── 1. 安裝套件（幂等，已裝則跳過）─────────────────────────
echo "[1/4] 檢查/安裝 Python 套件..."
python - <<'PYCHECK'
import importlib, subprocess, sys
PKGS = {
    "lightgbm":  "lightgbm",
    "sklearn":   "scikit-learn",
    "pandas":    "pandas",
    "numpy":     "numpy",
    "streamlit": "streamlit",
    "plotly":    "plotly",
    "pyarrow":   "pyarrow",
}
missing = [pip for mod, pip in PKGS.items() if importlib.util.find_spec(mod) is None]
if missing:
    print(f"  安裝缺少套件：{missing}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + missing)
else:
    print("  所有套件已就緒 ✓")
PYCHECK
echo ""

# ── 2. 確認資料表存在 ────────────────────────────────────────
echo "[2/4] 確認資料表..."
python - <<'PYCHECK'
from pathlib import Path
BASE = Path("/home/ec2-user/SageMaker") if Path("/home/ec2-user/SageMaker").exists() else Path(".")
DATA = BASE / "data"
REQUIRED = ["user_info","twd_transfer","crypto_transfer",
            "usdt_twd_trading","train_label","predict_label"]
ok = True
for t in REQUIRED:
    found = list((DATA / t).rglob("*.csv")) if (DATA / t).exists() else []
    status = f"✓ ({len(found)} 檔)" if found else "✗ 缺少！"
    print(f"  {t:25s} {status}")
    if not found:
        ok = False
if not ok:
    raise SystemExit("資料表不完整，請先執行資料下載")
PYCHECK
echo ""

# ── 3. 執行模型訓練 ──────────────────────────────────────────
echo "[3/4] 執行 LightGBM 訓練管線..."
echo "  (約需 3~8 分鐘，依機器規格而定)"
echo ""
python lgb_pipeline.py
echo ""

# ── 4. 顯示結果 ──────────────────────────────────────────────
echo "[4/4] 執行結果摘要："
python - <<'PYSUM'
import json, pandas as pd
from pathlib import Path
BASE = Path("/home/ec2-user/SageMaker") if Path("/home/ec2-user/SageMaker").exists() else Path(".")

# 指標
with open(BASE / "cv_report_lgb.json") as f:
    r = json.load(f)
m = r["oof_metrics"]
s = r["submission"]
print(f"  AUC      : {m['auc']:.4f}")
print(f"  F1-Score : {m['f1']:.4f}")
print(f"  Precision: {m['precision']:.4f}  (預測黑名單中真陽性比例)")
print(f"  Recall   : {m['recall']:.4f}  (實際黑名單被抓到比例)")
print(f"  門檻值   : {m['threshold']:.2f}")
print(f"  黑名單   : {s['blacklist']:,} / {s['total']:,} 用戶")

# 驗證 submission 格式
df = pd.read_csv(BASE / "submission.csv")
assert list(df.columns) == ["user_id","status"], "欄位格式錯誤！"
assert df["status"].isin([0,1]).all(), "status 值必須是 0 或 1！"
print(f"\n  submission.csv 格式驗證 ✓")
print(f"  存放位置：{BASE / 'submission.csv'}")
PYSUM
echo ""

# ── 5. 啟動 Streamlit UI ─────────────────────────────────────
if [ "$NO_UI" = false ]; then
    echo "============================================================"
    echo "  啟動 Streamlit 儀表板..."
    echo "============================================================"

    # 找可用 port
    PORT=8501

    if [ -d "/home/ec2-user/SageMaker" ]; then
        # SageMaker：透過 proxy 存取
        NOTEBOOK_NAME=$(cat /opt/ml/metadata/resource-metadata.json 2>/dev/null \
                        | python -c "import json,sys; d=json.load(sys.stdin); print(d.get('ResourceName','your-notebook'))" \
                        2>/dev/null || echo "your-notebook")
        REGION=$(curl -s http://169.254.169.254/latest/meta-data/placement/region 2>/dev/null || echo "ap-northeast-1")
        echo ""
        echo "  瀏覽器開啟（SageMaker Proxy URL）："
        echo "  https://${NOTEBOOK_NAME}.notebook.${REGION}.sagemaker.aws/proxy/${PORT}/"
        echo ""
        echo "  ⚠️  若無法連線，請至 EC2 Security Group 確認 port ${PORT} 已開放"
        echo "  （Inbound Rules → 加入 Custom TCP, Port ${PORT}, Source: My IP）"
        echo ""
        streamlit run app.py \
            --server.port $PORT \
            --server.address 0.0.0.0 \
            --server.headless true \
            --browser.gatherUsageStats false
    else
        # 本機
        echo ""
        echo "  瀏覽器開啟：http://localhost:${PORT}"
        echo ""
        streamlit run app.py --server.port $PORT
    fi
else
    echo "============================================================"
    echo "  ✅ 訓練完成！(--no-ui 模式，未啟動介面)"
    echo ""
    echo "  手動啟動 UI："
    echo "    streamlit run app.py --server.port 8501 --server.address 0.0.0.0"
    echo "============================================================"
fi
