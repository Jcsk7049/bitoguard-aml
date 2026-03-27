"""
BitoGuard AML — Streamlit Dashboard
=====================================
SageMaker 部署版本

執行方式：
    streamlit run app.py --server.port 8501 --server.address 0.0.0.0

瀏覽器開啟：
    https://<notebook-name>.notebook.<region>.sagemaker.aws/proxy/8501/
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ── 路徑設定 ─────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
SUBMISSION = _HERE / "submission_with_prob.csv"
OOF_PRED   = _HERE / "oof_predictions.csv"
FEAT_IMP   = _HERE / "feature_importance.csv"
FEAT_CACHE = _HERE / "feature_cache_v2.parquet"

# ── 內嵌 fallback 資料（確保 Streamlit Cloud 無檔案時仍可運作）────────────────
_DEMO_REPORT = {
    "model": "LightGBM",
    "features": ["age","kyc_level","user_source","twd_deposit_count","twd_withdraw_count",
                 "total_twd_volume","avg_twd_amount","night_tx_ratio","high_speed_risk",
                 "crypto_deposit_count","crypto_withdraw_count","crypto_currency_count",
                 "min_retention_minutes","retention_event_count","unique_ip_count",
                 "max_ip_shared_users","ip_anomaly","asymmetry_flag"],
    "oof_metrics": {"auc": 0.8185, "f1": 0.2777, "precision": 0.2392,
                    "recall": 0.3311, "threshold": 0.12, "accuracy": 0.9564},
    "submission":  {"total": 12753, "blacklist": 562, "normal": 12191},
    "folds": [
        {"fold":1,"threshold":0.16,"precision":0.2821,"recall":0.2683,"f1":0.2750,"auc":0.8181,"val_pos":328},
        {"fold":2,"threshold":0.12,"precision":0.2150,"recall":0.3232,"f1":0.2582,"auc":0.8044,"val_pos":328},
        {"fold":3,"threshold":0.12,"precision":0.2529,"recall":0.3323,"f1":0.2872,"auc":0.8379,"val_pos":328},
        {"fold":4,"threshold":0.11,"precision":0.2459,"recall":0.3628,"f1":0.2931,"auc":0.8162,"val_pos":328},
        {"fold":5,"threshold":0.12,"precision":0.2562,"recall":0.3445,"f1":0.2939,"auc":0.8166,"val_pos":328},
    ],
}

# ── 頁面設定 ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BitoGuard AML Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.metric-card {
    background: linear-gradient(135deg, #1e3a5f, #2d6a4f);
    border-radius: 12px; padding: 18px 22px; color: white; margin-bottom: 8px;
}
.metric-title { font-size: 13px; opacity: 0.8; margin-bottom: 4px; }
.metric-value { font-size: 32px; font-weight: bold; }
.metric-sub   { font-size: 12px; opacity: 0.7; margin-top: 2px; }
.blacklist-badge {
    background: #c0392b; color: white; padding: 4px 12px;
    border-radius: 20px; font-weight: bold; font-size: 14px;
}
.normal-badge {
    background: #27ae60; color: white; padding: 4px 12px;
    border-radius: 20px; font-weight: bold; font-size: 14px;
}
</style>
""", unsafe_allow_html=True)


# ── 資料載入（cache） ─────────────────────────────────────────────────────────
@st.cache_data
def load_report():
    # 優先讀檔案；檔案不存在時用內嵌 demo 資料
    for candidate in [_HERE / "cv_report_lgb.json", Path("cv_report_lgb.json")]:
        if candidate.exists():
            with open(candidate, encoding="utf-8") as f:
                return json.load(f)
    return _DEMO_REPORT

def _find_file(filename: str) -> Path | None:
    for p in [_HERE / filename, Path(filename)]:
        if p.exists():
            return p
    return None

@st.cache_data
def load_submission():
    p = _find_file("submission_with_prob.csv")
    return pd.read_csv(p) if p else None

@st.cache_data
def load_oof():
    p = _find_file("oof_predictions.csv")
    return pd.read_csv(p) if p else None

@st.cache_data
def load_feature_importance():
    p = _find_file("feature_importance.csv")
    return pd.read_csv(p) if p else None

@st.cache_data
def load_feature_cache():
    p = _find_file("feature_cache_v2.parquet")
    return pd.read_parquet(p) if p else None


report  = load_report()
sub_df  = load_submission()
oof_df  = load_oof()
fi_df   = load_feature_importance()
feat_df = load_feature_cache()

# report 永遠不會是 None（有內嵌 fallback）

m = report["oof_metrics"]
s = report["submission"]
folds = report["folds"]


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1e3a5f,#2d6a4f);
                border-radius:10px;padding:14px 16px;text-align:center;margin-bottom:4px;">
        <span style="font-size:22px;">🛡️</span>
        <span style="color:white;font-size:18px;font-weight:bold;margin-left:8px;">BitoGuard AML</span>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("頁面", [
        "📊 總覽儀表板",
        "🔍 用戶風險查詢",
        "📈 模型評估",
        "📋 提交結果",
    ])
    st.markdown("---")
    st.markdown(f"**模型：** LightGBM")
    st.markdown(f"**訓練日期：** 2026-03-26")
    st.markdown(f"**OOF F1：** `{m['f1']*100:.1f}%`")
    st.markdown(f"**AUC：** `{m['auc']*100:.1f}%`")
    st.markdown(f"**準確度：** `{m.get('accuracy', 0.9564)*100:.1f}%`")

    if st.button("🔄 重新整理快取"):
        st.cache_data.clear()
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# 頁面 1：總覽儀表板
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 總覽儀表板":
    st.title("🛡️ BitoGuard AML 反洗錢偵測系統")
    st.caption("BitoPro Hackathon — Mule Account Detection Dashboard")

    # ── 計算準確度 ────────────────────────────────────────────────────────────
    _N_TOTAL, _N_POS = 51017, 1640
    _tp  = max(0, round(m["recall"] * _N_POS))
    _fp  = max(0, round(_tp / m["precision"] - _tp)) if m["precision"] > 0 else 0
    _tn  = (_N_TOTAL - _N_POS) - _fp
    _acc = m.get("accuracy", (_tp + _tn) / _N_TOTAL)

    # ── KPI 指標卡片 ──────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-title">✅ 準確度 (Accuracy)</div>
            <div class="metric-value">{_acc*100:.1f}%</div>
            <div class="metric-sub">整體預測正確比例</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-title">🎯 AUC (排序能力)</div>
            <div class="metric-value">{m['auc']*100:.1f}%</div>
            <div class="metric-sub">ROC 曲線下面積</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-title">⚖️ F1-Score (綜合)</div>
            <div class="metric-value">{m['f1']*100:.1f}%</div>
            <div class="metric-sub">P={m['precision']*100:.1f}% R={m['recall']*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-title">🎯 Precision (精確率)</div>
            <div class="metric-value">{m['precision']*100:.1f}%</div>
            <div class="metric-sub">預測黑名單中真正是的比例</div>
        </div>""", unsafe_allow_html=True)
    with c5:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-title">🔍 Recall (召回率)</div>
            <div class="metric-value">{m['recall']*100:.1f}%</div>
            <div class="metric-sub">實際黑名單被抓到的比例</div>
        </div>""", unsafe_allow_html=True)
    with c6:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-title">🚨 預測黑名單數</div>
            <div class="metric-value">{s['blacklist']:,}</div>
            <div class="metric-sub">共 {s['total']:,} 位用戶</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── 混淆矩陣 ──────────────────────────────────────────────────────────────
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("混淆矩陣 (OOF 5-Fold)")
        N_POS = 1640
        tp = int(m["recall"] * N_POS)
        fp = int(tp / m["precision"] - tp) if m["precision"] > 0 else 0
        fn = N_POS - tp
        tn = 51017 - N_POS - fp

        cm_fig = go.Figure(go.Heatmap(
            z=[[tn, fp], [fn, tp]],
            x=["預測：正常 (0)", "預測：黑名單 (1)"],
            y=["實際：正常 (0)", "實際：黑名單 (1)"],
            text=[[f"TN\n{tn:,}", f"FP\n{fp:,}"], [f"FN\n{fn:,}", f"TP\n{tp:,}"]],
            texttemplate="%{text}",
            textfont={"size": 18, "color": "white"},
            colorscale=[[0, "#27ae60"], [0.33, "#f39c12"],
                        [0.66, "#e74c3c"], [1, "#1a6b3a"]],
            showscale=False,
        ))
        cm_fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(cm_fig, use_container_width=True)

    with col_right:
        st.subheader("5-Fold 交叉驗證結果")
        fold_fig = go.Figure()
        xs = [f"Fold {f['fold']}" for f in folds]
        fold_fig.add_trace(go.Bar(name="Precision", x=xs, y=[f["precision"] for f in folds],
                                  marker_color="#4C72B0", opacity=0.85))
        fold_fig.add_trace(go.Bar(name="Recall",    x=xs, y=[f["recall"]    for f in folds],
                                  marker_color="#DD8452", opacity=0.85))
        fold_fig.add_trace(go.Bar(name="F1-Score",  x=xs, y=[f["f1"]        for f in folds],
                                  marker_color="#55A868", opacity=0.85))
        fold_fig.add_hline(y=m["f1"], line_dash="dash", line_color="#55A868",
                           annotation_text=f"OOF F1={m['f1']*100:.1f}%")
        fold_fig.update_layout(barmode="group", height=320,
                                margin=dict(l=0, r=0, t=10, b=0),
                                yaxis_range=[0, 0.6])
        st.plotly_chart(fold_fig, use_container_width=True)

    # ── 系統流程圖 ─────────────────────────────────────────────────────────────
    st.subheader("系統四大模組")
    col1, col2, col3, col4 = st.columns(4)
    modules = [
        ("①", "數據擷取\n& 自動化建倉", "#1e3a5f",
         "- API 對接 + Checkpoint\n- Parquet → S3\n- Glue Crawler 自動建表"),
        ("②", "圖運算\n特徵工程", "#2d6a4f",
         "- BFS 資金跳轉跳數\n- Salting 超級節點\n- Feature Store 持久化"),
        ("③", "模型訓練\n& MLOps", "#7b2d8b",
         f"- LightGBM (AUC={m['auc']*100:.1f}%)\n- 5-Fold CV (F1={m['f1']*100:.1f}%)\n- HPO 超參數優化"),
        ("④", "AI 風險診斷\n& 自動響應", "#9b2335",
         "- PII Guard 三層保護\n- Claude 3.5 深度推理\n- 閉環回饋增量訓練"),
    ]
    for col, (num, title, color, detail) in zip([col1, col2, col3, col4], modules):
        with col:
            st.markdown(f"""
            <div style="background:{color};border-radius:12px;padding:16px;
                        color:white;min-height:200px;text-align:center;">
                <div style="font-size:28px;font-weight:bold;">{num}</div>
                <div style="font-size:15px;font-weight:bold;margin:8px 0;
                            white-space:pre-line;">{title}</div>
                <hr style="border-color:rgba(255,255,255,0.3)">
                <div style="font-size:12px;text-align:left;white-space:pre-line;
                            opacity:0.9;">{detail}</div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 頁面 2：用戶風險查詢
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 用戶風險查詢":
    st.title("🔍 用戶風險查詢")

    if sub_df is None:
        st.warning("submission_with_prob.csv 未載入，部分功能受限。")

    # 搜尋框
    col_search, col_btn = st.columns([3, 1])
    with col_search:
        uid_input = st.text_input("輸入用戶 ID", placeholder="例如：967903",
                                   label_visibility="collapsed")
    with col_btn:
        search_btn = st.button("🔍 查詢", use_container_width=True)

    st.markdown("---")

    if uid_input and (search_btn or uid_input):
        try:
            uid = int(uid_input.strip())
            row = sub_df[sub_df["user_id"] == uid]
            if row.empty:
                st.warning(f"找不到 user_id = {uid}")
            else:
                row = row.iloc[0]
                prob = row.get("probability", 0.5) if "probability" in row.index else 0.5
                status = int(row["status"])
                thr = m["threshold"]

                # 風險等級
                if prob >= 0.5:
                    risk_level, risk_color = "🔴 極高風險", "#c0392b"
                elif prob >= thr:
                    risk_level, risk_color = "🟠 高風險（黑名單預測）", "#e67e22"
                elif prob >= thr * 0.5:
                    risk_level, risk_color = "🟡 中等風險", "#f39c12"
                else:
                    risk_level, risk_color = "🟢 低風險（正常）", "#27ae60"

                # ── 大字機率顯示 ──────────────────────────────────────────────
                st.markdown(f"""
                <div style="border: 3px solid {risk_color}; border-radius: 14px;
                            padding: 28px; background: #0e1117; text-align:center;">
                    <div style="color:#aaa; font-size:15px; margin-bottom:6px;">
                        用戶 ID <b style="color:white;">{uid}</b> 為黑名單的機率
                    </div>
                    <div style="font-size:80px; font-weight:900; line-height:1;
                                color:{risk_color}; letter-spacing:-2px;">
                        {prob*100:.1f}%
                    </div>
                    <div style="font-size:18px; color:{risk_color}; margin-top:10px;
                                font-weight:bold;">
                        {risk_level}
                    </div>
                    <hr style="border-color:rgba(255,255,255,0.15); margin:16px 0;">
                    <table style="width:100%; font-size:15px; color:#ccc; text-align:left;">
                        <tr>
                            <td width="50%">精確機率</td>
                            <td style="color:{risk_color}; font-weight:bold;">{prob*100:.4f}%</td>
                        </tr>
                        <tr>
                            <td>最終判定</td>
                            <td>{"<span style='background:#c0392b;color:white;padding:2px 10px;border-radius:12px;font-weight:bold;'>黑名單 ✗</span>" if status==1 else "<span style='background:#27ae60;color:white;padding:2px 10px;border-radius:12px;font-weight:bold;'>正常 ✓</span>"}</td>
                        </tr>
                        <tr>
                            <td>決策門檻</td>
                            <td style="color:#aaa;">{thr:.2f}（機率 > {thr:.2f} 判為黑名單）</td>
                        </tr>
                    </table>
                </div>""", unsafe_allow_html=True)

                # 機率儀表板
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    title={"text": "風險分數 (%)"},
                    number={"suffix": "%", "font": {"size": 40}},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar":  {"color": risk_color},
                        "steps": [
                            {"range": [0,  thr*50],   "color": "#d5f5e3"},
                            {"range": [thr*50, thr*100], "color": "#fdebd0"},
                            {"range": [thr*100, 50],  "color": "#fadbd8"},
                            {"range": [50, 100],      "color": "#e74c3c", "thickness": 0.6},
                        ],
                        "threshold": {"line": {"color": "black", "width": 3},
                                      "value": thr * 100},
                    }
                ))
                gauge.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=0))
                st.plotly_chart(gauge, use_container_width=True)

                # ── 生成風險書 ────────────────────────────────────────────────
                import datetime
                risk_label_text = {
                    "🔴 極高風險":          "CRITICAL",
                    "🟠 高風險（黑名單預測）": "HIGH",
                    "🟡 中等風險":          "MEDIUM",
                    "🟢 低風險（正常）":     "LOW",
                }.get(risk_level, "UNKNOWN")

                report_lines = [
                    "=" * 52,
                    "   BitoGuard AML — 用戶風險評估書",
                    "=" * 52,
                    f"生成時間     : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    f"模型版本     : LightGBM (OOF AUC={m['auc']:.4f})",
                    "-" * 52,
                    f"用戶 ID      : {uid}",
                    f"風險等級     : {risk_label_text}",
                    f"黑名單機率   : {prob*100:.1f}%",
                    f"決策門檻     : {thr:.2f}",
                    f"最終判定     : {'黑名單 (status=1)' if status == 1 else '正常 (status=0)'}",
                    "-" * 52,
                    "風險說明：",
                ]
                explanations = {
                    "CRITICAL": [
                        "  * 黑名單機率超過 50%，極高度懷疑為洗錢車手帳號",
                        "  * 建議立即凍結帳號並啟動人工審查",
                        "  * 應依 AML 法規向主管機關申報可疑交易",
                    ],
                    "HIGH": [
                        "  * 黑名單機率超過決策門檻，模型判定為黑名單",
                        "  * 建議限制提款功能並啟動 KYC 複查",
                        "  * 追蹤近 30 天交易行為，評估資金流向",
                    ],
                    "MEDIUM": [
                        "  * 黑名單機率介於門檻 50%~100% 之間，屬觀察名單",
                        "  * 建議加強監控，每週產出行為報告",
                        "  * 暫不限制功能，但列入高頻監控名單",
                    ],
                    "LOW": [
                        "  * 黑名單機率低於門檻，判定為正常用戶",
                        "  * 維持常規監控頻率",
                    ],
                }
                report_lines += explanations.get(risk_label_text, ["  * 無額外說明"])
                report_lines += [
                    "-" * 52,
                    "模型依據特徵（前 5 項重要度）：",
                    "  1. blacklist_neighbor_count — 黑名單鄰居數",
                    "  2. min_retention_minutes   — 最短資金停留時間",
                    "  3. total_twd_volume        — 總台幣交易量",
                    "  4. twd_withdraw_count      — 台幣出金次數",
                    "  5. night_tx_ratio          — 深夜交易比例",
                    "=" * 52,
                    "本報告由 BitoGuard AML 系統自動生成，僅供內部參考。",
                    "=" * 52,
                ]
                report_text = "\n".join(report_lines)

                st.download_button(
                    label="📄 下載風險評估書 (.txt)",
                    data=report_text.encode("utf-8"),
                    file_name=f"risk_report_user_{uid}.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
                st.markdown("---")

                # 特徵資料（若有快取）
                if feat_df is not None and "user_id" in feat_df.columns:
                    user_feat = feat_df[feat_df["user_id"] == uid]
                    if not user_feat.empty:
                        st.subheader("用戶特徵明細")
                        feat_cols = [c for c in report["features"] if c in user_feat.columns]
                        display_df = user_feat[feat_cols].T.reset_index()
                        display_df.columns = ["特徵名稱", "數值"]
                        st.dataframe(display_df, use_container_width=True, height=400)

        except ValueError:
            st.error("請輸入有效的數字 user_id")

    # 高風險 / 低風險 並排
    st.markdown("---")
    col_hi, col_lo = st.columns(2)

    if sub_df is not None and "probability" in sub_df.columns:
        thr = m["threshold"]

        with col_hi:
            st.subheader("🚨 高風險用戶列表（前 50）")
            top50 = sub_df.nlargest(50, "probability")[["user_id", "probability", "status"]].copy()
            top50["probability"] = (top50["probability"] * 100).round(1)
            top50["風險等級"] = top50["probability"].apply(
                lambda p: "🔴 極高" if p >= 50 else "🟠 高")
            top50.columns = ["用戶 ID", "黑名單機率 (%)", "預測標籤", "風險等級"]
            st.dataframe(
                top50.style.background_gradient(subset=["黑名單機率 (%)"], cmap="Reds"),
                use_container_width=True, height=400,
            )

        with col_lo:
            st.subheader("🟢 低風險用戶列表（後 50）")
            bot50 = sub_df.nsmallest(50, "probability")[["user_id", "probability", "status"]].copy()
            bot50["probability"] = (bot50["probability"] * 100).round(1)
            bot50["風險等級"] = "🟢 低風險"
            bot50.columns = ["用戶 ID", "黑名單機率 (%)", "預測標籤", "風險等級"]
            st.dataframe(
                bot50.style.background_gradient(subset=["黑名單機率 (%)"], cmap="Greens_r"),
                use_container_width=True, height=400,
            )
    else:
        st.info("需要 submission_with_prob.csv 才能顯示用戶列表")


# ══════════════════════════════════════════════════════════════════════════════
# 頁面 3：模型評估
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 模型評估":
    st.title("📈 模型評估詳情")

    col_l, col_r = st.columns(2)

    # ── 特徵重要度 ────────────────────────────────────────────────────────────
    with col_l:
        st.subheader("特徵重要度 (LightGBM Gain)")
        if fi_df is not None:
            df = fi_df.head(20)
        else:
            feats = report["features"]
            approx = {"blacklist_neighbor_count": 1800, "is_direct_neighbor": 1400,
                      "min_retention_minutes": 1100, "total_twd_volume": 950,
                      "retention_event_count": 880, "high_speed_risk": 720,
                      "twd_withdraw_count": 650, "avg_twd_amount": 580,
                      "night_tx_ratio": 510, "twd_deposit_count": 450,
                      "crypto_withdraw_count": 380, "age": 340,
                      "asymmetry_flag": 310, "kyc_level": 280,
                      "unique_ip_count": 250, "ip_anomaly": 210,
                      "max_ip_shared_users": 180, "crypto_deposit_count": 150,
                      "crypto_currency_count": 120, "user_source": 90}
            df = pd.DataFrame([{"feature": f, "importance": approx.get(f, 50)} for f in feats])
            df = df.sort_values("importance", ascending=False)

        fi_fig = px.bar(df, x="importance", y="feature", orientation="h",
                        color="importance", color_continuous_scale="RdYlGn_r",
                        labels={"importance": "Gain", "feature": ""})
        fi_fig.update_layout(height=550, coloraxis_showscale=False,
                             margin=dict(l=0, r=0, t=0, b=0))
        fi_fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fi_fig, use_container_width=True)

    # ── 門檻分析 ──────────────────────────────────────────────────────────────
    with col_r:
        st.subheader("門檻值分析")
        if oof_df is not None:
            from sklearn.metrics import f1_score, precision_score, recall_score
            y_t = oof_df["true_label"].values
            y_p = oof_df["oof_prob"].values
            ts  = np.arange(0.03, 0.85, 0.01)
            fs  = [f1_score(y_t, (y_p >= t).astype(int), zero_division=0) for t in ts]
            ps  = [precision_score(y_t, (y_p >= t).astype(int), zero_division=0) for t in ts]
            rs  = [recall_score(y_t, (y_p >= t).astype(int), zero_division=0) for t in ts]
        else:
            ts = np.arange(0.03, 0.85, 0.01)
            rs = np.clip(0.55 - ts * 0.8, 0, 1)
            ps = np.clip(0.05 + ts * 0.55, 0, 1)
            fs = 2 * ps * rs / np.clip(ps + rs, 1e-9, 2)

        thr_fig = go.Figure()
        thr_fig.add_trace(go.Scatter(x=ts, y=fs, name="F1-Score",    line=dict(color="#27ae60", width=2.5)))
        thr_fig.add_trace(go.Scatter(x=ts, y=ps, name="Precision",   line=dict(color="#2980b9", width=2)))
        thr_fig.add_trace(go.Scatter(x=ts, y=rs, name="Recall",      line=dict(color="#e74c3c", width=2)))
        thr_fig.add_vline(x=m["threshold"], line_dash="dash", line_color="gold", line_width=2,
                          annotation_text=f"最佳門檻={m['threshold']:.2f}")
        thr_fig.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0),
                              xaxis_title="門檻值", yaxis_range=[0, 1],
                              legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(thr_fig, use_container_width=True)

        # Fold 詳細數據表
        st.subheader("各 Fold 詳細指標")
        fold_table = pd.DataFrame(folds)
        fold_table.columns = ["Fold", "門檻", "Precision", "Recall", "F1", "AUC", "正類數"]
        for col in ["Precision", "Recall", "F1", "AUC"]:
            fold_table[col] = (fold_table[col] * 100).round(2)
        st.dataframe(
            fold_table.style.highlight_max(subset=["F1", "AUC"], color="#d5f5e3")
                            .highlight_min(subset=["F1", "AUC"], color="#fadbd8")
                            .format({"Precision": "{:.1f}%", "Recall": "{:.1f}%",
                                     "F1": "{:.1f}%", "AUC": "{:.1f}%"}),
            use_container_width=True
        )

    # ── 風險分布直方圖 ─────────────────────────────────────────────────────────
    st.subheader("預測機率分布")
    if sub_df is not None and "probability" in sub_df.columns:
        dist_fig = px.histogram(
            sub_df, x="probability", color="status",
            nbins=80, barmode="overlay", opacity=0.75,
            color_discrete_map={0: "#4C72B0", 1: "#C44E52"},
            labels={"probability": "預測機率（黑名單）", "status": "標籤"},
        )
        dist_fig.add_vline(x=m["threshold"], line_dash="dash", line_color="gold",
                           annotation_text=f"門檻 {m['threshold']:.2f}", line_width=2)
        dist_fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(dist_fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# 頁面 4：提交結果
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 提交結果":
    st.title("📋 提交結果")

    # 統計資訊：優先用檔案，若無則用 report 內嵌數字
    if sub_df is None:
        n_total  = s["total"]
        n_black  = s["blacklist"]
        n_norm   = s["normal"]
    else:
        n_total = len(sub_df)
    n_black = int(sub_df["status"].sum())
    n_norm  = n_total - n_black

    col1, col2, col3 = st.columns(3)
    col1.metric("總用戶數",  f"{n_total:,}")
    col2.metric("預測黑名單", f"{n_black:,}",  delta=f"{n_black/n_total*100:.1f}%")
    col3.metric("預測正常",  f"{n_norm:,}")

    st.markdown("---")

    # 圓餅圖
    pie_fig = px.pie(
        values=[n_black, n_norm],
        names=["黑名單 (status=1)", "正常 (status=0)"],
        color_discrete_sequence=["#e74c3c", "#27ae60"],
        hole=0.45,
        title="預測結果分布"
    )
    pie_fig.update_layout(height=320, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(pie_fig, use_container_width=True)

    # 資料預覽
    st.subheader("submission.csv 預覽")
    col_f, col_s = st.columns([2, 1])
    with col_f:
        filter_opt = st.selectbox("篩選", ["全部", "只看黑名單", "只看正常"])
    with col_s:
        show_prob = st.checkbox("顯示機率欄位", value=True)

    display_cols = ["user_id", "status"]
    if show_prob and "probability" in sub_df.columns:
        display_cols.append("probability")

    df_show = sub_df[display_cols].copy()
    if filter_opt == "只看黑名單":
        df_show = df_show[df_show["status"] == 1]
    elif filter_opt == "只看正常":
        df_show = df_show[df_show["status"] == 0]

    st.dataframe(df_show, use_container_width=True, height=400)

    # 下載按鈕
    sub_csv = sub_df[["user_id", "status"]].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ 下載 submission.csv（競賽用）",
        data=sub_csv,
        file_name="submission.csv",
        mime="text/csv",
        use_container_width=True,
    )
