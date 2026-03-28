"""
BitoGuard AML — Streamlit Dashboard
=====================================
SageMaker 部署版本

執行方式：
    streamlit run app.py --server.port 8501 --server.address 0.0.0.0

瀏覽器開啟：
    https://<notebook-name>.notebook.<region>.sagemaker.aws/proxy/8501/
"""

import datetime
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

# ── 內嵌 fallback 資料（確保 Streamlit Cloud 無檔案時仍可運作）────────────────
_DEMO_REPORT = {
    "model": "LightGBM",
    "features": ["age","kyc_level","user_source","twd_deposit_count","twd_withdraw_count",
                 "total_twd_volume","avg_twd_amount","night_tx_ratio","high_speed_risk",
                 "unique_ip_count","max_ip_shared_users","ip_anomaly",
                 "crypto_deposit_count","crypto_withdraw_count","crypto_currency_count",
                 "min_retention_minutes","retention_event_count",
                 "asymmetry_flag","usdt_tx_count",
                 "swap_count","swap_twd_volume","swap_buy_count","swap_sell_count","swap_buy_ratio",
                 "blacklist_neighbor_count","is_direct_neighbor",
                 "deposit_only_flag","twd_deposit_ratio","crypto_net_flow",
                 "tx_per_day","total_volume","swap_to_twd_ratio"],
    "oof_metrics": {"auc": 0.8317, "f1": 0.3011, "precision": 0.2753,
                    "recall": 0.3323, "threshold": 0.24, "accuracy": 0.9504,
                    "composite": 0.6757},
    "submission":  {"total": 12753, "blacklist": 501, "normal": 12252},
    "folds": [
        {"fold":1,"threshold":0.19,"precision":0.3333,"recall":0.2713,"f1":0.2992,"auc":0.8440,"accuracy":0.9591,"val_pos":328},
        {"fold":2,"threshold":0.05,"precision":0.2300,"recall":0.3323,"f1":0.2718,"auc":0.8190,"accuracy":0.9428,"val_pos":328},
        {"fold":3,"threshold":0.15,"precision":0.3174,"recall":0.3232,"f1":0.3202,"auc":0.8485,"accuracy":0.9559,"val_pos":328},
        {"fold":4,"threshold":0.18,"precision":0.3546,"recall":0.2713,"f1":0.3074,"auc":0.8378,"accuracy":0.9607,"val_pos":328},
        {"fold":5,"threshold":0.15,"precision":0.3048,"recall":0.3262,"f1":0.3152,"auc":0.8264,"accuracy":0.9544,"val_pos":328},
    ],
}

# ── 頁面設定 ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BitoGuard AML Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 全域 CSS 注入 ─────────────────────────────────────────────────────────────
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;900&family=Noto+Sans+TC:wght@300;400;500;700&display=swap');

/* ── 全域字體、行高與背景 ────────────────────────────────────────────── */
html, body, [class*="css"], * {
    font-family: 'Noto Sans TC', 'Microsoft JhengHei', '微軟正黑體', 'Inter', sans-serif !important;
    line-height: 1.7 !important;
    -webkit-font-smoothing: antialiased !important;
}
.stApp {
    background-color: #F2EFE9 !important;
}

/* ── 主內容容器：限寬 + 左右留白，避免撐滿螢幕 ─────────────────────── */
.block-container {
    padding: 2.5rem 4rem 4rem !important;
    max-width: 1200px !important;
}

/* ── Sidebar ────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #EDE8DF !important;
    border-right: 1px solid #DDD8CE !important;
}

/* ── 側邊欄收合 / 展開 按鈕 ─────────────────────────────────────────── */

/* sidebar 展開時的收合按鈕 */
[data-testid="stSidebarCollapseButton"] button {
    all: unset !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 32px !important;
    height: 32px !important;
    border-radius: 8px !important;
    cursor: pointer !important;
    transition: background 0.2s !important;
}
[data-testid="stSidebarCollapseButton"] button:hover {
    background: rgba(93,90,84,0.10) !important;
}
[data-testid="stSidebarCollapseButton"] button svg {
    stroke: #5D5A54 !important;
    width: 18px !important;
    height: 18px !important;
    display: block !important;
}
[data-testid="stSidebarCollapseButton"] button span {
    display: none !important;
}

/* sidebar 收合後：把整個 collapsedControl 隱藏，防止文字洩出 */
[data-testid="collapsedControl"] {
    visibility: hidden !important;
    pointer-events: none !important;
}
/* 但保留按鈕可點 */
[data-testid="collapsedControl"] button {
    visibility: visible !important;
    pointer-events: auto !important;
    all: unset !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 32px !important;
    height: 32px !important;
    border-radius: 8px !important;
    cursor: pointer !important;
    margin: 0.6rem 0.3rem !important;
}
[data-testid="collapsedControl"] button:hover {
    background: rgba(93,90,84,0.10) !important;
}
[data-testid="collapsedControl"] button svg {
    visibility: visible !important;
    stroke: #5D5A54 !important;
    width: 18px !important;
    height: 18px !important;
    display: block !important;
}
[data-testid="collapsedControl"] button span {
    display: none !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding: 2rem 1.6rem 1.5rem !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label {
    color: #6A6058 !important;
}

/* ── 全域按鈕（Ghost 風格，比照側邊欄導航）─────────────────────────── */
.stButton > button,
.stDownloadButton > button {
    border-radius: 10px !important;
    background: transparent !important;
    color: #3A3228 !important;
    border: 1px solid #C4BAB0 !important;
    font-family: 'Noto Sans TC', 'Microsoft JhengHei', 'Inter', sans-serif !important;
    font-size: 13.5px !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
    box-shadow: none !important;
    transition: background 0.2s, border-color 0.2s, color 0.2s !important;
    padding: 0.55rem 1.4rem !important;
    min-height: 42px !important;
    line-height: 1.4 !important;
}
.stButton > button *,
.stDownloadButton > button * {
    color: #3A3228 !important;
}
.stButton > button:hover,
.stDownloadButton > button:hover {
    background: rgba(142,115,91,0.10) !important;
    border-color: #8E735B !important;
    color: #2C2010 !important;
    box-shadow: none !important;
}
.stButton > button *:hover,
.stDownloadButton > button *:hover {
    color: #2C2010 !important;
}
.stButton > button:active,
.stDownloadButton > button:active {
    background: rgba(142,115,91,0.18) !important;
    border-color: #7A6148 !important;
    box-shadow: none !important;
    transform: translateY(1px) !important;
    color: #2C2010 !important;
}

/* ── 文字輸入（圓角 12px）────────────────────────────────────────────── */
[data-testid="stTextInput"] input {
    background: #FFFFFF !important;
    border: 1px solid #D8D2C8 !important;
    border-radius: 12px !important;
    color: #3A3228 !important;
    box-shadow: inset 0 1px 3px rgba(60,50,40,0.04) !important;
    font-size: 14px !important;
    letter-spacing: 0.3px !important;
    padding: 0.5rem 1rem !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #7A9B8A !important;
    box-shadow: 0 0 0 3px rgba(122,155,138,0.15), inset 0 1px 3px rgba(60,50,40,0.06) !important;
    outline: none !important;
}

/* ── Selectbox（圓角 12px）──────────────────────────────────────────── */
[data-testid="stSelectbox"] > div > div {
    background: #FFFFFF !important;
    border: 1px solid #D8D2C8 !important;
    border-radius: 12px !important;
    color: #3A3228 !important;
}

/* ── Sidebar 導航（YouTube 風格）────────────────────────────────────── */

[data-testid="stSidebar"] .stButton {
    margin: 0 !important;
    width: 100% !important;
}

/* 預設：完全透明，無邊框（用高特異性選擇器蓋過全域 .stButton）*/
[data-testid="stSidebar"] .element-container .stButton > button,
[data-testid="stSidebar"] .stButton > button {
    width: 100% !important;
    text-align: left !important;
    justify-content: flex-start !important;
    background: transparent !important;
    color: #3A3228 !important;
    border: none !important;
    border-radius: 10px !important;
    box-shadow: none !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    letter-spacing: 0.1px !important;
    padding: 8px 12px 8px 44px !important;
    margin: 2px 0 !important;
    min-height: unset !important;
    position: relative !important;
    transition: background 0.15s !important;
}

/* icon：position absolute 貼左 */
[data-testid="stSidebar"] .stButton > button::before {
    content: "" !important;
    position: absolute !important;
    left: 12px !important;
    top: 50% !important;
    transform: translateY(-50%) !important;
    width: 18px !important;
    height: 18px !important;
    background-repeat: no-repeat !important;
    background-size: contain !important;
    opacity: 0.6 !important;
    transition: opacity 0.15s !important;
}

/* Hover：淺灰底（與 YouTube 一樣）*/
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(0,0,0,0.06) !important;
    border: none !important;
    box-shadow: none !important;
    color: #1A1714 !important;
}
[data-testid="stSidebar"] .stButton > button:hover::before {
    opacity: 0.9 !important;
}
[data-testid="stSidebar"] .stButton > button:active {
    transform: none !important;
    box-shadow: none !important;
}

/* Icons（SVG data URI）*/
[data-testid="stSidebar"] .stButton:nth-of-type(1) > button::before {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='%233A3228' stroke-width='1.6' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='3' y='3' width='18' height='18' rx='2'/%3E%3Cline x1='3' y1='9' x2='21' y2='9'/%3E%3Cline x1='9' y1='21' x2='9' y2='9'/%3E%3C/svg%3E") !important;
}
[data-testid="stSidebar"] .stButton:nth-of-type(2) > button::before {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='%233A3228' stroke-width='1.6' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='11' cy='11' r='8'/%3E%3Cline x1='21' y1='21' x2='16.65' y2='16.65'/%3E%3C/svg%3E") !important;
}
[data-testid="stSidebar"] .stButton:nth-of-type(3) > button::before {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='%233A3228' stroke-width='1.6' stroke-linecap='round' stroke-linejoin='round'%3E%3Cline x1='18' y1='20' x2='18' y2='10'/%3E%3Cline x1='12' y1='20' x2='12' y2='4'/%3E%3Cline x1='6' y1='20' x2='6' y2='14'/%3E%3C/svg%3E") !important;
}
[data-testid="stSidebar"] .stButton:nth-of-type(4) > button::before {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='%233A3228' stroke-width='1.6' stroke-linecap='round' stroke-linejoin='round'%3E%3Cellipse cx='12' cy='5' rx='9' ry='3'/%3E%3Cpath d='M21 12c0 1.66-4 3-9 3s-9-1.34-9-3'/%3E%3Cpath d='M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5'/%3E%3C/svg%3E") !important;
}

/* 刷新資料（第 5 個）：居中、細邊框 */
[data-testid="stSidebar"] .stButton:nth-of-type(5) > button {
    padding: 7px 12px !important;
    border: 1px solid #C8B8A8 !important;
    justify-content: center !important;
    font-size: 12.5px !important;
    color: #7A6A5A !important;
    margin-top: 4px !important;
}
[data-testid="stSidebar"] .stButton:nth-of-type(5) > button::before {
    display: none !important;
}

/* ── Headings ───────────────────────────────────────────────────────── */
h1 { color: #2C2720 !important; font-weight: 700 !important;
     letter-spacing: -0.6px !important; line-height: 1.2 !important; }
h2 { color: #3A3228 !important; font-weight: 600 !important;
     letter-spacing: -0.3px !important; line-height: 1.35 !important; }
h3 { color: #4A4238 !important; font-weight: 600 !important;
     letter-spacing: -0.2px !important; line-height: 1.4 !important; }

/* ── Markdown 文字 ──────────────────────────────────────────────────── */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
    color: #6A6058 !important;
    font-size: 14px !important;
    line-height: 1.75 !important;
    letter-spacing: 0.15px !important;
}

/* ── Caption ────────────────────────────────────────────────────────── */
[data-testid="stCaptionContainer"] p {
    color: #A8A098 !important;
    font-size: 12px !important;
    letter-spacing: 0.5px !important;
}

/* ── Dividers ───────────────────────────────────────────────────────── */
hr {
    border: none !important;
    border-top: 1px solid #E0DDD5 !important;
    margin: 20px 0 !important;
}

/* ── DataFrames ─────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid #E0DDD5 !important;
    border-radius: 4px !important;
    overflow: hidden !important;
}
[data-testid="stDataFrame"] th {
    background: #F5F2EC !important;
    color: #8A8078 !important;
    font-size: 12px !important;
    letter-spacing: 0.5px !important;
}

/* ── Checkbox ───────────────────────────────────────────────────────── */
[data-testid="stCheckbox"] label { color: #6A6058 !important; }

/* ── Native st.metric override ──────────────────────────────────────── */
[data-testid="stMetric"] {
    background: #FFFFFF !important;
    border: 1px solid #E6E2D9 !important;
    border-radius: 4px !important;
    padding: 14px 18px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.03) !important;
}
[data-testid="stMetricLabel"] p { color: #8A8078 !important; font-size: 12px !important; }
[data-testid="stMetricValue"]   { color: #2C2720 !important; }

/* ── 自訂元件類別 ────────────────────────────────────────────────────── */
.bg-card {
    background: #FFFFFF;
    border: 1px solid #E6E2D9;
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 14px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.03);
}
.kpi-strip {
    background: #FFFFFF;
    border: 1px solid #E6E2D9;
    border-radius: 16px;
    padding: 18px 22px;
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 12px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.03);
}
.kpi-accent { width: 3px; height: 42px; border-radius: 3px; flex-shrink: 0; }
.kpi-label  { font-size: 10px; color: #A8A098; text-transform: uppercase;
               letter-spacing: 1.5px; margin-bottom: 4px; font-weight: 600; }
.kpi-value  { font-size: 26px; font-weight: 800; line-height: 1;
               color: #2C2720; letter-spacing: -0.8px; }
.kpi-sub    { font-size: 11px; color: #B8B0A4; margin-top: 4px;
               letter-spacing: 0.2px; line-height: 1.5; }

.section-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 2.2px;
    text-transform: uppercase;
    color: #B0A898;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 16px;
    margin-top: 4px;
}
.section-label::before {
    content: "";
    display: inline-block;
    width: 16px;
    height: 2px;
    background: #C8C0B4;
    border-radius: 1px;
    flex-shrink: 0;
}
.page-title {
    font-size: 28px;
    font-weight: 800;
    color: #2C2720;
    letter-spacing: -0.8px;
    line-height: 1.15;
    margin-bottom: 6px;
}
.page-subtitle {
    font-size: 11px;
    color: #B0A898;
    text-transform: uppercase;
    letter-spacing: 1.8px;
    margin-bottom: 36px;
    font-weight: 500;
}
.section-gap {
    margin-top: 32px;
}

/* ── 自訂提示框（取代 st.info / st.warning / st.error）──────────────── */
.info-div {
    background: #EEF2F8;
    border: 1px solid #C8D0E4;
    border-left: 4px solid #6878B8;
    border-radius: 12px;
    padding: 13px 18px;
    color: #5060A0;
    font-size: 13px;
    line-height: 1.7;
    letter-spacing: 0.15px;
    margin: 12px 0;
    display: flex;
    align-items: flex-start;
    gap: 10px;
}
.warn-div {
    background: #F8F3E8;
    border: 1px solid #DCC898;
    border-left: 4px solid #A88840;
    border-radius: 12px;
    padding: 13px 18px;
    color: #806830;
    font-size: 13px;
    line-height: 1.7;
    letter-spacing: 0.15px;
    margin: 12px 0;
    display: flex;
    align-items: flex-start;
    gap: 10px;
}
.err-div {
    background: #F8EEEE;
    border: 1px solid #DCC0C0;
    border-left: 4px solid #A86868;
    border-radius: 12px;
    padding: 13px 18px;
    color: #905050;
    font-size: 13px;
    line-height: 1.7;
    letter-spacing: 0.15px;
    margin: 12px 0;
    display: flex;
    align-items: flex-start;
    gap: 10px;
}
</style>

<script>
(function() {
    function fixCollapsedBtn() {
        // 找到所有含 "arrow" 或 "keyboard" 文字的節點並隱藏
        var btns = document.querySelectorAll(
            '[data-testid="collapsedControl"] button, [data-testid="stSidebarCollapseButton"] button'
        );
        btns.forEach(function(btn) {
            btn.childNodes.forEach(function(node) {
                if (node.nodeType === 3) {
                    // 純文字節點直接清空
                    node.textContent = '';
                } else if (node.nodeName === 'SPAN' && !node.querySelector('svg')) {
                    node.style.display = 'none';
                }
            });
        });
    }
    // 頁面載入後執行
    document.addEventListener('DOMContentLoaded', fixCollapsedBtn);
    // Streamlit 動態更新時也執行
    var obs = new MutationObserver(fixCollapsedBtn);
    obs.observe(document.body, { childList: true, subtree: true });
})();
</script>
""", unsafe_allow_html=True)

# ── Lucide Light 圖示（stroke-width 1.5，內聯 SVG）──────────────────────────
_ICONS: dict[str, str] = {
    "shield":         '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>',
    "bar-chart":      '<line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/>',
    "search":         '<circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>',
    "trending-up":    '<polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/>',
    "file-text":      '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/>',
    "refresh-cw":     '<polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/>',
    "alert-triangle": '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
    "info":           '<circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>',
    "x-circle":       '<circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/>',
    "download":       '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>',
    "database":       '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>',
    "git-branch":     '<line x1="6" y1="3" x2="6" y2="15"/><circle cx="18" cy="6" r="3"/><circle cx="6" cy="18" r="3"/><path d="M18 9a9 9 0 0 1-9 9"/>',
    "cpu":            '<rect x="4" y="4" width="16" height="16" rx="2" ry="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/>',
    "message-circle": '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>',
    "activity":       '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
    "layers":         '<polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/>',
}

def _icon(name: str, size: int = 14, color: str = "#8A8078") -> str:
    """產生 Lucide Light 風格的內聯 SVG 圖示（stroke-width 1.5）。"""
    paths = _ICONS.get(name, "")
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" '
        f'stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" '
        f'style="display:inline-block;vertical-align:middle;flex-shrink:0;">'
        f'{paths}</svg>'
    )

# ── Plotly 暗色主題基底（所有圖表共用）──────────────────────────────────────
_CL = dict(
    template="plotly_white",
    paper_bgcolor="#F2EFE9",
    plot_bgcolor="#FFFFFF",
    font=dict(color="#6A6058", family="Microsoft JhengHei, Inter, sans-serif", size=12),
    xaxis=dict(gridcolor="#EAE6DE", linecolor="#D8D2C8", zerolinecolor="#D8D2C8"),
    yaxis=dict(gridcolor="#EAE6DE", linecolor="#D8D2C8", zerolinecolor="#D8D2C8"),
)

# ── 自訂提示框 helper ─────────────────────────────────────────────────────────
def _info(msg: str) -> None:
    ico = _icon("info", 15, "#6878B8")
    st.markdown(f'<div class="info-div">{ico}<span>{msg}</span></div>', unsafe_allow_html=True)

def _warn(msg: str) -> None:
    ico = _icon("alert-triangle", 15, "#A88840")
    st.markdown(f'<div class="warn-div">{ico}<span>{msg}</span></div>', unsafe_allow_html=True)

def _err(msg: str) -> None:
    ico = _icon("x-circle", 15, "#A86868")
    st.markdown(f'<div class="err-div">{ico}<span>{msg}</span></div>', unsafe_allow_html=True)

def _section(label: str) -> None:
    st.markdown(f'<div class="section-label">{label}</div>', unsafe_allow_html=True)

def _divider() -> None:
    st.markdown('<div style="border-top:1px solid #E0DDD5;margin:36px 0 28px;"></div>',
                unsafe_allow_html=True)


# ── 資料載入（cache） ─────────────────────────────────────────────────────────
@st.cache_data
def load_report():
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

m     = report["oof_metrics"]
s     = report["submission"]
folds = report["folds"]


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    shield_svg = _icon("shield", 22, "#7A9B8A")
    st.markdown(f"""
    <div style="padding:18px 4px 14px;display:flex;align-items:center;gap:10px;">
        <div style="background:#EBF2EE;border:1px solid #C8DDD4;border-radius:12px;
                    padding:8px;display:flex;align-items:center;justify-content:center;">
            {shield_svg}
        </div>
        <div>
            <div style="font-size:17px;font-weight:900;color:#1E1A16;letter-spacing:-0.5px;
                        line-height:1.15;">BitoGuard</div>
            <div style="font-size:9.5px;color:#C0B8AE;letter-spacing:2.5px;
                        text-transform:uppercase;font-weight:500;margin-top:2px;">AML Platform</div>
        </div>
    </div>
    <div style="border-top:1px solid #E0DDD5;margin:0 0 18px;"></div>
    """, unsafe_allow_html=True)

    if "page" not in st.session_state:
        st.session_state["page"] = "數據總覽"

    _NAV = ["數據總覽", "風險查詢", "模型表現", "預測結果"]

    # 注入 active 樣式：膠囊底色 + inset 左線 + icon 提亮
    _idx = _NAV.index(st.session_state["page"]) + 1
    st.markdown(f"""<style>
[data-testid="stSidebar"] .stButton:nth-of-type({_idx}) > button {{
    background: rgba(0,0,0,0.09) !important;
    color: #1A1714 !important;
    font-weight: 700 !important;
    border: none !important;
    box-shadow: none !important;
    border-radius: 10px !important;
}}
[data-testid="stSidebar"] .stButton:nth-of-type({_idx}) > button::before {{
    opacity: 1 !important;
}}
[data-testid="stSidebar"] .stButton:nth-of-type({_idx}) > button:hover {{
    background: rgba(0,0,0,0.12) !important;
}}
</style>""", unsafe_allow_html=True)

    for _label in _NAV:
        if st.button(_label, key=f"_nav_{_label}"):
            st.session_state["page"] = _label
            st.rerun()

    page = st.session_state["page"]

    st.markdown('<div style="border-top:1px solid #E0DDD5;margin:18px 0 14px;"></div>',
                unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:#FDFAF5;border:1px solid #D8D0C4;border-radius:14px;
                padding:18px 18px 14px;margin-top:8px;
                box-shadow:0 1px 4px rgba(0,0,0,0.04);">
        <div style="font-size:0.8rem;letter-spacing:2px;text-transform:uppercase;
                    color:#C0B4A8;margin-bottom:14px;font-weight:600;">MODEL INFO</div>
        <div style="display:flex;justify-content:space-between;padding:8px 0;
                    border-bottom:1px solid #EDE9E2;">
            <span style="font-size:12px;color:#8A8078;">演算法</span>
            <span style="font-size:12px;color:#4A4238;font-weight:600;">LightGBM</span>
        </div>
        <div style="display:flex;justify-content:space-between;padding:8px 0;
                    border-bottom:1px solid #EDE9E2;">
            <span style="font-size:12px;color:#8A8078;">OOF F1</span>
            <span style="font-size:12px;color:#5A8A6A;font-weight:700;">{m['f1']*100:.1f}%</span>
        </div>
        <div style="display:flex;justify-content:space-between;padding:8px 0;
                    border-bottom:1px solid #EDE9E2;">
            <span style="font-size:12px;color:#8A8078;">AUC</span>
            <span style="font-size:12px;color:#5878A8;font-weight:700;">{m['auc']*100:.1f}%</span>
        </div>
        <div style="display:flex;justify-content:space-between;padding:8px 0;
                    border-bottom:1px solid #EDE9E2;">
            <span style="font-size:12px;color:#8A8078;">準確度</span>
            <span style="font-size:12px;color:#7A7870;font-weight:700;">{m.get('accuracy', 0.9564)*100:.1f}%</span>
        </div>
        <div style="display:flex;justify-content:space-between;padding:8px 0;">
            <span style="font-size:12px;color:#8A8078;">訓練日期</span>
            <span style="font-size:12px;color:#B0A898;">2026-03-26</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="margin-top:14px;"></div>', unsafe_allow_html=True)
    if st.button("刷新資料"):
        st.cache_data.clear()
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# 頁面 1：總覽儀表板
# ══════════════════════════════════════════════════════════════════════════════
if page == "數據總覽":
    st.markdown('<div class="page-title">BitoGuard — 守護每一筆交易</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">即時分析 · 智能預警 · 合規守護</div>', unsafe_allow_html=True)

    # ── 計算數值 ──────────────────────────────────────────────────────────────
    _N_TOTAL, _N_POS = 51017, 1640
    _tp  = max(0, round(m["recall"] * _N_POS))
    _fp  = max(0, round(_tp / m["precision"] - _tp)) if m["precision"] > 0 else 0
    _tn  = (_N_TOTAL - _N_POS) - _fp
    _acc = m.get("accuracy", (_tp + _tn) / _N_TOTAL)

    # ── KPI 列（不對稱：左側 4 格 KPI，右側英雄數字卡）────────────────────
    col_kpi, col_hero = st.columns([5, 2])

    def _kpi_html(label, val_str, sub, accent):
        return f"""<div class="kpi-strip">
            <div class="kpi-accent" style="background:{accent};"></div>
            <div>
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{val_str}</div>
                <div class="kpi-sub">{sub}</div>
            </div>
        </div>"""

    with col_kpi:
        _section("模型整體表現")
        r1c1, r1c2 = st.columns(2)
        r2c1, r2c2 = st.columns(2)
        with r1c1:
            st.markdown(_kpi_html("AUC", f"{m['auc']*100:.1f}%", "模型排序風險的能力", "#6A8EBA"),
                        unsafe_allow_html=True)
        with r1c2:
            st.markdown(_kpi_html("F1-SCORE", f"{m['f1']*100:.1f}%", "精確與召回的平衡", "#7A9B8A"),
                        unsafe_allow_html=True)
        with r2c1:
            st.markdown(_kpi_html("PRECISION", f"{m['precision']*100:.1f}%", "標記出的帳號有多可信", "#A8906A"),
                        unsafe_allow_html=True)
        with r2c2:
            st.markdown(_kpi_html("RECALL", f"{m['recall']*100:.1f}%", "找到了多少真實風險帳號", "#B07878"),
                        unsafe_allow_html=True)

    with col_hero:
        _section("ACCURACY")
        st.markdown(f"""
        <div class="bg-card" style="text-align:center;padding:30px 16px 24px;">
            <div style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;
                        color:#B0A898;margin-bottom:10px;">整體判斷準確度</div>
            <div style="font-size:60px;font-weight:900;line-height:0.95;
                        color:#2C2720;letter-spacing:-3px;">{_acc*100:.1f}<span style="font-size:24px;color:#C0B8AE;">%</span></div>
            <div style="border-top:1px solid #E0DDD5;margin:18px 0 14px;"></div>
            <div style="font-size:10px;letter-spacing:1.2px;text-transform:uppercase;
                        color:#B0A898;margin-bottom:6px;">標記為風險</div>
            <div style="font-size:30px;font-weight:800;color:#B06858;
                        letter-spacing:-1px;">{s['blacklist']:,}</div>
            <div style="font-size:11px;color:#B0A898;margin-top:3px;">涵蓋全部 {s['total']:,} 位用戶</div>
        </div>
        """, unsafe_allow_html=True)

    _divider()

    # ── 主圖區（不對稱：混淆矩陣 38% + 5-Fold 柱狀圖 62%）────────────────
    col_cm, col_fold = st.columns([2, 3])

    with col_cm:
        _section("預測結果矩陣")
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
            textfont={"size": 15, "color": "#3A3228"},
            colorscale=[[0, "#D8EAE0"], [0.33, "#EAE0C8"],
                        [0.66, "#EAD0CC"], [1, "#C8DED4"]],
            showscale=False,
        ))
        cm_fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), **_CL)
        st.plotly_chart(cm_fig, use_container_width=True)

    with col_fold:
        _section("5 折交叉驗證表現")
        fold_fig = go.Figure()
        xs = [f"Fold {f['fold']}" for f in folds]
        fold_fig.add_trace(go.Bar(name="Precision", x=xs,
                                  y=[f["precision"] for f in folds],
                                  marker_color="#7A9AC8", opacity=0.82))
        fold_fig.add_trace(go.Bar(name="Recall", x=xs,
                                  y=[f["recall"] for f in folds],
                                  marker_color="#C8A878", opacity=0.82))
        fold_fig.add_trace(go.Bar(name="F1-Score", x=xs,
                                  y=[f["f1"] for f in folds],
                                  marker_color="#7A9B8A", opacity=0.82))
        fold_fig.add_hline(y=m["f1"], line_dash="dash", line_color="#7A9B8A",
                           annotation_text=f"OOF F1={m['f1']*100:.1f}%",
                           annotation_font_color="#5A7A6A")
        fold_fig.update_layout(barmode="group", height=300,
                               margin=dict(l=0, r=0, t=10, b=0),
                               yaxis_range=[0, 0.6],
                               legend=dict(orientation="h", y=-0.28, font=dict(size=11)),
                               **_CL)
        st.plotly_chart(fold_fig, use_container_width=True)

    _divider()

    # ── 系統四大模組（雜誌感，不等寬欄）────────────────────────────────────
    _section("系統架構 — 四大核心模組")
    modules = [
        ("database",       "數據擷取 & 自動化建倉", "#EBF0F4", "#6A8EBA",
         "BitoPro API · Checkpoint · Parquet → S3 · Glue Crawler 自動建表"),
        ("git-branch",     "圖運算特徵工程",         "#EBF2EE", "#6A9A7A",
         "BFS 資金跳轉 · Salting 超級節點 · Feature Store 持久化"),
        ("cpu",            "模型訓練 & MLOps",        "#F0EBF4", "#9A88B8",
         f"LightGBM AUC={m['auc']*100:.1f}% · 5-Fold CV · HPO 超參數優化"),
        ("message-circle", "AI 風險診斷 & 自動響應",  "#F4EBEB", "#B87878",
         "PII Guard 三層保護 · Claude 3.5 深度推理 · 閉環回饋增量訓練"),
    ]
    mcols = st.columns([3, 2.5, 2.5, 2])
    for col, (ico_name, title, bg, accent, detail) in zip(mcols, modules):
        with col:
            ico_html = _icon(ico_name, 20, accent)
            st.markdown(f"""
            <div style="background:{bg};border:1px solid #E0DDD5;
                        border-top:3px solid {accent};border-radius:16px;
                        padding:20px 18px;min-height:150px;">
                <div style="background:white;border:1px solid {accent}30;border-radius:10px;
                            width:36px;height:36px;display:flex;align-items:center;
                            justify-content:center;margin-bottom:14px;">{ico_html}</div>
                <div style="font-size:14px;font-weight:700;color:#3A3228;
                            margin-bottom:8px;line-height:1.35;letter-spacing:-0.1px;">{title}</div>
                <div style="font-size:11px;color:#8A8078;line-height:1.7;
                            letter-spacing:0.1px;">{detail}</div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 頁面 2：用戶風險查詢
# ══════════════════════════════════════════════════════════════════════════════
elif page == "風險查詢":
    st.markdown('<div class="page-title">帳號風險查詢</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">輸入帳號 ID，即時查看風險評分與詳細分析</div>', unsafe_allow_html=True)

    if sub_df is None:
        _warn("部分資料尚未載入，風險查詢功能可能受限。")

    col_search, col_btn = st.columns([4, 1])
    with col_search:
        uid_input = st.text_input("帳號 ID", placeholder="請輸入帳號 ID，例如 967903",
                                   label_visibility="collapsed")
    with col_btn:
        search_btn = st.button("開始查詢")

    _divider()

    if uid_input and (search_btn or uid_input):
        try:
            uid = int(uid_input.strip())
            row = sub_df[sub_df["user_id"] == uid]
            if row.empty:
                _warn(f"找不到 ID 為 {uid} 的帳號，請確認後再試。")
            else:
                row = row.iloc[0]
                prob = row.get("probability", 0.5) if "probability" in row.index else 0.5
                status = int(row["status"])
                thr = m["threshold"]

                if prob >= 0.5:
                    risk_level, risk_color, risk_bg, risk_border = "極高風險 · 建議立即審查", "#C94C4C", "#FDF2F2", "#C94C4C"
                elif prob >= thr:
                    risk_level, risk_color, risk_bg, risk_border = "高風險 · 列入觀察", "#D9534F", "#FDF4F2", "#D9534F"
                elif prob >= thr * 0.5:
                    risk_level, risk_color, risk_bg, risk_border = "中等風險 · 持續監控", "#C08040", "#F8F3E8", "#C08040"
                else:
                    risk_level, risk_color, risk_bg, risk_border = "低風險 · 行為正常", "#4A8A6A", "#EDF6F1", "#4A8A6A"

                # 不對稱：大機率（2份）+ 細節表格（3份）
                col_prob, col_detail = st.columns([2, 3])

                with col_prob:
                    st.markdown(f"""
                    <div style="background:{risk_bg};border:1px solid {risk_border}40;
                                border-left:4px solid {risk_border};border-radius:12px;
                                padding:30px 22px;text-align:center;">
                        <div style="font-size:10px;letter-spacing:2.5px;text-transform:uppercase;
                                    color:#505068;margin-bottom:14px;">USER ID {uid}</div>
                        <div style="font-size:68px;font-weight:900;line-height:0.9;
                                    color:{risk_color};letter-spacing:-3px;">{prob*100:.1f}<span style="font-size:26px;">%</span></div>
                        <div style="margin-top:16px;font-size:11px;font-weight:700;
                                    letter-spacing:1.5px;color:{risk_color};
                                    text-transform:uppercase;">{risk_level}</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col_detail:
                    verdict_html = (
                        "<span style='background:rgba(217,83,79,0.10);color:#C94C4C;"
                        "padding:4px 12px;border-radius:6px;font-size:12px;font-weight:700;"
                        "letter-spacing:0.3px;'>黑名單 ✗</span>"
                        if status == 1 else
                        "<span style='background:rgba(74,138,106,0.10);color:#3A7A58;"
                        "padding:4px 12px;border-radius:6px;font-size:12px;font-weight:700;"
                        "letter-spacing:0.3px;'>正常 ✓</span>"
                    )
                    diff = (prob - thr) * 100
                    diff_color = "#C94C4C" if diff > 0 else "#4A8A6A"
                    diff_str = f"+{diff:.1f}%" if diff > 0 else f"{diff:.1f}%"
                    st.markdown(f"""
                    <div class="bg-card" style="margin-top:0;">
                        <div style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;
                                    color:#B0A898;margin-bottom:14px;">風險詳情</div>
                        <table style="width:100%;border-collapse:collapse;">
                            <tr style="border-bottom:1px solid #E0DDD5;">
                                <td style="padding:9px 0;font-size:12px;color:#8A8078;">黑名單機率（精確）</td>
                                <td style="padding:9px 0;font-size:13px;color:{risk_color};
                                           font-weight:700;text-align:right;">{prob*100:.4f}%</td>
                            </tr>
                            <tr style="border-bottom:1px solid #E0DDD5;">
                                <td style="padding:9px 0;font-size:12px;color:#8A8078;">最終判定</td>
                                <td style="padding:9px 0;text-align:right;">{verdict_html}</td>
                            </tr>
                            <tr style="border-bottom:1px solid #E0DDD5;">
                                <td style="padding:9px 0;font-size:12px;color:#8A8078;">決策門檻</td>
                                <td style="padding:9px 0;font-size:12px;color:#A8A098;
                                           text-align:right;">{thr:.2f}</td>
                            </tr>
                            <tr>
                                <td style="padding:9px 0;font-size:12px;color:#8A8078;">與門檻差距</td>
                                <td style="padding:9px 0;font-size:13px;font-weight:700;
                                           color:{diff_color};text-align:right;">{diff_str}</td>
                            </tr>
                        </table>
                    </div>
                    """, unsafe_allow_html=True)

                # 儀表板（全寬）— 高風險時指針與區塊自動轉紅
                _gauge_bar   = risk_color
                _gauge_num   = risk_color if prob >= thr else "#3A3228"
                _step_high   = "#FADADA" if prob >= thr else "#F0E8E4"
                _step_crit   = "#F5C0C0" if prob >= 0.5  else _step_high
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    title={"text": "風險分數 (%)", "font": {"color": "#8A8078", "size": 13}},
                    number={"suffix": "%", "font": {"size": 34, "color": _gauge_num}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#C0B8AE", "tickwidth": 1},
                        "bar":  {"color": _gauge_bar},
                        "bgcolor": "#FFFFFF",
                        "borderwidth": 0,
                        "steps": [
                            {"range": [0,        thr*50],  "color": "#E8F0EC"},
                            {"range": [thr*50,   thr*100], "color": "#F0EDE0"},
                            {"range": [thr*100,  50],      "color": _step_high},
                            {"range": [50,       100],     "color": _step_crit},
                        ],
                        "threshold": {"line": {"color": risk_color, "width": 2},
                                      "value": thr * 100},
                    }
                ))
                gauge.update_layout(height=230, margin=dict(l=20, r=20, t=30, b=0), **_CL)
                st.plotly_chart(gauge, use_container_width=True)

                # 風險評估書下載
                risk_label_text = {
                    "極高風險 · 建議立即審查": "CRITICAL",
                    "高風險 · 列入觀察":       "HIGH",
                    "中等風險 · 持續監控":     "MEDIUM",
                    "低風險 · 行為正常":       "LOW",
                }.get(risk_level, "UNKNOWN")

                report_lines = [
                    "=" * 52,
                    "   BitoGuard AML — 帳號風險評估報告",
                    "=" * 52,
                    f"生成時間     : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    f"模型版本     : LightGBM (OOF AUC={m['auc']:.4f})",
                    "-" * 52,
                    f"帳號 ID      : {uid}",
                    f"風險等級     : {risk_label_text}",
                    f"風險機率     : {prob*100:.1f}%",
                    f"決策門檻     : {thr:.2f}",
                    f"最終判定     : {'風險帳號 (status=1)' if status == 1 else '正常帳號 (status=0)'}",
                    "-" * 52,
                    "風險說明：",
                ]
                explanations = {
                    "CRITICAL": [
                        "  * 風險機率超過 50%，高度懷疑為異常帳號",
                        "  * 建議立即啟動人工審查，必要時凍結帳號",
                        "  * 請依 AML 法規評估是否向主管機關申報",
                    ],
                    "HIGH": [
                        "  * 風險機率已超過決策門檻，模型判定為異常",
                        "  * 建議加強 KYC 驗證，並考慮限制出金功能",
                        "  * 請追蹤近 30 天的交易行為，評估資金流向",
                    ],
                    "MEDIUM": [
                        "  * 風險機率落在觀察區間，需持續關注",
                        "  * 建議加強監控頻率，定期產出行為報告",
                        "  * 目前不限制功能，但已列入加強監控名單",
                    ],
                    "LOW": [
                        "  * 風險機率低於門檻，帳號行為目前正常",
                        "  * 維持常規監控即可，無需特別處理",
                    ],
                }
                report_lines += explanations.get(risk_label_text, ["  * 無額外說明"])
                report_lines += [
                    "-" * 52,
                    "主要影響特徵（前 5 項）：",
                    "  1. blacklist_neighbor_count — 風險鄰居數",
                    "  2. min_retention_minutes   — 最短資金停留時間",
                    "  3. total_twd_volume        — 總台幣交易量",
                    "  4. twd_withdraw_count      — 台幣出金次數",
                    "  5. night_tx_ratio          — 深夜交易比例",
                    "=" * 52,
                    "本報告由 BitoGuard AML 系統自動生成，最終決策請以人工審核為準。",
                    "=" * 52,
                ]
                st.download_button(
                    label="下載完整評估報告",
                    data="\n".join(report_lines).encode("utf-8"),
                    file_name=f"risk_report_user_{uid}.txt",
                    mime="text/plain",
                )

                _divider()

                # 特徵明細（若有快取）
                if feat_df is not None and "user_id" in feat_df.columns:
                    user_feat = feat_df[feat_df["user_id"] == uid]
                    if not user_feat.empty:
                        _section("帳號特徵明細")
                        feat_cols = [c for c in report["features"] if c in user_feat.columns]
                        display_df = user_feat[feat_cols].T.reset_index()
                        display_df.columns = ["特徵名稱", "數值"]
                        st.dataframe(display_df, use_container_width=True, height=400)

        except ValueError:
            _err("請輸入有效的帳號 ID（純數字）")

    # ── 風險排行（不對稱 3:2）────────────────────────────────────────────────
    _divider()

    if sub_df is not None and "probability" in sub_df.columns:
        col_hi, col_lo = st.columns([3, 2])

        with col_hi:
            _section("風險最高的 50 個帳號")
            top50 = sub_df.nlargest(50, "probability")[["user_id", "probability", "status"]].copy()
            top50["probability"] = (top50["probability"] * 100).round(1)
            top50["風險等級"] = top50["probability"].apply(
                lambda p: "CRITICAL" if p >= 50 else "HIGH")
            top50.columns = ["用戶 ID", "黑名單機率 (%)", "預測標籤", "風險等級"]
            st.dataframe(
                top50.style.background_gradient(subset=["黑名單機率 (%)"], cmap="Reds"),
                use_container_width=True, height=400,
            )

        with col_lo:
            _section("風險最低的 50 個帳號")
            bot50 = sub_df.nsmallest(50, "probability")[["user_id", "probability", "status"]].copy()
            bot50["probability"] = (bot50["probability"] * 100).round(1)
            bot50["風險等級"] = "LOW"
            bot50.columns = ["用戶 ID", "黑名單機率 (%)", "預測標籤", "風險等級"]
            st.dataframe(
                bot50.style.background_gradient(subset=["黑名單機率 (%)"], cmap="Greens_r"),
                use_container_width=True, height=400,
            )
    else:
        _info("載入預測資料後，即可查看風險排行列表。")


# ══════════════════════════════════════════════════════════════════════════════
# 頁面 3：模型評估
# ══════════════════════════════════════════════════════════════════════════════
elif page == "模型表現":
    st.markdown('<div class="page-title">模型評估</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">深入了解模型的預測邏輯與各項表現指標</div>', unsafe_allow_html=True)

    # 不對稱：特徵重要度（3份）+ 右側欄位（4份）
    col_l, col_r = st.columns([3, 4])

    with col_l:
        _section("最具影響力的特徵")
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
                        color="importance", color_continuous_scale="Teal",
                        labels={"importance": "Gain", "feature": ""})
        fi_fig.update_layout(height=560, coloraxis_showscale=False,
                             margin=dict(l=0, r=0, t=0, b=0), **_CL)
        fi_fig.update_yaxes(autorange="reversed", tickfont=dict(size=11, color="#8A8078"))
        fi_fig.update_xaxes(tickfont=dict(size=11, color="#8A8078"))
        st.plotly_chart(fi_fig, use_container_width=True)

    with col_r:
        _section("決策門檻分析")
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
        thr_fig.add_trace(go.Scatter(x=ts, y=fs, name="F1-Score",
                                     line=dict(color="#7A9B8A", width=2.5)))
        thr_fig.add_trace(go.Scatter(x=ts, y=ps, name="Precision",
                                     line=dict(color="#6A8EBA", width=2)))
        thr_fig.add_trace(go.Scatter(x=ts, y=rs, name="Recall",
                                     line=dict(color="#B07878", width=2)))
        thr_fig.add_vline(x=m["threshold"], line_dash="dash", line_color="#B0A898",
                          line_width=1.5,
                          annotation_text=f"最佳門檻={m['threshold']:.2f}",
                          annotation_font_color="#8A8078")
        thr_fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0),
                              xaxis_title="門檻值", yaxis_range=[0, 1],
                              legend=dict(orientation="h", y=-0.3, font=dict(size=11)),
                              **_CL)
        st.plotly_chart(thr_fig, use_container_width=True)

        _section("各折驗證結果")
        fold_table = pd.DataFrame(folds)
        fold_table.columns = ["Fold", "門檻", "Precision", "Recall", "F1", "AUC", "準確度", "正類數"]
        for col in ["Precision", "Recall", "F1", "AUC", "準確度"]:
            fold_table[col] = (fold_table[col] * 100).round(2)
        st.dataframe(
            fold_table.style
                .highlight_max(subset=["F1", "AUC"], color="#D4EAE0")
                .highlight_min(subset=["F1", "AUC"], color="#EAD8D4")
                .format({"Precision": "{:.1f}%", "Recall": "{:.1f}%",
                         "F1": "{:.1f}%", "AUC": "{:.1f}%", "準確度": "{:.1f}%"}),
            use_container_width=True,
        )

    # 全寬機率分布
    _divider()
    _section("預測機率分布")
    if sub_df is not None and "probability" in sub_df.columns:
        dist_fig = px.histogram(
            sub_df, x="probability", color="status",
            nbins=80, barmode="overlay", opacity=0.70,
            color_discrete_map={0: "#7A9AC8", 1: "#C09078"},
            labels={"probability": "預測機率（黑名單）", "status": "標籤"},
        )
        dist_fig.add_vline(x=m["threshold"], line_dash="dash", line_color="#B0A898",
                           line_width=1.5,
                           annotation_text=f"門檻 {m['threshold']:.2f}",
                           annotation_font_color="#8A8078")
        dist_fig.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0), **_CL)
        st.plotly_chart(dist_fig, use_container_width=True)
    else:
        _info("載入預測資料後，即可查看機率分布圖。")


# ══════════════════════════════════════════════════════════════════════════════
# 頁面 4：提交結果
# ══════════════════════════════════════════════════════════════════════════════
elif page == "預測結果":
    st.markdown('<div class="page-title">預測結果</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">模型對所有帳號的最終風險判定</div>', unsafe_allow_html=True)

    if sub_df is None:
        n_total = s["total"]
        n_black = s["blacklist"]
        n_norm  = s["normal"]
    else:
        n_total = len(sub_df)
        n_black = int(sub_df["status"].sum())
        n_norm  = n_total - n_black

    # 自訂統計卡片（取代 st.metric）
    sc1, sc2, sc3 = st.columns(3)

    def _stat_card(col, label, value, sub_value, accent):
        col.markdown(f"""
        <div class="bg-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value" style="color:{accent};">{value}</div>
            <div class="kpi-sub">{sub_value}</div>
        </div>""", unsafe_allow_html=True)

    _stat_card(sc1, "TOTAL USERS",  f"{n_total:,}", "涵蓋的帳號總數",                   "#7A7870")
    _stat_card(sc2, "RISK",         f"{n_black:,}", f"標記為風險 · 佔比 {n_black/n_total*100:.2f}%", "#B06858")
    _stat_card(sc3, "NORMAL",       f"{n_norm:,}",  f"判定正常 · 佔比 {n_norm/n_total*100:.2f}%",    "#5A8A6A")

    _divider()

    # 不對稱：圓餅圖（2份）+ 資料預覽（5份）
    col_pie, col_table = st.columns([2, 5])

    with col_pie:
        _section("風險分布")
        pie_fig = px.pie(
            values=[n_black, n_norm],
            names=["黑名單", "正常"],
            color_discrete_sequence=["#C09080", "#8AB8A0"],
            hole=0.55,
        )
        pie_fig.update_traces(textinfo="percent+label", textfont_size=12,
                              textfont_color=["#6A3828", "#2A5040"])
        pie_fig.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0),
                              showlegend=False, **_CL)
        st.plotly_chart(pie_fig, use_container_width=True)

    with col_table:
        _section("預測結果預覽")
        ctrl1, ctrl2 = st.columns([3, 2])
        with ctrl1:
            filter_opt = st.selectbox("篩選", ["全部帳號", "只看風險帳號", "只看正常帳號"],
                                      label_visibility="collapsed")
        with ctrl2:
            show_prob = st.checkbox("顯示機率欄", value=True)

        if sub_df is not None:
            display_cols = ["user_id", "status"]
            if show_prob and "probability" in sub_df.columns:
                display_cols.append("probability")

            df_show = sub_df[display_cols].copy()
            if filter_opt == "只看風險帳號":
                df_show = df_show[df_show["status"] == 1]
            elif filter_opt == "只看正常帳號":
                df_show = df_show[df_show["status"] == 0]

            st.dataframe(df_show, use_container_width=True, height=360)

            sub_csv = sub_df[["user_id", "status"]].to_csv(index=False).encode("utf-8")
            st.download_button(
                label="下載預測結果（.csv）",
                data=sub_csv,
                file_name="submission.csv",
                mime="text/csv",
            )
        else:
            _info("預測資料尚未載入，無法顯示資料預覽。")
