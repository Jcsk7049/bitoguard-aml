"""
XAI 風險診斷書生成器 v3 — Amazon Bedrock + SHAP + Batch Scoring

╔══════════════════════════════════════════════════════════════════╗
║  Batch Scoring 路由策略                                          ║
║  ─────────────────────────────────────────────────────────────  ║
║  EXTREME  P > 0.90          → Claude 3 Haiku  (制式報告，低成本) ║
║  BOUNDARY 0.65 ≤ P ≤ 0.75  → Claude 3.5 Sonnet (深度診斷)      ║
║  HIGH     0.75 < P ≤ 0.90  → Claude 3.5 Sonnet (標準分析)      ║
║  MEDIUM   0.50 ≤ P < 0.65  → Rule-Based Only  (不調用 LLM)     ║
║  LOW      P < 0.50          → 跳過                              ║
╠══════════════════════════════════════════════════════════════════╣
║  Structured Output（JSON Mode）                                  ║
║  所有 Bedrock 回應均以 JSON Schema 強制輸出，包含：              ║
║    • primary_action  : 機器可執行的動作代碼                      ║
║    • auto_executable : 後台是否可自動執行鎖帳                    ║
║    • execution_priority: 緊急程度 1（立即）~ 5（低優先）         ║
║    • str_required    : 是否需提交可疑交易報告                    ║
╚══════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import boto3
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from botocore.exceptions import ClientError

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ══════════════════════════════════════════════════════════════════════════════
#  設定
# ══════════════════════════════════════════════════════════════════════════════

BEDROCK_REGION     = os.environ.get("BEDROCK_REGION",     "us-east-1")
COMPREHEND_REGION  = os.environ.get("COMPREHEND_REGION",  "us-east-1")
RISK_THRESHOLD     = 0.50          # 最低觸發門檻

# 模型路由對照
MODEL_SONNET = "anthropic.claude-3-5-sonnet-20241022-v2:0"
MODEL_HAIKU  = "anthropic.claude-3-haiku-20240307-v1:0"

# ── 本專案唯一允許呼叫的模型白名單 ──────────────────────────────────────────
# 防止誤啟用非必要模型（競賽規定：最小化模型存取請求）。
# 所有 _BaseWriter 子類別的 model_id 必須在此集合內，否則拒絕初始化。
_ALLOWED_MODELS: frozenset[str] = frozenset({
    MODEL_SONNET,   # Claude 3.5 Sonnet — BOUNDARY / HIGH 深度分析
    MODEL_HAIKU,    # Claude 3 Haiku    — EXTREME 制式報告（低成本高速）
})

# 機率區間定義
TIER_EXTREME_MIN  = 0.90   # P > 0.90 → Haiku
TIER_BOUNDARY_LO  = 0.65   # ┐
TIER_BOUNDARY_HI  = 0.75   # ┘ 0.7 ± 0.05 → Sonnet 深度
TIER_HIGH_HI      = 0.90   # 0.75 < P ≤ 0.90 → Sonnet 標準
TIER_MEDIUM_LO    = 0.50   # 0.50 ≤ P < 0.65 → Rule-Based

# 特徵中文對照表
FEATURE_LABELS: dict[str, str] = {
    "min_retention_minutes":  "最短資金滯留時間（分鐘）",
    "retention_event_count":  "快進快出事件次數",
    "high_speed_risk":        "高速資金風險旗標",
    "unique_ip_count":        "不同 IP 數量",
    "ip_anomaly":             "IP 異常旗標",
    "total_twd_volume":       "總交易量（TWD）",
    "volume_zscore":          "交易量 Z-score（同 KYC 群組偏離度）",
    "asymmetry_flag":         "量能不對稱旗標",
    "kyc_level":              "KYC 等級（0=未驗證, 1=手機, 2=身份）",
    "twd_deposit_count":      "台幣入金次數",
    "twd_withdraw_count":     "台幣出金次數",
    "crypto_deposit_count":   "虛幣入金次數",
    "crypto_withdraw_count":  "虛幣出金次數",
    "night_tx_ratio":         "深夜交易比例（22:00–06:00）",
    "mule_risk_score":        "人頭戶風險綜合評分（0–3）",
    "min_hops_to_blacklist":  "距黑名單最短跳轉數",
    "is_direct_neighbor":     "直接黑名單鄰居旗標",
    "blacklist_neighbor_count": "直接相連黑名單節點數",
    "in_blacklist_network":   "在黑名單連通分量中",
}


# ══════════════════════════════════════════════════════════════════════════════
#  SHAP 自然語言解讀範本
#
#  格式：feature_name → (高風險描述模板, 低風險描述模板)
#    模板中 {value} 代換實際特徵值，{shap:.3f} 代換 SHAP 值
#    高風險模板：SHAP > 0（該特徵提高了風險評分）
#    低風險模板：SHAP < 0（該特徵降低了風險評分）
# ══════════════════════════════════════════════════════════════════════════════

FEATURE_INTERPRETATION: dict[str, tuple[str, str]] = {
    "min_retention_minutes": (
        "資金僅滯留約 {value:.0f} 分鐘即出金，遠低於正常用戶平均（高速過境特徵，疑似人頭洗幣）",
        "資金滯留達 {value:.0f} 分鐘，符合正常交易者持倉行為，無快進快出風險",
    ),
    "retention_event_count": (
        "累計發生 {value:.0f} 次快進快出事件，顯示用戶長期存在高速資金流轉模式",
        "快進快出事件次數僅 {value:.0f} 次，屬正常範圍",
    ),
    "high_speed_risk": (
        "系統偵測到高速資金風險旗標（旗標值={value:.0f}），確認存在 < 10 分鐘的閃電出金紀錄",
        "無高速資金風險旗標，出入金時序正常",
    ),
    "unique_ip_count": (
        "登入 IP 多達 {value:.0f} 個不同節點，異常 IP 跳躍可能代表多人共用帳號或使用跳板",
        "僅使用 {value:.0f} 個 IP，登入行為穩定",
    ),
    "ip_anomaly": (
        "IP 異常旗標觸發（旗標值={value:.0f}），與已知代理節點或異常地理位置相符",
        "IP 地理位置與登入行為無異常",
    ),
    "total_twd_volume": (
        "台幣交易總量達 {value:,.0f} 元，對應 KYC 等級而言顯著偏高（資金流量遠超同群體均值）",
        "台幣交易量 {value:,.0f} 元，符合同 KYC 等級用戶的正常交易規模",
    ),
    "volume_zscore": (
        "交易量 Z-score={value:.2f}，偏離同群體均值逾 {value:.1f} 個標準差，屬統計異常值",
        "交易量 Z-score={value:.2f}，在同 KYC 群組的正常分布範圍內",
    ),
    "asymmetry_flag": (
        "量能不對稱旗標觸發（值={value:.0f}），入金後短時間內大額出金，符合「快速提領」洗幣模式",
        "入出金量能平衡，無不對稱資金轉移跡象",
    ),
    "kyc_level": (
        "KYC 等級僅 {value:.0f}（未完成進階身份驗證），在高風險交易量下缺乏身份保障",
        "KYC 等級 {value:.0f}，已完成身份驗證，帳戶實名可信度較高",
    ),
    "twd_deposit_count": (
        "台幣入金 {value:.0f} 次，頻繁入金搭配快速出金是典型洗幣前段作業特徵",
        "台幣入金 {value:.0f} 次，屬正常交易頻率",
    ),
    "twd_withdraw_count": (
        "台幣出金 {value:.0f} 次，密集出金行為可能對應多筆資金快速轉出",
        "台幣出金 {value:.0f} 次，出金頻率正常",
    ),
    "crypto_deposit_count": (
        "虛幣入金 {value:.0f} 次，搭配台幣大額出金暗示「虛幣接收→法幣套現」的洗幣路徑",
        "虛幣入金 {value:.0f} 次，無異常",
    ),
    "crypto_withdraw_count": (
        "虛幣出金 {value:.0f} 次，頻繁將虛幣轉出至外部錢包是洗幣後段典型操作",
        "虛幣出金 {value:.0f} 次，頻率正常",
    ),
    "night_tx_ratio": (
        "深夜交易比例達 {value:.1%}，逾四成交易發生於 22:00–06:00，規避監控的可能性較高",
        "深夜交易比例僅 {value:.1%}，交易時段符合正常日間作息",
    ),
    "mule_risk_score": (
        "人頭戶綜合風險評分 {value:.0f}/3，多項人頭指標同時觸發，高度懷疑為人頭帳戶",
        "人頭戶風險評分 {value:.0f}/3，無人頭帳戶特徵",
    ),
    "min_hops_to_blacklist": (
        "距已知黑名單用戶僅 {value:.0f} 跳，資金網絡與洗幣帳戶高度關聯",
        "距黑名單用戶達 {value:.0f} 跳，尚在安全距離之外",
    ),
    "is_direct_neighbor": (
        "直接黑名單鄰居旗標觸發（值={value:.0f}），與黑名單用戶有一度直接資金往來",
        "非黑名單用戶的直接鄰居，無一度直接關聯",
    ),
    "blacklist_neighbor_count": (
        "直接相連的黑名單節點達 {value:.0f} 個，關聯黑名單的廣度異常高",
        "直接相連黑名單節點僅 {value:.0f} 個，關聯程度有限",
    ),
    "in_blacklist_network": (
        "確認位於黑名單連通分量之中（值={value:.0f}），屬於已知洗幣網絡的組成節點",
        "未進入黑名單連通分量，與洗幣網絡無結構性關聯",
    ),
    "ip_shared_user_count": (
        "與 {value:.0f} 個其他帳號共用同一 IP，可能為詐騙集團批量操控的帳號群",
        "共用 IP 的帳號數僅 {value:.0f}，無帳號集群風險",
    ),
    "has_high_speed_risk": (
        "Athena 圖分析確認存在高速交易風險（值={value:.0f}），與行為特徵層相互印證",
        "Athena 圖分析未偵測到高速交易風險",
    ),
}


def _shap_to_narrative(
    feature_name: str,
    feature_value: float,
    shap_value: float,
    contribution_pct: float,
) -> str:
    """
    將單一 SHAP 貢獻轉換為合規人員易懂的繁體中文自然語言說明。

    優先使用 FEATURE_INTERPRETATION 中的模板；若無對應模板，
    則產生通用格式說明，確保所有特徵都有輸出。

    Parameters
    ----------
    feature_name     : 特徵英文名稱（與 FEATURE_LABELS key 一致）
    feature_value    : 該特徵的實際數值
    shap_value       : SHAP 值（>0 增加風險，<0 降低風險）
    contribution_pct : 貢獻百分比（0–100）

    Returns
    -------
    自然語言說明字串（繁體中文）
    """
    is_risk = shap_value > 0
    label   = FEATURE_LABELS.get(feature_name, feature_name)
    pct_str = f"（佔模型決策貢獻 {contribution_pct:.1f}%）"

    templates = FEATURE_INTERPRETATION.get(feature_name)
    if templates:
        template = templates[0] if is_risk else templates[1]
        try:
            narrative = template.format(value=feature_value, shap=shap_value)
        except (KeyError, ValueError):
            narrative = template
        direction = "⬆ 風險加分" if is_risk else "⬇ 風險減分"
        return f"{direction} {narrative}{pct_str}"

    # 無範本時的通用描述
    direction = "增加" if is_risk else "降低"
    return (
        f"{'⬆' if is_risk else '⬇'} {label}＝{feature_value}，"
        f"SHAP={shap_value:+.4f}，此特徵{direction}了風險評分{pct_str}"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  PII Filter — 個人資訊保護（PDPA / GDPR / 台灣個資法 合規）
#
#  設計原則：縱深防禦（Defense-in-Depth），四層攔截
#
#    Layer 1 — filter_pii_context()
#      ・ context dict 鍵名白名單（僅允許 _ALLOWED_CONTEXT_KEYS）
#      ・ 值型別強制：所有欄位必須為數值（int/float/bool）
#        或屬於 _ALLOWED_CATEGORICAL_VALUES 中明確核准的字串標籤
#      ・ user_info 等原始資料表的任何字串欄位在此被完全阻斷
#
#    Layer 2 — filter_pii_contributions()
#      ・ SHAP 特徵名稱白名單（_ALLOWED_FEATURE_NAMES 嚴格允許名單）
#      ・ 不在名單中的特徵整體移除，不得流入 Prompt
#
#    Layer 3 — scan_prompt_for_pii()
#      ・ Regex 掃描固定格式 PII（IPv4、Email、台灣手機、國際電話、TW 身份證）
#      ・ 上游遺漏的結構化 PII 在此遮罩為 [REDACTED]
#
#    Layer 4 — comprehend_scan_prompt()          ← 新增：語義層防線
#      ・ Amazon Comprehend detect_pii_entities API
#      ・ 偵測 Regex 無法覆蓋的非結構化 PII（姓名、地址、護照、銀行帳號…）
#      ・ 所有偵測結果替換為 [REDACTED]，並記錄 SECURITY_WARNING 日誌
#      ・ Comprehend API 異常時 fallback（記錄 WARNING，不阻斷主流程）
#
#  嚴禁傳送至 Bedrock 的資訊：
#    - 姓名（真實姓名、帳號名稱）
#    - 電話號碼（台灣 09xx、國際 +xx）
#    - Email 地址
#    - 完整 IPv4 位址（僅允許傳送共用帳號計數）
#    - 台灣身份證字號 / 護照號碼
#    - 錢包鏈上地址 / 銀行帳號
#    - user_info 等原始資料表的任何字串型原始欄位
#  允許傳送：user_id（整數）、數值統計特徵、核准分類標籤
# ══════════════════════════════════════════════════════════════════════════════

# ── Layer 1：context dict 允許名單 ────────────────────────────────────────────
# 只有下列欄位可以出現在傳給 Bedrock 的 context dict 中。
# 這些都是純數值統計或分類標籤，不含任何可識別個人的原始資訊。
_ALLOWED_CONTEXT_KEYS: frozenset[str] = frozenset({
    # 交易行為（數值）
    "min_retention_minutes",
    "retention_event_count",
    "twd_deposit_count",
    "twd_withdraw_count",
    "crypto_deposit_count",
    "crypto_withdraw_count",
    "total_twd_volume",
    "volume_zscore",
    "night_tx_ratio",
    "mule_risk_score",
    "unique_ip_count",       # 不同 IP 的「數量」（整數），非原始 IP 字串
    "ip_anomaly",
    "asymmetry_flag",
    "high_speed_risk",
    # 圖分析欄位（Athena BFS 衍生統計，非原始識別資訊）
    "min_hops_to_blacklist",
    "is_direct_neighbor",
    "blacklist_neighbor_count",
    "in_blacklist_network",
    "ip_shared_user_count",  # 共用同一 IP 的帳號「數量」（整數），非 IP 字串本身
    "has_high_speed_risk",
    "hop_risk_level",
    "weighted_risk_label",
    # KYC 等級（0/1/2 分類代碼，不含身份文件內容）
    "kyc_level",
})

# 鍵名中包含這些字串的欄位，無論是否在允許名單中，一律封鎖。
# 防止開發者誤將 PII 欄位命名為允許名單中的相似名稱。
_BLOCKED_KEY_SUBSTRINGS: tuple[str, ...] = (
    "name",        # full_name, account_name, etc.
    "email",       # email, email_address
    "mail",        # mail_to
    "phone",       # phone, phone_number
    "tel",         # tel, telephone
    "mobile",      # mobile_number
    "address",     # email_address, wallet_address, ip_address
    "birthday",    # birthday, birth_day
    "birth",       # birth_date, dob
    "dob",         # date of birth
    "passport",    # passport_number
    "id_card",     # id_card_number
    "national_id", # national_id
    "kyc_doc",     # kyc_document
    "source_ip",   # 原始 IP 欄位（非計數）
    "ip_addr",     # ip_address
    "wallet",      # wallet_address（鏈上地址）
)

# 分類標籤核准值（Layer 1 型別驗證使用）
# 欄位值若為字串，必須完全符合此表中的核准標籤，否則視同 PII 並封鎖。
# 目的：防止 user_info 等原始資料表中的任意字串值（如真實姓名）透過合法鍵名流入。
_ALLOWED_CATEGORICAL_VALUES: dict[str, frozenset[str]] = {
    "hop_risk_level": frozenset({
        "blacklist", "direct", "indirect_2", "indirect_3", "isolated",
    }),
    "weighted_risk_label": frozenset({
        "BLACKLIST", "HIGH_WEIGHTED", "HIGH", "MEDIUM", "LOW",
    }),
}

# ── Layer 2：SHAP 特徵名稱允許名單 ────────────────────────────────────────────
# 只有 FEATURE_LABELS 中定義的特徵可傳送，其他特徵（如姓名衍生特徵）一律排除。
# 此集合在模組載入時與 FEATURE_LABELS 合併，確保保持同步。
_ALLOWED_FEATURE_NAMES: frozenset[str] = frozenset(FEATURE_LABELS.keys()) | frozenset({
    # Athena 新增特徵（與 FEATURE_LABELS 保持同步的補充）
    "ip_shared_user_count",
    "has_high_speed_risk",
    "hop_risk_level",
    "weighted_risk_label",
    "in_blacklist_network",
})

# ── Layer 3：Prompt 字串正則掃描模式 ──────────────────────────────────────────
_RE_IPV4       = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
_RE_EMAIL      = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
_RE_TW_PHONE   = re.compile(r"\b09\d{8}\b")                    # 台灣手機（09xxxxxxxx）
_RE_INTL_PHONE = re.compile(r"\+\d{8,15}\b")                   # 國際電話（+886...）
_RE_TW_ID      = re.compile(r"\b[A-Z][12]\d{8}\b")             # 台灣身份證字號


def filter_pii_context(context: dict) -> dict:
    """
    Layer 1：context dict PII 過濾器（允許名單策略）。

    只保留 _ALLOWED_CONTEXT_KEYS 中的欄位，同時對所有鍵名執行
    _BLOCKED_KEY_SUBSTRINGS 掃描，確保不意外帶入 PII 欄位。

    Args:
        context: 原始 context dict，可能含有姓名、IP 字串、電話等 PII。

    Returns:
        只含允許欄位的乾淨 dict，所有值皆為數值或分類標籤。
    """
    filtered: dict = {}
    blocked: list[str] = []

    for key, value in context.items():
        key_lower = key.lower()
        # 規則 A：不在允許名單中 → 排除
        if key not in _ALLOWED_CONTEXT_KEYS:
            blocked.append(key)
            continue
        # 規則 B：鍵名含 PII 關鍵字 → 排除（防誤命名混入）
        if any(sub in key_lower for sub in _BLOCKED_KEY_SUBSTRINGS):
            blocked.append(f"{key}[blocked-keyword]")
            continue
        # 規則 C：值型別驗證——只允許數值或已核准的分類標籤字串
        if isinstance(value, str):
            approved_values = _ALLOWED_CATEGORICAL_VALUES.get(key)
            if approved_values is None or value not in approved_values:
                blocked.append(f"{key}[string-not-whitelisted]")
                continue
        elif not isinstance(value, (int, float, bool)):
            blocked.append(f"{key}[non-numeric-type:{type(value).__name__}]")
            continue
        filtered[key] = value

    if blocked:
        log.info(
            "[PII_FILTER] context 已排除 %d 個非允許欄位：%s",
            len(blocked), blocked,
        )
    return filtered


def filter_pii_contributions(
    contributions: list,
) -> list:
    """
    Layer 2：SHAP 貢獻清單 PII 過濾器（允許名單策略）。

    驗證每個 ShapContribution.feature_name 是否在 _ALLOWED_FEATURE_NAMES 中。
    不在允許名單的特徵整個移除，不傳至 Bedrock。

    Args:
        contributions: 原始 ShapContribution 清單。

    Returns:
        只含允許特徵的乾淨清單。
    """
    clean = []
    blocked: list[str] = []

    for c in contributions:
        if c.feature_name in _ALLOWED_FEATURE_NAMES:
            clean.append(c)
        else:
            blocked.append(c.feature_name)

    if blocked:
        log.warning(
            "[PII_FILTER] SHAP 特徵已排除 %d 個非允許名稱：%s "
            "（若為合法特徵，請將其加入 FEATURE_LABELS 並同步更新 _ALLOWED_FEATURE_NAMES）",
            len(blocked), blocked,
        )
    return clean


def scan_prompt_for_pii(text: str) -> str:
    """
    Layer 3：Prompt 字串正則掃描（最後防線）。

    在 Prompt 即將傳送至 Bedrock 的最後一刻進行掃描。
    若上游過濾器有任何遺漏，此層會將殘留 PII 遮罩為標記字串，
    並以 ERROR 等級記錄警告，供安全審計追蹤。

    掃描範圍：
      - IPv4 位址           → [IP-REDACTED]
      - Email 地址          → [EMAIL-REDACTED]
      - 台灣手機號碼        → [PHONE-REDACTED]
      - 國際電話格式        → [PHONE-REDACTED]
      - 台灣身份證字號      → [ID-REDACTED]

    Args:
        text: 即將傳送至 Bedrock 的完整 Prompt 字串。

    Returns:
        經過掃描與遮罩的安全字串（若無 PII 則原樣返回）。
    """
    findings: list[str] = []

    if _RE_IPV4.search(text):
        findings.append(f"IPv4×{len(_RE_IPV4.findall(text))}")
        text = _RE_IPV4.sub("[IP-REDACTED]", text)

    if _RE_EMAIL.search(text):
        findings.append(f"Email×{len(_RE_EMAIL.findall(text))}")
        text = _RE_EMAIL.sub("[EMAIL-REDACTED]", text)

    tw_phones = _RE_TW_PHONE.findall(text) + _RE_INTL_PHONE.findall(text)
    if tw_phones:
        findings.append(f"Phone×{len(tw_phones)}")
        text = _RE_TW_PHONE.sub("[PHONE-REDACTED]", text)
        text = _RE_INTL_PHONE.sub("[PHONE-REDACTED]", text)

    if _RE_TW_ID.search(text):
        findings.append(f"TW-ID×{len(_RE_TW_ID.findall(text))}")
        text = _RE_TW_ID.sub("[ID-REDACTED]", text)

    if findings:
        log.error(
            "[PII_FILTER] ⛔ Prompt 中偵測到 PII 殘留，已遮罩（上游過濾器未攔截）：%s",
            ", ".join(findings),
        )
    return text


def comprehend_scan_prompt(text: str) -> str:
    """
    Layer 4：Amazon Comprehend 語義 PII 掃描（最終防線）。

    呼叫 detect_pii_entities API 偵測 Regex 無法覆蓋的非結構化 PII，包括：
      - 人名（PERSON）
      - 地址（ADDRESS）
      - 護照號碼（PASSPORT_NUMBER）
      - 銀行帳號（BANK_ACCOUNT_NUMBER）
      - 信用卡號（CREDIT_DEBIT_NUMBER）
      - 出生日期（DATE_TIME，高風險情境）
      - 及其他 Comprehend 支援的 PII 實體類型

    設計原則：
      ・只替換，不阻斷——偵測到 PII 時將敏感片段替換為 [REDACTED] 並繼續
      ・Fail-Open——API 異常時記錄 WARNING 後返回原始文字，不中斷主流程
      ・每次偵測均記錄 SECURITY_WARNING 日誌以供稽核追蹤

    Args:
        text: 已通過 Layer 1–3 的 Prompt 字串。

    Returns:
        替換敏感片段後的安全字串（API 異常時返回原始輸入）。
    """
    if not text:
        return text

    try:
        response = _comprehend.detect_pii_entities(Text=text, LanguageCode="en")
        entities = response.get("Entities", [])
    except Exception as exc:
        log.warning(
            "[PII_FILTER] Layer 4 Comprehend 掃描失敗，Fail-Open 繼續執行：%s", exc
        )
        return text

    if not entities:
        return text

    # 依 BeginOffset 倒序替換，避免偏移量因前段替換而偏移
    entities_sorted = sorted(entities, key=lambda e: e["BeginOffset"], reverse=True)
    text_chars = list(text)
    found: list[str] = []

    for entity in entities_sorted:
        begin  = entity["BeginOffset"]
        end    = entity["EndOffset"]
        e_type = entity.get("Type", "PII")
        score  = entity.get("Score", 0.0)
        found.append(f"{e_type}(score={score:.2f})")
        text_chars[begin:end] = list("[REDACTED]")

    redacted_text = "".join(text_chars)
    log.warning(
        "[SECURITY_WARNING] Layer 4 Comprehend 偵測到 %d 個 PII 實體，已遮罩：%s",
        len(found), ", ".join(found),
    )
    return redacted_text


# ══════════════════════════════════════════════════════════════════════════════
#  Bedrock Rate Limiter（1 RPS，跨所有執行緒共享）
# ══════════════════════════════════════════════════════════════════════════════

class _BedrockRateLimiter:
    """
    執行緒安全的嚴格 < 1 RPS 速率限制器（mutex-based sliding window）。

    設計：持有鎖期間計算等待時間並 sleep，確保任意兩次 invoke_model 呼叫
    之間至少間隔 MIN_INTERVAL 秒（預設 1.1s，嚴格低於 1 RPS 上限）。
    多執行緒環境下自動序列化請求，不會有任何請求並發送出。

    為何用 1.1s 而非 1.0s：
      Bedrock TPS 限制為「每秒 1 個」，計量精度為毫秒級。
      使用 1.0s 間隔在時鐘精度與 OS 排程抖動下有機率觸發 ThrottlingException。
      1.1s 提供 100ms 安全邊際，在 90+ 個用戶的批次中完全避免限流。
    """
    MIN_INTERVAL: float = 1.1   # 秒，嚴格低於 1 RPS（比 1.0 多 100ms 安全邊際）

    def __init__(self, rps: float = 1.0):
        # 取 MAX：不管 rps 怎麼設，間隔至少 MIN_INTERVAL
        self._min_interval = max(1.0 / rps, self.MIN_INTERVAL)
        self._lock         = threading.Lock()
        self._last_call_ts = 0.0
        log.info(
            f"[RateLimiter] Bedrock 速率限制初始化：間隔 {self._min_interval:.2f}s "
            f"（{1/self._min_interval:.3f} RPS）"
        )

    def acquire(self) -> None:
        """
        呼叫 Bedrock 前必須先呼叫此方法。
        若距上次呼叫不足 min_interval 秒，則在持有鎖的狀態下 sleep 補足差額。
        """
        with self._lock:
            now  = time.monotonic()
            gap  = now - self._last_call_ts
            wait = self._min_interval - gap
            if wait > 0:
                time.sleep(wait)            # 強制 1.1s 間隔
            self._last_call_ts = time.monotonic()


# ── 模組級單例：所有 _BaseWriter 實例共用同一個限流器 ────────────────────────
# 此設計確保即使同時有多個 Writer 物件，全域 Bedrock TPS 仍受控於 1.1s。
_bedrock_rate_limiter = _BedrockRateLimiter(rps=1.0)

# ── Amazon Comprehend 客戶端（Layer 4 PII 掃描） ────────────────────────────
# 模組載入時初始化一次，後續重用（避免每次呼叫重建 Session）。
# 若 Comprehend 不可用（非 us-east-1/us-west-2），會在 comprehend_scan_prompt()
# 中 fallback 並記錄 WARNING，不阻斷主流程。
_comprehend = boto3.client("comprehend", region_name=COMPREHEND_REGION)

# ── Throttling 指數退避設定 ───────────────────────────────────────────────────
_THROTTLE_MAX_RETRIES: int   = 5      # ThrottlingException 最大重試次數
_THROTTLE_BASE_WAIT:   float = 2.0    # 指數退避基數（秒）：2, 4, 8, 16, 32 …
_THROTTLE_MAX_WAIT:    float = 60.0   # 單次等待上限（秒）


# ══════════════════════════════════════════════════════════════════════════════
#  模型存取權檢核（verify_model_access）
# ══════════════════════════════════════════════════════════════════════════════

def verify_model_access(region: str = BEDROCK_REGION) -> dict[str, bool]:
    """
    啟動批次推論前，透過 Bedrock list_foundation_models API 確認
    本專案所用模型均已在 Bedrock Console 啟用。

    設計原則：
      - 僅驗證 _ALLOWED_MODELS 白名單中的模型，不掃描其他模型
      - 若模型未啟用 → 拋出 RuntimeError（讓呼叫端在 Job 啟動前中止）
      - 若 API 本身失敗（無 bedrock:ListFoundationModels IAM 權限）→
        記錄 WARNING 並 assume 可存取（降級），不中斷流程

    呼叫時機（建議）：
      XAIReportGenerator.__init__ 中呼叫 verify_model_access()，
      確保 Job 在第一個 invoke_model 失敗前就能快速失敗並提示原因。

    Returns:
        {model_id: is_accessible} — 每個白名單模型的存取狀態

    Raises:
        RuntimeError: 任一白名單模型未在目標 Region 啟用時
    """
    log.info(
        f"[ModelCheck] 開始驗證模型存取權（region={region}，"
        f"共 {len(_ALLOWED_MODELS)} 個模型）"
    )
    try:
        client = boto3.client("bedrock", region_name=region)
        resp   = client.list_foundation_models(byOutputModality="TEXT")
        active_ids: set[str] = {
            m["modelId"]
            for m in resp.get("modelSummaries", [])
            if m.get("modelLifecycle", {}).get("status") == "ACTIVE"
        }
    except ClientError as exc:
        log.warning(
            f"[ModelCheck] list_foundation_models 失敗（{exc.response['Error']['Code']}），"
            f"跳過驗證，assume 所有白名單模型可存取。"
        )
        return {m: True for m in _ALLOWED_MODELS}
    except Exception as exc:
        log.warning(f"[ModelCheck] API 呼叫異常（{exc}），跳過驗證。")
        return {m: True for m in _ALLOWED_MODELS}

    result: dict[str, bool] = {}
    for model_id in sorted(_ALLOWED_MODELS):
        accessible = model_id in active_ids
        result[model_id] = accessible
        symbol = "✓" if accessible else "✗ 未啟用"
        log.info(f"[ModelCheck] {symbol}  {model_id}")

    blocked = [m for m, ok in result.items() if not ok]
    if blocked:
        raise RuntimeError(
            f"[ModelCheck] 以下 {len(blocked)} 個模型在 {region} 尚未啟用，"
            f"請至 Bedrock Console → Model access 申請後重新執行：\n"
            + "\n".join(f"  ✗  {m}" for m in blocked)
        )

    log.info(f"[ModelCheck] ✓ 所有 {len(_ALLOWED_MODELS)} 個白名單模型驗證通過")
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  Enum 定義
# ══════════════════════════════════════════════════════════════════════════════

class ScoringTier(str, Enum):
    """用戶風險機率 → Bedrock 路由決策。"""
    EXTREME  = "EXTREME"    # P > 0.90
    HIGH     = "HIGH"       # 0.75 < P ≤ 0.90
    BOUNDARY = "BOUNDARY"   # 0.65 ≤ P ≤ 0.75  ← 邊界案例
    MEDIUM   = "MEDIUM"     # 0.50 ≤ P < 0.65
    LOW      = "LOW"        # P < 0.50


class PrimaryAction(str, Enum):
    """後台可自動執行的動作代碼。"""
    FREEZE_ACCOUNT   = "FREEZE_ACCOUNT"    # 立即凍結帳戶
    LIMIT_WITHDRAWAL = "LIMIT_WITHDRAWAL"  # 限制提領金額
    CALL_VERIFY      = "CALL_VERIFY"       # 人工電訪核實
    FORCE_KYC        = "FORCE_KYC"         # 強制重新 KYC
    MONITOR          = "MONITOR"           # 列入觀察名單
    NO_ACTION        = "NO_ACTION"         # 維持現狀


class ThreatPattern(str, Enum):
    """識別出的威脅模式（後台系統可按此分類處理）。"""
    MULE_ACCOUNT    = "MULE_ACCOUNT"        # 人頭戶
    LAYERING        = "LAYERING"            # 層進式洗錢
    HACK_LAUNDERING = "HACK_LAUNDERING"     # 黑客洗錢路徑
    ARBITRAGE_ABUSE = "ARBITRAGE_ABUSE"     # 異常大額套利
    CLUSTER_FRAUD   = "CLUSTER_FRAUD"       # 機房集體詐騙（ip_shared>3 + hops≤2 + 高速交易）
    UNKNOWN         = "UNKNOWN"


# ══════════════════════════════════════════════════════════════════════════════
#  資料結構
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ShapContribution:
    feature_name:     str
    feature_label:    str
    shap_value:       float
    contribution_pct: float
    feature_value:    float
    direction:        str     # "增加風險" / "降低風險"


@dataclass
class ActionDirective:
    """
    機器可執行的風控指令。

    後台系統解析此結構後可直接執行：
      if directive.auto_executable and directive.primary_action == "FREEZE_ACCOUNT":
          account_service.freeze(user_id, duration_days=directive.freeze_duration_days)
          str_service.submit(user_id) if directive.str_required else None
    """
    primary_action:            str      # PrimaryAction enum string
    auto_executable:           bool     # True = 後台可自動執行，False = 需人工確認
    execution_priority:        int      # 1（立即）~ 5（低優先）
    steps:                     list[str]

    # STR 可疑交易報告
    str_required:              bool
    str_deadline_hours:        Optional[int]   # None = 不需提交

    # 凍結參數
    freeze_duration_days:      Optional[int]   # None = 不凍結

    # 限制提領
    daily_withdrawal_limit_twd: Optional[int]  # None = 不限制

    # KYC 升級
    require_kyc_upgrade:       bool

    # 觀察名單
    watchlist_days:            Optional[int]   # None = 不列入


@dataclass
class StructuredDiagnosis:
    """
    完整結構化診斷結果（JSON 可序列化，後台直接消費）。
    """
    # 身份資訊
    user_id:          int
    probability:      float
    scoring_tier:     str    # ScoringTier enum string
    model_used:       str    # MODEL_SONNET / MODEL_HAIKU
    generated_at:     str    # ISO-8601

    # 威脅分析
    threat_pattern:             str    # ThreatPattern enum string
    threat_pattern_zh:          str    # 中文名稱
    threat_pattern_description: str    # 中文說明

    # 診斷文字
    risk_diagnosis:   str    # 風險成因診斷段落

    # 機器可執行指令
    action:           ActionDirective

    # SHAP 資訊
    shap_contributions: list[dict]
    raw_shap_sum:     float

    # 相容舊欄位（backward compat）
    risk_level:         str    # HIGH / MEDIUM / LOW
    narrative_risk_cause: str  # = risk_diagnosis
    fund_flow_analysis:   str  # = threat_pattern_description
    action_directive:     str  # = "\n".join(action.steps)
    risk_diagnosis_text:  str  # alias
    interception_advice:  str  # alias

    # ── 合規人員友善摘要（新增）──────────────────────────────────────────
    # compliance_summary   : 白話摘要，供非技術合規主管一眼判讀
    # graph_risk_narrative : 資金流向圖結論（Athena BFS 一句話摘要）
    # risk_factors         : 分項風險因子清單，含 category / description / severity
    compliance_summary:    str        = ""
    graph_risk_narrative:  str        = ""
    risk_factors:          list       = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
#  ModelRouter — 路由決策
# ══════════════════════════════════════════════════════════════════════════════

class ModelRouter:
    """
    根據預測機率決定：
      ① 使用哪個 Bedrock 模型（Sonnet / Haiku / 無）
      ② 用戶屬於哪個 ScoringTier
      ③ 預設 ActionDirective（當 LLM 未能解析時的 fallback）
    """

    @staticmethod
    def classify(probability: float) -> ScoringTier:
        if probability > TIER_EXTREME_MIN:
            return ScoringTier.EXTREME
        if TIER_BOUNDARY_LO <= probability <= TIER_BOUNDARY_HI:
            return ScoringTier.BOUNDARY
        if probability > TIER_BOUNDARY_HI:
            return ScoringTier.HIGH
        if probability >= TIER_MEDIUM_LO:
            return ScoringTier.MEDIUM
        return ScoringTier.LOW

    @staticmethod
    def select_model(tier: ScoringTier) -> Optional[str]:
        """回傳 Bedrock model_id；MEDIUM / LOW 不調用 LLM 回傳 None。"""
        return {
            ScoringTier.EXTREME:  MODEL_HAIKU,
            ScoringTier.HIGH:     MODEL_SONNET,
            ScoringTier.BOUNDARY: MODEL_SONNET,
            ScoringTier.MEDIUM:   None,
            ScoringTier.LOW:      None,
        }[tier]

    @staticmethod
    def default_action(tier: ScoringTier, probability: float) -> ActionDirective:
        """
        Rule-Based 預設動作（MEDIUM 層不調 LLM 時使用；也作為 LLM 解析失敗的 fallback）。
        """
        if tier == ScoringTier.EXTREME:
            return ActionDirective(
                primary_action="FREEZE_ACCOUNT", auto_executable=True,
                execution_priority=1,
                steps=[
                    "立即凍結帳戶所有交易功能（凍結期 30 日）",
                    "24 小時內向調查局洗錢防制處提交 STR",
                    "完整保全 KYC 文件、IP 日誌、交易紀錄",
                    "啟動緊急帳戶調查程序（EAR）",
                ],
                str_required=True, str_deadline_hours=24,
                freeze_duration_days=30, daily_withdrawal_limit_twd=None,
                require_kyc_upgrade=False, watchlist_days=None,
            )
        if tier in (ScoringTier.HIGH, ScoringTier.BOUNDARY):
            return ActionDirective(
                primary_action="CALL_VERIFY", auto_executable=False,
                execution_priority=2,
                steps=[
                    "限制單日提領上限（5 萬元台幣）直至核實完成",
                    "3 個工作日內由合規專員進行電話訪查",
                    "要求補件：資金來源佐證文件",
                    "30 日後重新評估風險等級",
                ],
                str_required=False, str_deadline_hours=None,
                freeze_duration_days=None, daily_withdrawal_limit_twd=50_000,
                require_kyc_upgrade=False, watchlist_days=30,
            )
        return ActionDirective(
            primary_action="MONITOR", auto_executable=True,
            execution_priority=4,
            steps=[
                "納入強化監控名單，觀察期 60 日",
                "設置交易警報：單筆 > 10 萬元或單日 > 50 萬元觸發人工複審",
                "期滿後重新評分",
            ],
            str_required=False, str_deadline_hours=None,
            freeze_duration_days=None, daily_withdrawal_limit_twd=None,
            require_kyc_upgrade=False, watchlist_days=60,
        )

    @staticmethod
    def legacy_risk_level(probability: float) -> str:
        if probability >= 0.75:
            return "HIGH"
        if probability >= RISK_THRESHOLD:
            return "MEDIUM"
        return "LOW"


# ══════════════════════════════════════════════════════════════════════════════
#  SHAP 計算層（不變）
# ══════════════════════════════════════════════════════════════════════════════

class ShapAnalyzer:
    """使用 TreeExplainer 計算 XGBoost 模型的 SHAP 貢獻度。"""

    def __init__(self, model: xgb.Booster, feature_names: list[str]):
        self.feature_names = feature_names
        self.explainer     = shap.TreeExplainer(model)

    def explain_single(
        self, user_id: int, feature_row: np.ndarray,
        probability: float, top_n: int = 8,
    ) -> list[ShapContribution]:
        dmat      = xgb.DMatrix(feature_row, feature_names=self.feature_names)
        shap_vals = self.explainer.shap_values(dmat)[0]
        abs_sum   = np.abs(shap_vals).sum() or 1.0

        contributions = [
            ShapContribution(
                feature_name=name,
                feature_label=FEATURE_LABELS.get(name, name),
                shap_value=round(float(sv), 6),
                contribution_pct=round(abs(sv) / abs_sum * 100, 2),
                feature_value=round(float(feature_row[0, i]), 4),
                direction="增加風險" if sv > 0 else "降低風險",
            )
            for i, (name, sv) in enumerate(zip(self.feature_names, shap_vals))
        ]
        contributions.sort(key=lambda c: c.contribution_pct, reverse=True)
        return contributions[:top_n]

    def explain_batch(
        self, user_ids: list[int], feature_matrix: np.ndarray,
        probabilities: np.ndarray, top_n: int = 8,
    ) -> dict[int, list[ShapContribution]]:
        return {
            uid: self.explain_single(uid, feature_matrix[i:i+1], probabilities[i], top_n)
            for i, uid in enumerate(user_ids)
        }


# ══════════════════════════════════════════════════════════════════════════════
#  JSON Schema 定義（注入 Prompt）
# ══════════════════════════════════════════════════════════════════════════════

# 後台系統期望的 JSON Schema（以 JSON Schema Draft-7 格式說明給 Claude）
_OUTPUT_JSON_SCHEMA = """\
{
  "compliance_summary": "string（合規人員白話摘要，3–5 句；使用『此帳戶』開頭；說明最關鍵的 2–3 個可疑行為及建議處置動作；嚴禁出現『SHAP值』『特徵貢獻』『模型預測機率』等技術詞彙）",
  "graph_risk_narrative": "string（1–2 句，描述 Athena BFS 資金流向圖的核心結論；例如：『此帳戶與 3 個已知黑名單帳戶僅隔 1 層交易關係，且共用 IP 的帳號群中有同期洗錢案例』；若為孤立節點則說明無直接圖關聯）",
  "risk_factors": [
    {
      "category": "IP 異常 | 快進快出 | 黑名單關聯 | 身份驗證不足 | 自動化交易 | 資金集中 | 冷錢包轉移 | 機房集體詐騙 | 深夜異常",
      "description": "string（30 字以內；使用合規人員能理解的語言；範例：『資金存入後平均 12 分鐘即提出，遠低於正常用戶』）",
      "severity": "HIGH | MEDIUM | LOW"
    }
  ],
  "risk_diagnosis": "string（繁體中文，2–4 段技術性風險成因分析，供風控主管深度閱覽；可引用特徵名稱與數值；邏輯需嚴密）",
  "threat_pattern": "MULE_ACCOUNT | LAYERING | HACK_LAUNDERING | ARBITRAGE_ABUSE | CLUSTER_FRAUD | UNKNOWN",
  "threat_pattern_zh": "string（威脅模式中文名稱，例如：人頭戶洗錢、機房集體詐騙、層進式洗錢）",
  "threat_pattern_description": "string（1–2 句描述此模式的典型特徵與辨識方式）",
  "recommended_actions": {
    "primary_action": "FREEZE_ACCOUNT | LIMIT_WITHDRAWAL | CALL_VERIFY | FORCE_KYC | MONITOR | NO_ACTION",
    "auto_executable": true | false,
    "execution_priority": 1 | 2 | 3 | 4 | 5,
    "steps": ["string", "..."],
    "str_required": true | false,
    "str_deadline_hours": integer | null,
    "freeze_duration_days": integer | null,
    "daily_withdrawal_limit_twd": integer | null,
    "require_kyc_upgrade": true | false,
    "watchlist_days": integer | null
  }
}

【欄位規則】
- compliance_summary 寫作規範（合規主管閱讀，重要）：
    ✓ 主語範例：「此帳戶在過去 30 天內...」「帳戶使用的 IP...」「根據資金往來記錄...」
    ✓ 行為描述範例：「資金存入後平均 X 分鐘即提出」「同一 IP 與 Y 個帳號共用」「與 N 個已知洗錢帳戶有直接交易往來」
    ✗ 禁止：「SHAP 值」「特徵貢獻度」「模型預測機率為 X%」「XGBoost」
- risk_factors 分項準則：
    IP 異常      → ip_shared_user_count > 1 或 unique_ip_count 異常高
    快進快出     → min_retention_minutes < 30，或 has_high_speed_risk = true
    黑名單關聯   → min_hops_to_blacklist ≤ 2，is_direct_neighbor = true
    機房集體詐騙 → ip_shared > 3 且 hops ≤ 2 且 has_high_speed_risk（三重訊號）
    冷錢包轉移   → 虛幣出金佔總出金 > 70%，且虛幣出金次數 ≥ 3
    深夜異常     → night_tx_ratio > 0.40
- primary_action 選擇依據：
    FREEZE_ACCOUNT   → 風險機率極高（> 90%）或直接黑名單鄰居（hop=1）
    LIMIT_WITHDRAWAL → 風險高但需進一步核實，暫時降低出金上限
    CALL_VERIFY      → 邊界案例（65–75%），電訪後決定是否升級
    FORCE_KYC        → 身份文件可疑或 KYC Level 0
    MONITOR          → 低度風險，列入觀察
    NO_ACTION        → 無顯著異常
- auto_executable = true 僅限 primary_action 為 FREEZE_ACCOUNT 或 MONITOR
- execution_priority: 1=立即（1 小時內）, 2=緊急（24 小時）, 3=一般（3 日）, 4=低（7 日）, 5=觀察（30 日）
- str_deadline_hours: 若 str_required=true，依法規應在此小時數內提交（通常 24 或 72）
- 所有 null 欄位表示「不適用」，後台系統應忽略"""


# ══════════════════════════════════════════════════════════════════════════════
#  _BaseWriter — Bedrock 呼叫基底類別
# ══════════════════════════════════════════════════════════════════════════════

class _BaseWriter:
    """
    Bedrock 呼叫共用邏輯：
      - JSON 模式強制輸出（Prefill + Schema 注入）
      - JSON 解析 + 三層 fallback
      - ActionDirective 建構
    """

    _KYC_DESC = {
        0: "L0（未驗證，不具法幣交易資格）",
        1: "L1（手機驗證，單日上限 3 萬元）",
        2: "L2（完整身份驗證）",
    }

    def __init__(self, model_id: str, region: str = BEDROCK_REGION):
        # ── 模型白名單守衛：拒絕初始化任何不在 _ALLOWED_MODELS 的模型 ──────────
        if model_id not in _ALLOWED_MODELS:
            raise ValueError(
                f"[ModelCheck] model_id='{model_id}' 不在本專案允許清單中。\n"
                f"允許的模型：{sorted(_ALLOWED_MODELS)}\n"
                f"請勿啟用未經審核的模型（競賽最小存取原則）。"
            )
        self.model_id = model_id
        self.client   = boto3.client("bedrock-runtime", region_name=region)
        log.info(f"[_BaseWriter] 初始化完成：model={model_id}  region={region}")

    # ── 特徵表格格式化 ────────────────────────────────────────────────────────

    @staticmethod
    def _feature_table(contributions: list[ShapContribution]) -> str:
        header  = f"{'排名':<4}  {'特徵名稱':<26}  {'實際值':>12}  {'貢獻':>7}  方向    風險解讀"
        divider = "-" * 110
        rows    = [header, divider]
        for rank, c in enumerate(contributions, 1):
            arrow = "▲" if c.direction == "增加風險" else "▼"
            narrative = _shap_to_narrative(
                c.feature_name, c.feature_value, c.shap_value, c.contribution_pct
            )
            rows.append(
                f"{rank:<4}  {c.feature_label[:24]:<26}  {c.feature_value:>12.4g}"
                f"  {c.contribution_pct:>6.2f}%  {arrow} {c.direction}"
            )
            rows.append(f"      → {narrative}")
        return "\n".join(rows)

    # ── Athena 圖分析區塊格式化 ──────────────────────────────────────────────

    @staticmethod
    def _graph_context_block(context: dict) -> str:
        """
        將 Athena BFS 查詢結果（athena_graph_hops.sql 輸出欄位）格式化為
        Prompt 中的可讀分析區塊，供 Claude 進行三層交叉驗證。

        讀取的 context 欄位：
          min_hops_to_blacklist  — BFS 跳轉數（0=黑名單自身，4=孤立）
          blacklist_neighbor_count — 直接黑名單鄰居數
          in_blacklist_network   — 是否在黑名單連通分量中
          ip_shared_user_count   — 最高共用 IP 的帳號數（Athena S9）
          has_high_speed_risk    — 是否有 < 10 分鐘交易（Athena S10）
          hop_risk_level         — blacklist/direct/indirect_2/indirect_3/isolated
          weighted_risk_label    — HIGH_WEIGHTED/BLACKLIST/HIGH/MEDIUM/LOW
          is_direct_neighbor     — min_hops == 1
        """
        hops          = context.get("min_hops_to_blacklist")
        bl_count      = int(context.get("blacklist_neighbor_count", 0))
        in_bl_network = bool(context.get("in_blacklist_network", False))
        ip_shared     = int(context.get("ip_shared_user_count", 0))
        high_speed    = bool(context.get("has_high_speed_risk", False))
        hop_risk      = context.get("hop_risk_level", "isolated")
        weighted      = context.get("weighted_risk_label", "LOW")

        # Hop 距離語意說明
        if hops is None or hops >= 4:
            hop_desc = "孤立節點——BFS 無法到達任何黑名單（≥ 4 跳）"
        elif hops == 0:
            hop_desc = "⛔ 帳戶本身即已知黑名單"
        elif hops == 1:
            hop_desc = (
                f"🚨 直接黑名單鄰居——"
                f"與 {bl_count} 個已知黑名單帳戶有直接資金往來，風險最高"
            )
        elif hops == 2:
            hop_desc = "⚠ 二層關聯——透過 1 個中間帳戶與黑名單相連（中介洗錢路徑）"
        else:
            hop_desc = "△ 三層關聯——透過 2 個中間帳戶與黑名單相連（遠端關聯）"

        # IP 共用說明（Athena S9：ip_shared_user_count）
        if ip_shared > 10:
            ip_desc = (
                f"🚨 極高風險：同一 IP 被 {ip_shared} 個帳號共用"
                "（機房大規模集體詐騙強力訊號）"
            )
        elif ip_shared > 3:
            ip_desc = (
                f"⚠ 高風險：同一 IP 被 {ip_shared} 個帳號共用"
                "（疑似機房伺服器或代理 VPN 群控）"
            )
        elif ip_shared > 1:
            ip_desc = f"輕度異常：IP 與另 {ip_shared - 1} 個帳號共用（需進一步確認）"
        else:
            ip_desc = "正常（IP 獨佔或低共用，無群控跡象）"

        # 三重訊號警示：機房集體詐騙的核心判斷條件
        triple_signal = ""
        if ip_shared > 3 and hops is not None and hops <= 2 and high_speed:
            triple_signal = (
                "  🔴 三重訊號警示：[ip_shared > 3] + [hops ≤ 2] + [高速交易]\n"
                "     → 高度符合機房集體詐騙模式，建議標記 CLUSTER_FRAUD"
            )

        lines = [
            "═══ 資金關聯圖分析（Athena BFS） ═══",
            f"  黑名單距離       : {hops if hops is not None else '≥4'} 跳  [{hop_risk}]",
            f"  關聯說明         : {hop_desc}",
            f"  直接黑名單鄰居   : {bl_count} 個",
            f"  在黑名單連通分量  : {'是（帳戶在同一犯罪資金網路）' if in_bl_network else '否'}",
            f"  IP 共用分析      : {ip_desc}",
            f"  高速交易旗標     : {'是（< 10 分鐘完成，自動化腳本特徵）' if high_speed else '否'}",
            f"  複合風險標籤     : {weighted}",
        ]
        if triple_signal:
            lines.append(triple_signal)
        return "\n".join(lines)

    # ── 核心 Bedrock 呼叫（JSON Prefill Mode） ───────────────────────────────

    def _call_bedrock(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
    ) -> str:
        """
        呼叫 Bedrock 並強制 JSON 輸出，內建 Rate Limiting + ThrottlingException 退避。

        JSON Mode 實作策略：
          1. System Prompt 包含嚴格的 JSON-only 指令與 Schema
          2. 在 messages 中加入 assistant prefill `{`，迫使 Claude 從 `{` 開始生成
          3. 解析時補回開頭的 `{`

        速率控制（三層防護）：
          L1 — _bedrock_rate_limiter.acquire()：呼叫前強制 1.1s 間隔（< 1 RPS）
          L2 — ThrottlingException 指數退避：2s, 4s, 8s, 16s, 32s（最多重試 5 次）
          L3 — 超過重試上限仍失敗則拋出，由 generate() 捕獲並降級 Rule-Based
        """
        # Layer 3 PII：正則掃描 Prompt，遮罩固定格式殘留 PII
        user_prompt = scan_prompt_for_pii(user_prompt)
        # Layer 4 PII：Comprehend 語義掃描，偵測 Regex 無法覆蓋的非結構化 PII
        user_prompt = comprehend_scan_prompt(user_prompt)

        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": [
                {"role": "user",      "content": user_prompt},
                {"role": "assistant", "content": "{"},   # ← JSON Prefill
            ],
        })

        for attempt in range(_THROTTLE_MAX_RETRIES + 1):
            _bedrock_rate_limiter.acquire()   # L1：強制 1.1s 間隔（< 1 RPS）
            try:
                resp = self.client.invoke_model(
                    modelId=self.model_id, body=body,
                    contentType="application/json", accept="application/json",
                )
                raw = json.loads(resp["body"].read())["content"][0]["text"]
                return "{" + raw    # 補回 prefill 的開頭 `{`

            except ClientError as exc:
                code = exc.response["Error"]["Code"]
                is_throttle = code in ("ThrottlingException", "TooManyRequestsException")

                if is_throttle and attempt < _THROTTLE_MAX_RETRIES:
                    # L2：指數退避（2^attempt × base，上限 60s）
                    wait = min(_THROTTLE_BASE_WAIT * (2 ** attempt), _THROTTLE_MAX_WAIT)
                    log.warning(
                        f"[Bedrock] {code} model={self.model_id} "
                        f"attempt={attempt + 1}/{_THROTTLE_MAX_RETRIES + 1}，"
                        f"等待 {wait:.1f}s 後重試…"
                    )
                    time.sleep(wait)
                else:
                    # 不可重試的錯誤（AccessDeniedException、ModelNotReadyException…）
                    # 或已超過重試上限 → 向上拋出，由 generate() 降級
                    log.error(
                        f"[Bedrock] {code} model={self.model_id} "
                        f"attempt={attempt + 1}，不再重試，拋出例外。"
                    )
                    raise

        # 理論上不會到達此處（for 迴圈已 raise），加入以滿足型別檢查
        raise RuntimeError(f"[Bedrock] 超過最大重試次數 {_THROTTLE_MAX_RETRIES}")

    # ── JSON 解析（三層 fallback） ───────────────────────────────────────────

    @staticmethod
    def _parse_json(raw: str) -> Optional[dict]:
        """
        嘗試解析 Bedrock 回傳的 JSON 字串。三層降級策略：
          Layer 1: 直接 json.loads（最快路徑）
          Layer 2: regex 擷取第一個完整 JSON 物件（處理前後有雜文字的情況）
          Layer 3: 回傳 None（由 _to_action_directive 填入預設值）
        """
        # Layer 1
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Layer 2：找出最外層 {} 配對
        try:
            m = re.search(r"\{[\s\S]*\}", raw)
            if m:
                return json.loads(m.group())
        except (json.JSONDecodeError, AttributeError):
            pass

        log.warning("[JSON Parse] 三層解析均失敗，使用預設值。raw前100字：%s", raw[:100])
        return None

    # ── 從解析結果建構 ActionDirective ────────────────────────────────────────

    @staticmethod
    def _to_action_directive(
        parsed: Optional[dict],
        fallback: ActionDirective,
    ) -> ActionDirective:
        """
        從 LLM 回傳的 parsed dict 建構 ActionDirective。
        任何欄位解析失敗都使用 fallback 對應欄位（防禦性設計）。
        """
        if not parsed:
            return fallback

        ra = parsed.get("recommended_actions", {})

        def _get(key, default):
            v = ra.get(key, default)
            return default if v is None else v

        # 驗證 primary_action 是否合法
        raw_action = _get("primary_action", fallback.primary_action)
        valid_actions = {a.value for a in PrimaryAction}
        if raw_action not in valid_actions:
            raw_action = fallback.primary_action

        # 驗證 execution_priority 範圍
        priority = int(_get("execution_priority", fallback.execution_priority))
        priority = max(1, min(5, priority))

        return ActionDirective(
            primary_action             = raw_action,
            auto_executable            = bool(_get("auto_executable",            fallback.auto_executable)),
            execution_priority         = priority,
            steps                      = _get("steps",                           fallback.steps),
            str_required               = bool(_get("str_required",               fallback.str_required)),
            str_deadline_hours         = _get("str_deadline_hours",               fallback.str_deadline_hours),
            freeze_duration_days       = _get("freeze_duration_days",             fallback.freeze_duration_days),
            daily_withdrawal_limit_twd = _get("daily_withdrawal_limit_twd",       fallback.daily_withdrawal_limit_twd),
            require_kyc_upgrade        = bool(_get("require_kyc_upgrade",         fallback.require_kyc_upgrade)),
            watchlist_days             = _get("watchlist_days",                   fallback.watchlist_days),
        )

    # ── 從解析結果建構 StructuredDiagnosis ───────────────────────────────────

    def _build_diagnosis(
        self,
        user_id:      int,
        probability:  float,
        tier:         ScoringTier,
        contributions: list[ShapContribution],
        parsed:       Optional[dict],
        fallback_action: ActionDirective,
    ) -> StructuredDiagnosis:
        p = parsed or {}
        action = self._to_action_directive(p.get("recommended_actions"), fallback_action)

        # 威脅模式驗證
        raw_threat = p.get("threat_pattern", ThreatPattern.UNKNOWN.value)
        valid_threats = {t.value for t in ThreatPattern}
        if raw_threat not in valid_threats:
            raw_threat = ThreatPattern.UNKNOWN.value

        risk_diag  = p.get("risk_diagnosis", "（LLM 解析失敗，請人工審核）")
        threat_zh  = p.get("threat_pattern_zh", "未知威脅模式")
        threat_desc = p.get("threat_pattern_description", "")
        action_text = "\n".join(action.steps)

        # ── 新增欄位：合規人員友善摘要 & 圖分析 ──────────────────────────
        compliance_summary = p.get(
            "compliance_summary",
            "（合規摘要生成失敗，請人工填寫）",
        )
        graph_risk_narrative = p.get("graph_risk_narrative", "")
        risk_factors_raw = p.get("risk_factors", [])
        if not isinstance(risk_factors_raw, list):
            risk_factors_raw = []
        # 確保每個 factor 有 category / description / severity
        risk_factors = [
            {
                "category":    f.get("category",    "未分類"),
                "description": f.get("description", ""),
                "severity":    f.get("severity",    "MEDIUM"),
            }
            for f in risk_factors_raw
            if isinstance(f, dict)
        ]

        return StructuredDiagnosis(
            user_id=user_id,
            probability=round(float(probability), 6),
            scoring_tier=tier.value,
            model_used=self.model_id,
            generated_at=datetime.now(timezone.utc).isoformat(),
            threat_pattern=raw_threat,
            threat_pattern_zh=threat_zh,
            threat_pattern_description=threat_desc,
            risk_diagnosis=risk_diag,
            action=action,
            shap_contributions=[asdict(c) for c in contributions],
            raw_shap_sum=round(sum(c.shap_value for c in contributions), 6),
            # backward compat
            risk_level=ModelRouter.legacy_risk_level(probability),
            narrative_risk_cause=risk_diag,
            fund_flow_analysis=threat_desc,
            action_directive=action_text,
            risk_diagnosis_text=risk_diag,
            interception_advice=action_text,
            # 新增欄位
            compliance_summary=compliance_summary,
            graph_risk_narrative=graph_risk_narrative,
            risk_factors=risk_factors,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  _SonnetWriter — Claude 3.5 Sonnet 深度診斷（邊界 & 高風險）
# ══════════════════════════════════════════════════════════════════════════════

class _SonnetWriter(_BaseWriter):
    """
    Claude 3.5 Sonnet 深度診斷。

    適用：
      BOUNDARY (0.65–0.75) — 模型最不確定的邊界案例，需要 Sonnet 最強推理力
      HIGH     (0.75–0.90) — 確定高風險，但仍需詳細的攻擊模式識別

    Prompt 策略：
      - System: AML 調查員角色 + 嚴格 JSON-Only 指令 + 完整 Schema
      - User:   特徵表格 + 上下文 + 邊界案例特殊提示（僅 BOUNDARY tier）
      - max_tokens: BOUNDARY=1200, HIGH=900（邊界需要更多推理）
    """

    _SYSTEM = f"""\
你是「BitoGuard」，幣託 (BitoPro) 的資深 AML 調查員與合規風控專家。
你整合機器學習模型（XGBoost + SHAP 特徵貢獻）與資金關聯圖（Athena BFS 圖遍歷）的分析結果，
為每個高風險帳戶產出一份結構化風險診斷書，同時服務技術風控主管與非技術合規人員。

【嚴格格式規定】
你的回應必須且只能是一個符合以下 Schema 的 JSON 物件。
不得輸出任何 markdown、解釋性文字、或 JSON 以外的任何內容。
JSON 必須可被 json.loads() 直接解析。

【輸出 Schema】
{_OUTPUT_JSON_SCHEMA}

【三層交叉驗證分析框架】

▌ 層一：行為異常（XGBoost SHAP）
  - 解讀最高貢獻特徵，量化描述真實行為（如「平均滯留 X 分鐘」）
  - 勿直接引用特徵名稱或 SHAP 數值於 compliance_summary（技術詞彙對合規主管無意義）
  - 若 SHAP 貢獻與圖分析方向矛盾，在 risk_diagnosis 中明確說明

▌ 層二：關聯網路（Athena BFS 圖）
  - hop=1（直接鄰居）：最高警戒，帳戶與已知犯罪帳戶有直接資金往來
  - hop=2（二層）：中度警戒，透過一個中間人連結到黑名單
  - ip_shared > 3 + hops ≤ 2 + 高速交易 = 三重訊號 → 標記 CLUSTER_FRAUD（機房集體詐騙）
  - in_blacklist_network = true 需在診斷書中特別標注

▌ 層三：加密貨幣交易實務邏輯校準
  正常基準線（防止誤報）：
  · 一般個人用戶：資金滯留 > 30 分鐘；IP 獨佔或家用固定 IP；KYC L2；深夜比例 < 10%
  · 快速套利交易有正當動機（< 5 分鐘、金額適中、低頻）勿直接等同洗錢
  · 高 KYC 等級（L2）不代表無風險，仍須看行為特徵

  高風險模式識別：
  · 快進快出（< 10 分鐘）+ IP 共用 > 3 → 自動化腳本、機房群控
  · 虛幣出金比例 > 70%（≥ 3 次）→ 資金轉入冷錢包，逃避平台追蹤
  · KYC L0 + 高交易量 → 違規操作（L0 受嚴格額度限制）
  · volume_zscore > 3 → 遠超同 KYC 群組，交易量異常集中
  · 深夜交易比例 > 40% → 自動化腳本行為特徵（人工交易難以持續深夜高頻）
  · is_direct_neighbor = true → 不論機率高低，應列管並啟動人工審查

【compliance_summary 寫作要求】（最重要欄位）
  目標讀者：無技術背景的合規主管，閱讀後須能立即判斷是否啟動調查
  必須涵蓋：
    ① 最關鍵的 2–3 個可疑行為（用一般人語言描述）
    ② 資金流向關聯風險（若有 hop ≤ 2 或 ip_shared > 3 必須提及）
    ③ 末句給出建議處置動作
  語言範例：「此帳戶在過去 30 天內將資金存入後平均 12 分鐘即提出，
    明顯異於一般用戶；其使用的 IP 地址與另外 6 個帳戶共用，高度疑似由同一機房控制；
    帳戶與 2 個已知洗錢黑名單帳戶僅隔一層交易關係。
    建議合規人員於 24 小時內啟動電訪核實，並暫停大額提領。」

【邊界案例特殊要求（機率 65–75%）】
  在 risk_diagnosis 中必須明確分析：
    (a) 哪些特徵支持判定為高風險
    (b) 哪些特徵使模型存疑（反向證據）
    (c) 最終建議的決策依據"""

    def generate(
        self,
        user_id:      int,
        probability:  float,
        tier:         ScoringTier,
        contributions: list[ShapContribution],
        kyc_level:    int,
        context:      dict,
    ) -> StructuredDiagnosis:

        # Layer 1 & 2：PII 過濾（允許名單策略，必須在組裝 Prompt 前執行）
        context       = filter_pii_context(context)
        contributions = filter_pii_contributions(contributions)

        pct       = round(probability * 100, 1)
        kyc_desc  = self._KYC_DESC.get(kyc_level, f"L{kyc_level}")
        ret       = context.get("min_retention_minutes")
        hops      = context.get("min_hops_to_blacklist")
        crypto_w  = int(context.get("crypto_withdraw_count", 0))
        twd_w     = int(context.get("twd_withdraw_count", 0))
        total_w   = crypto_w + twd_w
        cr        = crypto_w / total_w if total_w > 0 else 0.0

        boundary_note = ""
        if tier == ScoringTier.BOUNDARY:
            boundary_note = (
                "\n【邊界案例特殊指令】本案機率落在 65–75% 不確定區間。"
                "請在 risk_diagnosis 中明確說明：(a) 哪些特徵支持判定為高風險，"
                "(b) 哪些特徵使模型存疑，(c) 最終建議依據何種考量。"
            )

        cold_wallet = ""
        if cr >= 0.7 and crypto_w >= 3:
            cold_wallet = (
                f"\n⚠ 冷錢包警示：虛幣出金 {crypto_w} 次，佔出金 {cr*100:.0f}%，"
                "資金可能已轉入無需實名之外部冷錢包（需比對 OFAC SDN 清單）。"
            )

        ret_str = (
            f"{ret:.1f} 分鐘"
            f"{'  ⚠ 極短，快進快出訊號' if ret < 30 else ''}"
            if ret is not None else "N/A"
        )
        twd_d   = int(context.get("twd_deposit_count",    0))
        twd_w_n = int(context.get("twd_withdraw_count",   0))
        cry_d   = int(context.get("crypto_deposit_count", 0))

        user_prompt = f"""分析以下帳戶並輸出風險診斷書 JSON：

═══ 用戶基本資訊 ═══
用戶 ID    : {user_id}
風險機率   : {pct}%（Tier: {tier.value}）
KYC 等級   : {kyc_desc}

{self._graph_context_block(context)}

═══ 交易行為摘要 ═══
資金滯留時間    : {ret_str}
台幣出入金次數  : 入金 {twd_d} / 出金 {twd_w_n}
虛幣出入金次數  : 入金 {cry_d} / 出金 {crypto_w}
{cold_wallet}{boundary_note}

═══ SHAP 特徵貢獻分析（模型決策依據，供技術分析層使用） ═══
{self._feature_table(contributions)}

請依三層分析框架產出 JSON 風險診斷書（直接從 {{ 開始，不要有任何前綴）："""

        max_tokens = 1400 if tier == ScoringTier.BOUNDARY else 1100
        fallback   = ModelRouter.default_action(tier, probability)

        try:
            raw    = self._call_bedrock(self._SYSTEM, user_prompt, max_tokens)
            parsed = self._parse_json(raw)
        except Exception as e:
            log.error("[Sonnet] user_id=%s 呼叫失敗：%s", user_id, e)
            parsed = None

        return self._build_diagnosis(user_id, probability, tier, contributions, parsed, fallback)


# ══════════════════════════════════════════════════════════════════════════════
#  _HaikuWriter — Claude 3 Haiku 制式報告（極高風險 P > 0.9）
# ══════════════════════════════════════════════════════════════════════════════

class _HaikuWriter(_BaseWriter):
    """
    Claude 3 Haiku 制式報告生成器。

    適用：EXTREME tier（P > 0.90）

    設計哲學：
      - 極高風險案例「攔截決策已定」，不需要深度推理
      - Haiku 成本約為 Sonnet 的 1/15，吞吐量更高
      - Prompt 更簡短，聚焦於生成「制式公文語氣的凍結通知」
      - max_tokens 縮減至 600（成本控制）

    制式報告特點：
      - primary_action 強制為 FREEZE_ACCOUNT（P > 0.9 必然凍結）
      - auto_executable = true（後台自動執行）
      - str_required = true（高風險必提 STR）
    """

    _SYSTEM = f"""\
你是「BitoGuard」AML 自動合規機器人，專為幣託 (BitoPro) 生成極高風險帳戶的制式凍結通知 JSON。

【強制規則】
1. 回應只能是一個 JSON 物件，可被 json.loads() 直接解析，不得有任何其他文字。
2. 由於風險機率超過 90%，primary_action 必須為 "FREEZE_ACCOUNT"。
3. auto_executable 必須為 true。
4. str_required 必須為 true，str_deadline_hours 為 24。
5. freeze_duration_days 必須為 30。
6. compliance_summary 必須以「此帳戶風險評估達極高等級」開頭，
   用白話描述前 2 個最關鍵的可疑行為（禁止使用 SHAP、特徵貢獻等技術詞彙），
   末句說明「已啟動自動凍結程序」。
7. graph_risk_narrative 描述資金關聯圖的核心結論（1 句話）。
8. risk_factors 至少包含 2 項，描述語言需讓合規主管一眼理解。
9. 語言：繁體中文。

【輸出 Schema】
{_OUTPUT_JSON_SCHEMA}"""

    def generate(
        self,
        user_id:      int,
        probability:  float,
        tier:         ScoringTier,
        contributions: list[ShapContribution],
        kyc_level:    int,
        context:      dict,
    ) -> StructuredDiagnosis:

        # Layer 1 & 2：PII 過濾（允許名單策略，必須在組裝 Prompt 前執行）
        context       = filter_pii_context(context)
        contributions = filter_pii_contributions(contributions)

        top3     = contributions[:3]
        top3_str = "；".join(
            f"{c.feature_label}={c.feature_value}（貢獻 {c.contribution_pct:.1f}%）"
            for c in top3
        )

        user_prompt = f"""極高風險帳戶凍結通知（制式報告）

═══ 用戶基本資訊 ═══
用戶 ID    : {user_id}
風險機率   : {round(probability * 100, 1)}%（超過 90% 強制凍結門檻）
KYC 等級   : {self._KYC_DESC.get(kyc_level, f"L{kyc_level}")}

{self._graph_context_block(context)}

═══ 前三大 SHAP 風險驅動因素 ═══
{top3_str}

生成凍結通知 JSON（直接從 {{ 開始）："""

        fallback = ModelRouter.default_action(tier, probability)

        try:
            raw    = self._call_bedrock(self._SYSTEM, user_prompt, max_tokens=600)
            parsed = self._parse_json(raw)
        except Exception as e:
            log.error("[Haiku] user_id=%s 呼叫失敗：%s", user_id, e)
            parsed = None

        # 強制覆蓋凍結相關欄位（即使 LLM 輸出有誤）
        if parsed and "recommended_actions" in parsed:
            ra = parsed["recommended_actions"]
            ra["primary_action"]       = PrimaryAction.FREEZE_ACCOUNT.value
            ra["auto_executable"]      = True
            ra["str_required"]         = True
            ra["str_deadline_hours"]   = 24
            ra["freeze_duration_days"] = 30

        return self._build_diagnosis(user_id, probability, tier, contributions, parsed, fallback)


# ══════════════════════════════════════════════════════════════════════════════
#  BatchScoringEngine — 並行 Batch 路由引擎
# ══════════════════════════════════════════════════════════════════════════════

class BatchScoringEngine:
    """
    批次評分引擎：根據機率路由到適合的 Writer，並行處理所有用戶。

    並行策略（ThreadPoolExecutor）：
      - 每個 Bedrock 呼叫約 3–15 秒（網路 I/O 瓶頸）
      - max_workers=12：12 個並行 HTTP 連線
      - EXTREME 與 HIGH/BOUNDARY 分開 Pool，Haiku 先完成後印出結果

    成本統計：
      - 每批完成後輸出 Token 估算與成本（便於 hackathon 預算控制）
    """

    # 各 tier 的 max_workers（避免 Bedrock throttling）
    _WORKERS = {
        ScoringTier.EXTREME:  8,   # Haiku 快，可多並行
        ScoringTier.BOUNDARY: 4,   # Sonnet 慢，控制並發
        ScoringTier.HIGH:     6,
        ScoringTier.MEDIUM:   0,   # 不調 LLM
        ScoringTier.LOW:      0,
    }

    def __init__(self, region: str = BEDROCK_REGION):
        self._sonnet = _SonnetWriter(MODEL_SONNET, region)
        self._haiku  = _HaikuWriter(MODEL_HAIKU,  region)
        self._router = ModelRouter()

    def _get_writer(self, tier: ScoringTier) -> Optional[_BaseWriter]:
        if tier in (ScoringTier.BOUNDARY, ScoringTier.HIGH):
            return self._sonnet
        if tier == ScoringTier.EXTREME:
            return self._haiku
        return None

    def process_batch(
        self,
        tasks: list[dict],   # [{"user_id", "probability", "contributions", "kyc_level", "context"}]
        show_progress: bool = True,
    ) -> list[StructuredDiagnosis]:
        """
        並行處理 tasks，依 tier 分組後並發呼叫 Bedrock。

        Parameters
        ----------
        tasks : list of dicts，每個 dict 包含：
            user_id, probability, contributions, kyc_level, context
        show_progress : 顯示進度日誌

        Returns
        -------
        list[StructuredDiagnosis]，依機率由高至低排序
        """
        if show_progress:
            tier_counts: dict[str, int] = {}
            for t in tasks:
                tier = ModelRouter.classify(t["probability"]).value
                tier_counts[tier] = tier_counts.get(tier, 0) + 1
            log.info("[BatchScoring] %d 筆用戶 | 分層：%s", len(tasks), tier_counts)

        # 分組
        tier_groups: dict[ScoringTier, list[dict]] = {t: [] for t in ScoringTier}
        for task in tasks:
            tier_groups[ModelRouter.classify(task["probability"])].append(task)

        results: list[StructuredDiagnosis] = []
        t_start = time.time()

        # 各 tier 並行處理
        for tier, group in tier_groups.items():
            if not group:
                continue

            writer = self._get_writer(tier)

            # MEDIUM / LOW：Rule-Based，不調 LLM
            if writer is None:
                for task in group:
                    fallback_action = ModelRouter.default_action(tier, task["probability"])
                    diag = StructuredDiagnosis(
                        user_id=task["user_id"],
                        probability=round(float(task["probability"]), 6),
                        scoring_tier=tier.value,
                        model_used="rule-based",
                        generated_at=datetime.now(timezone.utc).isoformat(),
                        threat_pattern=ThreatPattern.UNKNOWN.value,
                        threat_pattern_zh="（觀察中）",
                        threat_pattern_description="機率低於深度分析門檻，使用規則型評估。",
                        risk_diagnosis="本案預測機率未達深度診斷門檻，已依規則型邏輯建議處置措施。",
                        action=fallback_action,
                        shap_contributions=[asdict(c) for c in task["contributions"]],
                        raw_shap_sum=round(sum(c.shap_value for c in task["contributions"]), 6),
                        risk_level=ModelRouter.legacy_risk_level(task["probability"]),
                        narrative_risk_cause="（規則型評估）",
                        fund_flow_analysis="",
                        action_directive="\n".join(fallback_action.steps),
                        risk_diagnosis_text="（規則型評估）",
                        interception_advice="\n".join(fallback_action.steps),
                    )
                    results.append(diag)
                continue

            # 並行呼叫 LLM
            workers = self._WORKERS.get(tier, 4)
            log.info("[BatchScoring] Tier=%s  筆數=%d  workers=%d  model=%s",
                     tier.value, len(group), workers, writer.model_id)

            with ThreadPoolExecutor(max_workers=workers) as pool:
                future_to_task = {
                    pool.submit(
                        writer.generate,
                        task["user_id"],
                        task["probability"],
                        tier,
                        task["contributions"],
                        task["kyc_level"],
                        task["context"],
                    ): task
                    for task in group
                }
                for i, fut in enumerate(as_completed(future_to_task), 1):
                    task = future_to_task[fut]
                    try:
                        diag = fut.result()
                        results.append(diag)
                        if show_progress:
                            log.info(
                                "  [%d/%d] user_id=%-8d  prob=%.4f  tier=%-8s  action=%s",
                                i, len(group),
                                task["user_id"], task["probability"],
                                tier.value, diag.action.primary_action,
                            )
                    except Exception as e:
                        log.error("[BatchScoring] user_id=%s 失敗：%s",
                                  task["user_id"], e)

        elapsed = round(time.time() - t_start, 1)
        log.info("[BatchScoring] 完成。總耗時 %ss  成功 %d/%d 筆",
                 elapsed, len(results), len(tasks))

        # 成本估算（Token 數近似）
        n_sonnet = sum(1 for r in results if r.model_used == MODEL_SONNET)
        n_haiku  = sum(1 for r in results if r.model_used == MODEL_HAIKU)
        cost_est = n_sonnet * 0.003 + n_haiku * 0.00025  # USD 估算
        log.info("[Cost] Sonnet×%d + Haiku×%d ≈ $%.4f USD", n_sonnet, n_haiku, cost_est)

        results.sort(key=lambda r: r.probability, reverse=True)
        return results


# ══════════════════════════════════════════════════════════════════════════════
#  XAIReportGenerator — 端到端整合入口
# ══════════════════════════════════════════════════════════════════════════════

class XAIReportGenerator:
    """
    端到端 XAI 報告生成器（v3）。

    使用方式：
        generator = XAIReportGenerator(model, feature_names)
        reports   = generator.generate_reports(predict_df, probabilities,
                                               feature_cols=FEAT_COLS)
        generator.save_reports(reports, "xai_reports.json")
        generator.save_action_queue(reports, "action_queue.json")
    """

    _CONTEXT_FIELDS = [
        "min_retention_minutes", "crypto_withdraw_count",
        "twd_deposit_count",     "twd_withdraw_count",
        "min_hops_to_blacklist", "is_direct_neighbor",
        "total_twd_volume",      "blacklist_neighbor_count",
    ]

    def __init__(
        self,
        model: xgb.Booster,
        feature_names: list[str],
        region: str = BEDROCK_REGION,
    ):
        self.shap_analyzer   = ShapAnalyzer(model, feature_names)
        self.batch_engine    = BatchScoringEngine(region)

    def _extract_context(self, row: pd.Series) -> dict:
        ctx = {}
        for f in self._CONTEXT_FIELDS:
            val = row.get(f, None)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                ctx[f] = val
        return ctx

    def generate_reports(
        self,
        predict_df:      pd.DataFrame,
        probabilities:   np.ndarray,
        feature_cols:    list[str],
        min_probability: float = RISK_THRESHOLD,
        top_n_shap:      int   = 8,
    ) -> list[StructuredDiagnosis]:
        """
        對所有機率 >= min_probability 的用戶執行批次診斷。

        流程：
          1. SHAP 批次計算（全部高風險用戶）
          2. 建構 tasks list
          3. BatchScoringEngine 依 tier 並行路由 → Bedrock
          4. 依機率排序後返回
        """
        X        = predict_df[feature_cols].values.astype(float)
        user_ids = predict_df["user_id"].tolist()

        mask   = probabilities >= min_probability
        hr_ids   = [uid for uid, m in zip(user_ids, mask) if m]
        hr_probs = probabilities[mask]
        hr_X     = X[mask]

        # 計算各 tier 分布
        tiers = [ModelRouter.classify(p) for p in hr_probs]
        tier_dist = {t.value: tiers.count(t) for t in ScoringTier if tiers.count(t) > 0}
        log.info("[XAI] 觸發用戶：%d / %d  Tier 分布：%s",
                 len(hr_ids), len(user_ids), tier_dist)

        # 批次 SHAP
        log.info("[XAI] 計算 SHAP 值...")
        shap_map = self.shap_analyzer.explain_batch(hr_ids, hr_X, hr_probs, top_n_shap)

        # 組裝 tasks
        tasks = []
        for uid, prob in zip(hr_ids, hr_probs):
            row = predict_df[predict_df["user_id"] == uid].iloc[0]
            tasks.append({
                "user_id":      uid,
                "probability":  float(prob),
                "contributions": shap_map[uid],
                "kyc_level":    int(row.get("kyc_level", 0)),
                "context":      self._extract_context(row),
            })

        return self.batch_engine.process_batch(tasks)

    @staticmethod
    def save_reports(
        reports: list[StructuredDiagnosis],
        path: str = "xai_reports.json",
    ) -> None:
        """完整診斷報告（人工閱讀用）。"""
        import numpy as np
        
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.floating, np.float32, np.float64)):
                    return float(obj)
                if isinstance(obj, (np.integer, np.int32, np.int64)):
                    return int(obj)
                return super().default(obj)
        
        records = []
        for r in reports:
            d = asdict(r)
            d["action"] = asdict(r.action)
            records.append(d)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
        log.info("[Save] %s  (%d 筆)", path, len(reports))

    @staticmethod
    def save_action_queue(
        reports: list[StructuredDiagnosis],
        path: str = "action_queue.json",
    ) -> None:
        """
        機器可執行動作佇列（後台系統消費用）。

        只輸出 auto_executable=True 的記錄，格式精簡，
        後台 Worker 可直接依此執行鎖帳 / 提交 STR。

        格式：
        [
          {
            "user_id": 12345,
            "probability": 0.9412,
            "scoring_tier": "EXTREME",
            "primary_action": "FREEZE_ACCOUNT",
            "auto_executable": true,
            "execution_priority": 1,
            "str_required": true,
            "str_deadline_hours": 24,
            "freeze_duration_days": 30,
            "daily_withdrawal_limit_twd": null,
            "require_kyc_upgrade": false,
            "watchlist_days": null,
            "generated_at": "2026-03-23T10:00:00+00:00"
          },
          ...
        ]
        """
        queue = [
            {
                "user_id":                   r.user_id,
                "probability":               r.probability,
                "scoring_tier":              r.scoring_tier,
                "threat_pattern":            r.threat_pattern,
                "primary_action":            r.action.primary_action,
                "auto_executable":           r.action.auto_executable,
                "execution_priority":        r.action.execution_priority,
                "str_required":              r.action.str_required,
                "str_deadline_hours":        r.action.str_deadline_hours,
                "freeze_duration_days":      r.action.freeze_duration_days,
                "daily_withdrawal_limit_twd": r.action.daily_withdrawal_limit_twd,
                "require_kyc_upgrade":       r.action.require_kyc_upgrade,
                "watchlist_days":            r.action.watchlist_days,
                "generated_at":              r.generated_at,
            }
            for r in reports if r.action.auto_executable
        ]
        queue.sort(key=lambda x: x["execution_priority"])
        with open(path, "w", encoding="utf-8") as f:
            json.dump(queue, f, ensure_ascii=False, indent=2)
        log.info("[Save] %s  (%d 筆 auto_executable)", path, len(queue))

    @staticmethod
    def print_report(report: StructuredDiagnosis) -> None:
        """終端機格式化預覽。"""
        TIER_BADGE = {
            "EXTREME":  "🔴 EXTREME",
            "HIGH":     "🟠 HIGH",
            "BOUNDARY": "🟡 BOUNDARY",
            "MEDIUM":   "🟢 MEDIUM",
            "LOW":      "⚪ LOW",
        }
        ACTION_BADGE = {
            "FREEZE_ACCOUNT":   "🔒 FREEZE_ACCOUNT",
            "LIMIT_WITHDRAWAL": "⛔ LIMIT_WITHDRAWAL",
            "CALL_VERIFY":      "📞 CALL_VERIFY",
            "FORCE_KYC":        "🪪 FORCE_KYC",
            "MONITOR":          "👁 MONITOR",
            "NO_ACTION":        "✅ NO_ACTION",
        }
        sep = "═" * 66
        a   = report.action

        print(f"\n{sep}")
        print(f"  【風險診斷報告】  user_id: {report.user_id}")
        print(f"  機率: {report.probability:.2%}   Tier: {TIER_BADGE.get(report.scoring_tier, report.scoring_tier)}")
        print(f"  模型: {report.model_used.split('.')[-1]}   生成時間: {report.generated_at[:19]}")
        print(sep)

        # SHAP 橫條圖
        print("\n  SHAP 貢獻度分析")
        print(f"  {'排名':<4} {'貢獻':>6}  特徵")
        print("  " + "─" * 58)
        for rank, c in enumerate(report.shap_contributions[:6], 1):
            bar   = "█" * max(1, int(c["contribution_pct"] / 5))
            arrow = "▲" if c["direction"] == "增加風險" else "▼"
            print(f"  {rank:<4} {c['contribution_pct']:5.1f}%  {bar:<12} "
                  f"{arrow} {c['feature_label'][:28]}  ({c['feature_value']})")

        print(f"\n  威脅模式：{report.threat_pattern_zh}（{report.threat_pattern}）")
        print(f"  {report.threat_pattern_description}")

        print(f"\n  風險成因診斷：")
        for line in report.risk_diagnosis.split("\n"):
            print(f"    {line}")

        print(f"\n  風控建議行動：{ACTION_BADGE.get(a.primary_action, a.primary_action)}")
        print(f"  auto_executable={a.auto_executable}  priority={a.execution_priority}"
              f"  STR={'是' if a.str_required else '否'}"
              f"  凍結={'N/A' if not a.freeze_duration_days else f'{a.freeze_duration_days}天'}")
        for step in a.steps:
            print(f"    • {step}")
        print(f"\n{sep}")


# ══════════════════════════════════════════════════════════════════════════════
#  __main__
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os
    from bito_data_manager import BitoDataManager
    from train_sagemaker import build_features

    manager  = BitoDataManager()
    features = build_features(manager)

    predict_label = manager._load_raw("predict_label")
    predict_label["user_id"] = pd.to_numeric(predict_label["user_id"], errors="coerce")

    FEAT_COLS  = [c for c in features.columns
                  if c not in ("user_id", "asymmetry_reason", "ip_source")]
    predict_df = (
        predict_label
        .merge(features[["user_id"] + FEAT_COLS], on="user_id", how="left")
        .fillna(0)
    )

    MODEL_PATH = os.environ.get("XGB_MODEL_PATH", "model.json")
    model      = xgb.Booster()
    model.load_model(MODEL_PATH)

    dmat  = xgb.DMatrix(predict_df[FEAT_COLS].values, feature_names=FEAT_COLS)
    probs = model.predict(dmat)

    generator = XAIReportGenerator(model, FEAT_COLS)
    reports   = generator.generate_reports(
        predict_df, probs, feature_cols=FEAT_COLS, min_probability=0.5,
    )

    if reports:
        generator.print_report(reports[0])

    generator.save_reports(reports,      "xai_reports.json")
    generator.save_action_queue(reports, "action_queue.json")
