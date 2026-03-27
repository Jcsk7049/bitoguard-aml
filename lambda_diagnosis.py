"""
lambda_diagnosis.py
幣託人頭戶偵測 — 自動診斷書生成 Lambda 函數

══════════════════════════════════════════════════════════════════════════════
  觸發條件（S3 Event Trigger）
  ────────────────────────────────────────────────────────────────────────
  當 s3://{BUCKET}/predictions/*.csv 上傳時自動觸發。
  檔案格式（submission_with_prob.csv）：
      user_id, probability[, feat_col_1, feat_col_2, ...]
      12345,   0.94,        3.2, 15, 3200000, ...
      67890,   0.72,        ...

  觸發後流程：
    1. 從 S3 讀取預測結果 CSV
    2. 過濾高風險用戶（probability ≥ RISK_THRESHOLD）
    3. 按風險層級路由至對應 Bedrock 模型
       EXTREME (P>0.90) → Claude 3 Haiku   制式報告（低成本）
       BOUNDARY (0.65–0.75) → Claude 3.5 Sonnet 深度診斷
       HIGH (0.75–0.90) → Claude 3.5 Sonnet 標準分析
    4. 生成 JSON 格式結構化診斷書
    5. 寫入 DynamoDB（供前端儀表板即時查詢）
    6. 更新 CloudWatch 自訂指標

  DynamoDB 資料模型
  ────────────────────────────────────────────────────────────────────────
  Table     : bito-diagnoses
  PK        : user_id       (String)
  SK        : generated_at  (String, ISO-8601)
  GSI-1     : risk_level-generated_at-index   （按風險等級查所有診斷）
  GSI-2     : scoring_tier-probability-index  （按機率排序高危用戶）
  TTL field : expire_at     （90 天後自動刪除，節省儲存費用）

  Lambda 部署規格
  ────────────────────────────────────────────────────────────────────────
  Runtime  : python3.12
  Memory   : 512 MB
  Timeout  : 900 s（最大值，Bedrock 並發呼叫較耗時）
  IAM      : s3:GetObject + bedrock:InvokeModel + dynamodb:PutItem/BatchWriteItem
             + cloudwatch:PutMetricData
══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import io
import json
import logging
import os
import random
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Optional

import boto3
from botocore.exceptions import ClientError

# ══════════════════════════════════════════════════════════════════════════════
#  設定（優先讀取環境變數，方便 IaC 注入）
# ══════════════════════════════════════════════════════════════════════════════

_ALLOWED_REGIONS: frozenset[str] = frozenset({"us-east-1", "us-west-2"})

def _validate_region(region: str, var_name: str = "Region") -> str:
    """Region 合規閘門：若非 us-east-1 或 us-west-2 則立即拋出 RuntimeError 並終止程序。"""
    if region not in _ALLOWED_REGIONS:
        raise RuntimeError(
            f"[Region 合規] {var_name}='{region}' 不在核准清單中。\n"
            f"  允許值：{sorted(_ALLOWED_REGIONS)}\n"
            "  請更新 Lambda 環境變數後重新部署。"
        )
    return region


LOG_LEVEL       = os.environ.get("LOG_LEVEL",        "INFO")
DYNAMO_TABLE    = os.environ.get("DYNAMO_TABLE",      "bito-diagnoses")
BEDROCK_REGION  = _validate_region(os.environ.get("BEDROCK_REGION", "us-east-1"), "BEDROCK_REGION")
S3_REGION       = _validate_region(os.environ.get("S3_REGION",      "us-east-1"), "S3_REGION")
CW_NAMESPACE    = os.environ.get("CW_NAMESPACE",      "BitoMuleDetection")
SNS_ALERT_ARN   = os.environ.get("SNS_ALERT_ARN",     "")

# ── SQS 節流控制（取代 time.sleep 方案） ──────────────────────────────────────
# 架構說明：
#   Producer 路徑（S3 觸發）：將每個用戶診斷請求發至 SQS FIFO 佇列
#   Consumer 路徑（SQS 觸發）：Lambda Reserved Concurrency=1 保證序列執行
#   → 無需任何 sleep 即可確保 Bedrock < 1 RPS（由基礎設施層保證）
#
#   SQS FIFO 設定建議（template.yaml）：
#     FifoQueue: true
#     ContentBasedDeduplication: false
#     VisibilityTimeout: 120     # > Lambda Timeout（給 Bedrock 足夠時間）
#     MessageRetentionPeriod: 3600
#     RedrivePolicy: maxReceiveCount=2, deadLetterTargetArn: ...
#   Lambda 設定：
#     ReservedConcurrentExecutions: 1  ← 限制單一執行個體，序列化 Bedrock 呼叫
SQS_QUEUE_URL   = os.environ.get("SQS_QUEUE_URL",     "")        # FIFO Queue URL
SQS_REGION      = _validate_region(
    os.environ.get("SQS_REGION", os.environ.get("S3_REGION", "us-east-1")), "SQS_REGION"
)

# Bedrock 模型 ID
MODEL_SONNET    = "anthropic.claude-3-5-sonnet-20241022-v2:0"
MODEL_HAIKU     = "anthropic.claude-3-haiku-20240307-v1:0"

# 機率門檻
RISK_THRESHOLD  = float(os.environ.get("RISK_THRESHOLD",  "0.65"))
TIER_EXTREME    = float(os.environ.get("TIER_EXTREME",    "0.90"))
TIER_BOUNDARY_LO = float(os.environ.get("TIER_BOUNDARY_LO", "0.65"))
TIER_BOUNDARY_HI = float(os.environ.get("TIER_BOUNDARY_HI", "0.75"))

# 並發上限（Lambda 記憶體 512MB 下安全並發數）
MAX_WORKERS_HAIKU  = 8
MAX_WORKERS_SONNET = 4

# DynamoDB TTL（秒）
DYNAMO_TTL_DAYS = int(os.environ.get("DYNAMO_TTL_DAYS", "90"))

# 特徵中文對照（與 xai_bedrock.py 保持同步）
FEATURE_LABELS: dict[str, str] = {
    "min_retention_minutes":    "最短資金滯留時間（分鐘）",
    "retention_event_count":    "快進快出事件次數",
    "high_speed_risk":          "高速資金風險旗標",
    "unique_ip_count":          "不同 IP 數量",
    "ip_anomaly":               "IP 異常旗標",
    "ip_shared_user_count":     "同 IP 共用帳號數",
    "has_high_speed_risk":      "高速交易風險旗標（<10分鐘）",
    "weighted_risk_label":      "複合風險加權標籤",
    "total_twd_volume":         "總交易量（TWD）",
    "volume_zscore":            "交易量 Z-score（同 KYC 群組偏離度）",
    "asymmetry_flag":           "量能不對稱旗標",
    "kyc_level":                "KYC 等級",
    "twd_deposit_count":        "台幣入金次數",
    "twd_withdraw_count":       "台幣出金次數",
    "crypto_deposit_count":     "虛幣入金次數",
    "crypto_withdraw_count":    "虛幣出金次數",
    "night_tx_ratio":           "深夜交易比例（22:00–06:00）",
    "mule_risk_score":          "人頭戶風險綜合評分（0–3）",
    "min_hops_to_blacklist":    "距黑名單最短跳轉數",
    "is_direct_neighbor":       "直接黑名單鄰居旗標",
    "blacklist_neighbor_count": "直接相連黑名單節點數",
}

# 非特徵欄位（CSV 中存在但不應視為特徵）
NON_FEATURE_COLS  = frozenset({"user_id", "status", "probability", "incident_id", "predicted_at"})
SHAP_COL_PREFIX   = "shap_"   # SHAP 值欄位前綴（SageMaker 輸出時加此前綴）
TOP_FEATURES_N    = 5         # 注入 Prompt 的最高貢獻特徵數量（節省 Input Token）

# ══════════════════════════════════════════════════════════════════════════════
#  日誌設定
# ══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(funcName)s — %(message)s",
)
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
#  AWS 客戶端（Lambda 冷啟動時初始化一次，後續重用）
# ══════════════════════════════════════════════════════════════════════════════

_s3       = boto3.client("s3",              region_name=S3_REGION)
_bedrock  = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
_dynamo   = boto3.resource("dynamodb",      region_name=S3_REGION)
_cw       = boto3.client("cloudwatch",      region_name=S3_REGION)
# SQS client：僅在 SQS_QUEUE_URL 有設定時初始化（節省冷啟動資源）
_sqs      = boto3.client("sqs",             region_name=SQS_REGION) if SQS_QUEUE_URL else None

# ══════════════════════════════════════════════════════════════════════════════
#  Bedrock ThrottlingException 指數退避（Full-Jitter）
#
#  節流控制架構（已從 time.sleep 方案升級至 SQS + Reserved Concurrency=1）：
#    - 舊方案：Lambda 內部 _bedrock_acquire() 強制 sleep(1.1s)
#      缺點：多 Lambda 實例時跨實例無法協調，仍可能觸發 ThrottlingException
#    - 新方案：SQS FIFO + Lambda Reserved Concurrency=1
#      ① S3 觸發 → Lambda 將每個用戶請求發至 SQS FIFO 佇列（Producer 模式）
#      ② SQS 觸發 → Lambda 消費單一訊息（Consumer 模式，Reserved Concurrency=1）
#      ③ 由 AWS 基礎設施保證序列執行，完全無需任何 sleep
#
#  本層仍保留指數退避作為最後防線（當 Bedrock 短暫過載或 SQS 未啟用時）。
#  退避策略：Full-Jitter（random.uniform(0, cap)）
#    優點：分散多實例的重試時間，避免「Thundering Herd」同步化衝擊波
#
#  降級策略：超過 _THROTTLE_MAX_RETRIES 次後拋出 _BedrockThrottled，
#    由 _diagnose_one() 捕獲並切換至 Rule-Based 診斷，DynamoDB 標記 is_ai_generated=False
# ══════════════════════════════════════════════════════════════════════════════

_THROTTLE_MAX_RETRIES: int   = 5      # ThrottlingException 最大重試次數（超過後降級）
_THROTTLE_BASE_WAIT:   float = 2.0    # Full-Jitter 退避基數（秒）
_THROTTLE_MAX_WAIT:    float = 60.0   # 單次退避上限（秒）


class _BedrockThrottled(Exception):
    """
    Bedrock ThrottlingException 超過 _THROTTLE_MAX_RETRIES 次後拋出。
    由 _diagnose_one() 捕獲 → 自動切換 Rule-Based 降級輸出。
    DynamoDB 欄位：is_ai_generated=False, degradation_reason="THROTTLE_DEGRADED"。
    """


# ── 本 Lambda 唯一允許呼叫的模型白名單 ─────────────────────────────────────
# 防止誤啟用非必要模型（最小存取原則）。
_ALLOWED_MODELS: frozenset[str] = frozenset({MODEL_SONNET, MODEL_HAIKU})


def _verify_model_access() -> None:
    """
    Lambda 冷啟動時一次性驗證白名單模型是否已在 Bedrock Console 啟用。

    設計原則（與 xai_bedrock.verify_model_access 保持一致）：
      - 僅查詢 _ALLOWED_MODELS 中的模型
      - 未啟用 → log.error（不拋例外，避免 Lambda 整體初始化失敗；
                 _invoke_bedrock 收到 AccessDeniedException 後會降級 Rule-Based）
      - API 失敗 → log.warning，assume 可存取（降級）
    """
    log.info(
        f"[ModelCheck] 冷啟動模型存取權驗證（region={BEDROCK_REGION}，"
        f"{len(_ALLOWED_MODELS)} 個模型）"
    )
    try:
        client = boto3.client("bedrock", region_name=BEDROCK_REGION)
        resp   = client.list_foundation_models(byOutputModality="TEXT")
        active = {
            m["modelId"]
            for m in resp.get("modelSummaries", [])
            if m.get("modelLifecycle", {}).get("status") == "ACTIVE"
        }
        for mid in sorted(_ALLOWED_MODELS):
            if mid in active:
                log.info(f"[ModelCheck] ✓  {mid}")
            else:
                log.error(
                    f"[ModelCheck] ✗  {mid} 尚未啟用。"
                    f"請至 Bedrock Console → Model access 申請後重新部署 Lambda。"
                )
    except ClientError as exc:
        log.warning(
            f"[ModelCheck] list_foundation_models 失敗"
            f"（{exc.response['Error']['Code']}），跳過驗證。"
        )
    except Exception as exc:
        log.warning(f"[ModelCheck] API 異常（{exc}），跳過驗證。")


# ── 冷啟動時執行一次模型驗證（非同步告警，不中斷 Lambda 初始化） ──────────
_verify_model_access()
_sns      = boto3.client("sns",             region_name=S3_REGION) if SNS_ALERT_ARN else None
_table    = _dynamo.Table(DYNAMO_TABLE)


# ══════════════════════════════════════════════════════════════════════════════
#  S3 事件解析
# ══════════════════════════════════════════════════════════════════════════════

def _parse_s3_event(event: dict) -> list[tuple[str, str]]:
    """
    解析 S3 Event Notification，回傳 [(bucket, key), ...] 列表。

    支援格式：
      - 標準 S3 Event（s3:ObjectCreated:*）
      - SQS 包裝的 S3 Event（SQS → Lambda）
      - SNS 包裝的 S3 Event（SNS → SQS → Lambda）
    """
    sources: list[tuple[str, str]] = []

    for record in event.get("Records", []):
        # ── 直接 S3 觸發 ──────────────────────────────────────────────────
        if "s3" in record:
            bucket = record["s3"]["bucket"]["name"]
            key    = record["s3"]["object"]["key"]
            # URL decode（S3 key 中的空格會被編碼成 +）
            key    = key.replace("+", " ")
            sources.append((bucket, key))
            continue

        # ── SQS 包裝 ──────────────────────────────────────────────────────
        if "body" in record:
            body = record["body"]
            if isinstance(body, str):
                try:
                    body = json.loads(body)
                except json.JSONDecodeError:
                    continue
            # SNS 包裝在 SQS 內
            if "Message" in body:
                inner = body["Message"]
                if isinstance(inner, str):
                    inner = json.loads(inner)
                for s3r in inner.get("Records", []):
                    if "s3" in s3r:
                        sources.append((
                            s3r["s3"]["bucket"]["name"],
                            s3r["s3"]["object"]["key"].replace("+", " "),
                        ))
            # 直接 S3 在 SQS Body
            elif "Records" in body:
                for s3r in body["Records"]:
                    if "s3" in s3r:
                        sources.append((
                            s3r["s3"]["bucket"]["name"],
                            s3r["s3"]["object"]["key"].replace("+", " "),
                        ))

    log.info(f"解析到 {len(sources)} 個 S3 來源")
    return sources


# ══════════════════════════════════════════════════════════════════════════════
#  S3 資料讀取
# ══════════════════════════════════════════════════════════════════════════════

def _load_predictions(bucket: str, key: str) -> list[dict]:
    """
    從 S3 讀取預測結果 CSV，回傳高風險用戶列表（probability ≥ RISK_THRESHOLD）。

    CSV 格式（submission_with_prob.csv）：
        user_id, probability [, feature_cols...]

    回傳格式：
        [{"user_id": 123, "probability": 0.94, "features": {...}}, ...]
    """
    log.info(f"讀取 s3://{bucket}/{key}")
    try:
        obj      = _s3.get_object(Bucket=bucket, Key=key)
        raw      = obj["Body"].read().decode("utf-8")
    except ClientError as e:
        log.error(f"S3 GetObject 失敗: {e}")
        raise

    import csv
    reader = csv.DictReader(io.StringIO(raw))
    rows   = []

    for row in reader:
        try:
            prob = float(row.get("probability", 0))
        except (ValueError, TypeError):
            continue

        if prob < RISK_THRESHOLD:
            continue

        user_id = row.get("user_id", "")
        try:
            user_id = int(float(user_id))
        except (ValueError, TypeError):
            continue

        # 特徵欄位與 SHAP 欄位分開解析
        # SHAP 欄位（shap_xxx）：記錄每個特徵對模型輸出的邊際貢獻度
        features: dict = {}
        shap_values: dict = {}
        for col, val in row.items():
            if col in NON_FEATURE_COLS:
                continue
            try:
                numeric = float(val)
            except (ValueError, TypeError):
                features[col] = val   # 保留字串型特徵
                continue
            if col.startswith(SHAP_COL_PREFIX):
                shap_values[col[len(SHAP_COL_PREFIX):]] = numeric  # 去掉前綴
            else:
                features[col] = numeric

        rows.append({
            "user_id":     user_id,
            "probability": prob,
            "features":    features,
            "shap_values": shap_values,   # 可為空 dict（CSV 無 SHAP 欄位時）
            "incident_id": row.get("incident_id", str(uuid.uuid4())),
            "predicted_at": row.get("predicted_at", ""),  # 用於去重時排序
        })

    log.info(f"高風險用戶：{len(rows)} 筆 / 門檻 {RISK_THRESHOLD:.2f}")
    return rows


# ══════════════════════════════════════════════════════════════════════════════
#  風險路由
# ══════════════════════════════════════════════════════════════════════════════

def _classify(prob: float) -> tuple[str, Optional[str], int]:
    """
    回傳 (scoring_tier, model_id, max_tokens)。
    model_id = None 表示不呼叫 Bedrock（Rule-Based 回應）。
    """
    if prob > TIER_EXTREME:
        return "EXTREME",  MODEL_HAIKU,  600
    if prob > TIER_BOUNDARY_HI:
        return "HIGH",     MODEL_SONNET, 900
    if prob >= TIER_BOUNDARY_LO:
        return "BOUNDARY", MODEL_SONNET, 1200
    return "MEDIUM", None, 0


def _default_action(tier: str) -> dict:
    """
    Rule-Based 預設動作（MEDIUM 層或 LLM 解析失敗時）。

    幣託內部核准行動代碼對照：
      FREEZE_ACCOUNT  — 完全凍結帳戶（P > 0.90，EXTREME 層）
      LOCK_ACCOUNT    — 鎖定提領與交易，保留查看（HIGH 層嚴重案例）
      ENHANCED_KYC    — 強化 KYC 審查（BOUNDARY 層預設）
      WATCH_ONLY      — 僅監控，不限制操作（MEDIUM 層）
      CALL_VERIFY     — 電訪核實（HIGH 層標準程序）
    """
    if tier == "EXTREME":
        return {
            "primary_action":             "FREEZE_ACCOUNT",
            "auto_executable":            True,
            "execution_priority":         1,
            "str_required":               True,
            "str_deadline_hours":         24,
            "freeze_duration_days":       30,
            "daily_withdrawal_limit_twd": None,
            "require_kyc_upgrade":        False,
            "watchlist_days":             None,
            "steps": [
                "立即凍結帳戶所有交易功能（凍結期 30 日）",
                "24 小時內向調查局洗錢防制處提交 STR",
                "完整保全 KYC 文件、IP 日誌、交易紀錄",
                "啟動緊急帳戶調查程序（EAR）",
            ],
        }
    if tier == "HIGH":
        return {
            "primary_action":             "LOCK_ACCOUNT",
            "auto_executable":            True,
            "execution_priority":         2,
            "str_required":               False,
            "str_deadline_hours":         None,
            "freeze_duration_days":       7,
            "daily_withdrawal_limit_twd": None,
            "require_kyc_upgrade":        True,
            "watchlist_days":             None,
            "steps": [
                "鎖定帳戶提領與交易功能（鎖定期 7 日，保留查看）",
                "3 個工作日內啟動電訪核實程序，留存通話紀錄",
                "要求用戶上傳資金來源佐證文件（銀行對帳單 / 勞健保異動紀錄）",
                "強制重新 KYC（升級至 Level 2 完整驗證）",
            ],
        }
    if tier == "BOUNDARY":
        return {
            "primary_action":             "ENHANCED_KYC",
            "auto_executable":            False,
            "execution_priority":         3,
            "str_required":               False,
            "str_deadline_hours":         None,
            "freeze_duration_days":       None,
            "daily_withdrawal_limit_twd": 30000,
            "require_kyc_upgrade":        True,
            "watchlist_days":             60,
            "steps": [
                "降低每日提領上限至 3 萬元台幣，直至增強 KYC 完成",
                "發送站內通知要求用戶補件（資金來源聲明 + 第二身分文件）",
                "5 個工作日內完成增強 KYC 審核，留存審核紀錄",
                "審核期間持續監控 IP 登入異常與資金流向，納入 60 日觀察名單",
            ],
        }
    # MEDIUM / 其他
    return {
        "primary_action":             "WATCH_ONLY",
        "auto_executable":            True,
        "execution_priority":         4,
        "str_required":               False,
        "str_deadline_hours":         None,
        "freeze_duration_days":       None,
        "daily_withdrawal_limit_twd": None,
        "require_kyc_upgrade":        False,
        "watchlist_days":             60,
        "steps": [
            "納入強化監控名單（觀察期 60 日，不限制帳戶操作）",
            "設定異常觸發警報：單筆 > 10 萬元台幣或 IP 異動時通知合規團隊",
        ],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Bedrock Prompt 建構
# ══════════════════════════════════════════════════════════════════════════════

# ── 通用系統提示（EXTREME / HIGH 層） ──────────────────────────────────────
_SYSTEM_PROMPT = """\
你是一位資深的虛擬資產反洗錢 (AML) 調查員。你的任務是根據「機器學習模型的預測結果」\
與「特徵數值」，為幣託 (BitoPro) 的風控團隊撰寫一份專業的【風險診斷報告】。
你必須遵循以下嚴格準則：
1. **事實一致性**：報告內容必須嚴格對應輸入的 Feature Value。嚴禁虛構數據。
2. **行為模式識別**：識別出特定攻擊模式（人頭戶洗錢、高速資金穿透、分散式 IP 攻擊）。
3. **證據清單**：evidence_summary 必須列出前三大可疑特徵，每項需包含數值與偏差說明。
4. **專業語氣**：使用金融合規術語，保持客觀、專業、精確。
5. **行動代碼合規**：primary_action 必須使用幣託內部核准的行動代碼。
6. **結構化輸出**：必須以合法 JSON 格式輸出，不得包含任何 JSON 以外的文字。"""

# ── BOUNDARY 層專屬系統提示（虛擬貨幣合規官視角） ─────────────────────────
_SYSTEM_PROMPT_BOUNDARY = """\
你是幣託 (BitoPro) 的資深虛擬貨幣合規官 (Virtual Asset Compliance Officer, VACO)。
你的任務是針對模型信心度落在 65–75% 的【邊界案例】用戶，進行深度合規審查，
判斷其行為是否構成洗錢防制法（AML）及虛擬資產服務提供者（VASP）監管規定的違規疑慮。

邊界案例的挑戰在於：模型無法確定，但人工合規官必須透過細節做出裁決。

你必須遵循以下嚴格準則：
1. **法幣進、虛幣出時間差分析**：
   計算或估算「台幣入金（TWD Deposit）」到「虛擬貨幣出金（Crypto Withdrawal）」
   的時間差（min_retention_minutes 為核心指標）。
   ・ < 10 分鐘：極高速穿透，典型人頭戶洗錢手法
   ・ 10–30 分鐘：高速穿透，需深入審查
   ・ > 120 分鐘：持有行為，降低即時洗錢疑慮
2. **IP 變動頻率分析**：
   結合 unique_ip_count（不同 IP 數量）與 ip_shared_user_count（共用 IP 帳號數）
   進行多維度 IP 異常判斷：
   ・ unique_ip_count > 5 且交易頻繁：多地點登入 / 帳戶共用 / 代理風險
   ・ ip_shared_user_count > 3：機房集體詐騙高度疑慮
   ・ ip_anomaly = 1：IP 異常旗標觸發，應列為核心疑點
3. **證據清單導向**：evidence_summary 必須列出前三大可疑行為特徵，
   每項包含特徵名稱、實際數值、與正常用戶的偏差、合規意義。
4. **合規裁決精準性**：action.primary_action 必須使用幣託內部核准行動代碼，
   邊界案例通常適用 ENHANCED_KYC 或 WATCH_ONLY，嚴重異常才升至 LOCK_ACCOUNT。
5. **結構化輸出**：必須以合法 JSON 格式輸出，不得包含任何 JSON 以外的文字。"""

# ── 通用 JSON Schema（EXTREME / HIGH 層） ──────────────────────────────────
# 欄位設計原則：
#   risk_summary        : Claude 負責的分析性描述（人類可讀）
#   key_evidences       : Top 5 SHAP 特徵的可疑性解讀（結構化）
#   action_recommendation: 給合規官的自然語言建議（不含機器碼）
# 機器可執行的合規行動（primary_action / auto_executable 等）
# 由 _default_action(tier) 規則引擎決定，不由 LLM 輸出，以降低幻覺風險。
_JSON_SCHEMA = """\
{
  "risk_summary": "（字串）風險成因摘要，50–100字，結合特徵數值說明判定高風險的主要原因與威脅模式",
  "key_evidences": [
    {"feature": "（特徵中文名稱）", "value": "（實際值，含單位）", "interpretation": "（為何可疑，20–40字）"},
    {"feature": "（特徵中文名稱）", "value": "（實際值）",        "interpretation": "（可疑原因）"},
    {"feature": "（特徵中文名稱）", "value": "（實際值）",        "interpretation": "（可疑原因）"},
    {"feature": "（特徵中文名稱）", "value": "（實際值）",        "interpretation": "（可疑原因）"},
    {"feature": "（特徵中文名稱）", "value": "（實際值）",        "interpretation": "（可疑原因）"}
  ],
  "action_recommendation": "（字串）給合規官的自然語言處置建議，30–60字，說明應採取何種措施及理由"
}"""

# ── BOUNDARY 層專屬 JSON Schema（含法幣/IP 深度分析欄位） ──────────────────
_JSON_SCHEMA_BOUNDARY = """\
{
  "risk_summary": "（字串）從合規官視角出發的邊界案例摘要，80–150字，重點說明法幣/虛幣時間差與IP模式的核心疑點",
  "key_evidences": [
    {"feature": "（最具說服力的可疑特徵中文名稱）", "value": "（實際值，含單位）", "interpretation": "（合規意義，20–50字）"},
    {"feature": "（第二可疑特徵）",                 "value": "（實際值）",        "interpretation": "（合規意義）"},
    {"feature": "（第三可疑特徵）",                 "value": "（實際值）",        "interpretation": "（合規意義）"},
    {"feature": "（第四可疑特徵）",                 "value": "（實際值）",        "interpretation": "（合規意義）"},
    {"feature": "（第五可疑特徵）",                 "value": "（實際值）",        "interpretation": "（合規意義）"}
  ],
  "action_recommendation": "（字串）給合規官的處置建議，50–80字，說明邊界案例應優先審查哪些方向及所需文件",
  "fiat_crypto_analysis": {
    "estimated_throughput_minutes": "（數值或 null）法幣入金到虛幣出金的估計時間差（分鐘）",
    "risk_verdict": "（字串，必須是以下之一）HIGH_SPEED_THROUGHPUT / MODERATE_SPEED / NORMAL_HOLD / INSUFFICIENT_DATA",
    "explanation": "（字串）50–100字，說明資金穿透速度與洗錢風險關聯"
  },
  "ip_analysis": {
    "unique_ip_count": "（整數）不同 IP 數量（來自特徵值）",
    "ip_shared_user_count": "（整數）同 IP 共用帳號數（來自特徵值）",
    "risk_verdict": "（字串，必須是以下之一）DATACENTER_FRAUD / MULTI_LOCATION / PROXY_SUSPECTED / BOT_PATTERN / NORMAL",
    "explanation": "（字串）40–80字，說明 IP 異常模式與機房詐騙或帳戶共用風險"
  }
}"""


def _build_feature_table(features: dict) -> str:
    """將特徵 dict 格式化為易讀表格，注入 Prompt。"""
    lines = [
        f"{'特徵名稱':<32}  {'實際值':>14}",
        "-" * 48,
    ]
    for col, val in sorted(features.items()):
        label = FEATURE_LABELS.get(col, col)[:30]
        if isinstance(val, float):
            val_str = f"{val:.4g}"
        elif isinstance(val, bool):
            val_str = "是" if val else "否"
        else:
            val_str = str(val)
        lines.append(f"{label:<32}  {val_str:>14}")
    return "\n".join(lines)


def _extract_ip_context(features: dict) -> str:
    """為 BOUNDARY 層提示建構 IP 變動分析上下文段落。"""
    unique_ip   = features.get("unique_ip_count", None)
    ip_shared   = features.get("ip_shared_user_count", None)
    ip_anomaly  = features.get("ip_anomaly", None)
    has_hsr     = features.get("has_high_speed_risk", None)

    lines = []
    if unique_ip is not None:
        risk_note = "（⚠ 多地點 / 代理 / 帳戶共用疑慮）" if unique_ip > 5 else "（正常範圍）"
        lines.append(f"  ・ 不同 IP 數量：{int(unique_ip)} 個  {risk_note}")
    if ip_shared is not None:
        risk_note = "（🔴 機房集體詐騙高度疑慮）" if ip_shared > 3 else "（正常範圍）"
        lines.append(f"  ・ 同 IP 共用帳號數：{int(ip_shared)} 個  {risk_note}")
    if ip_anomaly is not None:
        flag = "觸發（異常）" if ip_anomaly else "未觸發"
        lines.append(f"  ・ IP 異常旗標：{flag}")
    if has_hsr is not None:
        flag = "是（<10 分鐘，自動化腳本疑慮）" if has_hsr else "否"
        lines.append(f"  ・ 高速交易旗標：{flag}")

    return "\n".join(lines) if lines else "  ・ IP 資料不足，無法評估"


def _top5_by_shap(
    features:    dict,
    shap_values: dict,
    n:           int = TOP_FEATURES_N,
) -> dict:
    """
    依 SHAP 絕對值取貢獻度最高的前 n 個特徵，回傳子集 dict。

    策略
    ─────
    ① 若 shap_values 非空：以 |shap| 降序排序，選 Top n 特徵名稱
    ② 若 shap_values 為空（CSV 無 shap_ 欄位）：
       以特徵數值絕對值作為近似貢獻度排序（數值大的特徵通常影響較大）
       僅對數值型特徵排序，字串型特徵補至末位

    回傳
    ─────
    dict — 最多 n 個 {feature_name: feature_value}，保留原始特徵值（非 SHAP）
    """
    if shap_values:
        # 以 SHAP 絕對值排序，選 Top n 出現在 features 中的特徵名
        ordered = sorted(
            [(k, abs(v)) for k, v in shap_values.items() if k in features],
            key=lambda x: x[1],
            reverse=True,
        )
        top_keys = [k for k, _ in ordered[:n]]
    else:
        # Fallback：以特徵數值絕對值排序（字串型補末位）
        def _sort_key(item):
            v = item[1]
            return (0, -abs(v)) if isinstance(v, (int, float)) else (1, 0)
        ordered = sorted(features.items(), key=_sort_key)
        top_keys = [k for k, _ in ordered[:n]]

    return {k: features[k] for k in top_keys if k in features}


def _dedup_records_by_user(records: list[dict]) -> list[dict]:
    """
    去重：同一 user_id 的多筆記錄，僅保留最新一筆。

    排序依據（優先順序）：
    ① predicted_at 欄位（ISO 時間字串）→ 較大值 = 較新
    ② 清單位置（後出現的視為較新，適用無 predicted_at 的 CSV）

    為何需要去重
    ─────────────
    若同一批次 submission.csv 中同一 user_id 出現多次（SageMaker 多次推論、
    資料管線重跑等），重複呼叫 Bedrock 會浪費 token 並可能造成 DynamoDB 競寫。
    """
    seen: dict[int, dict] = {}
    for record in records:
        uid = record["user_id"]
        if uid not in seen:
            seen[uid] = record
        else:
            # 比較 predicted_at（字串字典序，ISO 格式可直接比較）
            prev_ts = seen[uid].get("predicted_at", "")
            curr_ts = record.get("predicted_at", "")
            if curr_ts >= prev_ts:   # 相等時以後出現的覆蓋（位置較新）
                seen[uid] = record

    original_count = len(records)
    deduped_count  = len(seen)
    if original_count != deduped_count:
        log.info(
            f"[Dedup] user_records 去重：{original_count} → {deduped_count} 筆"
            f"（移除 {original_count - deduped_count} 筆重複 user_id）"
        )
    return list(seen.values())


def _build_user_prompt(
    user_id:     int,
    prob:        float,
    tier:        str,
    features:    dict,
    model_id:    str,
    shap_values: dict | None = None,
) -> str:
    """
    依風險層級選擇診斷深度建構 Prompt。

    上下文過濾：注入 Prompt 前，先以 SHAP 值篩選 Top 5 貢獻特徵，
    減少 Input Token 消耗，並讓 Claude 聚焦在最有判別力的信號。

    輸出格式：強制要求 JSON，固定欄位：
      {"risk_summary": "...", "key_evidences": [...], "action_recommendation": "..."}
    BOUNDARY 層額外含 fiat_crypto_analysis + ip_analysis。

    EXTREME（Haiku）  ：精簡版，Top 5 特徵，降低 token 消耗
    HIGH（Sonnet）     ：完整版，Top 5 特徵，標準 AML 分析
    BOUNDARY（Sonnet）：合規官深度版，Top 5 特徵，法幣/IP 深度分析
    """
    shap_values   = shap_values or {}
    prob_pct      = round(prob * 100, 1)
    kyc_level     = int(features.get("kyc_level", 0))
    kyc_desc      = {0: "L0 未驗證", 1: "L1 手機驗證", 2: "L2 完整驗證"}.get(kyc_level, f"L{kyc_level}")
    retention     = features.get("min_retention_minutes", None)
    retention_str = f"{retention:.1f} 分鐘" if retention is not None else "N/A"
    hops          = features.get("min_hops_to_blacklist", None)
    hops_str      = str(int(hops)) if hops is not None else "N/A"
    twd_dep       = features.get("twd_deposit_count", None)
    crypto_with   = features.get("crypto_withdraw_count", None)

    # ── 上下文過濾：僅取 Top 5 SHAP 特徵注入 Prompt ──────────────────────
    top5_features = _top5_by_shap(
        {k: v for k, v in features.items() if k in FEATURE_LABELS},
        shap_values,
    )
    feat_table = _build_feature_table(top5_features)

    tier_note = {
        "EXTREME":  "（P > 90%，極高風險，制式報告模式）",
        "HIGH":     "（P 75–90%，高風險，標準分析模式）",
        "BOUNDARY": "（P 65–75%，邊界案例，合規官深度診斷模式）",
    }.get(tier, "")

    # ── BOUNDARY 層：合規官視角，深度分析法幣/IP ──────────────────────────
    if tier == "BOUNDARY":
        fiat_crypto_note = ""
        if retention is not None:
            if retention < 10:
                fiat_crypto_note = f"⚠ 極高速穿透（{retention:.1f} 分鐘），典型人頭戶洗錢手法"
            elif retention < 30:
                fiat_crypto_note = f"⚠ 高速穿透（{retention:.1f} 分鐘），需深入審查"
            else:
                fiat_crypto_note = f"持有時間 {retention:.1f} 分鐘，即時洗錢疑慮相對低"
        else:
            fiat_crypto_note = "滯留時間資料不足，請參閱 retention_event_count"

        ip_context = _extract_ip_context(features)

        prompt = f"""---
【合規審查目標 — 用戶 ID】: {user_id}  {tier_note}
【模型風險機率】: {prob_pct}%（邊界案例，需合規官裁決）

══ 一、全部風險特徵數值 ══
{feat_table}

══ 二、法幣進、虛幣出（Fiat-In / Crypto-Out）時間差分析 ══
- 最短資金滯留時間（min_retention_minutes）: {retention_str}
- 評估：{fiat_crypto_note}
- 台幣入金次數（twd_deposit_count）: {int(twd_dep) if twd_dep is not None else "N/A"}
- 虛幣出金次數（crypto_withdraw_count）: {int(crypto_with) if crypto_with is not None else "N/A"}
- 請分析：資金是否呈現「法幣入→即轉虛幣出」的短時間穿透模式？

══ 三、IP 變動頻率分析 ══
{ip_context}
- KYC 等級: {kyc_desc}
- 距黑名單最短跳數: {hops_str}
- 請分析：IP 模式是否顯示多帳戶共用、機房代理或自動化腳本行為？

══ 四、合規裁決依據 ══
邊界案例的合規裁決需基於明確證據，請：
① 將以上 Top {TOP_FEATURES_N} SHAP 特徵中最具說服力的可疑行為列入 key_evidences
② 針對法幣/虛幣時間差給出 fiat_crypto_analysis 判決
③ 針對 IP 異常給出 ip_analysis 判決
④ 在 action_recommendation 給出自然語言建議（機器碼由系統規則引擎自動決定）

【輸出規範 — 必須嚴格遵守】:
・ 只輸出 JSON，不得有任何前言、說明、markdown 或代碼區塊
・ 欄位名稱固定為：risk_summary、key_evidences、action_recommendation、fiat_crypto_analysis、ip_analysis
・ 不得新增或省略任何頂層欄位
{_JSON_SCHEMA_BOUNDARY}
---"""
        return prompt

    # ── EXTREME / HIGH 層：標準 AML 報告格式 ─────────────────────────────
    prompt = f"""---
【待分析用戶 ID】: {user_id}  {tier_note}
【模型預測風險機率】: {prob_pct}%  【風險層級】: {tier}

【Top {TOP_FEATURES_N} 高 SHAP 貢獻特徵】:
{feat_table}

【上下文環境】:
- KYC 等級: {kyc_desc}
- 最短資金滯留時間: {retention_str}
- 與黑名單地址最小跳數: {hops_str}

【輸出規範 — 必須嚴格遵守】:
・ 只輸出 JSON，不得有任何前言、說明、markdown 或代碼區塊
・ 欄位名稱固定為：risk_summary、key_evidences、action_recommendation
・ key_evidences 必須包含 {TOP_FEATURES_N} 個條目，依風險貢獻度排序
・ 不得新增或省略任何頂層欄位
{_JSON_SCHEMA}
---"""

    return prompt


# ══════════════════════════════════════════════════════════════════════════════
#  Bedrock 呼叫
# ══════════════════════════════════════════════════════════════════════════════

def _invoke_bedrock(
    user_id:     int,
    prob:        float,
    tier:        str,
    features:    dict,
    model_id:    str,
    max_tokens:  int,
    shap_values: dict | None = None,
) -> dict:
    """
    呼叫 Bedrock 並回傳已解析的 JSON 診斷結果。
    使用 Prefill 技術（assistant 欄位以 "{" 開頭）強制 JSON 輸出。

    系統提示依層級選擇：
      BOUNDARY → _SYSTEM_PROMPT_BOUNDARY（虛擬貨幣合規官視角，深度法幣/IP 分析）
      EXTREME / HIGH → _SYSTEM_PROMPT（資深 AML 調查員，制式報告）
    """
    # ── 依層級選擇系統提示 ──────────────────────────────────────────────────
    system_prompt = (
        _SYSTEM_PROMPT_BOUNDARY if tier == "BOUNDARY" else _SYSTEM_PROMPT
    )

    user_prompt = _build_user_prompt(user_id, prob, tier, features, model_id, shap_values)

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": [
            {"role": "user",      "content": user_prompt},
            {"role": "assistant", "content": "{"},   # ← Prefill，強制從 { 開始
        ],
    })

    # ── 模型白名單守衛 ────────────────────────────────────────────────────────
    if model_id not in _ALLOWED_MODELS:
        raise ValueError(
            f"[ModelCheck] model_id='{model_id}' 不在允許清單中。"
            f"允許：{sorted(_ALLOWED_MODELS)}"
        )

    log.debug(f"user={user_id} tier={tier} model={model_id} max_tokens={max_tokens}")

    # ── ThrottlingException Full-Jitter 指數退避 ──────────────────────────────
    # 節流控制已移至基礎設施層（SQS FIFO + Reserved Concurrency=1）。
    # 本層保留退避作為最後防線，使用 Full-Jitter 策略分散多實例的重試時間點。
    #
    # Full-Jitter 公式：wait = random.uniform(0, min(base * 2^attempt, cap))
    #   - 相較於確定性退避，Full-Jitter 將重試分散在 [0, cap] 之間
    #   - 有效避免「Thundering Herd」問題（多實例同時重試造成的衝擊波）
    #
    # 退避序列（期望值，非確定值）：
    #   attempt 0: [0, 2s]   attempt 1: [0, 4s]   attempt 2: [0, 8s]
    #   attempt 3: [0, 16s]  attempt 4: [0, 32s]  → 第 5 次失敗 → _BedrockThrottled
    _throttle_count = 0

    for attempt in range(_THROTTLE_MAX_RETRIES + 1):
        try:
            resp     = _bedrock.invoke_model(
                modelId      = model_id,
                body         = body,
                contentType  = "application/json",
                accept       = "application/json",
            )
            raw_text = "{" + json.loads(resp["body"].read())["content"][0]["text"]
            return _parse_json_response(raw_text, tier)

        except ClientError as exc:
            code        = exc.response["Error"]["Code"]
            is_throttle = code in (
                "ThrottlingException",
                "TooManyRequestsException",
                "ServiceUnavailableException",
            )

            if is_throttle:
                _throttle_count += 1

                if attempt < _THROTTLE_MAX_RETRIES:
                    # Full-Jitter：wait ∈ [0, min(base × 2^attempt, cap)]
                    cap  = min(_THROTTLE_BASE_WAIT * (2 ** attempt), _THROTTLE_MAX_WAIT)
                    wait = random.uniform(0.0, cap)
                    log.warning(
                        f"[Bedrock] {code} user={user_id} model={model_id} "
                        f"attempt={attempt+1}/{_THROTTLE_MAX_RETRIES+1}，"
                        f"Full-Jitter 退避 {wait:.2f}s（cap={cap:.0f}s，"
                        f"累計限流 {_throttle_count} 次）…"
                    )
                    time.sleep(wait)
                else:
                    # 超過重試上限 → 拋出降級異常，由 _diagnose_one 切換 Rule-Based
                    log.error(
                        f"[Bedrock] ThrottlingException 累計 {_throttle_count} 次，"
                        f"已達上限 {_THROTTLE_MAX_RETRIES}，user={user_id} "
                        f"觸發降級模式（is_ai_generated=False）。"
                    )
                    raise _BedrockThrottled(
                        f"ThrottlingException × {_throttle_count}，user={user_id}"
                    )
            else:
                # 非限流錯誤（AccessDeniedException、ModelNotReadyException 等）
                # 不重試，直接向上傳遞，由 _diagnose_one 的通用 except 處理
                log.error(
                    f"[Bedrock] {code} user={user_id} attempt={attempt+1}，不重試。"
                )
                raise


def _parse_json_response(raw: str, tier: str) -> dict:
    """
    三層 Parse Fallback：
      ① json.loads（完整解析）
      ② regex 擷取第一個 {...}（Bedrock 偶爾在 JSON 前後多餘文字）
      ③ 預設值（完全無法解析時，回傳 Rule-Based 結果，欄位依層級補足）
    """
    # 嘗試①：直接解析
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # 嘗試②：regex 擷取
    m = re.search(r'\{[\s\S]+\}', raw)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass

    # 嘗試③：降級到預設值
    log.warning(f"JSON 解析失敗（tier={tier}），使用 Rule-Based 預設值。原文（前200字）：{raw[:200]}")

    _empty_evidence = [
        {"rank": i, "feature_name": "—", "observed_value": "—",
         "deviation": "—", "compliance_implication": "LLM 輸出異常，請人工複審"}
        for i in (1, 2, 3)
    ]

    base = {
        "risk_diagnosis":             "（LLM 解析失敗，使用 Rule-Based 預設診斷）",
        "threat_pattern":             "UNKNOWN",
        "threat_pattern_zh":          "未知",
        "threat_pattern_description": "模型輸出格式異常，請人工複審。",
        "evidence_summary":           _empty_evidence,
        "action":                     _default_action(tier),
    }

    # BOUNDARY 層補充法幣/IP 分析欄位
    if tier == "BOUNDARY":
        base["fiat_crypto_analysis"] = {
            "estimated_throughput_minutes": None,
            "risk_verdict":                 "INSUFFICIENT_DATA",
            "explanation":                  "LLM 輸出解析失敗，無法自動評估，請人工審查。",
        }
        base["ip_analysis"] = {
            "unique_ip_count":    None,
            "ip_shared_user_count": None,
            "risk_verdict":       "NORMAL",
            "explanation":        "LLM 輸出解析失敗，無法自動評估，請人工審查。",
        }

    return base


# ══════════════════════════════════════════════════════════════════════════════
#  EXTREME 層強制覆蓋（Haiku 制式報告後確保行動代碼正確）
# ══════════════════════════════════════════════════════════════════════════════

def _enforce_extreme_action(diagnosis: dict) -> dict:
    """
    P > 0.90 的用戶無論 LLM 輸出什麼，強制設定：
      primary_action = FREEZE_ACCOUNT
      auto_executable = true
      execution_priority = 1
    此設計保障 EXTREME 層不因 LLM 幻覺而降低行動等級。
    """
    action = diagnosis.get("action", {})
    action["primary_action"]    = "FREEZE_ACCOUNT"
    action["auto_executable"]   = True
    action["execution_priority"] = 1
    action["str_required"]      = True
    if not action.get("str_deadline_hours"):
        action["str_deadline_hours"] = 24
    if not action.get("freeze_duration_days"):
        action["freeze_duration_days"] = 30
    diagnosis["action"] = action
    return diagnosis


# ══════════════════════════════════════════════════════════════════════════════
#  單用戶完整診斷流程
# ══════════════════════════════════════════════════════════════════════════════

def _diagnose_one(user_record: dict, source_key: str) -> dict:
    """
    對單一用戶執行完整診斷流程，回傳可直接寫入 DynamoDB 的 item dict。
    """
    user_id     = user_record["user_id"]
    prob        = user_record["probability"]
    features    = user_record.get("features", {})
    shap_values = user_record.get("shap_values", {})
    incident_id = user_record.get("incident_id", str(uuid.uuid4()))

    tier, model_id, max_tokens = _classify(prob)
    generated_at = datetime.now(timezone.utc).isoformat()

    # ── 呼叫 Bedrock 或 Rule-Based ────────────────────────────────────────
    is_ai_generated   = False   # 預設 False，Bedrock 成功後才設 True
    degradation_reason = ""     # 空字串代表無降級

    if model_id:
        try:
            diagnosis       = _invoke_bedrock(
                user_id, prob, tier, features, model_id, max_tokens,
                shap_values=shap_values,
            )
            is_ai_generated = True   # ← Bedrock 成功，標記為 AI 生成

        except _BedrockThrottled as exc:
            # ── 降級路徑 1：ThrottlingException 超過重試上限 ──────────────
            log.warning(
                f"[降級模式] user={user_id} tier={tier} "
                f"Bedrock 持續限流，切換至 Rule-Based 診斷。原因：{exc}"
            )
            is_ai_generated   = False
            degradation_reason = "THROTTLE_DEGRADED"
            _empty_kev = [
                {"feature": "—", "value": "—",
                 "interpretation": "Bedrock 限流降級，Rule-Based 自動輸出，請人工複審"}
            ] * TOP_FEATURES_N
            diagnosis = {
                "risk_summary": (
                    f"⚠ Bedrock 持續限流超過 {_THROTTLE_MAX_RETRIES} 次，已自動切換至 Rule-Based "
                    f"診斷模式（機率 {prob:.2%}，{tier} 層）。此報告為系統預設值，請合規人員人工複審。"
                ),
                "key_evidences":         _empty_kev,
                "action_recommendation": "Bedrock 限流降級，建議於系統恢復後重新觸發診斷，或人工審查。",
            }
            if tier == "BOUNDARY":
                diagnosis["fiat_crypto_analysis"] = {
                    "estimated_throughput_minutes": None,
                    "risk_verdict":                 "INSUFFICIENT_DATA",
                    "explanation":                  "Bedrock 限流降級，無法自動評估，請人工審查。",
                }
                diagnosis["ip_analysis"] = {
                    "unique_ip_count":      int(features.get("unique_ip_count", 0) or 0),
                    "ip_shared_user_count": int(features.get("ip_shared_user_count", 0) or 0),
                    "risk_verdict":         "NORMAL",
                    "explanation":          "Bedrock 限流降級，無法自動評估，請人工審查。",
                }

        except Exception as exc:
            # ── 降級路徑 2：其他 Bedrock 錯誤（AccessDenied、Network 等）──
            log.error(
                f"[降級模式] user={user_id} Bedrock 呼叫失敗（{type(exc).__name__}: {exc}），"
                f"切換至 Rule-Based 診斷。"
            )
            is_ai_generated    = False
            degradation_reason = f"EXCEPTION:{type(exc).__name__}"
            _empty_kev = [
                {"feature": "—", "value": "—",
                 "interpretation": "Bedrock 呼叫失敗，請人工複審"}
            ] * TOP_FEATURES_N
            diagnosis = {
                "risk_summary":          f"Bedrock 呼叫失敗（{type(exc).__name__}），使用 Rule-Based 預設診斷，請人工複審。",
                "key_evidences":         _empty_kev,
                "action_recommendation": "請人工複審此用戶，確認是否需要進一步調查。",
            }
            if tier == "BOUNDARY":
                diagnosis["fiat_crypto_analysis"] = {
                    "estimated_throughput_minutes": None,
                    "risk_verdict":                 "INSUFFICIENT_DATA",
                    "explanation":                  "Bedrock 呼叫失敗，無法自動評估，請人工審查。",
                }
                diagnosis["ip_analysis"] = {
                    "unique_ip_count":      int(features.get("unique_ip_count", 0) or 0),
                    "ip_shared_user_count": int(features.get("ip_shared_user_count", 0) or 0),
                    "risk_verdict":         "NORMAL",
                    "explanation":          "Bedrock 呼叫失敗，無法自動評估，請人工審查。",
                }
    else:
        # MEDIUM 層：Rule-Based，不調用 LLM（正常行為，非降級）
        degradation_reason = "RULE_BASED_TIER"
        diagnosis = {
            "risk_summary":          f"機率 {prob:.2%}，屬中度風險，自動列入觀察名單。",
            "key_evidences":         [
                {"feature": "probability", "value": f"{prob:.2%}",
                 "interpretation": "中度風險，系統自動監控"}
            ],
            "action_recommendation": "列入 60 日觀察名單，期間如有異常交易再行升級。",
        }

    # EXTREME 層強制覆蓋行動代碼（rule-based，不依賴 LLM 輸出）
    action = _default_action(tier)
    if tier == "EXTREME":
        diagnosis = _enforce_extreme_action(diagnosis)
        action    = diagnosis.get("action", action)

    # ── 組裝 DynamoDB item ────────────────────────────────────────────────
    # action 來自 rule-based _default_action(tier)，不由 LLM 決定：
    # 避免 LLM 幻覺導致不符規範的合規代碼寫入後台執行系統。
    item = {
        # Keys
        "user_id":      str(user_id),           # PK（String 型）
        "generated_at": generated_at,            # SK

        # 路由資訊
        "scoring_tier": tier,
        "risk_level":   _tier_to_level(tier),   # HIGH / MEDIUM / LOW（GSI 用）
        "model_used":   model_id or "rule-based",
        "probability":  _to_decimal(prob),       # DynamoDB 不支援 float，轉 Decimal

        # Claude 分析結果（3 個固定欄位）
        "risk_summary":          diagnosis.get("risk_summary", ""),
        "key_evidences":         json.dumps(diagnosis.get("key_evidences", []),  ensure_ascii=False),
        "action_recommendation": diagnosis.get("action_recommendation", ""),

        # 機器可執行指令（由 rule-based 規則引擎決定，前端可直接 parse）
        "action_json": json.dumps(action, ensure_ascii=False),
        "primary_action":     action.get("primary_action", "NO_ACTION"),
        "auto_executable":    bool(action.get("auto_executable", False)),
        "execution_priority": int(action.get("execution_priority", 5)),

        # AI / Rule-Based 標記（合規儀表板用）
        # is_ai_generated=True  → Bedrock 正常輸出
        # is_ai_generated=False → 降級輸出（限流/異常），需人工複審
        "is_ai_generated":   is_ai_generated,
        "degradation_reason": degradation_reason,   # THROTTLE_DEGRADED / EXCEPTION:xxx / RULE_BASED_TIER / ""

        # 溯源資訊
        "incident_id":  incident_id,
        "source_file":  source_key,

        # BOUNDARY 層深度分析欄位（非 BOUNDARY 為空字串，DynamoDB 不存 null）
        "fiat_crypto_analysis": json.dumps(
            diagnosis.get("fiat_crypto_analysis", {}), ensure_ascii=False
        ),
        "ip_analysis": json.dumps(
            diagnosis.get("ip_analysis", {}), ensure_ascii=False
        ),

        # 特徵快照（Top 5 SHAP 特徵，供前端儀表板顯示）
        "feature_snapshot": json.dumps(
            _top5_by_shap(
                {k: v for k, v in features.items() if k in FEATURE_LABELS},
                shap_values,
            ),
            ensure_ascii=False,
        ),

        # DynamoDB TTL（90 天後自動刪除）
        "expire_at": int(time.time()) + DYNAMO_TTL_DAYS * 86400,
    }

    log.info(
        f"user={user_id}  prob={prob:.3f}  tier={tier}"
        f"  action={item['primary_action']}  auto={item['auto_executable']}"
    )
    return item


def _tier_to_level(tier: str) -> str:
    """將 ScoringTier 轉為 GSI 用的粗粒度 risk_level。"""
    return {"EXTREME": "HIGH", "HIGH": "HIGH", "BOUNDARY": "HIGH",
            "MEDIUM": "MEDIUM"}.get(tier, "MEDIUM")


def _to_decimal(val) -> Decimal:
    """float → Decimal（DynamoDB 要求）。"""
    return Decimal(str(round(val, 6)))


# ══════════════════════════════════════════════════════════════════════════════
#  SQS Producer：將診斷請求發至 FIFO 佇列
# ══════════════════════════════════════════════════════════════════════════════

def _enqueue_to_sqs(user_records: list[dict], source_key: str) -> int:
    """
    將每個用戶的診斷請求單獨發至 SQS FIFO 佇列（Producer 模式）。

    架構說明
    --------
    每筆 user_record 成為一個獨立 SQS 訊息，搭配 Lambda Reserved Concurrency=1，
    確保 Consumer Lambda 序列處理，Bedrock 呼叫自然維持 < 1 RPS，無需 sleep。

    訊息格式
    --------
    {
        "message_type": "DIAGNOSIS_JOB",
        "user_record":  {...},        # 單一用戶的診斷所需資料
        "source_key":   "predictions/2026-03-24/submission.csv"
    }

    FIFO Queue 設定（template.yaml 對應欄位）
    -----------------------------------------
    MessageGroupId = "bedrock-diagnosis"  → 所有訊息同組，嚴格序列消費
    MessageDeduplicationId = uuid4()      → 每筆唯一，避免 FIFO 去重誤判

    回傳：成功入隊的訊息數
    """
    if not _sqs or not SQS_QUEUE_URL:
        log.debug("[SQS] SQS_QUEUE_URL 未設定，跳過入隊（直接在 Lambda 內處理）")
        return 0

    enqueued = 0
    is_fifo  = SQS_QUEUE_URL.endswith(".fifo")

    for record in user_records:
        body = json.dumps(
            {"message_type": "DIAGNOSIS_JOB", "user_record": record, "source_key": source_key},
            ensure_ascii=False,
            default=str,
        )
        kwargs: dict = {"QueueUrl": SQS_QUEUE_URL, "MessageBody": body}
        if is_fifo:
            kwargs["MessageGroupId"]         = "bedrock-diagnosis"
            kwargs["MessageDeduplicationId"] = str(uuid.uuid4())   # 保證唯一

        try:
            _sqs.send_message(**kwargs)
            enqueued += 1
        except ClientError as exc:
            log.error(
                f"[SQS] 訊息入隊失敗 user={record.get('user_id', '?')}: "
                f"{exc.response['Error']['Code']} — {exc.response['Error']['Message']}"
            )

    log.info(f"[SQS] 入隊完成：{enqueued}/{len(user_records)} 筆 → {SQS_QUEUE_URL}")
    return enqueued


def _is_sqs_diagnosis_job(event: dict) -> bool:
    """
    判斷 Lambda 事件是否為 SQS 診斷工作訊息（Consumer 模式）。

    區分邏輯：
    ① eventSource == "aws:sqs"  → SQS 觸發
    ② body.message_type == "DIAGNOSIS_JOB"  → 我們自己的診斷工作（非 S3 事件包裝）
    """
    records = event.get("Records", [])
    if not records:
        return False
    r = records[0]
    if r.get("eventSource") != "aws:sqs":
        return False
    try:
        body = json.loads(r.get("body", "{}"))
        return isinstance(body, dict) and body.get("message_type") == "DIAGNOSIS_JOB"
    except (json.JSONDecodeError, AttributeError):
        return False


def _handle_sqs_consumer(event: dict, context: Any) -> dict:
    """
    SQS Consumer 模式（Reserved Concurrency=1）。

    每次呼叫只處理一個用戶的診斷，由 SQS 觸發。
    Reserved Concurrency=1 由 Lambda 設定保證，此處無需任何 sleep 或鎖。

    若診斷失敗（_BedrockThrottled 或其他例外），回傳非 2xx 狀態碼，
    讓 SQS 按 VisibilityTimeout 重新投遞（最多 maxReceiveCount 次後進 DLQ）。
    """
    written = errors = 0

    for record in event.get("Records", []):
        try:
            body        = json.loads(record.get("body", "{}"))
            user_record = body["user_record"]
            source_key  = body.get("source_key", "unknown")
        except (json.JSONDecodeError, KeyError) as exc:
            log.error(f"[SQS Consumer] 訊息格式錯誤，略過：{exc}")
            errors += 1
            continue

        item = _diagnose_one(user_record, source_key)
        ok, err = _write_to_dynamo([item])
        written += ok
        errors  += err

        log.info(
            f"[SQS Consumer] user={user_record.get('user_id')} "
            f"is_ai={item.get('is_ai_generated')}  dynamo={'OK' if ok else 'FAIL'}"
        )

    return {
        "statusCode": 200 if errors == 0 else 207,
        "body": {"dynamo_written": written, "errors": errors},
    }


# ══════════════════════════════════════════════════════════════════════════════
#  DynamoDB 批次寫入
# ══════════════════════════════════════════════════════════════════════════════

def _write_to_dynamo(items: list[dict]) -> tuple[int, int]:
    """
    批次寫入 DynamoDB（batch_writer 自動處理 25 筆限制與重試）。
    回傳 (success_count, error_count)。
    """
    if not items:
        return 0, 0

    success = error = 0
    try:
        with _table.batch_writer() as batch:
            for item in items:
                batch.put_item(Item=item)
                success += 1
    except ClientError as e:
        log.error(f"DynamoDB batch_writer 失敗: {e}")
        error += len(items) - success

    log.info(f"DynamoDB 寫入完成：{success} 成功 / {error} 失敗")
    return success, error


# ══════════════════════════════════════════════════════════════════════════════
#  CloudWatch 自訂指標
# ══════════════════════════════════════════════════════════════════════════════

def _put_metrics(
    total: int,
    processed: int,
    errors: int,
    tier_counts: dict[str, int],
    source_key: str,
) -> None:
    """
    發送批次診斷指標至 CloudWatch，供 Dashboard 與 Alarm 使用。

    指標：
      DiagnosisTotal     — 觸發診斷的高風險用戶總數
      DiagnosisProcessed — 成功生成診斷書的數量
      DiagnosisErrors    — 失敗數量
      ExtremeCount       — EXTREME 層用戶數
      BoundaryCount      — BOUNDARY 層（邊界案例）數量
    """
    now    = datetime.now(timezone.utc)
    bucket = source_key.split("/")[0] if "/" in source_key else "unknown"

    metric_data = [
        {"MetricName": "DiagnosisTotal",     "Value": total,     "Unit": "Count"},
        {"MetricName": "DiagnosisProcessed", "Value": processed, "Unit": "Count"},
        {"MetricName": "DiagnosisErrors",    "Value": errors,    "Unit": "Count"},
        {"MetricName": "ExtremeCount",   "Value": tier_counts.get("EXTREME", 0),   "Unit": "Count"},
        {"MetricName": "BoundaryCount",  "Value": tier_counts.get("BOUNDARY", 0),  "Unit": "Count"},
        # 降級計數：用於 CloudWatch Alarm 偵測 Bedrock 持續限流
        {"MetricName": "DegradedCount",  "Value": tier_counts.get("DEGRADED", 0),  "Unit": "Count"},
    ]
    for m in metric_data:
        m["Timestamp"] = now
        m["Dimensions"] = [
            {"Name": "Function",    "Value": os.environ.get("AWS_LAMBDA_FUNCTION_NAME", "local")},
            {"Name": "SourceBucket","Value": bucket},
        ]

    try:
        # CloudWatch 每次最多 20 筆
        for i in range(0, len(metric_data), 20):
            _cw.put_metric_data(
                Namespace=CW_NAMESPACE,
                MetricData=metric_data[i:i+20],
            )
    except ClientError as e:
        log.warning(f"CloudWatch 指標發送失敗（不影響主流程）: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  SNS 告警（批次失敗率 > 20% 時）
# ══════════════════════════════════════════════════════════════════════════════

def _alert_if_high_error_rate(
    total: int, errors: int, source_key: str
) -> None:
    if not _sns or not SNS_ALERT_ARN or total == 0:
        return
    error_rate = errors / total
    if error_rate < 0.2:
        return
    try:
        _sns.publish(
            TopicArn=SNS_ALERT_ARN,
            Subject=f"[Lambda Alert] 診斷書生成錯誤率 {error_rate:.0%}",
            Message=json.dumps({
                "event":      "HIGH_ERROR_RATE",
                "source_key": source_key,
                "total":      total,
                "errors":     errors,
                "error_rate": round(error_rate, 4),
            }, ensure_ascii=False),
        )
    except ClientError:
        pass


# ══════════════════════════════════════════════════════════════════════════════
#  Lambda Handler
# ══════════════════════════════════════════════════════════════════════════════

def lambda_handler(event: dict, context: Any) -> dict:
    """
    Lambda 主入口（雙模式）。

    ┌─────────────────────────────────────────────────────────────────┐
    │  觸發模式 1：S3 ObjectCreated（Producer 模式）                  │
    │    s3://{bucket}/predictions/*.csv 上傳 → 此 Lambda 觸發        │
    │    流程：讀取 CSV → 每個用戶 → SQS FIFO 入隊                   │
    │    （SQS_QUEUE_URL 未設定時退化為舊的直接處理模式）             │
    ├─────────────────────────────────────────────────────────────────┤
    │  觸發模式 2：SQS 診斷工作（Consumer 模式）                     │
    │    SQS FIFO → 此 Lambda（Reserved Concurrency=1）              │
    │    流程：取出單一 user_record → 呼叫 Bedrock → 寫入 DynamoDB   │
    │    Reserved Concurrency=1 保證序列執行，取代 time.sleep 方案   │
    └─────────────────────────────────────────────────────────────────┘

    降級模式（is_ai_generated=False）觸發條件：
      - ThrottlingException 超過 _THROTTLE_MAX_RETRIES（5）次
      - Bedrock AccessDeniedException / 其他 ClientError
    降級時自動切換 Rule-Based 輸出，DynamoDB 標記 is_ai_generated=False。
    """
    log.info(f"Lambda 觸發  RequestId={getattr(context, 'aws_request_id', 'local')}")
    log.debug(f"Event: {json.dumps(event, default=str)[:500]}")

    # ── 模式偵測：SQS Consumer 優先 ──────────────────────────────────────────
    if _is_sqs_diagnosis_job(event):
        log.info("[路由] SQS Consumer 模式（Reserved Concurrency=1）")
        return _handle_sqs_consumer(event, context)

    sources = _parse_s3_event(event)
    if not sources:
        log.warning("事件中無有效 S3 來源，略過。")
        return {"statusCode": 204, "body": {"message": "no s3 sources"}}

    total_processed = total_errors = total_written = 0
    source_files = []

    for bucket, key in sources:
        source_files.append(f"s3://{bucket}/{key}")
        log.info(f"處理檔案：s3://{bucket}/{key}")

        # ── 讀取預測結果 ──────────────────────────────────────────────────
        try:
            user_records = _load_predictions(bucket, key)
        except Exception as e:
            log.error(f"讀取 S3 檔案失敗: {e}")
            total_errors += 1
            continue

        if not user_records:
            log.info("無高風險用戶，略過本次診斷。")
            continue

        # ── 去重：同一 user_id 只保留最新一筆（節省 Bedrock token） ──────
        user_records = _dedup_records_by_user(user_records)

        # ── SQS Producer 模式：入隊後返回，由 Consumer Lambda 處理 ───────
        if SQS_QUEUE_URL:
            enqueued = _enqueue_to_sqs(user_records, key)
            log.info(
                f"[Producer] s3://{bucket}/{key} → SQS 入隊 {enqueued}/{len(user_records)} 筆。"
                f"Consumer Lambda（Reserved Concurrency=1）將序列處理。"
            )
            total_processed += enqueued
            total_written   += 0   # Consumer Lambda 負責寫入 DynamoDB
            continue   # 本 Lambda 在 Producer 模式下不直接處理

        # ── 直接處理模式（SQS_QUEUE_URL 未設定時的向後相容路徑） ─────────
        # 注意：此模式下多 Lambda 實例仍可能觸發 ThrottlingException。
        # 建議正式部署時設定 SQS_QUEUE_URL。

        # ── 按 ScoringTier 分組（決定並發參數） ──────────────────────────
        extreme_records  = [r for r in user_records if _classify(r["probability"])[0] == "EXTREME"]
        sonnet_records   = [r for r in user_records if _classify(r["probability"])[0] in ("HIGH", "BOUNDARY")]
        medium_records   = [r for r in user_records if _classify(r["probability"])[0] == "MEDIUM"]

        tier_counts = {
            "EXTREME":  len(extreme_records),
            "HIGH+BOUNDARY": len(sonnet_records),
            "MEDIUM":   len(medium_records),
        }
        log.info(f"分層統計：{tier_counts}")

        dynamo_items       = []
        errors             = 0
        degraded_count     = 0   # 降級（非 AI 生成）的診斷書數量

        def _safe_diagnose(record):
            try:
                return _diagnose_one(record, key), None
            except Exception as e:
                log.error(f"user={record['user_id']} 診斷失敗: {e}")
                return None, e

        with ThreadPoolExecutor(max_workers=MAX_WORKERS_HAIKU) as ex:
            futures = {ex.submit(_safe_diagnose, r): r for r in extreme_records}
            for future in as_completed(futures):
                item, err = future.result()
                if item:
                    dynamo_items.append(item)
                    if not item.get("is_ai_generated", True):
                        degraded_count += 1
                else:
                    errors += 1

        with ThreadPoolExecutor(max_workers=MAX_WORKERS_SONNET) as ex:
            futures = {ex.submit(_safe_diagnose, r): r for r in sonnet_records}
            for future in as_completed(futures):
                item, err = future.result()
                if item:
                    dynamo_items.append(item)
                    if not item.get("is_ai_generated", True):
                        degraded_count += 1
                else:
                    errors += 1

        # MEDIUM 層不需並發（Rule-Based，幾乎無 I/O）
        for record in medium_records:
            item, err = _safe_diagnose(record)
            if item:
                dynamo_items.append(item)
                # MEDIUM 層 Rule-Based 屬正常行為，不計入降級數
            else:
                errors += 1

        # ── 批次寫入 DynamoDB ─────────────────────────────────────────────
        written, write_err = _write_to_dynamo(dynamo_items)
        errors += write_err

        if degraded_count > 0:
            log.warning(
                f"[降級統計] 本批次共 {degraded_count}/{len(dynamo_items)} 筆降級為 Rule-Based "
                f"（is_ai_generated=False），請檢查 Bedrock 配額或啟用 SQS_QUEUE_URL。"
            )

        # ── CloudWatch 指標 ───────────────────────────────────────────────
        batch_tier_counts = {
            "EXTREME":  len(extreme_records),
            "BOUNDARY": sum(1 for r in user_records if _classify(r["probability"])[0] == "BOUNDARY"),
            "DEGRADED": degraded_count,
        }
        _put_metrics(
            total=len(user_records),
            processed=len(dynamo_items),
            errors=errors,
            tier_counts=batch_tier_counts,
            source_key=key,
        )

        # ── SNS 告警（高錯誤率） ──────────────────────────────────────────
        _alert_if_high_error_rate(len(user_records), errors, key)

        total_processed += len(dynamo_items)
        total_errors    += errors
        total_written   += written

        log.info(
            f"本批次完成：processed={len(dynamo_items)}"
            f"  degraded={degraded_count}  errors={errors}  dynamo_written={written}"
        )

    response_body = {
        "processed":      total_processed,
        "errors":         total_errors,
        "dynamo_written": total_written,
        "source_files":   source_files,
        "sqs_mode":       bool(SQS_QUEUE_URL),   # true=Producer模式，Consumer Lambda負責寫入
    }
    log.info(f"Lambda 完成：{response_body}")
    return {
        "statusCode": 200 if total_errors == 0 else 207,
        "body": response_body,
    }
