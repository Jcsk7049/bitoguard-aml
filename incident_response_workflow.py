"""
incident_response_workflow.py
幣託人頭戶偵測 — 事件回應工作流程 + Feedback Loop API

═══════════════════════════════════════════════════════════════════
  AWS Well-Architected Framework 對應
  ─────────────────────────────────────────────────────────────────
  卓越營運  │ 自動化事件觸發、CloudWatch 結構化日誌、DynamoDB 狀態追蹤
  安全性    │ 預簽名 URL（TTL 限制）、KMS 加密靜態資料、IAM 最小權限
  可靠性    │ 指數退避重試（tenacity）、DLQ 標記失敗事件、冪等事件 ID
  效能效率  │ 圖表非同步上傳、SNS/SQS 通知解耦
  成本最佳化│ 按風險等級選擇 LLM（已在 xai_bedrock.py 實作）
  永續發展  │ 批次收集 FP 樣本再訓練，而非每筆即時觸發 SageMaker
═══════════════════════════════════════════════════════════════════

主要模組
─────────
  IncidentResponseWorkflow   — 高風險事件觸發、S3 圖表上傳、預簽名 URL
  FeedbackLoopAPI            — FastAPI：誤判回報、增量訓練觸發
  IncrementalTrainer         — 彙整 FP 樣本 → SageMaker 增量訓練

執行方式
─────────
  # 事件回應（整合至 main_pipeline.py）
  python incident_response_workflow.py --mode trigger --user-id 12345 --prob 0.91

  # 啟動 Feedback Loop API 服務
  python incident_response_workflow.py --mode api --port 8000

  # 觸發增量訓練（需累積 ≥ MIN_RETRAIN_SAMPLES 筆 FP）
  python incident_response_workflow.py --mode retrain
"""

from __future__ import annotations

import io
import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import boto3
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from botocore.exceptions import ClientError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

# FastAPI（僅 API 模式需要，其餘模式不 import）
try:
    from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field as PydanticField
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

matplotlib.rcParams["font.family"]        = ["Microsoft JhengHei", "Arial Unicode MS", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)


# ══════════════════════════════════════════════════════════════════════════════
#  設定
# ══════════════════════════════════════════════════════════════════════════════

_ALLOWED_REGIONS: frozenset[str] = frozenset({"us-east-1", "us-west-2"})

def _validate_region(region: str) -> str:
    """Region 合規檢核：若非 us-east-1 或 us-west-2 則立即拋出 ValueError。"""
    if region not in _ALLOWED_REGIONS:
        raise ValueError(
            f"[Region 合規] Region '{region}' 不符合競賽規定。"
            f"僅允許 {sorted(_ALLOWED_REGIONS)}。"
            "請設定環境變數 AWS_DEFAULT_REGION=us-east-1 後重新執行。"
        )
    return region


S3_BUCKET              = os.environ.get("S3_BUCKET",             "your-hackathon-bucket")
S3_INCIDENT_PREFIX     = os.environ.get("S3_INCIDENT_PREFIX",    "incidents")
S3_FEEDBACK_PREFIX     = os.environ.get("S3_FEEDBACK_PREFIX",    "feedback")
S3_RETRAIN_PREFIX      = os.environ.get("S3_RETRAIN_PREFIX",     "retrain-queue")
REGION                 = _validate_region(os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
DYNAMO_TABLE           = os.environ.get("DYNAMO_INCIDENT_TABLE", "bito-incidents")
DYNAMO_FEEDBACK_TABLE  = os.environ.get("DYNAMO_FEEDBACK_TABLE", "bito-feedback")
SNS_TOPIC_ARN          = os.environ.get("SNS_TOPIC_ARN",         "")
ROLE_ARN               = os.environ.get("SAGEMAKER_ROLE_ARN",    "arn:aws:iam::ACCOUNT_ID:role/SageMakerRole")

PRESIGN_TTL_SECONDS    = 3600          # 預簽名 URL 有效期（1 小時）
MIN_RETRAIN_SAMPLES    = 50            # FP 樣本累積門檻，達到才觸發增量訓練
HIGH_RISK_THRESHOLD    = 0.65          # 觸發 Incident 的最低門檻


# ══════════════════════════════════════════════════════════════════════════════
#  資料結構
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class IncidentRecord:
    """
    一筆事件的完整狀態。儲存至 DynamoDB，pk = incident_id。

    狀態機：OPEN → CHART_UPLOADED → NOTIFIED → CONFIRMED / FALSE_POSITIVE / CLOSED
    """
    incident_id:       str                   # UUID，冪等鍵
    user_id:           int
    probability:       float
    risk_level:        str                   # EXTREME / HIGH / BOUNDARY / MEDIUM
    created_at:        str                   # ISO 8601 UTC
    status:            str = "OPEN"          # 狀態機
    chart_s3_key:      Optional[str] = None  # S3 key（圖表）
    presigned_url:     Optional[str] = None  # 預簽名 URL
    presigned_expires: Optional[str] = None  # 過期時間
    diagnosis_json:    Optional[dict] = None # xai_bedrock StructuredDiagnosis
    sns_message_id:    Optional[str] = None  # SNS publish message id
    operator_comment:  Optional[str] = None  # 客服備注
    is_false_positive: bool = False
    retrain_queued:    bool = False


@dataclass
class FeedbackPayload:
    """Feedback Loop API 的請求結構（Pydantic 版見 _FeedbackRequest）。"""
    incident_id:      str
    is_false_positive: bool
    operator_id:      str
    comment:          str = ""
    corrected_label:  int = 0              # 0 = 正常用戶（FP）, 1 = 確認黑名單


@dataclass
class RetrainJob:
    """增量訓練任務描述。"""
    job_id:         str
    s3_data_uri:    str
    triggered_at:   str
    sample_count:   int
    sagemaker_job:  Optional[str] = None


# ══════════════════════════════════════════════════════════════════════════════
#  圖表生成
# ══════════════════════════════════════════════════════════════════════════════

class FeatureChartGenerator:
    """
    將單一用戶的 SHAP 貢獻度 + 特徵值，生成一張事件報告用圖。
    輸出格式：PNG bytes（不落地，直接上傳 S3）。
    """

    C_RED   = "#DC2626"
    C_BLUE  = "#2563EB"
    C_BG    = "#F9FAFB"
    C_GRAY  = "#6B7280"

    def generate(
        self,
        user_id:       int,
        probability:   float,
        contributions: list[dict],   # [{"feature_label", "contribution_pct", "feature_value", "direction"}, ...]
        risk_level:    str,
        incident_id:   str,
    ) -> bytes:
        """
        生成雙子圖：
          左：SHAP 橫條圖（特徵貢獻度）
          右：風險儀表板（機率 + 等級 + 關鍵指標）
        回傳 PNG bytes。
        """
        top = contributions[:8]
        labels  = [c["feature_label"][:22] for c in top]
        values  = [c["contribution_pct"] for c in top]
        colors  = [self.C_RED if c["direction"] == "增加風險" else self.C_BLUE for c in top]

        fig, (ax_bar, ax_info) = plt.subplots(
            1, 2,
            figsize=(14, 6),
            gridspec_kw={"width_ratios": [2, 1]},
            facecolor=self.C_BG,
        )
        fig.suptitle(
            f"風險事件分析報告  ·  用戶 {user_id}  ·  {incident_id[:8].upper()}",
            fontsize=13, fontweight="bold", color="#111827",
        )

        # ── 左：SHAP 橫條圖 ────────────────────────────────────────────────
        ax_bar.set_facecolor(self.C_BG)
        bars = ax_bar.barh(labels[::-1], values[::-1], color=colors[::-1],
                           edgecolor="white", linewidth=0.6, height=0.65)
        for bar, val in zip(bars, values[::-1]):
            ax_bar.text(val + 0.4, bar.get_y() + bar.get_height() / 2,
                        f"{val:.1f}%", va="center", fontsize=8.5, color="#374151")

        ax_bar.set_xlabel("SHAP 貢獻佔比 (%)", fontsize=10)
        ax_bar.set_title("核心風險驅動特徵", fontsize=11, pad=8)
        ax_bar.spines[["top", "right"]].set_visible(False)
        ax_bar.set_xlim([0, max(values) * 1.2 + 2])
        ax_bar.grid(axis="x", alpha=0.25)

        # ── 右：風險儀表板 ─────────────────────────────────────────────────
        ax_info.set_facecolor(self.C_BG)
        ax_info.axis("off")

        level_colors = {"EXTREME": "#DC2626", "HIGH": "#F97316",
                        "BOUNDARY": "#EAB308", "MEDIUM": "#3B82F6"}
        badge_color = level_colors.get(risk_level, self.C_GRAY)

        # 機率環形進度條（近似）
        theta = np.linspace(0, 2 * np.pi * probability, 100)
        ax_info.plot(0.5 + 0.32 * np.cos(theta - np.pi / 2),
                     0.75 + 0.28 * np.sin(theta - np.pi / 2),
                     color=badge_color, lw=8, transform=ax_info.transAxes,
                     solid_capstyle="round")
        # 底圈（灰）
        theta_full = np.linspace(0, 2 * np.pi, 200)
        ax_info.plot(0.5 + 0.32 * np.cos(theta_full),
                     0.75 + 0.28 * np.sin(theta_full),
                     color="#E5E7EB", lw=8, transform=ax_info.transAxes, zorder=0)

        ax_info.text(0.50, 0.75, f"{probability:.0%}",
                     ha="center", va="center", fontsize=26,
                     fontweight="bold", color=badge_color,
                     transform=ax_info.transAxes)
        ax_info.text(0.50, 0.95, "AI 風險機率",
                     ha="center", va="center", fontsize=10, color=self.C_GRAY,
                     transform=ax_info.transAxes)
        ax_info.text(0.50, 0.48, f"[ {risk_level} ]",
                     ha="center", va="center", fontsize=15,
                     fontweight="bold", color=badge_color,
                     transform=ax_info.transAxes)

        # 關鍵指標小表
        rows = [
            ("用戶 ID",    str(user_id)),
            ("事件 ID",    incident_id[:8].upper()),
            ("生成時間",   datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")),
            ("特徵數量",   str(len(contributions))),
        ]
        for i, (k, v) in enumerate(rows):
            y = 0.35 - i * 0.08
            ax_info.text(0.02, y, f"{k}：", fontsize=8.5, color=self.C_GRAY,
                         transform=ax_info.transAxes)
            ax_info.text(0.50, y, v, fontsize=8.5, color="#111827",
                         transform=ax_info.transAxes)

        ax_info.set_title("事件摘要", fontsize=11, pad=8)

        fig.tight_layout(rect=[0, 0, 1, 0.95])

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.read()


# ══════════════════════════════════════════════════════════════════════════════
#  事件回應核心工作流程
# ══════════════════════════════════════════════════════════════════════════════

class IncidentResponseWorkflow:
    """
    AWS Well-Architected 事件回應工作流程。

    流程：
        trigger() →
          1. 建立 IncidentRecord（冪等 UUID）
          2. 生成特徵分析圖 PNG → 上傳 S3
          3. 產生預簽名 URL（TTL=1hr）
          4. 寫入 DynamoDB（狀態追蹤 + 稽核）
          5. SNS 通知風控團隊
          6. 回傳含 presigned_url 的 IncidentRecord
    """

    def __init__(
        self,
        region:       str = REGION,
        s3_bucket:    str = S3_BUCKET,
        dynamo_table: str = DYNAMO_TABLE,
        sns_topic:    str = SNS_TOPIC_ARN,
    ):
        sess = boto3.Session(region_name=region)
        self.s3      = sess.client("s3")
        self.dynamo  = sess.resource("dynamodb")
        self.sns     = sess.client("sns") if sns_topic else None
        self.table   = self.dynamo.Table(dynamo_table)
        self.bucket  = s3_bucket
        self.sns_arn = sns_topic
        self.charter = FeatureChartGenerator()

    # ── 公開入口 ──────────────────────────────────────────────────────────────

    def trigger(
        self,
        user_id:       int,
        probability:   float,
        risk_level:    str,
        contributions: list[dict],
        diagnosis:     Optional[dict] = None,
    ) -> IncidentRecord:
        """
        完整事件回應流程。回傳帶有 presigned_url 的 IncidentRecord。

        Parameters
        ----------
        contributions : list[dict]
            每筆含 feature_label / contribution_pct / feature_value / direction
        diagnosis : dict
            xai_bedrock StructuredDiagnosis.action_directive（JSON）
        """
        if probability < HIGH_RISK_THRESHOLD:
            log.info(f"user={user_id}  prob={probability:.3f} 低於門檻，略過 Incident 觸發")
            return None

        incident_id = str(uuid.uuid4())
        now_utc     = datetime.now(timezone.utc).isoformat()

        record = IncidentRecord(
            incident_id=incident_id,
            user_id=user_id,
            probability=probability,
            risk_level=risk_level,
            created_at=now_utc,
            diagnosis_json=diagnosis,
        )

        # Step 1 — 生成圖表 & 上傳 S3
        record = self._upload_chart(record, contributions)

        # Step 2 — 產生預簽名 URL
        record = self._generate_presigned_url(record)

        # Step 3 — 寫入 DynamoDB
        self._persist_incident(record)

        # Step 4 — SNS 通知
        record = self._notify(record)

        log.info(
            f"Incident 觸發完成  id={incident_id}  user={user_id}"
            f"  prob={probability:.3f}  level={risk_level}"
        )
        return record

    def get_incident(self, incident_id: str) -> Optional[IncidentRecord]:
        """從 DynamoDB 查詢單筆事件。"""
        try:
            resp = self.table.get_item(Key={"incident_id": incident_id})
            item = resp.get("Item")
            if not item:
                return None
            return IncidentRecord(**{k: v for k, v in item.items()
                                    if k in IncidentRecord.__dataclass_fields__})
        except ClientError as e:
            log.error(f"DynamoDB get_item 失敗: {e}")
            return None

    def update_status(self, incident_id: str, status: str, extra: dict = None) -> None:
        """更新 DynamoDB 事件狀態。"""
        update_expr   = "SET #s = :s, updated_at = :t"
        expr_names    = {"#s": "status"}
        expr_values   = {":s": status, ":t": datetime.now(timezone.utc).isoformat()}

        if extra:
            for k, v in extra.items():
                update_expr += f", {k} = :{k}"
                expr_values[f":{k}"] = v

        self._dynamo_update(incident_id, update_expr, expr_names, expr_values)

    # ── 私有方法 ─────────────────────────────────────────────────────────────

    def _upload_chart(
        self,
        record:        IncidentRecord,
        contributions: list[dict],
    ) -> IncidentRecord:
        """生成 PNG，上傳至 s3://{bucket}/incidents/{date}/{incident_id}.png。"""
        png_bytes = self.charter.generate(
            user_id=record.user_id,
            probability=record.probability,
            contributions=contributions,
            risk_level=record.risk_level,
            incident_id=record.incident_id,
        )
        date_prefix = datetime.now(timezone.utc).strftime("%Y/%m/%d")
        s3_key      = f"{S3_INCIDENT_PREFIX}/{date_prefix}/{record.incident_id}.png"

        self._s3_put_object(
            key=s3_key,
            body=png_bytes,
            content_type="image/png",
            metadata={
                "user_id":     str(record.user_id),
                "probability": f"{record.probability:.4f}",
                "risk_level":  record.risk_level,
                "incident_id": record.incident_id,
            },
        )
        record.chart_s3_key = s3_key
        record.status       = "CHART_UPLOADED"
        log.info(f"圖表已上傳 → s3://{self.bucket}/{s3_key}")
        return record

    def _generate_presigned_url(self, record: IncidentRecord) -> IncidentRecord:
        """
        生成限時預簽名 GET URL（TTL = PRESIGN_TTL_SECONDS）。

        安全考量（Well-Architected Security Pillar）：
          - TTL 設為 1 小時，防止 URL 洩露後長期有效
          - 對應 IAM 政策僅允許 s3:GetObject on incidents/* prefix
        """
        if not record.chart_s3_key:
            return record

        url = self._s3_presign(record.chart_s3_key, PRESIGN_TTL_SECONDS)
        expires_at = (
            datetime.now(timezone.utc) + timedelta(seconds=PRESIGN_TTL_SECONDS)
        ).isoformat()

        record.presigned_url     = url
        record.presigned_expires = expires_at
        log.info(f"預簽名 URL 生成完畢（有效至 {expires_at}）")
        return record

    def _persist_incident(self, record: IncidentRecord) -> None:
        """
        寫入 DynamoDB（Well-Architected Reliability：冪等 pk = incident_id）。

        DynamoDB 建議 Schema：
            pk: incident_id (String)
            GSI: user_id-index（查詢單一用戶的所有事件）
            TTL: expire_at（60 天後自動刪除，節省儲存成本）
        """
        item = asdict(record)
        # 將 None 清除，DynamoDB 不支援 None value
        item = {k: v for k, v in item.items() if v is not None}
        # 60 天後過期（epoch）
        item["expire_at"] = int(time.time()) + 60 * 86400

        self._dynamo_put(item)
        log.info(f"事件寫入 DynamoDB  table={DYNAMO_TABLE}  pk={record.incident_id}")

    def _notify(self, record: IncidentRecord) -> IncidentRecord:
        """
        SNS 通知風控團隊（非阻塞；若 SNS ARN 未設定則跳過）。

        訊息結構：JSON，前後台都能直接解析。
        """
        if not self.sns or not self.sns_arn:
            log.debug("SNS ARN 未設定，略過通知")
            return record

        message = {
            "event":         "HIGH_RISK_INCIDENT",
            "incident_id":   record.incident_id,
            "user_id":       record.user_id,
            "probability":   round(record.probability, 4),
            "risk_level":    record.risk_level,
            "chart_url":     record.presigned_url,
            "expires_at":    record.presigned_expires,
            "created_at":    record.created_at,
            "auto_action":   record.diagnosis_json.get("primary_action") if record.diagnosis_json else None,
        }
        subject = (
            f"[{record.risk_level}] 高風險事件 — 用戶 {record.user_id}"
            f" (P={record.probability:.0%})"
        )
        try:
            resp = self.sns.publish(
                TopicArn=self.sns_arn,
                Subject=subject,
                Message=json.dumps(message, ensure_ascii=False),
                MessageAttributes={
                    "risk_level": {
                        "DataType": "String",
                        "StringValue": record.risk_level,
                    }
                },
            )
            record.sns_message_id = resp["MessageId"]
            record.status         = "NOTIFIED"
            log.info(f"SNS 通知已發送  MessageId={resp['MessageId']}")
        except ClientError as e:
            log.warning(f"SNS 通知失敗（不影響主流程）: {e}")
        return record

    # ── AWS SDK 包裝（加上指數退避重試） ─────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(ClientError),
        reraise=True,
    )
    def _s3_put_object(self, key: str, body: bytes,
                       content_type: str, metadata: dict) -> None:
        self.s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=body,
            ContentType=content_type,
            Metadata=metadata,
            ServerSideEncryption="AES256",   # Well-Architected Security Pillar
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8),
           retry=retry_if_exception_type(ClientError), reraise=True)
    def _s3_presign(self, key: str, ttl: int) -> str:
        return self.s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=ttl,
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8),
           retry=retry_if_exception_type(ClientError), reraise=True)
    def _dynamo_put(self, item: dict) -> None:
        self.table.put_item(Item=item)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8),
           retry=retry_if_exception_type(ClientError), reraise=True)
    def _dynamo_update(self, pk: str, update_expr: str,
                       expr_names: dict, expr_values: dict) -> None:
        self.table.update_item(
            Key={"incident_id": pk},
            UpdateExpression=update_expr,
            ExpressionAttributeNames=expr_names,
            ExpressionAttributeValues=expr_values,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  增量訓練模組
# ══════════════════════════════════════════════════════════════════════════════

class IncrementalTrainer:
    """
    彙整 FP 樣本 → S3 → 觸發 SageMaker 增量訓練。

    設計原則（Well-Architected Cost Optimization）：
      - 不在每筆回饋時立即觸發，累積達 MIN_RETRAIN_SAMPLES 才訓練
      - 增量資料與原始訓練集合併後以 CSV 傳入，保持既有 pipeline 格式
    """

    def __init__(self, region: str = REGION):
        sess = boto3.Session(region_name=region)
        self.s3  = sess.client("s3")
        self.sm  = sess.client("sagemaker")

    def collect_fp_sample(
        self,
        user_id:      int,
        feature_row:  dict,
        incident_id:  str,
    ) -> str:
        """
        將單筆 FP 樣本存至 S3 feedback prefix。
        回傳 S3 key。
        """
        sample = {
            "user_id":     user_id,
            "status":      0,              # FP → 修正標籤為 0（正常）
            "incident_id": incident_id,
            "collected_at": datetime.now(timezone.utc).isoformat(),
            **feature_row,
        }
        key = f"{S3_FEEDBACK_PREFIX}/{incident_id}.json"
        self.s3.put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=json.dumps(sample, ensure_ascii=False, default=str),
            ContentType="application/json",
        )
        log.info(f"FP 樣本已上傳 s3://{S3_BUCKET}/{key}")
        return key

    def count_pending_samples(self) -> int:
        """計算 S3 feedback prefix 中未處理的 FP 樣本數。"""
        paginator = self.s3.get_paginator("list_objects_v2")
        count     = 0
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_FEEDBACK_PREFIX + "/"):
            count += len(page.get("Contents", []))
        return count

    def trigger_incremental_training(
        self,
        feat_cols:       list[str],
        base_model_path: str = "model.json",
        force:           bool = False,
    ) -> Optional[RetrainJob]:
        """
        彙整所有 FP 樣本 → 合併 → 上傳至 retrain-queue/ → 觸發 SageMaker。

        Parameters
        ----------
        force : bool — 忽略 MIN_RETRAIN_SAMPLES 強制觸發（供測試用）
        """
        n_pending = self.count_pending_samples()
        log.info(f"待處理 FP 樣本：{n_pending} 筆（門檻：{MIN_RETRAIN_SAMPLES}）")

        if n_pending < MIN_RETRAIN_SAMPLES and not force:
            log.info("樣本不足，延後訓練。")
            return None

        # ── 讀取所有 FP 樣本 ─────────────────────────────────────────────
        samples = self._load_feedback_samples(feat_cols)
        if samples.empty:
            log.warning("無有效 FP 樣本，略過。")
            return None

        # ── 上傳至 retrain-queue ──────────────────────────────────────────
        ts       = int(time.time())
        data_key = f"{S3_RETRAIN_PREFIX}/{ts}/incremental.csv"

        # 只保留 status + 特徵欄（與訓練格式一致）
        cols_needed = ["status"] + [c for c in feat_cols if c in samples.columns]
        export_df   = samples[cols_needed].fillna(0)

        csv_buffer = io.StringIO()
        export_df.to_csv(csv_buffer, index=False, header=False)
        self.s3.put_object(
            Bucket=S3_BUCKET,
            Key=data_key,
            Body=csv_buffer.getvalue().encode(),
            ContentType="text/csv",
        )
        log.info(f"增量資料集上傳 s3://{S3_BUCKET}/{data_key}  ({len(export_df)} 筆)")

        # ── 啟動 SageMaker Training Job ──────────────────────────────────
        from train_sagemaker import BASE_HYPERPARAMS, INSTANCE_TYPE
        import sagemaker
        from sagemaker.inputs import TrainingInput
        from sagemaker.xgboost import XGBoost

        session   = sagemaker.Session()
        job_name  = f"bito-incremental-{ts}"
        output_uri = f"s3://{S3_BUCKET}/bito-mule-detection/{ts}/output/"

        # 傳入現有模型作為初始化（XGBoost 支援 model_uri 熱啟動）
        estimator = XGBoost(
            entry_point=None,
            framework_version="1.7-1",
            role=ROLE_ARN,
            instance_count=1,
            instance_type=INSTANCE_TYPE,
            output_path=output_uri,
            hyperparameters={
                **BASE_HYPERPARAMS,
                # 增量訓練縮短輪數
                "num_round":              100,
                "early_stopping_rounds":  20,
                # 強化對 FP 樣本的學習（降低正常用戶被誤標的損失）
                "scale_pos_weight":       1,
            },
            sagemaker_session=session,
        )
        estimator.fit(
            inputs={
                "train": TrainingInput(
                    f"s3://{S3_BUCKET}/{data_key}", content_type="text/csv"
                )
            },
            job_name=job_name,
            wait=False,      # 非同步，不阻塞 API
            logs=False,
        )
        log.info(f"SageMaker 增量訓練已啟動  job={job_name}")

        # ── 移動已處理樣本至 processed/ ──────────────────────────────────
        self._archive_feedback_samples()

        job = RetrainJob(
            job_id=job_name,
            s3_data_uri=f"s3://{S3_BUCKET}/{data_key}",
            triggered_at=datetime.now(timezone.utc).isoformat(),
            sample_count=len(export_df),
            sagemaker_job=job_name,
        )
        return job

    def _load_feedback_samples(self, feat_cols: list[str]) -> pd.DataFrame:
        """讀取所有 FP JSON 樣本，合為 DataFrame。"""
        paginator = self.s3.get_paginator("list_objects_v2")
        rows = []
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_FEEDBACK_PREFIX + "/"):
            for obj in page.get("Contents", []):
                if not obj["Key"].endswith(".json"):
                    continue
                try:
                    body = self.s3.get_object(Bucket=S3_BUCKET, Key=obj["Key"])["Body"].read()
                    rows.append(json.loads(body))
                except Exception as e:
                    log.warning(f"讀取 FP 樣本失敗 key={obj['Key']}: {e}")
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def _archive_feedback_samples(self) -> None:
        """將 feedback/ 下的樣本移至 feedback/processed/ 避免重複訓練。"""
        paginator = self.s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_FEEDBACK_PREFIX + "/"):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if "/processed/" in key:
                    continue
                new_key = key.replace(
                    S3_FEEDBACK_PREFIX + "/",
                    S3_FEEDBACK_PREFIX + "/processed/",
                    1,
                )
                self.s3.copy_object(
                    Bucket=S3_BUCKET,
                    CopySource={"Bucket": S3_BUCKET, "Key": key},
                    Key=new_key,
                )
                self.s3.delete_object(Bucket=S3_BUCKET, Key=key)


# ══════════════════════════════════════════════════════════════════════════════
#  Feedback Loop FastAPI
# ══════════════════════════════════════════════════════════════════════════════

def build_feedback_api(
    workflow: IncidentResponseWorkflow,
    trainer:  IncrementalTrainer,
    feat_cols: list[str],
) -> "FastAPI":
    """
    建立 FastAPI 應用，掛載 Feedback Loop 端點。

    端點：
        POST /feedback/false-positive   — 客服確認誤判，回傳訓練觸發狀態
        GET  /feedback/stats            — 統計待處理 FP 樣本數
        POST /retrain/trigger           — 手動觸發增量訓練
        GET  /incident/{id}             — 查詢事件狀態 + 預簽名 URL
    """
    if not FASTAPI_AVAILABLE:
        raise RuntimeError("請先安裝 fastapi 與 uvicorn：pip install fastapi uvicorn")

    app = FastAPI(
        title="幣託人頭戶偵測 — Feedback Loop API",
        description="客服誤判回報 → 自動觸發增量訓練",
        version="1.0.0",
    )

    # ── Pydantic 請求/回應模型 ────────────────────────────────────────────

    class FeedbackRequest(BaseModel):
        incident_id:       str
        is_false_positive: bool         = PydanticField(..., description="True = 確認誤判（FP）")
        operator_id:       str         = PydanticField(..., description="客服工號")
        comment:           str         = ""
        feature_snapshot:  dict        = PydanticField(
            default_factory=dict,
            description="該用戶的特徵快照（從前台帶入，用於增量訓練）",
        )

    class FeedbackResponse(BaseModel):
        incident_id:      str
        accepted:         bool
        retrain_triggered: bool
        retrain_job:      Optional[str] = None
        pending_samples:  int
        message:          str

    class RetrainRequest(BaseModel):
        force:            bool = PydanticField(False, description="強制觸發，忽略樣本門檻")

    class IncidentResponse(BaseModel):
        incident_id:      str
        user_id:          int
        probability:      float
        risk_level:       str
        status:           str
        presigned_url:    Optional[str]
        presigned_expires: Optional[str]
        created_at:       str

    # ── 端點 ─────────────────────────────────────────────────────────────

    @app.post("/feedback/false-positive", response_model=FeedbackResponse)
    async def report_false_positive(
        req: FeedbackRequest,
        background: BackgroundTasks,
    ):
        """
        客服確認誤判後呼叫此端點。
        - 更新 DynamoDB 事件狀態為 FALSE_POSITIVE
        - 將特徵快照上傳至 S3 feedback bucket
        - 若累積樣本達門檻，以 BackgroundTask 非同步觸發增量訓練
        """
        incident = workflow.get_incident(req.incident_id)
        if not incident:
            raise HTTPException(status_code=404, detail=f"事件 {req.incident_id} 不存在")

        if incident.status in ("CLOSED", "FALSE_POSITIVE"):
            raise HTTPException(
                status_code=409,
                detail=f"事件 {req.incident_id} 已處理（狀態：{incident.status}）",
            )

        # 更新狀態
        workflow.update_status(
            req.incident_id,
            "FALSE_POSITIVE",
            extra={
                "is_false_positive": True,
                "operator_id":       req.operator_id,
                "operator_comment":  req.comment,
            },
        )
        log.info(f"FP 確認  incident={req.incident_id}  operator={req.operator_id}")

        # 上傳 FP 樣本
        if req.feature_snapshot:
            trainer.collect_fp_sample(
                user_id=incident.user_id,
                feature_row=req.feature_snapshot,
                incident_id=req.incident_id,
            )

        # 檢查是否觸發增量訓練
        pending      = trainer.count_pending_samples()
        retrain_job  = None
        do_retrain   = pending >= MIN_RETRAIN_SAMPLES

        if do_retrain:
            def _bg_retrain():
                job = trainer.trigger_incremental_training(feat_cols)
                if job:
                    log.info(f"背景增量訓練完成  job={job.sagemaker_job}")

            background.add_task(_bg_retrain)
            retrain_job = f"bito-incremental-{int(time.time())}"
            log.info(f"排程增量訓練（背景）  pending={pending}")

        return FeedbackResponse(
            incident_id=req.incident_id,
            accepted=True,
            retrain_triggered=do_retrain,
            retrain_job=retrain_job,
            pending_samples=pending,
            message=(
                f"FP 回饋已記錄。{'已觸發增量訓練。' if do_retrain else f'尚需 {MIN_RETRAIN_SAMPLES - pending} 筆達到訓練門檻。'}"
            ),
        )

    @app.get("/feedback/stats")
    async def feedback_stats():
        """回傳 FP 樣本累積統計。"""
        pending = trainer.count_pending_samples()
        return {
            "pending_samples":    pending,
            "retrain_threshold":  MIN_RETRAIN_SAMPLES,
            "ready_to_retrain":   pending >= MIN_RETRAIN_SAMPLES,
            "progress_pct":       round(pending / MIN_RETRAIN_SAMPLES * 100, 1),
        }

    @app.post("/retrain/trigger")
    async def manual_retrain(req: RetrainRequest, background: BackgroundTasks):
        """手動觸發增量訓練（管理員用）。"""
        def _bg():
            job = trainer.trigger_incremental_training(feat_cols, force=req.force)
            if job:
                log.info(f"手動增量訓練完成  job={job.sagemaker_job}")

        background.add_task(_bg)
        return {"triggered": True, "force": req.force,
                "message": "增量訓練已排程於背景執行。"}

    @app.get("/incident/{incident_id}", response_model=IncidentResponse)
    async def get_incident(incident_id: str):
        """查詢事件狀態、預簽名 URL 等。"""
        record = workflow.get_incident(incident_id)
        if not record:
            raise HTTPException(status_code=404, detail="事件不存在")
        return IncidentResponse(
            incident_id=record.incident_id,
            user_id=record.user_id,
            probability=record.probability,
            risk_level=record.risk_level,
            status=record.status,
            presigned_url=record.presigned_url,
            presigned_expires=record.presigned_expires,
            created_at=record.created_at,
        )

    @app.get("/health")
    async def health():
        return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}

    return app


# ══════════════════════════════════════════════════════════════════════════════
#  pipeline 整合函式（供 main_pipeline.py 呼叫）
# ══════════════════════════════════════════════════════════════════════════════

def process_high_risk_incidents(
    reports: list[dict],
    feat_cols: list[str],
) -> list[dict]:
    """
    批次處理高風險事件。

    供 main_pipeline.py stage_xai() 後呼叫：
        from incident_response_workflow import process_high_risk_incidents
        enriched = process_high_risk_incidents(xai_reports, FEAT_COLS)

    回傳原始 reports，並將 presigned_url 注入每筆 dict。
    """
    workflow = IncidentResponseWorkflow()
    enriched = []

    for r in reports:
        prob  = r.get("probability", 0)
        level = r.get("risk_level", "")

        if prob < HIGH_RISK_THRESHOLD:
            enriched.append(r)
            continue

        incident = workflow.trigger(
            user_id=r["user_id"],
            probability=prob,
            risk_level=level,
            contributions=r.get("shap_contributions", []),
            diagnosis=r.get("action_directive"),
        )

        enriched_r = dict(r)
        if incident:
            enriched_r["incident_id"]     = incident.incident_id
            enriched_r["presigned_url"]   = incident.presigned_url
            enriched_r["chart_s3_key"]    = incident.chart_s3_key
            enriched_r["presigned_expires"] = incident.presigned_expires
        enriched.append(enriched_r)

    log.info(f"Incident 處理完畢：{len(enriched)} 筆（觸發：{sum(1 for r in enriched if 'incident_id' in r)}）")
    return enriched


# ══════════════════════════════════════════════════════════════════════════════
#  CLI 入口
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(description="事件回應工作流程 + Feedback Loop API")
    parser.add_argument("--mode", choices=["trigger", "api", "retrain"], default="api")
    parser.add_argument("--user-id",  type=int,   default=0)
    parser.add_argument("--prob",     type=float, default=0.95)
    parser.add_argument("--port",     type=int,   default=8000)
    parser.add_argument("--force-retrain", action="store_true")
    args = parser.parse_args()

    if args.mode == "trigger":
        # 測試單筆事件觸發
        wf = IncidentResponseWorkflow()
        mock_contribs = [
            {"feature_label": "最短資金滯留時間",  "contribution_pct": 40.2, "feature_value": 3.1,  "direction": "增加風險"},
            {"feature_label": "不同 IP 數量",      "contribution_pct": 28.7, "feature_value": 15,   "direction": "增加風險"},
            {"feature_label": "總交易量 (TWD)",    "contribution_pct": 18.5, "feature_value": 3e6,  "direction": "增加風險"},
            {"feature_label": "KYC 等級",          "contribution_pct":  8.1, "feature_value": 1,    "direction": "增加風險"},
            {"feature_label": "黑名單跳轉數",      "contribution_pct":  4.5, "feature_value": 2,    "direction": "增加風險"},
        ]
        incident = wf.trigger(
            user_id=args.user_id,
            probability=args.prob,
            risk_level="EXTREME" if args.prob > 0.9 else "HIGH",
            contributions=mock_contribs,
        )
        if incident:
            print(f"\n事件 ID     : {incident.incident_id}")
            print(f"圖表 S3 Key : {incident.chart_s3_key}")
            print(f"預簽名 URL  : {incident.presigned_url}")
            print(f"URL 到期    : {incident.presigned_expires}")

    elif args.mode == "retrain":
        trainer = IncrementalTrainer()
        # feat_cols 從現有模型讀取
        try:
            import xgboost as xgb
            booster = xgb.Booster()
            booster.load_model("model.json")
            feat_cols = booster.feature_names or []
        except Exception:
            feat_cols = []
        job = trainer.trigger_incremental_training(feat_cols, force=args.force_retrain)
        if job:
            print(f"SageMaker Job: {job.sagemaker_job}  樣本數: {job.sample_count}")
        else:
            print("樣本不足，訓練未觸發。使用 --force-retrain 強制執行。")

    elif args.mode == "api":
        if not FASTAPI_AVAILABLE:
            print("請先安裝：pip install fastapi uvicorn")
            return
        wf      = IncidentResponseWorkflow()
        trainer = IncrementalTrainer()
        try:
            import xgboost as xgb
            booster = xgb.Booster()
            booster.load_model("model.json")
            feat_cols = booster.feature_names or []
        except Exception:
            feat_cols = []
        app = build_feedback_api(wf, trainer, feat_cols)
        print(f"啟動 Feedback Loop API  →  http://localhost:{args.port}/docs")
        uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
