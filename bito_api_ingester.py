"""
bito_api_ingester.py
幣託 BitoPro API → PyArrow Parquet → S3（Hive Partition）資料擷取器

設計要點
--------
  分頁策略  : 追蹤 response["next"] 游標欄位，為空字串／null 時停止。
               與 ingest_to_s3.py 的 offset/limit 策略獨立，適用於
               cursor-based pagination API（如 AWS Event API 新版端點）。

  精度策略  : twd_transfer.ori_samount 使用 Decimal("1e-8") 縮放，
               避免 IEEE 754 浮點誤差（如 100000000 → 1.00000000）。

  Schema    : 每張表有明確的 PyArrow Schema（衍生自 GLUE_COLUMN_DEFS），
               user_id 欄位強制 int64，nullable=False。

  Parquet   : Snappy 壓縮；超過 ROWS_PER_PART 列時自動切分為多個 part 檔。

  S3 路徑   : s3://{bucket}/{prefix}/{table}/dt=YYYY-MM-DD/part-{n:05d}.parquet
               符合 Hive Partition 格式，Athena / Glue Crawler 可直接識別。

使用範例
--------
  from bito_api_ingester import BitoApiIngester

  ingester = BitoApiIngester(
      bucket="your-hackathon-bucket",
      prefix="raw",
      region="us-east-1",   # 僅允許 us-east-1 / us-west-2
  )

  # 擷取單表
  result = ingester.ingest_table("twd_transfer")
  print(result)                  # IngestionResult(rows=150000, parts=1, ...)

  # 擷取全部表
  summary = ingester.ingest_all()
  print(summary.success_count)   # 7
"""

from __future__ import annotations

import io
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Iterator

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
import tenacity
from botocore.exceptions import ClientError

log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
#  Region 合規檢核（競賽規定：僅允許 us-east-1 / us-west-2）
# ══════════════════════════════════════════════════════════════════════════════

_ALLOWED_REGIONS: frozenset[str] = frozenset({"us-east-1", "us-west-2"})


def _validate_region(region: str) -> str:
    """
    Region 合規閘門：若 region 不在核准清單中，立即拋出 RuntimeError 並終止程序。

    所有 boto3 client / Session 初始化前都必須通過此函式。
    核准清單：us-east-1（預設）、us-west-2。
    任何其他 Region（包含 ap-northeast-1 等）一律拒絕。
    """
    if region not in _ALLOWED_REGIONS:
        raise RuntimeError(
            f"[Region 合規] Region '{region}' 不在核准清單中。\n"
            f"  允許值：{sorted(_ALLOWED_REGIONS)}\n"
            "  請設定環境變數 AWS_DEFAULT_REGION=us-east-1 後重新執行。"
        )
    return region


# ══════════════════════════════════════════════════════════════════════════════
#  Schema 定義（對應 ingest_to_s3.py GLUE_COLUMN_DEFS，PyArrow 型別版本）
# ══════════════════════════════════════════════════════════════════════════════

# Glue 型別 → PyArrow 型別
_GLUE_TO_PA: dict[str, pa.DataType] = {
    "bigint":    pa.int64(),
    "int":       pa.int32(),
    "double":    pa.float64(),
    "string":    pa.large_utf8(),
    "boolean":   pa.bool_(),
    "timestamp": pa.timestamp("us", tz="UTC"),
    "date":      pa.date32(),
}

# 每張表的 PyArrow Schema（user_id 固定 int64, nullable=False）
TABLE_SCHEMAS: dict[str, pa.Schema] = {
    "user_info": pa.schema([
        pa.field("user_id",            pa.int64(),            nullable=False),
        pa.field("status",             pa.int32()),
        pa.field("sex",                pa.large_utf8()),
        pa.field("birthday",           pa.date32()),
        pa.field("career",             pa.large_utf8()),
        pa.field("income_source",      pa.large_utf8()),
        pa.field("confirmed_at",       pa.timestamp("us", tz="UTC")),
        pa.field("level1_finished_at", pa.timestamp("us", tz="UTC")),
        pa.field("level2_finished_at", pa.timestamp("us", tz="UTC")),
        pa.field("user_source",        pa.large_utf8()),
        pa.field("kyc_level",          pa.int32()),
    ]),

    "twd_transfer": pa.schema([
        pa.field("created_at",  pa.timestamp("us", tz="UTC")),
        pa.field("user_id",     pa.int64(),   nullable=False),  # int64, non-null
        pa.field("kind",        pa.int32()),                    # 0=入金 1=出金
        pa.field("ori_samount", pa.float64()),                  # ×1e-8 縮放後，單位 TWD
        pa.field("source_ip",   pa.large_utf8()),
        pa.field("bank_code",   pa.large_utf8()),
        pa.field("status",      pa.int32()),
    ]),

    "crypto_transfer": pa.schema([
        pa.field("created_at",       pa.timestamp("us", tz="UTC")),
        pa.field("user_id",          pa.int64(),   nullable=False),
        pa.field("kind",             pa.int32()),
        pa.field("sub_kind",         pa.int32()),
        pa.field("ori_samount",      pa.float64()),              # ×1e-8 縮放
        pa.field("twd_srate",        pa.float64()),              # ×1e-8 縮放
        pa.field("currency",         pa.large_utf8()),
        pa.field("protocol",         pa.large_utf8()),
        pa.field("from_wallet",      pa.large_utf8()),
        pa.field("to_wallet",        pa.large_utf8()),
        pa.field("relation_user_id", pa.int64()),
        pa.field("source_ip",        pa.large_utf8()),
        pa.field("tx_hash",          pa.large_utf8()),
        pa.field("status",           pa.int32()),
    ]),

    "usdt_twd_trading": pa.schema([
        pa.field("updated_at",    pa.timestamp("us", tz="UTC")),
        pa.field("user_id",       pa.int64(),   nullable=False),
        pa.field("is_buy",        pa.bool_()),
        pa.field("trade_samount", pa.float64()),                 # ×1e-8 縮放
        pa.field("twd_srate",     pa.float64()),                 # ×1e-8 縮放
        pa.field("is_market",     pa.bool_()),
        pa.field("source",        pa.large_utf8()),
        pa.field("source_ip",     pa.large_utf8()),
    ]),

    "usdt_swap": pa.schema([
        pa.field("created_at",       pa.timestamp("us", tz="UTC")),
        pa.field("user_id",          pa.int64(),   nullable=False),
        pa.field("kind",             pa.int32()),
        pa.field("twd_samount",      pa.float64()),              # ×1e-8 縮放
        pa.field("currency_samount", pa.float64()),              # ×1e-8 縮放
        pa.field("currency",         pa.large_utf8()),
    ]),

    "train_label": pa.schema([
        pa.field("user_id", pa.int64(), nullable=False),
        pa.field("status",  pa.int32()),
    ]),

    "predict_label": pa.schema([
        pa.field("user_id", pa.int64(), nullable=False),
    ]),
}

# 需要 ×1e-8 縮放的欄位（Decimal 精度計算）
SCALE_FIELDS: dict[str, list[str]] = {
    "twd_transfer":     ["ori_samount"],
    "crypto_transfer":  ["ori_samount", "twd_srate"],
    "usdt_twd_trading": ["trade_samount", "twd_srate"],
    "usdt_swap":        ["twd_samount", "currency_samount"],
}

# ── 429 Rate-Limit 退避參數 ──────────────────────────────────────────────────
_RATE_LIMIT_BASE_WAIT = 30.0   # 429 指數退避基數（秒）：30 → 60 → 120 → ...
_RATE_LIMIT_MAX_WAIT  = 300.0  # 429 單次等待上限（秒，5 分鐘）


def _is_retryable(exc: Exception) -> bool:
    """
    tenacity retry 判定函式。

    重試條件：
    ─ HTTPError with 5xx / 429   → 重試
    ─ ConnectionError / Timeout  → 重試
    ─ ValueError（JSON 解析失敗） → 重試

    不重試條件：
    ─ HTTPError with 403 / 404   → 直接拋出
    """
    if isinstance(exc, requests.HTTPError):
        resp = exc.response
        if resp is not None and resp.status_code in (403, 404):
            return False   # 認證失敗或路徑不存在：無意義重試
        return True        # 5xx / 429 及其他 HTTP 錯誤
    return isinstance(exc, (requests.ConnectionError, requests.Timeout, ValueError))


# API endpoint 路徑（相對於 API_BASE）
TABLE_ENDPOINTS: dict[str, str] = {
    "user_info":        "/user_info",
    "twd_transfer":     "/twd_transfer",
    "crypto_transfer":  "/crypto_transfer",
    "usdt_twd_trading": "/usdt_twd_trading",
    "usdt_swap":        "/usdt_swap",
    "train_label":      "/train_label",
    "predict_label":    "/predict_label",
}

# ══════════════════════════════════════════════════════════════════════════════
#  回傳物件
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TableResult:
    """單張表的擷取結果。"""
    table:      str
    dt:         date
    rows:       int
    parts:      int
    s3_paths:   list[str] = field(default_factory=list)
    error:      str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def __str__(self) -> str:
        if self.ok:
            return (
                f"TableResult({self.table}  rows={self.rows:,}  parts={self.parts}"
                f"  dt={self.dt}  ok)"
            )
        return f"TableResult({self.table}  ERROR: {self.error})"


@dataclass
class IngestionSummary:
    """全部表的擷取匯總。"""
    results:       list[TableResult]
    started_at:    datetime
    finished_at:   datetime | None = None

    @property
    def success_count(self) -> int:
        return sum(1 for r in self.results if r.ok)

    @property
    def error_count(self) -> int:
        return sum(1 for r in self.results if not r.ok)

    @property
    def total_rows(self) -> int:
        return sum(r.rows for r in self.results)

    @property
    def failed_tables(self) -> list[str]:
        return [r.table for r in self.results if not r.ok]

    def __str__(self) -> str:
        elapsed = ""
        if self.finished_at:
            elapsed = f"  elapsed={round((self.finished_at - self.started_at).total_seconds())}s"
        return (
            f"IngestionSummary(success={self.success_count}  error={self.error_count}"
            f"  total_rows={self.total_rows:,}{elapsed})"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  主類別
# ══════════════════════════════════════════════════════════════════════════════

class BitoApiIngester:
    """
    BitoPro API → PyArrow Parquet → S3（Hive Partition）一站式擷取器。

    Parameters
    ----------
    bucket              : S3 目標桶名
    prefix              : S3 路徑前綴（不含尾斜線）
    api_base            : API 根網址
    region              : AWS Region
    rows_per_part       : 每個 Parquet part 檔的最大列數（超過後切分新檔）
    page_timeout        : 單次 API 請求逾時秒數
    max_retries         : HTTP 5xx / 網路錯誤最大重試次數
    retry_backoff       : 5xx 退避基數（秒），實際等待 = backoff × 2^(retry-1)
    tables              : 要擷取的表名列表（None = 全部）
    checkpoint_path     : 進度檢查點本地 JSON 路徑（預設 bito_checkpoint.json）
    checkpoint_s3_key   : （選填）同時將檢查點備份至 S3 的物件鍵值，
                          例如 "checkpoints/bito_ingest.json"；
                          留空代表不備份至 S3。
    """

    API_BASE      = os.environ.get("BITO_API_BASE", "https://aws-event-api.bitopro.com")
    ROWS_PER_PART = 500_000   # 每個 part 最多 50 萬列（約 50–200MB Parquet）

    def __init__(
        self,
        bucket:              str,
        prefix:              str        = "raw",
        api_base:            str        = "",
        region:              str        = "",
        rows_per_part:       int        = ROWS_PER_PART,
        page_timeout:        int        = 30,
        max_retries:         int        = 3,
        retry_backoff:       float      = 2.0,
        tables:              list[str] | None = None,
        checkpoint_path:     str        = "bito_checkpoint.json",
        checkpoint_s3_key:   str | None = None,
    ) -> None:
        # Region 解析優先順序：
        #   1. 明確傳入的 region 參數（非空字串）
        #   2. 環境變數 AWS_DEFAULT_REGION
        #   3. 硬性預設值 us-east-1
        resolved_region = region or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        # ▶ Region 合規閘門：非核准區域立即 RuntimeError 終止
        self._region             = _validate_region(resolved_region)

        self._bucket             = bucket
        self._prefix             = prefix.rstrip("/")
        self._api_base           = (api_base or self.API_BASE).rstrip("/")
        self._rows_per_part      = rows_per_part
        self._timeout            = page_timeout
        self._max_retries        = max_retries
        self._retry_backoff      = retry_backoff
        self._tables             = tables or list(TABLE_ENDPOINTS.keys())
        self._checkpoint_path    = checkpoint_path
        self._checkpoint_s3_key  = checkpoint_s3_key

        # requests.Session：共用 TCP 連線、header
        self._session = requests.Session()
        self._session.headers.update({
            "Accept":     "application/json",
            "User-Agent": "BitoApiIngester/1.0",
        })

        # boto3 client（S3 上傳）— 使用已通過 _validate_region() 核准的 self._region
        self._s3 = boto3.client("s3", region_name=self._region)

        log.info(
            f"BitoApiIngester 初始化  bucket={bucket}  prefix={prefix}"
            f"  api={self._api_base}  rows_per_part={rows_per_part:,}"
            f"  checkpoint={checkpoint_path}"
        )

    # ──────────────────────────────────────────────────────────────────────
    #  公開介面
    # ──────────────────────────────────────────────────────────────────────

    def ingest_table(self, table_name: str, dt: date | None = None) -> TableResult:
        """
        擷取單張表，回傳 TableResult。

        流程
        ─────
        1. 讀取檢查點：若本地 JSON 存有本表且日期吻合，從上次中斷的游標 + part 繼續
        2. 逐頁拉取 API（next cursor 分頁，支援游標恢復）
        3. 每頁轉為 PyArrow RecordBatch（含 1e-8 縮放 + Schema 強制轉型）
        4. 以 ParquetWriter 串流寫入 BytesIO buffer
        5. 達到 rows_per_part 時：flush → S3 上傳 → 存儲檢查點（原子操作）
        6. 完成後清除本表檢查點

        檢查點時機
        ──────────
        僅在 Part 成功上傳至 S3 後才儲存進度，確保狀態一致性：
          - Part N 已上傳 → 檢查點 part_num = N+1，cursor = 第 N+1 part 起始游標
          - 崩潰發生在兩次 Part 上傳之間 → 重啟後最多重新抓取 rows_per_part 筆資料
        """
        if table_name not in TABLE_ENDPOINTS:
            raise ValueError(f"未知的資料表：{table_name}。可用：{list(TABLE_ENDPOINTS)}")

        dt     = dt or date.today()
        schema = TABLE_SCHEMAS.get(table_name) or self._infer_schema(table_name)

        # ── 讀取並套用檢查點 ─────────────────────────────────────────────
        ckpt       = self._load_checkpoint().get(table_name, {})
        resuming   = bool(ckpt.get("cursor")) and ckpt.get("dt") == str(dt)
        resume_cursor = ckpt["cursor"]     if resuming else None
        part_num      = ckpt["part_num"]   if resuming else 0
        total_rows    = ckpt["total_rows"] if resuming else 0

        if resuming:
            log.info(
                f"[{table_name}] ✓ 從檢查點恢復  dt={dt}"
                f"  part_num={part_num}  total_rows={total_rows:,}"
                f"  cursor={str(resume_cursor)[:30]}..."
            )
        else:
            log.info(f"[{table_name}] 開始擷取（全新）  dt={dt}")

        s3_paths: list[str] = []
        rows_in_buf = 0
        buf:    io.BytesIO | None      = None
        writer: pq.ParquetWriter | None = None
        # 追蹤最新游標：_paginate yield 時更新，Part flush 後存入檢查點
        latest_cursor: str | None = resume_cursor

        try:
            for page_records, next_cursor in self._paginate(
                table_name, resume_cursor=resume_cursor
            ):
                # 每頁更新「下一頁游標」，Part flush 時用此值存入檢查點
                latest_cursor = next_cursor

                if not page_records:
                    continue

                batch      = self._to_record_batch(page_records, schema, table_name)
                batch_rows = batch.num_rows

                if writer is None:
                    buf    = io.BytesIO()
                    writer = pq.ParquetWriter(buf, schema, compression="snappy")

                writer.write_batch(batch)
                rows_in_buf += batch_rows
                total_rows  += batch_rows

                log.debug(f"[{table_name}] +{batch_rows} 列  累計 {total_rows:,}")

                # ── Part 切分：flush → S3 → 存儲檢查點 ───────────────────
                if rows_in_buf >= self._rows_per_part:
                    path = self._flush_part(writer, buf, table_name, dt, part_num)
                    s3_paths.append(path)
                    log.info(
                        f"[{table_name}] Part {part_num} 上傳完成"
                        f"  rows={rows_in_buf:,}  → {path}"
                    )
                    part_num   += 1
                    rows_in_buf = 0
                    writer = None
                    buf    = None

                    # 存儲檢查點（Part 已安全落地，cursor 指向下一頁）
                    self._save_table_checkpoint(
                        table_name, str(dt),
                        cursor=latest_cursor,
                        part_num=part_num,
                        total_rows=total_rows,
                    )

            # ── 最後一個 Part（剩餘不足 rows_per_part 的尾部資料）────────
            if writer is not None and rows_in_buf > 0:
                path = self._flush_part(writer, buf, table_name, dt, part_num)
                s3_paths.append(path)
                log.info(
                    f"[{table_name}] Part {part_num} 上傳完成（尾部）"
                    f"  rows={rows_in_buf:,}  → {path}"
                )
                part_num += 1

        except Exception as exc:
            # 擷取失敗時保留檢查點（下次可從上一個 Part 邊界繼續）
            log.error(f"[{table_name}] 擷取失敗：{exc}", exc_info=True)
            return TableResult(
                table=table_name, dt=dt, rows=total_rows,
                parts=part_num, s3_paths=s3_paths, error=str(exc),
            )

        # ── 成功：清除檢查點 ──────────────────────────────────────────────
        self._clear_table_checkpoint(table_name)

        if total_rows == 0:
            log.warning(f"[{table_name}] 無資料，跳過上傳。")

        log.info(
            f"[{table_name}] 完成  rows={total_rows:,}  parts={part_num}"
            f"  dt={dt}  paths={s3_paths}"
        )
        return TableResult(
            table=table_name, dt=dt, rows=total_rows,
            parts=part_num, s3_paths=s3_paths,
        )

    def ingest_all(self, dt: date | None = None) -> IngestionSummary:
        """
        擷取 self._tables 中的所有資料表，失敗的表不影響其他表繼續執行。
        """
        dt      = dt or date.today()
        started = datetime.now(timezone.utc)
        results: list[TableResult] = []

        for table_name in self._tables:
            result = self.ingest_table(table_name, dt=dt)
            results.append(result)
            if not result.ok:
                log.warning(f"[{table_name}] 失敗：{result.error}")

        summary = IngestionSummary(
            results=results,
            started_at=started,
            finished_at=datetime.now(timezone.utc),
        )
        log.info(str(summary))
        return summary

    # ──────────────────────────────────────────────────────────────────────
    #  進度檢查點（Checkpointing）
    # ──────────────────────────────────────────────────────────────────────

    def _load_checkpoint(self) -> dict:
        """
        從本地 JSON 檔案讀取全域檢查點。
        若本地檔案不存在，嘗試從 S3 下載（容器重啟後恢復）。
        回傳空 dict 代表無任何已存儲的進度。
        """
        # 1. 嘗試本地檔案
        if os.path.exists(self._checkpoint_path):
            try:
                with open(self._checkpoint_path, encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                log.warning(f"[Checkpoint] 本地檢查點讀取失敗（{exc}），忽略並從頭開始。")
                return {}

        # 2. 嘗試 S3 備份（適用容器重啟場景）
        if self._checkpoint_s3_key:
            try:
                resp = self._s3.get_object(Bucket=self._bucket, Key=self._checkpoint_s3_key)
                data = json.loads(resp["Body"].read().decode("utf-8"))
                # 同步回本地，方便後續讀寫
                with open(self._checkpoint_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                log.info(f"[Checkpoint] 從 S3 恢復檢查點  key={self._checkpoint_s3_key}")
                return data
            except ClientError as exc:
                if exc.response["Error"]["Code"] != "NoSuchKey":
                    log.warning(f"[Checkpoint] S3 讀取失敗：{exc}")
        return {}

    def _save_checkpoint(self, data: dict) -> None:
        """
        將全域檢查點寫入本地 JSON，並可選同步至 S3。

        寫入策略：先寫臨時檔再原子替換，避免寫入中途崩潰導致格式損壞。
        """
        tmp_path = self._checkpoint_path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, default=str, indent=2)
            os.replace(tmp_path, self._checkpoint_path)
        except OSError as exc:
            log.warning(f"[Checkpoint] 本地寫入失敗：{exc}")
            return

        # 可選備份至 S3
        if self._checkpoint_s3_key:
            try:
                self._s3.put_object(
                    Bucket=self._bucket,
                    Key=self._checkpoint_s3_key,
                    Body=json.dumps(data, ensure_ascii=False, default=str, indent=2).encode(),
                    ContentType="application/json",
                    ServerSideEncryption="AES256",
                )
            except ClientError as exc:
                log.warning(f"[Checkpoint] S3 備份失敗（本地已存）：{exc}")

    def _save_table_checkpoint(
        self,
        table_name: str,
        dt:         str,
        cursor:     str | None,
        part_num:   int,
        total_rows: int,
    ) -> None:
        """
        在 Part 成功上傳後，更新單張表的檢查點記錄。

        檢查點欄位說明：
        ─ dt         : 分區日期（字串），用來識別跨日重啟時是否應廢棄舊檢查點
        ─ cursor     : 下一頁的 next_cursor，None 表示已到最後一頁
        ─ part_num   : 下一個 part 編號（已上傳的 part 數量）
        ─ total_rows : 截至本次 flush 的累計行數
        ─ saved_at   : 儲存時間（UTC ISO 格式），便於排查
        """
        data = self._load_checkpoint()
        data[table_name] = {
            "dt":         dt,
            "cursor":     cursor,
            "part_num":   part_num,
            "total_rows": total_rows,
            "saved_at":   datetime.now(timezone.utc).isoformat(),
        }
        self._save_checkpoint(data)
        log.debug(
            f"[Checkpoint] {table_name} 已存  part={part_num}"
            f"  rows={total_rows:,}  cursor={str(cursor)[:20] if cursor else 'None'}"
        )

    def _clear_table_checkpoint(self, table_name: str) -> None:
        """表成功完成後移除其檢查點記錄，避免下次誤用舊游標。"""
        data = self._load_checkpoint()
        if table_name in data:
            del data[table_name]
            self._save_checkpoint(data)
            log.info(f"[Checkpoint] {table_name} 已清除（擷取完成）")

    # ──────────────────────────────────────────────────────────────────────
    #  next-Cursor 分頁器
    # ──────────────────────────────────────────────────────────────────────

    def _paginate(
        self,
        table_name:    str,
        resume_cursor: str | None = None,
    ) -> Iterator[tuple[list[dict], str | None]]:
        """
        追蹤 response["next"] 游標，逐頁 yield (page_records, next_cursor)。

        yields
        ------
        tuple  (records: list[dict], next_cursor: str | None)
               next_cursor 是下一頁的游標；None 表示這是最後一頁。
               呼叫方（ingest_table）持有 next_cursor 以決定是否存儲檢查點。

        參數
        ----
        resume_cursor : 若非 None，從此游標繼續（忽略第一頁，直接從游標位置抓取）。

        分頁終止條件（任一滿足）：
          ① next_cursor 為 None / 空字串 / 不存在
          ② data 為空列表
          ③ 已達 MAX_PAGES（防止無限迴圈）

        API 回應格式（相容多種）：
          {"data": [...], "next": "cursor_token"}
          {"data": [...], "nextCursor": "cursor_token"}
          [...]   （直接列表，無分頁）
        """
        MAX_PAGES = 2_000
        endpoint  = TABLE_ENDPOINTS[table_name]
        url       = f"{self._api_base}{endpoint}"
        params: dict[str, Any] = {"next": resume_cursor} if resume_cursor else {}
        page = 0

        if resume_cursor:
            log.info(
                f"[{table_name}] 從游標恢復分頁"
                f"  cursor={resume_cursor[:30]}{'...' if len(resume_cursor) > 30 else ''}"
            )

        while page < MAX_PAGES:
            body = self._fetch_page(url, params)

            # 相容多種 API 回應格式
            if isinstance(body, list):
                data        = body
                next_cursor = None
            elif isinstance(body, dict):
                data        = body.get("data", [])
                next_cursor = (
                    body.get("next") or body.get("nextCursor") or body.get("cursor") or None
                )
            else:
                log.warning(f"[{table_name}] 不預期的回應型別：{type(body)}，停止分頁。")
                break

            if not data:
                log.info(f"[{table_name}] 第 {page + 1} 頁空資料，停止分頁。")
                break

            log.info(
                f"[{table_name}] 第 {page + 1} 頁  {len(data)} 筆"
                f"  next={'有' if next_cursor else '無（最後一頁）'}"
            )
            yield data, next_cursor

            if not next_cursor:
                break

            params = {"next": next_cursor}
            page  += 1

        else:
            log.warning(f"[{table_name}] 已達最大頁數 {MAX_PAGES}，強制停止。")

    def _fetch_page(self, url: str, params: dict) -> Any:
        """
        以 tenacity 實作動態退避重試的單頁 GET 請求。

        退避策略（兩段式）
        ──────────────────
        HTTP 429 Too Many Requests（Rate Limit）：
          1. 優先讀取 Retry-After 回應 header（秒數或 HTTP 日期格式）
          2. header 缺失時使用指數退避，基數 = _RATE_LIMIT_BASE_WAIT（30 s）
             等待序列：30 → 60 → 120 → … 上限 300 s（5 分鐘）
          目的：BitoPro API 有 IP 頻率限制，等待時間必須夠長才能避免封鎖。

        HTTP 5xx / 網路錯誤（ConnectionError、Timeout）：
          使用指數退避，基數 = self._retry_backoff（預設 2 s）
          等待序列：2 → 4 → 8 → … 上限 60 s

        不重試（立即拋出）：
          HTTP 403 Forbidden / 404 Not Found（重試無意義）

        使用 tenacity.Retrying context manager 而非 decorator，
        以便在 instance method 中存取 self 的退避參數。
        """
        backoff = self._retry_backoff

        def _adaptive_wait(retry_state: tenacity.RetryCallState) -> float:
            exc = retry_state.outcome.exception()
            n   = retry_state.attempt_number

            # ── 429 Rate Limit ────────────────────────────────────────────
            if (
                isinstance(exc, requests.HTTPError)
                and exc.response is not None
                and exc.response.status_code == 429
            ):
                # 1. Retry-After header（整數秒）
                hdr = (exc.response.headers.get("Retry-After") or "").strip()
                if hdr.isdigit():
                    wait_secs = min(float(hdr), _RATE_LIMIT_MAX_WAIT)
                    log.warning(
                        f"[Rate Limit] HTTP 429  Retry-After header = {wait_secs:.0f}s"
                        f"  (第 {n} 次重試)"
                    )
                    return wait_secs
                # 2. 無 header → 指數退避（base 30 s）
                wait_secs = min(_RATE_LIMIT_BASE_WAIT * (2 ** (n - 1)), _RATE_LIMIT_MAX_WAIT)
                log.warning(
                    f"[Rate Limit] HTTP 429  指數退避 {wait_secs:.0f}s"
                    f"  (第 {n} 次重試，base={_RATE_LIMIT_BASE_WAIT}s)"
                )
                return wait_secs

            # ── 5xx / 網路錯誤：短指數退避 ───────────────────────────────
            wait_secs = min(backoff * (2 ** (n - 1)), 60.0)
            return wait_secs

        for attempt in tenacity.Retrying(
            retry=tenacity.retry_if_exception(_is_retryable),
            wait=_adaptive_wait,
            stop=tenacity.stop_after_attempt(self._max_retries),
            before_sleep=tenacity.before_sleep_log(log, logging.WARNING),
            reraise=True,
        ):
            with attempt:
                resp = self._session.get(url, params=params, timeout=self._timeout)
                if resp.status_code in (403, 404):
                    resp.raise_for_status()   # 不重試，直接拋出
                resp.raise_for_status()
                return resp.json()

    # ──────────────────────────────────────────────────────────────────────
    #  1e-8 精度縮放
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _scale_1e8(raw_value: Any) -> float | None:
        """
        以 Decimal 進行 1e-8 縮放，避免 IEEE 754 浮點誤差。

        輸入可為：
          - int / str (原始 API 整數，如 100000000 → 1.0)
          - float（已是小數，直接轉 Decimal 再縮放）
          - None / NaN → 回傳 None

        Decimal 運算流程：
          Decimal(str(100000000)) * Decimal("1e-8") = Decimal("1.00000000")
          → float(Decimal("1.00000000")) = 1.0
        """
        if raw_value is None:
            return None
        # 處理 pandas / numpy NaN
        try:
            if isinstance(raw_value, float) and (raw_value != raw_value):   # NaN check
                return None
        except Exception:
            pass

        try:
            return float(Decimal(str(raw_value)) * Decimal("1e-8"))
        except (InvalidOperation, ValueError, TypeError):
            log.debug(f"1e-8 縮放失敗，原始值={raw_value!r}，回傳 None")
            return None

    def _apply_scale_fields(self, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        """對 DataFrame 中的縮放欄位套用 1e-8 Decimal 轉換。"""
        for col in SCALE_FIELDS.get(table_name, []):
            if col in df.columns:
                df[col] = df[col].apply(self._scale_1e8)
        return df

    # ──────────────────────────────────────────────────────────────────────
    #  Schema 轉型：list[dict] → pa.RecordBatch
    # ──────────────────────────────────────────────────────────────────────

    def _to_record_batch(
        self,
        records:    list[dict],
        schema:     pa.Schema,
        table_name: str,
    ) -> pa.RecordBatch:
        """
        將 API 原始 list[dict] 轉換為符合 Schema 的 PyArrow RecordBatch。

        轉換策略：
          1. 建立 pandas DataFrame（利用 pandas 靈活的型別推斷）
          2. 套用 1e-8 縮放（Decimal 精度）
          3. 逐欄依 PyArrow 型別強制轉型
          4. 缺少的欄位補 null；多餘的欄位捨棄
        """
        df = pd.DataFrame(records)

        # Step 1：1e-8 縮放（必須在型別轉換前執行，否則 float 精度已損失）
        df = self._apply_scale_fields(df, table_name)

        arrays: list[pa.Array] = []

        for schema_field in schema:
            col_name = schema_field.name
            pa_type  = schema_field.type

            # 欄位不在 API 回應中 → 補全 null
            if col_name not in df.columns:
                arrays.append(pa.nulls(len(df), type=pa_type))
                continue

            series = df[col_name]
            arrays.append(self._cast_series(series, pa_type, col_name))

        return pa.record_batch(arrays, schema=schema)

    def _cast_series(
        self,
        series:   pd.Series,
        pa_type:  pa.DataType,
        col_name: str,
    ) -> pa.Array:
        """將 pandas Series 強制轉型為指定 PyArrow 型別。"""

        # ── 整數（user_id 必為 int64，不允許 null） ──────────────────────
        if pa.types.is_integer(pa_type):
            numeric = pd.to_numeric(series, errors="coerce")
            # 使用 nullable Int64 中間型別，再轉成 pa.int64
            arr = pa.array(
                numeric.where(numeric.notna(), other=None).tolist(),
                type=pa_type,
            )
            return arr

        # ── 浮點數（amount 欄位，已由 _apply_scale_fields 縮放） ─────────
        if pa.types.is_floating(pa_type):
            numeric = pd.to_numeric(series, errors="coerce")
            return pa.array(
                numeric.where(numeric.notna(), other=None).tolist(),
                type=pa_type,
            )

        # ── 時間戳（timestamp[us, UTC]）─────────────────────────────────
        if pa.types.is_timestamp(pa_type):
            ts = pd.to_datetime(series, utc=True, errors="coerce")
            # pa.array 能直接處理 pd.NaT → null
            return pa.array(ts.tolist(), type=pa_type)

        # ── 日期（date32）────────────────────────────────────────────────
        if pa.types.is_date(pa_type):
            dt_series = pd.to_datetime(series, errors="coerce").dt.date
            return pa.array(
                [v if not pd.isna(v) else None for v in dt_series],
                type=pa_type,
            )

        # ── 布林（"true"/"false" 字串或 bool/int）─────────────────────────
        if pa.types.is_boolean(pa_type):
            def _to_bool(v: Any) -> bool | None:
                if v is None:
                    return None
                if isinstance(v, bool):
                    return v
                if isinstance(v, (int, float)):
                    return bool(v) if v == v else None  # NaN guard
                if isinstance(v, str):
                    return v.strip().lower() in ("true", "1", "yes")
                return None

            return pa.array([_to_bool(v) for v in series.tolist()], type=pa_type)

        # ── 字串（large_utf8）─────────────────────────────────────────────
        def _to_str(v: Any) -> str | None:
            if v is None:
                return None
            if isinstance(v, float) and v != v:  # NaN
                return None
            return str(v)

        return pa.array([_to_str(v) for v in series.tolist()], type=pa_type)

    # ──────────────────────────────────────────────────────────────────────
    #  Parquet 寫出 + S3 上傳
    # ──────────────────────────────────────────────────────────────────────

    def _flush_part(
        self,
        writer:     pq.ParquetWriter,
        buf:        io.BytesIO,
        table_name: str,
        dt:         date,
        part_num:   int,
    ) -> str:
        """
        關閉 ParquetWriter，將 buffer 上傳至 S3，或回退至本地 CSV。

        S3 路徑格式（Hive Partition）：
          s3://{bucket}/{prefix}/{table}/dt=YYYY-MM-DD/part-{n:05d}.parquet

        本地回退路徑：
          ./data/{table}/dt={dt}/part-{n:05d}.csv
        """
        writer.close()
        buf.seek(0)

        s3_key = (
            f"{self._prefix}/{table_name}"
            f"/dt={dt.isoformat()}"
            f"/part-{part_num:05d}.parquet"
        )
        size_bytes = buf.getbuffer().nbytes

        log.debug(f"S3 上傳  key={s3_key}  size={size_bytes/1024/1024:.2f} MB")

        # 嘗試 S3 上傳
        try:
            self._s3.put_object(
                Bucket=self._bucket,
                Key=s3_key,
                Body=buf.getvalue(),
                ContentType="application/octet-stream",
                ServerSideEncryption="AES256",
                Metadata={
                    "source-table": table_name,
                    "partition-dt": dt.isoformat(),
                    "ingested-at":  datetime.now(timezone.utc).isoformat(),
                },
            )
            return f"s3://{self._bucket}/{s3_key}"
        except (ClientError, Exception) as exc:
            # S3 上傳失敗，回退至本地 CSV
            log.warning(
                f"S3 上傳失敗 ({type(exc).__name__})，回退至本地 CSV：{exc}"
            )
            
            # 建立本地目錄結構
            local_dir = f"./data/{table_name}/dt={dt.isoformat()}"
            os.makedirs(local_dir, exist_ok=True)
            
            # 將 Parquet buffer 轉換為 CSV
            buf.seek(0)
            table = pq.read_table(buf)
            df = table.to_pandas()
            
            local_path = f"{local_dir}/part-{part_num:05d}.csv"
            df.to_csv(local_path, index=False)
            
            log.info(
                f"本地 CSV 儲存完成  path={local_path}  rows={len(df):,}"
                f"  size={os.path.getsize(local_path)/1024/1024:.2f} MB"
            )
            
            return f"file://{os.path.abspath(local_path)}"

    # ──────────────────────────────────────────────────────────────────────
    #  未定義 Schema 的表：動態推斷
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _infer_schema(table_name: str) -> pa.Schema:
        """
        對不在 TABLE_SCHEMAS 中的表，動態推斷 Schema（只保證 user_id 為 int64）。
        實際生產環境建議在 TABLE_SCHEMAS 中明確定義。
        """
        log.warning(
            f"[{table_name}] 未在 TABLE_SCHEMAS 中定義，將動態推斷 Schema。"
            "建議在 TABLE_SCHEMAS 中明確定義以確保型別正確性。"
        )
        # 最小 Schema：僅保證 user_id
        return pa.schema([
            pa.field("user_id", pa.int64(), nullable=False),
        ])


# ══════════════════════════════════════════════════════════════════════════════
#  CLI 快速執行入口
# ══════════════════════════════════════════════════════════════════════════════

def _build_parser():
    import argparse
    p = argparse.ArgumentParser(
        description="BitoApiIngester — BitoPro API → S3 Parquet（Hive Partition）",
    )
    p.add_argument("--bucket",    required=True, help="S3 桶名")
    p.add_argument("--prefix",    default="raw", help="S3 前綴（預設：raw）")
    p.add_argument("--region",
                   default=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
                   choices=["us-east-1", "us-west-2"],
                   help="AWS Region（競賽規定僅允許 us-east-1 / us-west-2；"
                        "預設讀取 AWS_DEFAULT_REGION 環境變數，否則為 us-east-1）")
    p.add_argument("--api-base",  default="", help="API 根網址（留空使用環境變數）")
    p.add_argument("--tables",    nargs="*",  help="指定表名（留空=全部）")
    p.add_argument("--dt",        default="", help="分區日期 YYYY-MM-DD（預設今日）")
    p.add_argument("--rows-per-part",       type=int, default=BitoApiIngester.ROWS_PER_PART)
    p.add_argument("--max-retries",         type=int, default=3,
                   help="5xx / 網路錯誤最大重試次數（預設 3）")
    p.add_argument("--retry-backoff",       type=float, default=2.0,
                   help="5xx 退避基數秒數（預設 2.0）；429 固定使用 30s 基數")
    p.add_argument("--checkpoint",          default="bito_checkpoint.json",
                   help="進度檢查點本地 JSON 路徑（預設 bito_checkpoint.json）")
    p.add_argument("--checkpoint-s3-key",   default="",
                   help="（選填）檢查點備份至 S3 的物件鍵值，例如 checkpoints/bito_ingest.json")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main() -> None:
    import sys
    parser = _build_parser()
    args   = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        stream=sys.stdout,
    )

    dt = date.fromisoformat(args.dt) if args.dt else date.today()

    ingester = BitoApiIngester(
        bucket=args.bucket,
        prefix=args.prefix,
        api_base=args.api_base,
        region=args.region,
        rows_per_part=args.rows_per_part,
        max_retries=args.max_retries,
        retry_backoff=args.retry_backoff,
        tables=args.tables or None,
        checkpoint_path=args.checkpoint,
        checkpoint_s3_key=args.checkpoint_s3_key or None,
    )

    summary = ingester.ingest_all(dt=dt)
    print("\n" + "═" * 60)
    print(summary)
    for r in summary.results:
        status = "✓" if r.ok else "✗"
        print(f"  {status}  {r}")
    print("═" * 60)

    if summary.error_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
