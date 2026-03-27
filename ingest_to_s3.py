"""
ingest_to_s3.py
幣託人頭戶偵測 — 原始資料注入流程

═══════════════════════════════════════════════════════════════════════════
  流程概覽
  ───────────────────────────────────────────────────────────────────────
  1. DataIngester      — 從 BitoPro API / 本地 CSV 取得五張原始資料表
                         上傳至 S3（Parquet + CSV 雙格式），分區儲存
  2. GlueCatalogSetup  — boto3 建立 Glue Database、Crawler、執行爬取
                         完成後即可在 SageMaker / Athena 直接 SELECT

  S3 分區結構（Hive 格式，Athena / SageMaker 相容）：
    s3://{bucket}/raw/{table}/year={Y}/month={M}/day={D}/{table}.parquet
    s3://{bucket}/raw/{table}/year={Y}/month={M}/day={D}/{table}.csv

  Glue Data Catalog：
    Database : bito_hackathon
    Tables   : user_info / twd_transfer / crypto_transfer /
               usdt_twd_trading / usdt_swap / train_label / predict_label
═══════════════════════════════════════════════════════════════════════════

使用方式：
  # 完整流程（抓 API → 上傳 S3 → 建立 Glue Catalog）
  python ingest_to_s3.py

  # 僅上傳（使用本地 CSV 目錄）
  python ingest_to_s3.py --csv-dir ./data --skip-glue

  # 僅重建 Glue Catalog（資料已在 S3）
  python ingest_to_s3.py --only-glue

  # 查看 Glue Crawler 狀態
  python ingest_to_s3.py --crawler-status
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
import time
import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Optional

import boto3
import pandas as pd
import requests
from botocore.exceptions import ClientError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("ingest.log", encoding="utf-8"),
    ],
)


# ══════════════════════════════════════════════════════════════════════════════
#  設定
# ══════════════════════════════════════════════════════════════════════════════

S3_BUCKET        = os.environ.get("S3_BUCKET",            "your-hackathon-bucket")
S3_RAW_PREFIX    = os.environ.get("S3_RAW_PREFIX",        "raw")
# V-2 修正：預設 Region 改為競賽允許的 us-east-1（或 us-west-2）
# 競賽禁令：資源必須部署在 us-east-1 或 us-west-2
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


REGION           = _validate_region(os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
GLUE_DB          = os.environ.get("GLUE_DATABASE",        "bito_hackathon")
GLUE_CRAWLER     = os.environ.get("GLUE_CRAWLER_NAME",    "bito-raw-crawler")
# W-1 修正：必須透過環境變數注入真實 ARN，不得使用佔位符
# 正確設定：export GLUE_ROLE_ARN=arn:aws:iam::123456789012:role/GlueServiceRole
GLUE_ROLE_ARN    = os.environ.get("GLUE_ROLE_ARN", "")
if not GLUE_ROLE_ARN or "ACCOUNT_ID" in GLUE_ROLE_ARN:
    # 僅在使用 Glue 時強制失敗，其餘模式（--skip-glue）不受影響
    _GLUE_ROLE_ARN_INVALID = True
    log.warning(
        "GLUE_ROLE_ARN 未設定或仍含佔位符 'ACCOUNT_ID'。"
        " Glue Crawler 功能將無法使用，請設定環境變數："
        " export GLUE_ROLE_ARN=arn:aws:iam::<account-id>:role/GlueServiceRole"
    )
else:
    _GLUE_ROLE_ARN_INVALID = False

# ── API 設定 ──────────────────────────────────────────────────────────────────
API_BASE_URL  = os.environ.get("BITO_API_BASE", "https://aws-event-api.bitopro.com")
PAGE_LIMIT    = int(os.environ.get("API_PAGE_LIMIT", "1000"))  # 每頁筆數
PAGE_TIMEOUT  = int(os.environ.get("API_TIMEOUT",    "30"))    # 單次請求逾時（秒）
MAX_PAGES     = int(os.environ.get("API_MAX_PAGES",  "500"))   # 安全上限（500×1000=50萬筆）

# 不需分頁的標籤表（資料量小，一次拿完）
NO_PAGINATION_TABLES = frozenset({"train_label", "predict_label"})

# 需要 ×1e-8 精度轉換的金額欄位（依 aws-event 資料表及欄位說明.pdf）
AMOUNT_FIELDS: dict[str, list[str]] = {
    "twd_transfer":     ["ori_samount"],
    "crypto_transfer":  ["ori_samount", "twd_srate"],
    "usdt_twd_trading": ["trade_samount", "twd_srate"],
    "usdt_swap":        ["twd_samount", "currency_samount"],
}

# 上傳時同時保留 CSV（供人工稽核）與 Parquet（供 SageMaker/Athena 高效讀取）
UPLOAD_FORMATS   = ("parquet", "csv")

# 所有需要上傳的資料表（對應 BitoDataManager._load_raw 的 table 名稱）
RAW_TABLES: list[str] = [
    "user_info",
    "twd_transfer",
    "crypto_transfer",
    "usdt_twd_trading",
    "usdt_swap",
    "train_label",
    "predict_label",
]

# ── Glue 欄位型別對照（依 aws-event 資料表及欄位說明.pdf） ─────────────────────
# 格式：{table: [(column_name, glue_type, comment), ...]}
GLUE_COLUMN_DEFS: dict[str, list[tuple[str, str, str]]] = {
    "user_info": [
        ("user_id",              "bigint",  "用戶唯一識別碼"),
        ("status",               "int",     "帳戶狀態"),
        ("sex",                  "string",  "性別"),
        ("birthday",             "date",    "出生日期"),
        ("career",               "string",  "職業"),
        ("income_source",        "string",  "資金來源"),
        ("confirmed_at",         "timestamp","身份驗證完成時間"),
        ("level1_finished_at",   "timestamp","KYC Level1 完成時間"),
        ("level2_finished_at",   "timestamp","KYC Level2 完成時間"),
        ("user_source",          "string",  "用戶來源渠道"),
        ("kyc_level",            "int",     "KYC 等級 0/1/2"),
    ],
    "twd_transfer": [
        ("created_at",   "timestamp", "交易建立時間"),
        ("user_id",      "bigint",    "用戶唯一識別碼"),
        ("kind",         "int",       "0=入金 1=出金"),
        ("ori_samount",  "double",    "原始金額（已 ×1e-8，單位 TWD）"),
        ("source_ip",    "string",    "操作 IP"),
        ("bank_code",    "string",    "銀行代碼"),
        ("status",       "int",       "交易狀態"),
    ],
    "crypto_transfer": [
        ("created_at",       "timestamp", "交易建立時間"),
        ("user_id",          "bigint",    "用戶唯一識別碼"),
        ("kind",             "int",       "0=入金 1=出金"),
        ("sub_kind",         "int",       "子類型"),
        ("ori_samount",      "double",    "虛幣數量（已 ×1e-8）"),
        ("twd_srate",        "double",    "對台幣匯率（已 ×1e-8）"),
        ("currency",         "string",    "虛幣幣種"),
        ("protocol",         "string",    "區塊鏈協議"),
        ("from_wallet",      "string",    "來源錢包地址"),
        ("to_wallet",        "string",    "目標錢包地址"),
        ("relation_user_id", "bigint",    "關聯用戶 ID（站內轉帳）"),
        ("source_ip",        "string",    "操作 IP"),
        ("tx_hash",          "string",    "區塊鏈 Transaction Hash"),
        ("status",           "int",       "交易狀態"),
    ],
    "usdt_twd_trading": [
        ("updated_at",     "timestamp", "訂單更新時間"),
        ("user_id",        "bigint",    "用戶唯一識別碼"),
        ("is_buy",         "boolean",   "True=買幣 False=賣幣"),
        ("trade_samount",  "double",    "成交幣量（已 ×1e-8）"),
        ("twd_srate",      "double",    "成交匯率（已 ×1e-8）"),
        ("is_market",      "boolean",   "True=市價單"),
        ("source",         "string",    "下單來源（App/Web/API）"),
        ("source_ip",      "string",    "操作 IP"),
    ],
    "usdt_swap": [
        ("created_at",       "timestamp", "交易建立時間"),
        ("user_id",          "bigint",    "用戶唯一識別碼"),
        ("kind",             "int",       "0=買幣 1=賣幣"),
        ("twd_samount",      "double",    "台幣金額（已 ×1e-8）"),
        ("currency_samount", "double",    "虛幣數量（已 ×1e-8）"),
        ("currency",         "string",    "虛幣幣種"),
    ],
    "train_label": [
        ("user_id", "bigint", "用戶唯一識別碼"),
        ("status",  "int",    "標籤：0=正常 1=黑名單"),
    ],
    "predict_label": [
        ("user_id", "bigint", "用戶唯一識別碼（待預測）"),
    ],
}


# ══════════════════════════════════════════════════════════════════════════════
#  資料攝取模組
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class IngestionResult:
    """單一資料表的上傳結果。"""
    table:       str
    row_count:   int
    s3_keys:     list[str] = field(default_factory=list)
    elapsed_sec: float     = 0.0
    error:       Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None


class DataIngester:
    """
    從 BitoPro API（或本地 CSV）取得五張原始資料表，
    上傳至 S3 Parquet + CSV 雙格式，依日期分區。

    Well-Architected 設計考量：
      - 分頁抓取（limit=1000 / offset 循環）完整下載全量歷史資料
      - Decimal 精確轉換金額欄位（×1e-8），消除 IEEE 754 浮點誤差
      - Parquet 欄式壓縮（Snappy），Athena 掃描成本降低約 80%
      - Hive 分區鍵（year/month/day）加速時間範圍查詢
      - ServerSideEncryption AES256 確保靜態資料安全
      - tenacity 指數退避重試保護 S3 上傳可靠性

    分頁終止條件（任一成立即停止）：
      1. 本頁回傳筆數 < PAGE_LIMIT（最後一頁）
      2. 已知 total 且 offset + fetched >= total
      3. 達到 MAX_PAGES 安全上限
    """

    def __init__(
        self,
        bucket:   str = S3_BUCKET,
        prefix:   str = S3_RAW_PREFIX,
        region:   str = REGION,
        csv_dir:  Optional[str] = None,
        api_base: str = API_BASE_URL,
    ):
        self.bucket   = bucket
        self.prefix   = prefix
        self.csv_dir  = csv_dir
        self.api_base = api_base.rstrip("/")
        self.s3       = boto3.client("s3", region_name=region)
        self._session = requests.Session()
        self._session.headers.update({"Accept": "application/json"})

    # ── 公開方法 ──────────────────────────────────────────────────────────────

    def ingest_all(
        self,
        tables:   list[str]   = RAW_TABLES,
        formats:  tuple[str, ...] = UPLOAD_FORMATS,
        date:     Optional[datetime] = None,
    ) -> list[IngestionResult]:
        """
        批次注入所有資料表。

        Parameters
        ----------
        tables  : 要上傳的資料表名稱列表
        formats : 上傳格式，"parquet" 和/或 "csv"
        date    : 分區日期（預設今天 UTC）

        Returns
        -------
        list[IngestionResult]，每張表一筆結果
        """
        date     = date or datetime.now(timezone.utc)
        results  = []
        success  = 0

        log.info(f"開始注入 {len(tables)} 張資料表 → s3://{self.bucket}/{self.prefix}/")
        log.info(f"分區日期：{date.strftime('%Y-%m-%d')}  格式：{formats}")

        for table in tables:
            t0     = time.time()
            result = self._ingest_table(table, formats, date)
            result.elapsed_sec = round(time.time() - t0, 2)
            results.append(result)

            if result.success:
                success += 1
                log.info(
                    f"  ✓ {table:<22}  {result.row_count:>8,} 筆"
                    f"  {result.elapsed_sec:.1f}s  → {len(result.s3_keys)} 個 key"
                )
            else:
                log.error(f"  ✗ {table:<22}  失敗：{result.error}")

        log.info(f"注入完成：{success}/{len(tables)} 張成功")
        return results

    def ingest_table(
        self,
        table:   str,
        formats: tuple[str, ...] = UPLOAD_FORMATS,
        date:    Optional[datetime] = None,
    ) -> IngestionResult:
        """注入單一資料表（供外部直接呼叫）。"""
        date = date or datetime.now(timezone.utc)
        return self._ingest_table(table, formats, date)

    # ── 私有方法 ──────────────────────────────────────────────────────────────

    def _ingest_table(
        self,
        table:   str,
        formats: tuple[str, ...],
        date:    datetime,
    ) -> IngestionResult:
        """
        抓取單表 → Decimal 金額轉換 → 序列化 → 上傳 S3。

        流程：
          _fetch()               API 分頁抓取 / CSV 讀取
            ↓
          _apply_decimal_scale() 金額欄位以 Decimal 精確乘以 1e-8
            ↓
          _serialize()           轉為 Parquet（Snappy）+ CSV bytes
            ↓
          _put_object()          上傳 S3（AES256 加密，含重試）
        """
        try:
            df = self._fetch(table)
        except Exception as e:
            return IngestionResult(table=table, row_count=0, error=str(e))

        # 金額欄位精確轉換（Decimal × 1e-8 → float64）
        df = self._apply_decimal_scale(df, table)

        s3_keys   = []
        partition = self._partition_prefix(table, date)

        for fmt in formats:
            try:
                key  = f"{partition}/{table}.{fmt}"
                body = self._serialize(df, fmt)
                self._put_object(
                    key=key,
                    body=body,
                    content_type=("application/octet-stream" if fmt == "parquet"
                                  else "text/csv"),
                    metadata={
                        "table":        table,
                        "row_count":    str(len(df)),
                        "ingested_at":  date.isoformat(),
                        "format":       fmt,
                        "amount_scale": "1e-8",     # 標記已完成精度轉換
                    },
                )
                s3_keys.append(key)
            except Exception as e:
                log.warning(f"{table} [{fmt}] 上傳失敗: {e}")

        self._upload_schema(table, date)
        return IngestionResult(table=table, row_count=len(df), s3_keys=s3_keys)

    # ── 資料抓取（分頁 API / CSV） ────────────────────────────────────────────

    def _fetch(self, table: str) -> pd.DataFrame:
        """
        路由至對應資料源：
          csv_dir 設定 → 本地 CSV
          否則          → BitoPro API（帶分頁）
        """
        if self.csv_dir:
            return self._fetch_csv(table)
        if table in NO_PAGINATION_TABLES:
            return self._fetch_single_api(table)
        return self._fetch_paginated_api(table)

    def _fetch_paginated_api(self, table: str) -> pd.DataFrame:
        """
        以 limit=PAGE_LIMIT / offset 循環分頁抓取完整資料表。

        API 回應格式（兩種均支援）：
          {"data": [...], "total": 51017}   ← 含 total 時提前判斷終止
          [...]                             ← 直接 list

        終止條件（任一成立）：
          ① 本頁筆數 < PAGE_LIMIT（最後一頁）
          ② 已知 total 且已抓完
          ③ 達到 MAX_PAGES 安全上限（防無限迴圈）
          ④ 伺服器回傳空陣列
        """
        pages    : list[pd.DataFrame] = []
        offset   = 0
        total    : Optional[int] = None

        for page_num in range(MAX_PAGES):
            url    = f"{self.api_base}/{table}"
            params = {"limit": PAGE_LIMIT, "offset": offset}

            try:
                resp = self._session.get(url, params=params, timeout=PAGE_TIMEOUT)
                resp.raise_for_status()
            except requests.RequestException as e:
                if pages:
                    log.warning(f"{table} 第 {page_num+1} 頁請求失敗（{e}），使用已抓到的 {sum(len(p) for p in pages):,} 筆繼續")
                    break
                raise

            payload = resp.json()

            # 解析回應結構
            if isinstance(payload, dict):
                records = payload.get("data", payload.get("records", []))
                if total is None:
                    total = payload.get("total") or payload.get("count")
            else:
                records = payload  # 直接回傳 list

            if not records:
                log.debug(f"{table} 第 {page_num+1} 頁回傳空資料，結束分頁")
                break

            page_df = pd.DataFrame(records)
            pages.append(page_df)
            fetched = offset + len(records)

            log.info(
                f"  {table:<22} page {page_num+1:>3}"
                f"  offset={offset:>7,}  本頁={len(records):>5,}  累計={fetched:>7,}"
                + (f"  / total={total:,}" if total else "")
            )

            # 終止條件①：最後一頁（不足一整頁）
            if len(records) < PAGE_LIMIT:
                break

            # 終止條件②：已抓完（有 total 資訊）
            if total is not None and fetched >= total:
                break

            offset += len(records)

        else:
            log.warning(f"{table} 達到 MAX_PAGES={MAX_PAGES} 安全上限，可能未完整抓取")

        if not pages:
            log.warning(f"{table} API 回傳空資料")
            return pd.DataFrame()

        df = pd.concat(pages, ignore_index=True)
        log.info(f"  {table:<22} 分頁完成：共 {len(df):,} 筆（{len(pages)} 頁）")
        return df

    def _fetch_single_api(self, table: str) -> pd.DataFrame:
        """
        不分頁的單次 API 請求（用於 train_label / predict_label 等小表）。
        """
        url  = f"{self.api_base}/{table}"
        resp = self._session.get(url, timeout=PAGE_TIMEOUT)
        resp.raise_for_status()
        payload = resp.json()
        if isinstance(payload, dict):
            records = payload.get("data", payload)
        else:
            records = payload
        df = pd.DataFrame(records) if records else pd.DataFrame()
        log.info(f"  {table:<22} 單次抓取：{len(df):,} 筆")
        return df

    def _fetch_csv(self, table: str) -> pd.DataFrame:
        """從本地 CSV 目錄讀取（全部欄位先讀成 str，避免大整數精度損失）。"""
        path = os.path.join(self.csv_dir, f"{table}.csv")
        df   = pd.read_csv(path, dtype=str)
        log.info(f"  {table:<22} CSV 讀取：{len(df):,} 筆  ({path})")
        return df

    # ── Decimal 金額精確轉換 ──────────────────────────────────────────────────

    @staticmethod
    def _apply_decimal_scale(df: pd.DataFrame, table: str) -> pd.DataFrame:
        """
        將金額欄位以 decimal.Decimal 精確乘以 1e-8，結果存為 float64。

        為什麼使用 Decimal 而非直接 × 1e-8：
          原始 API 回傳整數字串，例如 "123456789012"（= 1234.56789012 TWD）。
          直接 int(x) * 1e-8 因 IEEE 754 double 精度限制，末位可能偏差 1–2 ULP。
          Decimal("123456789012") * Decimal("1e-8") 完整保留所有有效位，
          最終轉 float64 誤差 < 1e-15，符合金融合規要求。

        自動偵測策略：
          ① AMOUNT_FIELDS[table] 中明確定義的欄位
          ② 欄位名稱含 "amount" 或 "srate" 的欄位（防漏偵測）
        """
        # ① 明確定義 + ② 自動偵測
        explicit     = set(AMOUNT_FIELDS.get(table, []))
        auto_detect  = {
            c for c in df.columns
            if any(kw in c.lower() for kw in ("amount", "srate"))
        }
        target_cols = explicit | auto_detect

        if not target_cols:
            return df

        SCALE = Decimal("1e-8")
        df    = df.copy()

        def _safe_convert(val) -> float:
            """單值轉換：空值 → NaN，無效數字 → NaN，其餘精確轉換。"""
            if val is None or val != val:   # None 或 NaN
                return float("nan")
            s = str(val).strip()
            if s in ("", "null", "None", "NaN"):
                return float("nan")
            try:
                return float(Decimal(s) * SCALE)
            except InvalidOperation:
                return float("nan")

        converted = 0
        for col in sorted(target_cols):
            if col not in df.columns:
                continue
            df[col] = df[col].apply(_safe_convert).astype("float64")
            converted += 1
            log.debug(f"    {table}.{col}  ×1e-8 完成")

        if converted:
            log.info(f"  {table:<22} Decimal ×1e-8 轉換：{converted} 個欄位")

        return df

    def _partition_prefix(self, table: str, date: datetime) -> str:
        """生成 Hive 相容分區路徑。"""
        return (
            f"{self.prefix}/{table}"
            f"/year={date.year}"
            f"/month={date.month:02d}"
            f"/day={date.day:02d}"
        )

    @staticmethod
    def _serialize(df: pd.DataFrame, fmt: str) -> bytes:
        """DataFrame → bytes（parquet 或 csv）。"""
        buf = io.BytesIO()
        if fmt == "parquet":
            df.to_parquet(buf, index=False, engine="pyarrow", compression="snappy")
        else:
            df.to_csv(buf, index=False, encoding="utf-8")
        return buf.getvalue()

    def _upload_schema(self, table: str, date: datetime) -> None:
        """上傳欄位定義 JSON（輔助 Glue 理解欄位語意）。"""
        schema = {
            "table":   table,
            "columns": GLUE_COLUMN_DEFS.get(table, []),
            "generated_at": date.isoformat(),
        }
        key  = f"{self.prefix}/{table}/_schema.json"
        body = json.dumps(schema, ensure_ascii=False, indent=2).encode()
        try:
            self._put_object(key, body, "application/json", {})
        except Exception:
            pass  # schema 上傳失敗不阻斷主流程

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(ClientError),
        reraise=True,
    )
    def _put_object(self, key: str, body: bytes,
                    content_type: str, metadata: dict) -> None:
        self.s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=body,
            ContentType=content_type,
            Metadata=metadata,
            ServerSideEncryption="AES256",
        )

    def print_summary(self, results: list[IngestionResult]) -> None:
        """終端機列印注入摘要表格。"""
        print("\n" + "═" * 65)
        print(f"  S3 注入摘要  →  s3://{self.bucket}/{self.prefix}/")
        print("═" * 65)
        print(f"  {'資料表':<22} {'筆數':>8}  {'耗時':>6}  {'狀態'}")
        print("  " + "-" * 61)
        total_rows = 0
        for r in results:
            status = "✓ 完成" if r.success else f"✗ {r.error[:25]}"
            print(f"  {r.table:<22} {r.row_count:>8,}  {r.elapsed_sec:>5.1f}s  {status}")
            total_rows += r.row_count
        print("  " + "-" * 61)
        print(f"  {'合計':<22} {total_rows:>8,}")
        print("═" * 65 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
#  Glue Data Catalog 建置
# ══════════════════════════════════════════════════════════════════════════════

class GlueCatalogSetup:
    """
    使用 boto3 建立 Glue Database + Crawler，爬取 S3 Parquet 後產生 Data Catalog 表。

    完成後 SageMaker、Athena、EMR 可直接以 SQL 存取幣託資料。

    設計說明：
      - Crawler 採 Recrawl Policy = CRAWL_NEW_FOLDERS_ONLY，節省重複爬取費用
      - S3 Target 逐表設定，確保每張表的 Glue Table 名稱與 prefix 一一對應
      - 欄位型別由 _register_table_schema() 以 create_table API 預先定義，
        優先於 Crawler 自動推斷（避免 bigint 被推斷為 string 等問題）
    """

    # Glue 支援的 SerDe（Parquet 格式）
    _PARQUET_SERDE = "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe"
    _PARQUET_INPUT = "org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat"
    _PARQUET_OUTPUT = "org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat"

    def __init__(
        self,
        region:      str = REGION,
        glue_db:     str = GLUE_DB,
        crawler:     str = GLUE_CRAWLER,
        role_arn:    str = GLUE_ROLE_ARN,
        bucket:      str = S3_BUCKET,
        raw_prefix:  str = S3_RAW_PREFIX,
    ):
        self.glue       = boto3.client("glue", region_name=region)
        self.db         = glue_db
        self.crawler    = crawler
        self.role_arn   = role_arn
        self.bucket     = bucket
        self.raw_prefix = raw_prefix
        self.region     = region

    # ── 公開方法 ──────────────────────────────────────────────────────────────

    def setup_all(
        self,
        tables:      list[str] = RAW_TABLES,
        run_crawler: bool = True,
        wait:        bool = True,
    ) -> None:
        """
        完整建置流程：
          1. 建立 Glue Database（已存在則略過）
          2. 為每張表預先定義精確的欄位 Schema
          3. 建立 / 更新 Glue Crawler
          4. （可選）啟動 Crawler 並等待完成
        """
        # W-1 Guard：ARN 未設定時提早失敗，避免誤用佔位符
        if not self.role_arn or "ACCOUNT_ID" in self.role_arn:
            raise ValueError(
                "GLUE_ROLE_ARN 未正確設定（目前值含 'ACCOUNT_ID' 佔位符或為空）。"
                " 請先設定環境變數：export GLUE_ROLE_ARN=arn:aws:iam::<account-id>:role/GlueServiceRole"
            )

        log.info("═" * 55)
        log.info("Glue Data Catalog 建置開始")
        log.info("═" * 55)

        self._ensure_database()

        for table in tables:
            self._register_table_schema(table)

        self._upsert_crawler(tables)

        if run_crawler:
            self._run_crawler(wait=wait)

        log.info("\n建置完成！")
        log.info(f"  Glue Database : {self.db}")
        log.info(f"  Glue Crawler  : {self.crawler}")
        log.info(f"  Tables        : {', '.join(tables)}")
        log.info(f"\n  Athena 查詢範例：")
        log.info(f"    SELECT * FROM {self.db}.twd_transfer")
        log.info(f"    WHERE year='2026' AND month='03' LIMIT 100;")

    def get_crawler_status(self) -> dict:
        """查詢 Crawler 最新一次執行狀態。"""
        try:
            r = self.glue.get_crawler(Name=self.crawler)["Crawler"]
            return {
                "name":          r["Name"],
                "state":         r["State"],
                "last_run":      r.get("LastCrawl", {}).get("Status", "N/A"),
                "last_run_time": str(r.get("LastCrawl", {}).get("StartTime", "N/A")),
                "tables_created": r.get("LastCrawl", {}).get("TablesCreated", 0),
                "tables_updated": r.get("LastCrawl", {}).get("TablesUpdated", 0),
            }
        except ClientError:
            return {"error": f"Crawler '{self.crawler}' 不存在"}

    # ── 私有方法 ──────────────────────────────────────────────────────────────

    def _ensure_database(self) -> None:
        """建立 Glue Database（已存在則略過）。"""
        try:
            self.glue.create_database(
                DatabaseInput={
                    "Name":        self.db,
                    "Description": "幣託科技人頭戶偵測 — AWS Hackathon 2026",
                    "Parameters":  {
                        "classification":  "parquet",
                        "project":         "bito-mule-detection",
                        "created_by":      "ingest_to_s3.py",
                    },
                }
            )
            log.info(f"Glue Database 已建立：{self.db}")
        except self.glue.exceptions.AlreadyExistsException:
            log.info(f"Glue Database 已存在（略過）：{self.db}")

    def _register_table_schema(self, table: str) -> None:
        """
        以 Glue create_table API 預先定義精確欄位型別。

        優先於 Crawler 自動推斷，避免：
          - bigint 被推斷為 string（大 ID 欄位）
          - timestamp 被推斷為 string
          - double 精度損失

        分區欄位（year / month / day）額外定義於 PartitionKeys。
        """
        col_defs = GLUE_COLUMN_DEFS.get(table, [])
        columns  = [
            {"Name": name, "Type": gtype, "Comment": comment}
            for name, gtype, comment in col_defs
        ]
        s3_location = f"s3://{self.bucket}/{self.raw_prefix}/{table}/"

        table_input = {
            "Name":        table,
            "Description": f"幣託原始資料表 — {table}",
            "TableType":   "EXTERNAL_TABLE",
            "Parameters":  {
                "classification":               "parquet",
                "compressionType":              "snappy",
                "typeOfData":                   "file",
                "EXTERNAL":                     "TRUE",
                "parquet.compression":          "SNAPPY",
                "projection.enabled":           "false",
            },
            "StorageDescriptor": {
                "Columns":           columns,
                "Location":          s3_location,
                "InputFormat":       self._PARQUET_INPUT,
                "OutputFormat":      self._PARQUET_OUTPUT,
                "Compressed":        True,
                "SerdeInfo": {
                    "SerializationLibrary": self._PARQUET_SERDE,
                    "Parameters": {"serialization.format": "1"},
                },
                "StoredAsSubDirectories": True,
            },
            "PartitionKeys": [
                {"Name": "year",  "Type": "string", "Comment": "分區年份"},
                {"Name": "month", "Type": "string", "Comment": "分區月份"},
                {"Name": "day",   "Type": "string", "Comment": "分區日期"},
            ],
        }

        try:
            self.glue.create_table(DatabaseName=self.db, TableInput=table_input)
            log.info(f"  Glue Table 已建立：{self.db}.{table}")
        except self.glue.exceptions.AlreadyExistsException:
            # 更新現有 table schema
            self.glue.update_table(DatabaseName=self.db, TableInput=table_input)
            log.info(f"  Glue Table 已更新：{self.db}.{table}")
        except ClientError as e:
            log.warning(f"  Glue Table {table} 設定失敗：{e}")

    def _upsert_crawler(self, tables: list[str]) -> None:
        """
        建立或更新 Glue Crawler。

        Crawler 設定重點：
          - S3 Targets：每張表獨立 prefix，Glue 可正確對應 table 名稱
          - RecrawlPolicy = CRAWL_NEW_FOLDERS_ONLY：節省費用，僅爬新分區
          - SchemaChangePolicy = LOG：欄位變更記錄到 CloudWatch，不刪表
          - Schedule：可設定每日觸發，此處留空（手動觸發）
        """
        s3_targets = [
            {
                "Path":       f"s3://{self.bucket}/{self.raw_prefix}/{table}/",
                "Exclusions": ["**/_schema.json", "**/*.csv"],  # 僅爬 Parquet
            }
            for table in tables
        ]

        crawler_config = {
            "Name":        self.crawler,
            "Role":        self.role_arn,
            "DatabaseName": self.db,
            "Description": "爬取幣託原始 Parquet 資料，更新 Glue Data Catalog",
            "Targets": {"S3Targets": s3_targets},
            "RecrawlPolicy": {
                "RecrawlBehavior": "CRAWL_NEW_FOLDERS_ONLY",
            },
            "SchemaChangePolicy": {
                "UpdateBehavior": "UPDATE_IN_DATABASE",
                "DeleteBehavior": "LOG",
            },
            "Configuration": json.dumps({
                "Version": 1.0,
                "CrawlerOutput": {
                    "Partitions": {"AddOrUpdateBehavior": "InheritFromTable"},
                    "Tables":     {"AddOrUpdateBehavior": "MergeNewColumns"},
                },
                "Grouping": {
                    "TableGroupingPolicy": "CombineCompatibleSchemas",
                },
            }),
            "Tags": {
                "Project":     "bito-mule-detection",
                "Environment": "hackathon",
                "ManagedBy":   "ingest_to_s3.py",
            },
        }

        try:
            self.glue.create_crawler(**crawler_config)
            log.info(f"Glue Crawler 已建立：{self.crawler}")
        except self.glue.exceptions.AlreadyExistsException:
            # 更新現有 Crawler（刪除後重建，因 update_crawler 不支援所有欄位）
            try:
                state = self.glue.get_crawler(Name=self.crawler)["Crawler"]["State"]
                if state == "RUNNING":
                    log.warning("Crawler 正在執行中，略過更新。")
                    return
                # update_crawler 支援大部分欄位
                update_cfg = {k: v for k, v in crawler_config.items()
                              if k not in ("Name", "Tags")}
                self.glue.update_crawler(Name=self.crawler, **update_cfg)
                log.info(f"Glue Crawler 已更新：{self.crawler}")
            except ClientError as e:
                log.warning(f"Crawler 更新失敗：{e}")

    def _run_crawler(self, wait: bool = True, timeout: int = 600) -> None:
        """
        啟動 Crawler，可選等待至完成。

        Parameters
        ----------
        wait    : True = 阻塞等待 Crawler 完成（最多 timeout 秒）
        timeout : 等待上限（秒），預設 10 分鐘
        """
        try:
            self.glue.start_crawler(Name=self.crawler)
            log.info(f"Glue Crawler 已啟動：{self.crawler}")
        except self.glue.exceptions.CrawlerRunningException:
            log.info("Crawler 已在執行中。")
        except ClientError as e:
            log.error(f"Crawler 啟動失敗：{e}")
            return

        if not wait:
            log.info("（非同步模式，不等待 Crawler 完成）")
            return

        log.info("等待 Crawler 完成...")
        deadline = time.time() + timeout
        while time.time() < deadline:
            time.sleep(15)
            resp  = self.glue.get_crawler(Name=self.crawler)
            state = resp["Crawler"]["State"]
            log.info(f"  Crawler 狀態：{state}")
            if state == "READY":
                last = resp["Crawler"].get("LastCrawl", {})
                log.info(
                    f"Crawler 完成！"
                    f"  Status={last.get('Status')}"
                    f"  TablesCreated={last.get('TablesCreated', 0)}"
                    f"  TablesUpdated={last.get('TablesUpdated', 0)}"
                )
                return

        log.warning(f"等待超時（{timeout}s），Crawler 可能仍在執行中。")


# ══════════════════════════════════════════════════════════════════════════════
#  Glue Crawler 部署參考：CloudFormation YAML 片段（輸出供人工審核）
# ══════════════════════════════════════════════════════════════════════════════

def print_cloudformation_snippet() -> None:
    """
    輸出等效的 CloudFormation YAML。

    此片段可直接貼入 cfn_template.yaml，讓 IaC 管理 Glue 資源。
    """
    yaml_snippet = f"""
# ──────────────────────────────────────────────────────────────────────────
#  AWS CloudFormation — Glue Data Catalog（等效於 ingest_to_s3.py 的 boto3 操作）
# ──────────────────────────────────────────────────────────────────────────
Resources:

  BitoGlueDatabase:
    Type: AWS::Glue::Database
    Properties:
      CatalogId: !Ref AWS::AccountId
      DatabaseInput:
        Name: {GLUE_DB}
        Description: 幣託科技人頭戶偵測 — AWS Hackathon 2026

  BitoGlueCrawler:
    Type: AWS::Glue::Crawler
    Properties:
      Name: {GLUE_CRAWLER}
      Role: !Sub arn:aws:iam::${{AWS::AccountId}}:role/GlueServiceRole
      DatabaseName: !Ref BitoGlueDatabase
      Description: 爬取幣託原始 Parquet 資料，更新 Glue Data Catalog
      Targets:
        S3Targets:
          - Path: s3://{S3_BUCKET}/{S3_RAW_PREFIX}/user_info/
            Exclusions:
              - "**/_schema.json"
              - "**/*.csv"
          - Path: s3://{S3_BUCKET}/{S3_RAW_PREFIX}/twd_transfer/
            Exclusions: ["**/_schema.json", "**/*.csv"]
          - Path: s3://{S3_BUCKET}/{S3_RAW_PREFIX}/crypto_transfer/
            Exclusions: ["**/_schema.json", "**/*.csv"]
          - Path: s3://{S3_BUCKET}/{S3_RAW_PREFIX}/usdt_twd_trading/
            Exclusions: ["**/_schema.json", "**/*.csv"]
          - Path: s3://{S3_BUCKET}/{S3_RAW_PREFIX}/usdt_swap/
            Exclusions: ["**/_schema.json", "**/*.csv"]
          - Path: s3://{S3_BUCKET}/{S3_RAW_PREFIX}/train_label/
            Exclusions: ["**/_schema.json", "**/*.csv"]
          - Path: s3://{S3_BUCKET}/{S3_RAW_PREFIX}/predict_label/
            Exclusions: ["**/_schema.json", "**/*.csv"]
      RecrawlPolicy:
        RecrawlBehavior: CRAWL_NEW_FOLDERS_ONLY
      SchemaChangePolicy:
        UpdateBehavior: UPDATE_IN_DATABASE
        DeleteBehavior: LOG
      Configuration: |
        {{
          "Version": 1.0,
          "CrawlerOutput": {{
            "Partitions": {{"AddOrUpdateBehavior": "InheritFromTable"}},
            "Tables":     {{"AddOrUpdateBehavior": "MergeNewColumns"}}
          }},
          "Grouping": {{
            "TableGroupingPolicy": "CombineCompatibleSchemas"
          }}
        }}
      Tags:
        Project: bito-mule-detection
        Environment: hackathon

# SageMaker 訓練直接讀取 Glue Catalog 範例（train_sagemaker.py 可加入此呼叫）：
#
#   from sagemaker.inputs import DatasetDefinition, AthenaDatasetDefinition
#   athena_input = AthenaDatasetDefinition(
#       catalog="AwsDataCatalog",
#       database="{GLUE_DB}",
#       query_string="SELECT * FROM twd_transfer WHERE year='2026'",
#       output_s3_uri="s3://{S3_BUCKET}/athena-results/",
#       output_format="PARQUET",
#   )
"""
    print(yaml_snippet)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI 入口
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="幣託原始資料注入 S3 + Glue Catalog 自動建置",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--csv-dir",        default=None,
                        help="本地 CSV 目錄（不指定則從 BitoPro API 分頁抓取）")
    parser.add_argument("--api-base",       default=API_BASE_URL,
                        help=f"API 根網址（預設：{API_BASE_URL}）")
    parser.add_argument("--page-limit",     type=int, default=PAGE_LIMIT,
                        help=f"每頁筆數（預設：{PAGE_LIMIT}）")
    parser.add_argument("--bucket",         default=S3_BUCKET)
    parser.add_argument("--prefix",         default=S3_RAW_PREFIX)
    parser.add_argument("--skip-glue",      action="store_true",
                        help="僅上傳 S3，不建立 Glue Catalog（調試用）")
    parser.add_argument("--only-glue",      action="store_true",
                        help="僅建立 Glue Catalog（資料已在 S3）")
    parser.add_argument("--no-wait",        action="store_true",
                        help="啟動 Crawler 後不等待完成（背景執行）")
    parser.add_argument("--crawler-status", action="store_true",
                        help="查詢 Crawler 狀態後退出")
    parser.add_argument("--cfn",            action="store_true",
                        help="輸出等效 CloudFormation YAML 片段")
    parser.add_argument("--tables",         nargs="+", default=RAW_TABLES,
                        help="只注入指定的資料表")
    parser.add_argument("--formats",        nargs="+", default=list(UPLOAD_FORMATS),
                        choices=["parquet", "csv"])
    args = parser.parse_args()

    # ── CloudFormation 輸出模式 ───────────────────────────────────────────
    if args.cfn:
        print_cloudformation_snippet()
        return

    # ── Crawler 狀態查詢模式 ─────────────────────────────────────────────
    if args.crawler_status:
        glue_setup = GlueCatalogSetup(bucket=args.bucket)
        status     = glue_setup.get_crawler_status()
        print("\nGlue Crawler 狀態：")
        for k, v in status.items():
            print(f"  {k:<18}: {v}")
        return

    # ── Glue Only 模式 ───────────────────────────────────────────────────
    if args.only_glue:
        glue_setup = GlueCatalogSetup(bucket=args.bucket, raw_prefix=args.prefix)
        glue_setup.setup_all(
            tables=args.tables,
            run_crawler=True,
            wait=not args.no_wait,
        )
        return

    # ── 完整注入流程 ─────────────────────────────────────────────────────
    t_start  = time.time()
    ingester = DataIngester(
        bucket=args.bucket,
        prefix=args.prefix,
        csv_dir=args.csv_dir,
        api_base=args.api_base,
    )

    results = ingester.ingest_all(
        tables=args.tables,
        formats=tuple(args.formats),
    )
    ingester.print_summary(results)

    failed           = [r for r in results if not r.success]
    successful_tables = [r.table for r in results if r.success]

    if failed:
        log.warning(f"注意：{len(failed)} 張表上傳失敗 → {[r.table for r in failed]}")

    # ── 自動執行 Glue Catalog 建置（預設行為，--skip-glue 才跳過）────────
    if args.skip_glue:
        log.info("--skip-glue：略過 Glue Catalog 建置。")
    elif not successful_tables:
        log.error("所有表上傳失敗，略過 Glue 建置。")
    else:
        log.info(f"\n{'═'*55}")
        log.info("資料上傳完成，自動啟動 Glue Catalog 建置...")
        log.info(f"{'═'*55}")
        glue_setup = GlueCatalogSetup(bucket=args.bucket, raw_prefix=args.prefix)
        glue_setup.setup_all(
            tables=successful_tables,
            run_crawler=True,
            wait=not args.no_wait,
        )
        if not args.no_wait:
            log.info("\nAthena 現在可以直接查詢最新資料：")
            log.info(f"  SELECT * FROM {GLUE_DB}.twd_transfer LIMIT 10;")
            log.info(f"  SELECT * FROM {GLUE_DB}.user_info WHERE kyc_level = 1 LIMIT 10;")

    # 上傳失敗時回傳非零退出碼（供 CI/CD 偵測）
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
