"""
SageMaker Feature Store — 幣託人頭戶偵測特徵群組管理
=====================================================

功能：
  1. FeatureGroupManager   — 建立 / 刪除 Feature Group
  2. FeatureIngester       — 批次 & 即時攝取特徵
  3. FeatureRetriever      — 即時推論查詢（GetRecord）& 批次訓練查詢（Athena）
  4. RealtimeFeaturePipeline — 整合入口：推論時一次取得所有特徵向量

使用流程
--------
  # Step 1（只需跑一次）：建立 Feature Group
  python feature_store.py --action create

  # Step 2：攝取特徵（Glue Job 完成後手動觸發 or 排程）
  python feature_store.py --action ingest --csv hop_features.csv

  # Step 3：推論時呼叫（由 train_sagemaker.py / xai_bedrock.py 呼叫）
  from feature_store import RealtimeFeaturePipeline
  pipeline = RealtimeFeaturePipeline()
  context  = pipeline.get_user_context(user_id=12345)
"""

import os
import io
import time
import json
import logging
import argparse
from datetime import datetime, timezone
from typing import Optional

import boto3
import pandas as pd
import numpy as np

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

# ══════════════════════════════════════════════════════════════════════════════
#  Region 合規檢核（競賽規定：僅允許 us-east-1 / us-west-2）
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


# ── 設定（與其他模組共用的常數） ─────────────────────────────────────────────

REGION        = _validate_region(os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
S3_BUCKET     = os.environ.get("S3_BUCKET",           "your-hackathon-bucket")
ROLE_ARN      = os.environ.get("SAGEMAKER_ROLE_ARN",   "arn:aws:iam::ACCOUNT_ID:role/SageMakerRole")

# ── Feature Group 定義（兩個群組，對應不同特徵集） ────────────────────────────

FEATURE_GROUPS = {
    #
    # FG-1：圖跳轉特徵（由 Glue Job 產出）
    #
    "bito-mule-hops": {
        "description": "幣託人頭戶偵測 — 資金關聯深度 (Graph BFS Hops) 特徵群組",
        "record_id":   "user_id",
        "event_time":  "event_time",
        "features": [
            {"FeatureName": "user_id",                  "FeatureType": "Integral"},
            {"FeatureName": "min_hops_to_blacklist",    "FeatureType": "Integral"},
            {"FeatureName": "is_direct_neighbor",       "FeatureType": "Integral"},   # 0/1
            {"FeatureName": "blacklist_neighbor_count", "FeatureType": "Integral"},
            {"FeatureName": "in_blacklist_network",     "FeatureType": "Integral"},   # 0/1
            {"FeatureName": "hop_risk_level",           "FeatureType": "String"},
            {"FeatureName": "event_time",               "FeatureType": "String"},     # ISO-8601
        ],
        "offline_s3": f"s3://{S3_BUCKET}/feature-store/bito-mule-hops/",
    },
    #
    # FG-2：用戶行為特徵（由 BitoDataManager 產出）
    #
    "bito-mule-behavior": {
        "description": "幣託人頭戶偵測 — 用戶行為特徵（滯留時間、IP 跳動、量能不對稱）",
        "record_id":   "user_id",
        "event_time":  "event_time",
        "features": [
            {"FeatureName": "user_id",                  "FeatureType": "Integral"},
            {"FeatureName": "kyc_level",                "FeatureType": "Integral"},
            {"FeatureName": "min_retention_minutes",    "FeatureType": "Fractional"},
            {"FeatureName": "high_speed_risk",          "FeatureType": "Integral"},   # 0/1
            {"FeatureName": "unique_ip_count",          "FeatureType": "Integral"},
            {"FeatureName": "ip_anomaly",               "FeatureType": "Integral"},   # 0/1
            {"FeatureName": "total_twd_volume",         "FeatureType": "Fractional"},
            {"FeatureName": "volume_zscore",            "FeatureType": "Fractional"},
            {"FeatureName": "asymmetry_flag",           "FeatureType": "Integral"},   # 0/1
            {"FeatureName": "mule_risk_score",          "FeatureType": "Integral"},
            {"FeatureName": "crypto_withdraw_count",    "FeatureType": "Integral"},
            {"FeatureName": "twd_deposit_count",        "FeatureType": "Integral"},
            {"FeatureName": "twd_withdraw_count",       "FeatureType": "Integral"},
            {"FeatureName": "event_time",               "FeatureType": "String"},
        ],
        "offline_s3": f"s3://{S3_BUCKET}/feature-store/bito-mule-behavior/",
    },
}

# 推論時使用的特徵欄位（順序與 train_sagemaker.CANONICAL_FEATURE_COLS
# 及 train_xgboost_script.FEATURE_INDEX 完全一致）
# ⚠ 三處必須同步修改，永遠保持一致。
INFERENCE_FEATURE_COLUMNS = [
    # idx=0   用戶基本屬性
    "kyc_level",
    # idx=1~3 特徵①：資金滯留時間
    "min_retention_minutes",
    "retention_event_count",
    "high_speed_risk",
    # idx=4~5 特徵②：IP 異常跳動
    "unique_ip_count",
    "ip_anomaly",
    # idx=6~8 特徵③：量能不對稱
    "total_twd_volume",
    "volume_zscore",
    "asymmetry_flag",
    # idx=9~12 特徵④：圖跳轉
    "min_hops_to_blacklist",
    "is_direct_neighbor",
    "blacklist_neighbor_count",
    "in_blacklist_network",
    # idx=13  綜合風險評分
    "mule_risk_score",
    # idx=14~17 交易計數
    "twd_deposit_count",
    "twd_withdraw_count",
    "crypto_deposit_count",
    "crypto_withdraw_count",
    # idx=18  時間模式
    "night_tx_ratio",
]


# ══════════════════════════════════════════════════════════════════════════════
#  1. FeatureGroupManager — 建立 / 描述 / 刪除 Feature Group
# ══════════════════════════════════════════════════════════════════════════════

class FeatureGroupManager:
    """
    管理 SageMaker Feature Group 的生命周期。

    Feature Store 兩層儲存：
      ┌────────────────────┬─────────────────────────────────────────────────┐
      │ Online Store        │ DynamoDB，低延遲（< 10ms）即時讀取               │
      │ Offline Store       │ S3 + Glue Data Catalog，供 Athena 批次查詢      │
      └────────────────────┴─────────────────────────────────────────────────┘
    """

    def __init__(self, region: str = REGION, role_arn: str = ROLE_ARN):
        self.sm     = boto3.client("sagemaker",        region_name=region)
        self.region = region
        self.role   = role_arn

    def create(self, group_name: str, wait: bool = True) -> dict:
        """
        建立 Feature Group（Online + Offline Store）。

        Online Store 使用預設 DynamoDB，Offline Store 指向 S3。
        若 Feature Group 已存在則跳過（冪等操作）。
        """
        cfg = FEATURE_GROUPS[group_name]

        # ── 檢查是否已存在 ────────────────────────────────────────────────
        try:
            resp = self.sm.describe_feature_group(FeatureGroupName=group_name)
            status = resp["FeatureGroupStatus"]
            log.info(f"[FeatureGroup] {group_name} 已存在（狀態：{status}），跳過建立。")
            return resp
        except self.sm.exceptions.ResourceNotFound:
            pass

        log.info(f"[FeatureGroup] 建立 {group_name}...")
        resp = self.sm.create_feature_group(
            FeatureGroupName       = group_name,
            RecordIdentifierFeatureName = cfg["record_id"],
            EventTimeFeatureName        = cfg["event_time"],
            FeatureDefinitions     = cfg["features"],
            Description            = cfg["description"],
            OnlineStoreConfig      = {"EnableOnlineStore": True},
            OfflineStoreConfig     = {
                "S3StorageConfig": {
                    "S3Uri": cfg["offline_s3"],
                    "KmsKeyId": "",       # 可填入 KMS Key ARN 加密
                },
                "DisableGlueTableCreation": False,  # 自動在 Glue 建 Table
                "DataCatalogConfig": {
                    "TableName":    group_name.replace("-", "_"),
                    "Catalog":      "AwsDataCatalog",
                    "Database":     "bito_features",
                },
            },
            RoleArn                = self.role,
            Tags=[
                {"Key": "Project",     "Value": "bito-mule-detection"},
                {"Key": "ManagedBy",   "Value": "feature_store.py"},
            ],
        )

        if wait:
            self._wait_created(group_name)

        log.info(f"[FeatureGroup] {group_name} 建立完成。")
        return resp

    def create_all(self) -> None:
        """建立所有已定義的 Feature Groups。"""
        for name in FEATURE_GROUPS:
            self.create(name)

    def describe(self, group_name: str) -> dict:
        return self.sm.describe_feature_group(FeatureGroupName=group_name)

    def delete(self, group_name: str) -> None:
        """刪除 Feature Group（注意：Offline Store 資料不會被刪除）。"""
        log.warning(f"[FeatureGroup] 刪除 {group_name}（S3 Offline Store 資料保留）")
        self.sm.delete_feature_group(FeatureGroupName=group_name)

    def _wait_created(self, group_name: str, poll_interval: int = 5,
                      timeout: int = 300) -> None:
        """輪詢等待 Feature Group 狀態變為 Created。"""
        deadline = time.time() + timeout
        while time.time() < deadline:
            status = self.sm.describe_feature_group(
                FeatureGroupName=group_name
            )["FeatureGroupStatus"]
            if status == "Created":
                return
            elif "Failed" in status:
                raise RuntimeError(f"Feature Group 建立失敗：{status}")
            log.info(f"  等待 {group_name} ... 狀態={status}")
            time.sleep(poll_interval)
        raise TimeoutError(f"等待 {group_name} 超時（{timeout}s）")


# ══════════════════════════════════════════════════════════════════════════════
#  2. FeatureIngester — 將特徵從 DataFrame 攝取到 Feature Store
# ══════════════════════════════════════════════════════════════════════════════

class FeatureIngester:
    """
    將 Pandas DataFrame 的特徵攝取到 SageMaker Feature Store。

    支援兩種模式：
      ① 即時單筆（PutRecord）     — 用於即時特徵更新（< 100 筆）
      ② 批次並行（ThreadPoolExecutor）— 用於歷史特徵初始化（數萬~百萬筆）
    """

    def __init__(self, region: str = REGION):
        self.client = boto3.client(
            "sagemaker-featurestore-runtime", region_name=region
        )

    # ── 將 DataFrame 一列轉為 FeatureStore Record 格式 ───────────────────────
    @staticmethod
    def _row_to_record(row: pd.Series, schema: list[dict]) -> list[dict]:
        """
        把 DataFrame 的一行轉換為 Feature Store 的 Record 格式。
        Boolean 轉 "0"/"1"，數值轉 str，None 填 "0"。
        """
        record = []
        feature_names = {f["FeatureName"] for f in schema}

        for feat in schema:
            fname = feat["FeatureName"]
            if fname not in row.index:
                continue

            val = row[fname]

            # 處理 None / NaN
            if val is None or (isinstance(val, float) and np.isnan(val)):
                val = "0"
            elif isinstance(val, bool):
                val = "1" if val else "0"
            elif isinstance(val, (np.integer, np.int64, np.int32)):
                val = str(int(val))
            elif isinstance(val, (np.floating, np.float64, float)):
                val = f"{val:.8f}"
            elif isinstance(val, pd.Timestamp):
                val = val.isoformat()
            else:
                val = str(val)

            record.append({"FeatureName": fname, "ValueAsString": val})

        return record

    def put_record(self, group_name: str, row: pd.Series) -> None:
        """即時攝取單筆記錄。"""
        schema = FEATURE_GROUPS[group_name]["features"]
        record = self._row_to_record(row, schema)
        self.client.put_record(
            FeatureGroupName=group_name,
            Record=record,
        )

    def ingest_dataframe(
        self,
        df: pd.DataFrame,
        group_name: str,
        max_workers: int = 8,
        add_event_time: bool = True,
    ) -> dict:
        """
        批次攝取整個 DataFrame 到 Feature Store。

        使用 ThreadPoolExecutor 並行上傳，充分利用網路 I/O：
          max_workers=8 → 8 個並行 HTTP 連線，吞吐量約 200~500 RPS

        Parameters
        ----------
        df             : 待攝取的 DataFrame（需含 Feature Group 定義的所有欄位）
        group_name     : Feature Group 名稱
        max_workers    : 並行工作線程數
        add_event_time : 若 df 無 event_time 欄位，自動填入當前時間

        Returns
        -------
        {"success": int, "failed": int, "failed_ids": list}
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        df = df.copy()

        # 確保 event_time 存在
        if add_event_time and "event_time" not in df.columns:
            df["event_time"] = datetime.now(timezone.utc).isoformat()

        schema = FEATURE_GROUPS[group_name]["features"]
        record_id_col = FEATURE_GROUPS[group_name]["record_id"]

        success_count = 0
        failed_ids    = []

        def _put(row_tuple):
            idx, row = row_tuple
            try:
                record = self._row_to_record(row, schema)
                self.client.put_record(
                    FeatureGroupName=group_name,
                    Record=record,
                )
                return True, None
            except Exception as e:
                user_id = row.get(record_id_col, idx)
                log.warning(f"PutRecord 失敗 [{group_name}] user_id={user_id}: {e}")
                return False, user_id

        log.info(f"[Ingest] 開始攝取 {len(df):,} 筆 → {group_name} "
                 f"（並行度 {max_workers}）")
        t0 = time.time()

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_put, row_tuple): row_tuple
                for row_tuple in df.iterrows()
            }
            for i, fut in enumerate(as_completed(futures), 1):
                ok, uid = fut.result()
                if ok:
                    success_count += 1
                else:
                    failed_ids.append(uid)

                if i % 1000 == 0:
                    elapsed = round(time.time() - t0, 1)
                    log.info(f"  進度：{i:,}/{len(df):,}  耗時 {elapsed}s")

        elapsed = round(time.time() - t0, 1)
        log.info(f"[Ingest] 完成：成功 {success_count:,}  失敗 {len(failed_ids):,}  "
                 f"耗時 {elapsed}s  "
                 f"吞吐量 {round(success_count/elapsed):,} RPS")

        return {
            "success":    success_count,
            "failed":     len(failed_ids),
            "failed_ids": failed_ids,
        }

    def ingest_behavior_features(
        self,
        mule_df: pd.DataFrame,
        crypto_transfer: pd.DataFrame,
        twd_transfer: pd.DataFrame,
    ) -> dict:
        """
        從 extract_mule_features() 的輸出組裝 behavior Feature Group 所需欄位並攝取。

        Parameters
        ----------
        mule_df         : BitoDataManager.extract_mule_features() 的結果
        crypto_transfer : 正規化後的 crypto_transfer DataFrame
        twd_transfer    : 正規化後的 twd_transfer DataFrame
        """
        # 計算出金計數（需從原始表格統計）
        crypto_out = (
            crypto_transfer[crypto_transfer["stype"] == "out"]
            .groupby("user_id").size()
            .reset_index(name="crypto_withdraw_count")
        )
        twd_dep = (
            twd_transfer[twd_transfer["stype"] == "in"]
            .groupby("user_id").size()
            .reset_index(name="twd_deposit_count")
        )
        twd_out = (
            twd_transfer[twd_transfer["stype"] == "out"]
            .groupby("user_id").size()
            .reset_index(name="twd_withdraw_count")
        )

        df = (
            mule_df
            .merge(crypto_out, on="user_id", how="left")
            .merge(twd_dep,    on="user_id", how="left")
            .merge(twd_out,    on="user_id", how="left")
            .fillna(0)
        )
        df["event_time"] = datetime.now(timezone.utc).isoformat()

        return self.ingest_dataframe(df, "bito-mule-behavior")

    def ingest_hop_features(self, hop_df: pd.DataFrame) -> dict:
        """
        從 feature_graph_hops() 或 Glue Job 輸出的 DataFrame 攝取 hops Feature Group。
        """
        df = hop_df.copy()
        df["event_time"] = datetime.now(timezone.utc).isoformat()
        # Boolean → int
        for col in ("is_direct_neighbor", "in_blacklist_network"):
            if col in df.columns:
                df[col] = df[col].astype(int)
        return self.ingest_dataframe(df, "bito-mule-hops")


# ══════════════════════════════════════════════════════════════════════════════
#  3. FeatureRetriever — 即時推論查詢 & 批次訓練查詢
# ══════════════════════════════════════════════════════════════════════════════

class FeatureRetriever:
    """
    即時推論：GetRecord（Online Store，DynamoDB，P99 延遲 < 10ms）
    批次訓練：Athena 查詢（Offline Store，S3）
    """

    # Offline Store Athena 表名稱（Glue Data Catalog）
    OFFLINE_TABLES = {
        "bito-mule-hops":     "bito_features.bito_mule_hops",
        "bito-mule-behavior": "bito_features.bito_mule_behavior",
    }

    def __init__(self, region: str = REGION):
        self.runtime = boto3.client(
            "sagemaker-featurestore-runtime", region_name=region
        )
        self.athena  = boto3.client("athena", region_name=region)
        self.region  = region

    def get_record(
        self,
        group_name: str,
        user_id: int,
        feature_names: Optional[list[str]] = None,
    ) -> dict[str, str]:
        """
        從 Online Store（DynamoDB）即時讀取單筆記錄。

        Parameters
        ----------
        group_name    : Feature Group 名稱
        user_id       : 用戶 ID
        feature_names : 指定要取回的特徵欄位（None = 全部）

        Returns
        -------
        dict: {feature_name: value_string}
        空 dict = 此用戶無特徵紀錄（冷啟動）
        """
        params = {
            "FeatureGroupName": group_name,
            "RecordIdentifierValueAsString": str(int(user_id)),
        }
        if feature_names:
            params["FeatureNames"] = feature_names

        try:
            resp = self.runtime.get_record(**params)
            return {
                f["FeatureName"]: f["ValueAsString"]
                for f in resp.get("Record", [])
            }
        except self.runtime.exceptions.ResourceNotFound:
            log.debug(f"[GetRecord] user_id={user_id} 在 {group_name} 中無紀錄。")
            return {}

    def batch_get_records(
        self,
        group_name: str,
        user_ids: list[int],
    ) -> list[dict]:
        """
        批次讀取多筆記錄（每次 API 呼叫最多 100 筆）。
        超過 100 筆自動分批。

        Returns
        -------
        list[dict]: 每個元素為 {feature_name: value_string}，
                    無紀錄的用戶以空 dict 填入（保持順序一致）
        """
        id_index   = {uid: i for i, uid in enumerate(user_ids)}
        results    = [{}] * len(user_ids)
        batch_size = 100

        for i in range(0, len(user_ids), batch_size):
            batch = user_ids[i:i + batch_size]

            resp = self.runtime.batch_get_record(
                Identifiers=[{
                    "FeatureGroupName":               group_name,
                    "RecordIdentifiersValueAsString": [str(int(uid)) for uid in batch],
                }]
            )

            for rec in resp.get("Records", []):
                feature_rec = rec.get("Record", [])
                uid = int(next(
                    f["ValueAsString"] for f in feature_rec
                    if f["FeatureName"] == FEATURE_GROUPS[group_name]["record_id"]
                ))
                parsed = {f["FeatureName"]: f["ValueAsString"] for f in feature_rec}
                if uid in id_index:
                    results[id_index[uid]] = parsed

        return results

    def get_offline_features(
        self,
        group_name: str,
        output_s3: str,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        透過 Athena 查詢 Offline Store，回傳 DataFrame。
        適合訓練時取得全量歷史特徵。

        Parameters
        ----------
        group_name : Feature Group 名稱
        output_s3  : Athena 查詢結果輸出路徑（e.g. s3://bucket/athena-results/）
        limit      : 回傳筆數上限（None = 全量）
        """
        table   = self.OFFLINE_TABLES[group_name]
        lim_sql = f"LIMIT {limit}" if limit else ""
        sql     = f"""
            SELECT *
            FROM   "{table}"
            WHERE  write_time = (
                SELECT MAX(write_time) FROM "{table}"
            )
            {lim_sql}
        """

        log.info(f"[Athena] 查詢 {table} ...")
        exec_id = self.athena.start_query_execution(
            QueryString=sql,
            ResultConfiguration={"OutputLocation": output_s3},
            WorkGroup="primary",
        )["QueryExecutionId"]

        # 等待完成
        while True:
            status = self.athena.get_query_execution(
                QueryExecutionId=exec_id
            )["QueryExecution"]["Status"]["State"]
            if status in ("SUCCEEDED", "FAILED", "CANCELLED"):
                break
            time.sleep(3)

        if status != "SUCCEEDED":
            raise RuntimeError(f"Athena 查詢失敗：{status}")

        # 讀取結果 CSV
        result_key = f"{output_s3.split('/', 3)[-1]}/{exec_id}.csv"
        s3_client  = boto3.client("s3", region_name=self.region)
        obj        = s3_client.get_object(
            Bucket=output_s3.split("/")[2],
            Key=result_key,
        )
        df = pd.read_csv(io.BytesIO(obj["Body"].read()))
        log.info(f"[Athena] 回傳 {len(df):,} 筆。")
        return df


# ══════════════════════════════════════════════════════════════════════════════
#  4. RealtimeFeaturePipeline — 推論服務整合入口
# ══════════════════════════════════════════════════════════════════════════════

class RealtimeFeaturePipeline:
    """
    推論時的特徵組裝管線。

    呼叫方式（在 train_sagemaker.py 或推論端點內）：
        pipeline = RealtimeFeaturePipeline()
        context  = pipeline.get_user_context(user_id=12345)
        # context 包含：
        #   "feature_vector"    : list[float] — 供 XGBoost 推論的特徵向量
        #   "kyc_level"         : int
        #   "min_retention_minutes": float
        #   "min_hops_to_blacklist": int
        #   "mule_risk_score"   : int
        #   ...（完整特徵字典）

    冷啟動處理：
        若用戶在 Feature Store 尚無紀錄（新用戶），
        pipeline 會填入合理的預設值並記錄警告。
    """

    # 各特徵欄位的冷啟動預設值
    _COLD_START_DEFAULTS: dict[str, float] = {
        "kyc_level":                 0,
        "min_retention_minutes":     9999.0,    # 無資金流轉紀錄，填超大值
        "high_speed_risk":           0,
        "unique_ip_count":           1,
        "ip_anomaly":                0,
        "total_twd_volume":          0.0,
        "volume_zscore":             0.0,
        "asymmetry_flag":            0,
        "min_hops_to_blacklist":     4,          # ISOLATED
        "is_direct_neighbor":        0,
        "blacklist_neighbor_count":  0,
        "in_blacklist_network":      0,
        "mule_risk_score":           0,
        "crypto_withdraw_count":     0,
        "twd_deposit_count":         0,
        "twd_withdraw_count":        0,
    }

    def __init__(self, region: str = REGION):
        self.retriever = FeatureRetriever(region=region)

    def get_user_context(self, user_id: int) -> dict:
        """
        即時讀取用戶的完整特徵向量，供推論引擎使用。

        兩次 GetRecord 呼叫（Online Store，DynamoDB）：
          1. bito-mule-behavior  — 行為特徵
          2. bito-mule-hops      — 圖跳轉特徵

        合計延遲估計：< 20ms（雙 DynamoDB 讀取，東京 Region）

        Returns
        -------
        dict：
          "feature_vector"  → list[float]，對應 INFERENCE_FEATURE_COLUMNS 的順序
          "raw_features"    → dict，原始特徵鍵值對
          "is_cold_start"   → bool，是否使用了預設值
          其餘 key           → 常用特徵的直接存取（供 xai_bedrock.py 使用）
        """
        # ── 並行查詢兩個 Feature Group ──────────────────────────────────
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=2) as pool:
            f_behavior = pool.submit(
                self.retriever.get_record, "bito-mule-behavior", user_id
            )
            f_hops = pool.submit(
                self.retriever.get_record, "bito-mule-hops", user_id
            )
            behavior_raw = f_behavior.result()
            hops_raw     = f_hops.result()

        is_cold_start = not behavior_raw and not hops_raw
        if is_cold_start:
            log.warning(f"[RealtimePipeline] 冷啟動：user_id={user_id} 無 Feature Store 紀錄。")

        # ── 合併原始特徵，填預設值 ──────────────────────────────────────
        raw = {**behavior_raw, **hops_raw}

        def _get(key: str, dtype=float) -> float:
            """從 raw 讀取，找不到則用冷啟動預設值。"""
            val = raw.get(key)
            if val is None:
                val = self._COLD_START_DEFAULTS.get(key, 0.0)
            try:
                return dtype(val)
            except (ValueError, TypeError):
                return self._COLD_START_DEFAULTS.get(key, 0.0)

        # ── 建構推論特徵向量（順序必須與訓練集完全一致） ──────────────
        feature_vector = [
            _get("kyc_level",                int),
            _get("min_retention_minutes",    float),
            _get("high_speed_risk",          int),
            _get("unique_ip_count",          int),
            _get("ip_anomaly",               int),
            _get("total_twd_volume",         float),
            _get("volume_zscore",            float),
            _get("asymmetry_flag",           int),
            _get("min_hops_to_blacklist",    int),
            _get("is_direct_neighbor",       int),
            _get("blacklist_neighbor_count", int),
            _get("in_blacklist_network",     int),
            _get("mule_risk_score",          int),
            _get("crypto_withdraw_count",    int),
            _get("twd_deposit_count",        int),
            _get("twd_withdraw_count",       int),
        ]

        # ── 建構 context dict（供 xai_bedrock.py 使用） ──────────────
        context = {
            "feature_vector":            feature_vector,
            "raw_features":              raw,
            "is_cold_start":             is_cold_start,
            # 常用欄位直接暴露（xai_bedrock._build_user_prompt 使用）
            "kyc_level":                 int(_get("kyc_level")),
            "min_retention_minutes":     float(_get("min_retention_minutes")),
            "unique_ip_count":           int(_get("unique_ip_count")),
            "min_hops_to_blacklist":     int(_get("min_hops_to_blacklist")),
            "is_direct_neighbor":        bool(int(_get("is_direct_neighbor"))),
            "blacklist_neighbor_count":  int(_get("blacklist_neighbor_count")),
            "in_blacklist_network":      bool(int(_get("in_blacklist_network"))),
            "mule_risk_score":           int(_get("mule_risk_score")),
            "crypto_withdraw_count":     int(_get("crypto_withdraw_count")),
            "twd_withdraw_count":        int(_get("twd_withdraw_count")),
        }

        return context

    def get_batch_context(
        self,
        user_ids: list[int],
        max_workers: int = 10,
    ) -> list[dict]:
        """
        批次讀取多個用戶的特徵向量（推論服務批次請求）。

        使用 ThreadPoolExecutor 並行呼叫 get_user_context，
        適合一次推論 10~1000 個用戶的場景。
        """
        from concurrent.futures import ThreadPoolExecutor

        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(self.get_user_context, uid) for uid in user_ids]
            for fut in futures:
                try:
                    results.append(fut.result())
                except Exception as e:
                    log.error(f"get_user_context 失敗：{e}")
                    results.append({"feature_vector": [self._COLD_START_DEFAULTS.get(f, 0.0)
                                                        for f in INFERENCE_FEATURE_COLUMNS],
                                    "is_cold_start": True})
        return results


# ══════════════════════════════════════════════════════════════════════════════
#  CLI 入口
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SageMaker Feature Store 管理工具")
    parser.add_argument("--action", choices=["create", "describe", "delete",
                                             "ingest", "get", "test"],
                        required=True)
    parser.add_argument("--group",   default=None, help="Feature Group 名稱（留空=全部）")
    parser.add_argument("--csv",     default=None, help="攝取用 CSV 路徑")
    parser.add_argument("--user-id", default=None, type=int, help="查詢用戶 ID")
    parser.add_argument("--role",    default=ROLE_ARN)
    args = parser.parse_args()

    mgr       = FeatureGroupManager(role_arn=args.role)
    ingester  = FeatureIngester()
    retriever = FeatureRetriever()

    if args.action == "create":
        if args.group:
            mgr.create(args.group)
        else:
            mgr.create_all()

    elif args.action == "describe":
        for name in ([args.group] if args.group else FEATURE_GROUPS):
            desc = mgr.describe(name)
            print(f"\n{name}：{desc['FeatureGroupStatus']}")
            print(f"  Online  : {'啟用' if desc.get('OnlineStoreConfig') else '停用'}")
            print(f"  Offline : {desc.get('OfflineStoreConfig', {}).get('S3StorageConfig', {}).get('S3Uri', 'N/A')}")

    elif args.action == "delete":
        if not args.group:
            print("刪除操作需指定 --group。")
            return
        mgr.delete(args.group)

    elif args.action == "ingest":
        if not args.csv or not args.group:
            print("攝取操作需指定 --csv 和 --group。")
            return
        df = pd.read_csv(args.csv)
        result = ingester.ingest_dataframe(df, args.group)
        print(f"攝取完成：{result}")

    elif args.action == "get":
        if not args.user_id:
            print("查詢操作需指定 --user-id。")
            return
        pipeline = RealtimeFeaturePipeline()
        ctx      = pipeline.get_user_context(args.user_id)
        print(f"\nuser_id={args.user_id} 特徵：")
        for k, v in ctx.items():
            if k not in ("feature_vector", "raw_features"):
                print(f"  {k:<28} = {v}")
        print(f"\n特徵向量（{len(ctx['feature_vector'])} 維）：")
        for name, val in zip(INFERENCE_FEATURE_COLUMNS, ctx["feature_vector"]):
            print(f"  {name:<28} = {val}")

    elif args.action == "test":
        log.info("執行 Feature Store 連通性測試...")
        for gname in FEATURE_GROUPS:
            try:
                desc = mgr.describe(gname)
                log.info(f"  ✓ {gname}：{desc['FeatureGroupStatus']}")
            except Exception as e:
                log.warning(f"  ✗ {gname}：{e}")


if __name__ == "__main__":
    main()
