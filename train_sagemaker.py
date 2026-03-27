"""
SageMaker XGBoost 訓練與推論流程
命題文件：2026 去偽存真 AI 全民偵查黑客松

提交格式（needs to match 需提交的csv格式.csv）：
    user_id, status
    23, 1
    5645, 0
    ...
    status: 0=正常, 1=黑名單

資料來源（via BitoPro API）：
    train_label   — 51,017 筆，含 user_id + status（0/1）
    predict_label — 12,753 筆，含 user_id，status 待預測

MLOps 生命周期（Model Registry）：
    訓練完成後，若 F1-score 達到以下門檻，自動執行版本註冊：
      F1 ≥ 0.90  → Approved       （自動核准，可直接部署）
      F1 ≥ 0.80  → PendingManualApproval（等待風控主管審核）
      F1 < 0.80  → 不註冊（記錄原因至 CloudWatch，發送 SNS 告警）

    Model Package Group：bito-mule-detection-registry
    每個版本附帶：F1 / Precision / Recall / scale_pos_weight / 超參數快照
"""

import io
import json
import logging
import os
import time
from datetime import datetime, timezone
import boto3
import numpy as np
import pandas as pd
import sagemaker
from botocore.exceptions import ClientError
from sagemaker.inputs import TrainingInput
from sagemaker.tuner import (
    HyperparameterTuner,
    IntegerParameter,
    ContinuousParameter,
)
from sagemaker.xgboost import XGBoost
from bito_data_manager import BitoDataManager

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

# ── 設定區（依實際環境修改） ────────────────────────────────────────────────

S3_BUCKET      = os.environ.get("S3_BUCKET", "your-hackathon-bucket")
S3_PREFIX      = "bito-mule-detection"

# ── R-1 修正：Region 合規鎖定（與所有其他腳本保持一致） ─────────────────────
# 嚴禁部署至 us-east-1 / us-west-2 以外的 Region（競賽禁令）。
# 此驗證在模組載入時即執行，確保 SageMaker Job 啟動前就快速失敗。
_ALLOWED_REGIONS: frozenset[str] = frozenset({"us-east-1", "us-west-2"})

def _validate_region(region: str) -> str:
    """Region 合規檢核：若非 us-east-1 或 us-west-2 則立即拋出 ValueError。"""
    if region not in _ALLOWED_REGIONS:
        raise ValueError(
            f"[Region 合規] AWS_DEFAULT_REGION='{region}' 不符合競賽規定。\n"
            f"僅允許 {sorted(_ALLOWED_REGIONS)}。\n"
            "請執行：export AWS_DEFAULT_REGION=us-east-1"
        )
    return region

REGION         = _validate_region(os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))

# ── 自動取得 Notebook 的 IAM 角色（優先於環境變數） ──────────────────────────
# 若環境變數未設定，則使用 SageMaker Session 的預設角色（Notebook 本身的角色）
# 這避免了 trust relationship 問題，因為 Notebook 角色已經被 SageMaker 信任
def _get_sagemaker_role():
    """自動取得 SageMaker 執行角色。優先使用環境變數，否則使用 Notebook 角色。"""
    env_role = os.environ.get("SAGEMAKER_ROLE_ARN")
    if env_role and env_role != "arn:aws:iam::ACCOUNT_ID:role/SageMakerRole":
        return env_role
    
    # 若環境變數未設定或為預設值，使用 SageMaker Session 的角色
    try:
        session = sagemaker.Session()
        role = session.get_execution_role()
        if role:
            log.info(f"[IAM] 使用 Notebook 的 IAM 角色：{role}")
            return role
    except Exception as e:
        log.warning(f"[IAM] 無法自動取得 Notebook 角色：{e}")
    
    # 最後才使用環境變數的預設值
    return env_role or "arn:aws:iam::ACCOUNT_ID:role/SageMakerRole"

ROLE_ARN       = _get_sagemaker_role()
INSTANCE_TYPE  = "ml.m5.xlarge"  # 4 vCPU / 16 GB — XGBoost 訓練/推論/批次轉換的標準規格
OUTPUT_CSV     = "submission.csv"            # 最終提交檔
PROB_CSV       = "submission_with_prob.csv"  # 含機率的詳細版本

# ── XGBoost 固定超參數（不參與調優） ──────────────────────────────────────────
# max_depth / eta / gamma 由 HyperparameterTuner 決定，不在此設定
BASE_HYPERPARAMS: dict = {
    "objective":           "binary:logistic",
    "eval_metric":         "logloss",        # 內建 metric（feval=F1 才是 HPO 主角）
    "num_round":           500,              # 上限；early stopping 會提前截止
    "early_stopping_rounds": 30,            # 連續 30 輪 F1 無改善即停止此 job
    "min_child_weight":    5,
    "subsample":           0.8,
    "colsample_bytree":    0.8,
    "seed":                42,
}

# ── HPO 搜尋空間 ───────────────────────────────────────────────────────────────
HPO_PARAM_RANGES: dict = {
    "max_depth": IntegerParameter(3, 10),
    "eta":       ContinuousParameter(0.01, 0.30),
    "gamma":     ContinuousParameter(0.0,  5.0),
}

# ── HPO 目標指標（對應 train_xgboost_script.py 的 print 格式） ──────────────────
HPO_METRIC_NAME   = "validation:f1_score"
HPO_METRIC_REGEX  = r"\[CV\] f1_score: ([\d.]+)"
HPO_OBJECTIVE     = "Maximize"

# ── HPO 作業設定 ───────────────────────────────────────────────────────────────
HPO_MAX_JOBS         = 10   # 總共嘗試的超參數組合數（Hackathon 預算控制）
HPO_MAX_PARALLEL     = 2    # 同時並行的訓練 job 數（節省費用）
HPO_STRATEGY         = "Bayesian"      # Bayesian / Random / Hyperband
HPO_EARLY_STOP_TYPE  = "Auto"         # Auto = SageMaker 自動終止表現差的 job

# ══════════════════════════════════════════════════════════════════════════════
#  Instance Type 合規檢核 & 訓練資源配額守衛
# ══════════════════════════════════════════════════════════════════════════════

# 封鎖清單：競賽資源限制禁用的 GPU/加速器實例前綴
_GPU_INSTANCE_PREFIXES: tuple[str, ...] = (
    "ml.p3",    # NVIDIA V100 — 高成本 GPU
    "ml.p4d",   # NVIDIA A100 (multi-GPU)
    "ml.p4de",  # NVIDIA A100 EFA (multi-GPU)
    "ml.p5",    # NVIDIA H100
    "ml.g4dn",  # NVIDIA T4
    "ml.g5",    # NVIDIA A10G
    "ml.g5g",   # AWS Graviton2 + NVIDIA T4G
    "ml.trn1",  # AWS Trainium
    "ml.inf2",  # AWS Inferentia2
)


def _validate_instance_type(instance_type: str, *, allow_gpu: bool = False) -> str:
    """
    Instance Type 合規檢核。

    預設封鎖所有 GPU / 加速器實例，避免觸發競賽資源配額限制並防止意外高費用。
    若確認需要 GPU（例如深度學習任務），需明確傳入 allow_gpu=True。

    Args:
        instance_type: SageMaker instance type 字串（如 'ml.m5.xlarge'）
        allow_gpu:     True = 允許 GPU 實例（需明確聲明，防止誤用）

    Returns:
        原樣回傳 instance_type（檢核通過）

    Raises:
        ValueError: instance_type 屬於 GPU 實例且 allow_gpu=False
    """
    if not allow_gpu:
        it_lower = instance_type.lower()
        for prefix in _GPU_INSTANCE_PREFIXES:
            if it_lower.startswith(prefix):
                raise ValueError(
                    f"[ResourceCheck] 拒絕使用 GPU/加速器實例 '{instance_type}'。\n"
                    f"  競賽資源限制：請改用 CPU 實例（建議 ml.m5.xlarge 或 ml.c5.xlarge）。\n"
                    f"  若確認此任務需要 GPU，請在呼叫處明確傳入 allow_gpu=True。"
                )
    log.debug("[ResourceCheck] instance_type=%s 通過檢核（allow_gpu=%s）",
              instance_type, allow_gpu)
    return instance_type


# 在模組載入時立即對全域常數做檢核，確保任何導入此模組的程式也受到保護
INSTANCE_TYPE = _validate_instance_type(INSTANCE_TYPE)


class ResourceCheck:
    """
    SageMaker 訓練配額守衛。

    在提交 Training Job 或 HPO Tuner 前，透過 list_training_jobs 查詢
    帳號中目前 InProgress 的 Job 數量。若提交後預計超過上限，立即拋出
    RuntimeError，防止觸發 Account-level Service Quota 限制。

    使用方式：
        rc = ResourceCheck(sm_client, max_running=2)
        rc.assert_can_submit()                  # 提交 1 個 Job 前
        rc.assert_can_submit(n_new_jobs=4)      # HPO 並行 4 個 Job 前

    建議在每次呼叫 estimator.fit() 或 tuner.fit() 前執行。
    """

    MAX_RUNNING_DEFAULT = 2   # 帳號並行 Training Job 保守上限

    def __init__(self, sm_client, max_running: int = MAX_RUNNING_DEFAULT):
        """
        Args:
            sm_client:   boto3 SageMaker client
            max_running: 允許同時存在的最大 InProgress Training Job 數
        """
        self._sm          = sm_client
        self._max_running = max_running

    def count_running_jobs(self) -> int:
        """
        查詢帳號中目前 InProgress 的 Training Job 數量。

        使用分頁器（paginator）確保大量 Job 時也能正確計數。
        """
        count     = 0
        paginator = self._sm.get_paginator("list_training_jobs")
        for page in paginator.paginate(StatusEquals="InProgress"):
            count += len(page.get("TrainingJobSummaries", []))
        log.info("[ResourceCheck] 目前 InProgress Training Job 數：%d", count)
        return count

    def assert_can_submit(self, n_new_jobs: int = 1) -> None:
        """
        斷言提交 n_new_jobs 個 Job 後不會超過 max_running 上限。

        Args:
            n_new_jobs: 本次預計新增的 Job 數
                        （HPO 時傳入 max_parallel；單一訓練傳入預設值 1）

        Raises:
            RuntimeError: 若 currently_running + n_new_jobs > max_running
        """
        running = self.count_running_jobs()
        after   = running + n_new_jobs

        if after > self._max_running:
            raise RuntimeError(
                f"[ResourceCheck] ⛔ 配額不足，拒絕提交 Training Job。\n"
                f"  目前 InProgress Job 數 : {running}\n"
                f"  本次預計提交 Job 數    : {n_new_jobs}\n"
                f"  提交後總數             : {after}  >  上限 {self._max_running}\n"
                f"  請等待現有 Job 完成後重試，或調高 --max-running 參數。\n"
                f"  查詢現有 Job：\n"
                f"    aws sagemaker list-training-jobs --status-equals InProgress"
            )

        log.info(
            "[ResourceCheck] ✓ 配額充足：InProgress %d + 新增 %d = %d（上限 %d）",
            running, n_new_jobs, after, self._max_running,
        )

# ── Model Registry 設定 ────────────────────────────────────────────────────────
MODEL_PKG_GROUP      = os.environ.get("MODEL_PKG_GROUP",
                                      "bito-mule-detection-registry")
MODEL_PKG_DESC       = "幣託人頭戶偵測 XGBoost 模型 — AWS Hackathon 2026"
F1_AUTO_APPROVE      = 0.90   # F1 ≥ 0.90 自動核准部署
F1_REGISTER_MIN      = 0.80   # F1 ≥ 0.80 進入 PendingManualApproval
SNS_ALERT_ARN        = os.environ.get("SNS_TOPIC_ARN", "")   # F1 不足時告警


# ── 1. 特徵工程 ─────────────────────────────────────────────────────────────

# 訓練 CSV 欄位的唯一權威定義（順序必須與 train_xgboost_script.FEATURE_INDEX
# 及 feature_store.INFERENCE_FEATURE_COLUMNS 完全一致）。
# ⚠ 若新增/移除/重排任何特徵，必須同步修改上述兩個常數。
CANONICAL_FEATURE_COLS: list[str] = [
    # ── 用戶基本屬性 ──────────────────────────────────────────────────────
    "kyc_level",                  # idx=0

    # ── 特徵①：資金滯留時間 ────────────────────────────────────────────
    "min_retention_minutes",      # idx=1
    "retention_event_count",      # idx=2
    "high_speed_risk",            # idx=3

    # ── 特徵②：IP 異常跳動 ─────────────────────────────────────────────
    "unique_ip_count",            # idx=4
    "ip_anomaly",                 # idx=5

    # ── 特徵③：量能不對稱 ──────────────────────────────────────────────
    "total_twd_volume",           # idx=6
    "volume_zscore",              # idx=7
    "asymmetry_flag",             # idx=8

    # ── 特徵④：圖跳轉（BFS + in_blacklist_network） ─────────────────
    "min_hops_to_blacklist",      # idx=9
    "is_direct_neighbor",         # idx=10
    "blacklist_neighbor_count",   # idx=11
    "in_blacklist_network",       # idx=12

    # ── 綜合風險評分 ────────────────────────────────────────────────────
    "mule_risk_score",            # idx=13

    # ── 交易計數（build_features 補充） ─────────────────────────────
    "twd_deposit_count",          # idx=14
    "twd_withdraw_count",         # idx=15
    "crypto_deposit_count",       # idx=16
    "crypto_withdraw_count",      # idx=17

    # ── 時間模式 ────────────────────────────────────────────────────────
    "night_tx_ratio",             # idx=18
]


def build_features(
    manager: BitoDataManager,
    known_blacklist: Optional[set] = None,
) -> pd.DataFrame:
    """
    載入所有資料表，執行特徵提取，回傳以 user_id 為主鍵的特徵寬表。

    Parameters
    ----------
    manager          : BitoDataManager 實例
    known_blacklist  : 已知黑名單 user_id 集合（int）。
                       必須在呼叫此函數前由 train_label 提取並傳入，
                       否則圖跳轉特徵（idx 9-12）全部為 0，模型無法學習
                       關鍵的網絡結構資訊。

    Returns
    -------
    DataFrame，欄位 = ["user_id"] + CANONICAL_FEATURE_COLS
    欄位順序固定，與 CANONICAL_FEATURE_COLS 完全一致。
    """
    import logging
    log = logging.getLogger(__name__)

    if known_blacklist is None:
        log.warning(
            "[build_features] ⚠ known_blacklist=None！"
            "圖跳轉特徵（min_hops_to_blacklist / in_blacklist_network 等）"
            "將全部為 0，嚴重影響模型效果。"
            "請先載入 train_label 並傳入 known_blacklist。"
        )

    tables          = manager.load_all()
    users           = tables.get("users")
    twd_transfer    = tables.get("twd_transfer")
    crypto_transfer = tables.get("crypto_transfer")
    trades          = tables.get("trades")

    # ── 四大特徵提取（含 in_blacklist_network） ──────────────────────────
    if (users is not None and len(users) > 0 and
        twd_transfer is not None and len(twd_transfer) > 0 and
        crypto_transfer is not None and len(crypto_transfer) > 0):
        features = manager.extract_mule_features(
            users, twd_transfer, crypto_transfer, trades,
            known_blacklist=known_blacklist,   # ← 關鍵修正：傳入黑名單
        )
    else:
        log.warning("缺少必要表或表為空，使用最小特徵集（僅 user_id）")
        if users is not None and len(users) > 0:
            features = users[["user_id"]].drop_duplicates()
        else:
            raise ValueError("無法載入任何資料表，請確認資料來源（CSV 路徑或 API）")

    # ── 補充計數特徵（build_features 追加，不在 extract_mule_features 內） ──
    # 台幣入金次數 & 出金次數
    if twd_transfer is not None and len(twd_transfer) > 0:
        twd_in  = (twd_transfer[twd_transfer["kind"].astype(str) == "0"]
                   .groupby("user_id").size().reset_index(name="twd_deposit_count"))
        twd_out = (twd_transfer[twd_transfer["kind"].astype(str) == "1"]
                   .groupby("user_id").size().reset_index(name="twd_withdraw_count"))
        features = features.merge(twd_in,  on="user_id", how="left")
        features = features.merge(twd_out, on="user_id", how="left")

    # 虛幣入金次數 & 出金次數
    if crypto_transfer is not None and len(crypto_transfer) > 0:
        cr_in  = (crypto_transfer[crypto_transfer["kind"].astype(str) == "0"]
                  .groupby("user_id").size().reset_index(name="crypto_deposit_count"))
        cr_out = (crypto_transfer[crypto_transfer["kind"].astype(str) == "1"]
                  .groupby("user_id").size().reset_index(name="crypto_withdraw_count"))
        features = features.merge(cr_in,  on="user_id", how="left")
        features = features.merge(cr_out, on="user_id", how="left")

    # 深夜交易比例（22:00–06:00）
    if twd_transfer is not None and len(twd_transfer) > 0:
        _twd = twd_transfer.copy()
        _twd["hour"] = pd.to_datetime(_twd["created_at"], errors="coerce").dt.hour
        night_ratio = (
            _twd.assign(is_night=_twd["hour"].between(22, 23) | _twd["hour"].between(0, 6))
                .groupby("user_id")["is_night"]
                .mean()
                .reset_index(name="night_tx_ratio")
        )
        features = features.merge(night_ratio, on="user_id", how="left")

    # ── 確保所有 CANONICAL_FEATURE_COLS 欄位都存在，缺的補 0 ────────────────
    for col in CANONICAL_FEATURE_COLS:
        if col not in features.columns:
            features[col] = 0
            log.debug("[build_features] 補入缺失欄位 %s = 0", col)

    # ── 填補 NaN ──────────────────────────────────────────────────────────
    features[CANONICAL_FEATURE_COLS] = features[CANONICAL_FEATURE_COLS].fillna(0)

    return features


# ── 2. 標籤合併 & 計算 scale_pos_weight ─────────────────────────────────────

def build_hyperparams(train_label: pd.DataFrame) -> dict:
    """依不平衡比例動態計算 scale_pos_weight。"""
    neg = (train_label["status"] == 0).sum()
    pos = (train_label["status"] == 1).sum()
    ratio = round(neg / pos, 4)
    print(f"[imbalance] negative={neg:,}  positive={pos:,}  scale_pos_weight={ratio}")
    return {**BASE_HYPERPARAMS, "scale_pos_weight": ratio}


# ── 3. SageMaker HyperparameterTuner ────────────────────────────────────────

def run_hyperparameter_tuning(
    train_uri: str,
    output_uri: str,
    scale_pos_weight: float,
    session: sagemaker.Session,
    *,
    max_jobs: int        = HPO_MAX_JOBS,
    max_parallel: int    = HPO_MAX_PARALLEL,
    strategy: str        = HPO_STRATEGY,
    wait: bool           = True,
) -> tuple[str, dict]:
    """
    啟動 SageMaker HyperparameterTuner，搜尋最佳 max_depth / eta / gamma。

    目標函數：最大化 5-Fold CV F1-Score（由 train_xgboost_script.py 輸出）。

    Early Stopping 雙層設計：
      ① XGBoost 層（job 內部）：
            early_stopping_rounds=30 — 單個 job 連續 30 輪驗證 F1 無改善即中斷，
            防止在少量黑名單樣本（正樣本稀少）上過擬合。
      ② SageMaker Tuner 層（job 之間）：
            early_stopping_type="Auto" — Tuner 自動分析各 job 的學習曲線，
            在預測結果不會超越當前最佳時，提前終止整個 job，節省計算資源。

    Parameters
    ----------
    train_uri        : s3://bucket/key/train/  格式，訓練資料路徑
    output_uri       : s3://bucket/key/output/ 格式，模型輸出路徑
    scale_pos_weight : 負/正樣本比例（由 build_hyperparams 計算）
    session          : SageMaker Session 物件
    max_jobs         : 總嘗試組合數（預設 20）
    max_parallel     : 同時並行 job 數（預設 4）
    strategy         : 搜尋策略 Bayesian / Random / Hyperband（預設 Bayesian）
    wait             : 是否等待所有 job 完成（預設 True）

    Returns
    -------
    (best_job_name, best_hyperparams)
      best_job_name   : 最佳訓練 job 的名稱（可傳入 download_model.py）
      best_hyperparams: 最佳超參數 dict（含 scale_pos_weight 等固定參數）
    """
    # ── 固定超參數（注入 scale_pos_weight，其餘三項由 Tuner 注入） ──────────
    fixed_hyperparams = {
        **BASE_HYPERPARAMS,
        "scale_pos_weight": scale_pos_weight,
    }

    # ── 定義 Estimator（使用自訂訓練腳本） ───────────────────────────────
    estimator = XGBoost(
        entry_point="train_xgboost_script.py",  # 自訂腳本，負責輸出 F1
        source_dir=os.path.dirname(os.path.abspath(__file__)),
        framework_version="1.7-1",
        role=ROLE_ARN,
        instance_count=1,
        instance_type=INSTANCE_TYPE,
        output_path=output_uri,
        hyperparameters=fixed_hyperparams,
        sagemaker_session=session,
        # 讓 SageMaker 知道要從 stdout 捕捉 F1 指標
        metric_definitions=[
            {
                "Name":  HPO_METRIC_NAME,
                "Regex": HPO_METRIC_REGEX,
            }
        ],
    )

    # ── 定義 HyperparameterTuner ──────────────────────────────────────────
    tuner = HyperparameterTuner(
        estimator=estimator,

        # 目標：最大化 F1（來自 train_xgboost_script.py 的 stdout）
        objective_metric_name=HPO_METRIC_NAME,
        objective_type=HPO_OBJECTIVE,           # "Maximize"

        # 搜尋空間
        hyperparameter_ranges=HPO_PARAM_RANGES,

        # 作業設定
        max_jobs=max_jobs,
        max_parallel_jobs=max_parallel,
        strategy=strategy,

        # SageMaker 層 Early Stopping（在 job 之間）
        early_stopping_type=HPO_EARLY_STOP_TYPE,   # "Auto"

        # 讓 Tuner 知道要監控哪個指標（已在 estimator 的 metric_definitions 定義）
        base_tuning_job_name="bito-mule-hpo",
    )

    # ── 啟動調優作業 ──────────────────────────────────────────────────────
    print(f"\n[HPO] 啟動 HyperparameterTuner")
    print(f"  策略    : {strategy}")
    print(f"  總 Job  : {max_jobs}  並行: {max_parallel}")
    print(f"  目標    : {HPO_OBJECTIVE} {HPO_METRIC_NAME}")
    print(f"  搜尋空間:")
    print(f"    max_depth : IntegerParameter(3, 10)")
    print(f"    eta       : ContinuousParameter(0.01, 0.30)")
    print(f"    gamma     : ContinuousParameter(0.0,  5.0)")
    print(f"  XGBoost Early Stopping : {BASE_HYPERPARAMS['early_stopping_rounds']} 輪")
    print(f"  Tuner   Early Stopping : {HPO_EARLY_STOP_TYPE}")

    # ── ResourceCheck：提交前確認帳號配額充足 ───────────────────────────────
    # HPO 會同時啟動 max_parallel 個 Training Job，需以並行數計算配額消耗
    ResourceCheck(
        session.boto_session.client("sagemaker"),
        max_running=ResourceCheck.MAX_RUNNING_DEFAULT,
    ).assert_can_submit(n_new_jobs=max_parallel)

    tuner.fit(
        inputs={"train": TrainingInput(train_uri, content_type="text/csv")},
        wait=wait,
        logs=True,
    )

    if not wait:
        print(f"\n[HPO] 作業已提交（背景執行）。使用以下指令查詢進度：")
        print(f"  aws sagemaker describe-hyper-parameter-tuning-job \\")
        print(f"    --hyper-parameter-tuning-job-name {tuner.latest_tuning_job.name}")
        return tuner.latest_tuning_job.name, {}

    # ── 取得最佳 Job ──────────────────────────────────────────────────────
    best_job       = tuner.best_training_job()
    best_job_name  = best_job if isinstance(best_job, str) else best_job.name

    sm_client      = session.boto_session.client("sagemaker")
    best_desc      = sm_client.describe_training_job(TrainingJobName=best_job_name)
    best_hparams   = best_desc["HyperParameters"]
    best_metric    = best_desc["FinalMetricDataList"]

    print(f"\n[HPO 結果]")
    print(f"  最佳 Job     : {best_job_name}")
    for m in best_metric:
        if m["MetricName"] == HPO_METRIC_NAME:
            print(f"  最佳 F1-Score: {m['Value']:.6f}")
    print(f"  最佳超參數   :")
    for k, v in best_hparams.items():
        print(f"    {k}: {v}")

    # 合併固定超參數 + 最佳搜尋超參數
    # SageMaker HPO 回傳值全為字串，需依語義轉型
    _INT_PARAMS   = {"max_depth", "num_round", "early_stopping_rounds",
                     "min_child_weight", "seed"}
    _FLOAT_PARAMS = {"eta", "gamma", "subsample", "colsample_bytree",
                     "scale_pos_weight", "fn_weight", "fp_weight",
                     "label_smoothing", "adv_strength", "adv_ratio"}
    merged = {**fixed_hyperparams}
    for k, v in best_hparams.items():
        if k in _INT_PARAMS:
            merged[k] = int(float(v))        # float() 先處理 "500.0" 格式
        elif k in _FLOAT_PARAMS:
            merged[k] = float(v)
        else:
            try:
                merged[k] = float(v)
            except (ValueError, TypeError):
                merged[k] = v

    return best_job_name, merged


# ── 4. S3 上傳工具 ───────────────────────────────────────────────────────────

def upload_df_to_s3(
    df: pd.DataFrame,
    s3_client,
    bucket: str,
    key: str,
    header: bool = False,
) -> str:
    """
    將 DataFrame 以 CSV 格式上傳至 S3，回傳 s3:// URI。

    SageMaker XGBoost 內建容器訓練格式要求：
      - 第一欄為 label
      - 無 header 列
      - 無 index
    """
    buf = io.StringIO()
    df.to_csv(buf, index=False, header=header)
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=buf.getvalue().encode("utf-8"),
    )
    uri = f"s3://{bucket}/{key}"
    print(f"  uploaded → {uri}")
    return uri


# ── 5. SageMaker Model Registry ──────────────────────────────────────────────

class ModelRegistryManager:
    """
    SageMaker Model Registry 生命周期管理。

    職責：
      1. ensure_package_group()   — 建立 Model Package Group（冪等）
      2. register()               — 將訓練完成的模型版本注入 Registry
      3. _extract_metrics()       — 從 Training Job FinalMetricDataList 擷取 F1
      4. _notify_low_f1()         — F1 不達標時發送 SNS 告警

    MLOps 合規設計：
      - 每個 ModelPackage 記錄：超參數快照、訓練 Job 名稱、資料 S3 URI、F1/P/R
      - 審核狀態機：PendingManualApproval → Approved / Rejected（由風控主管操作）
      - 自動核准條件：F1 ≥ F1_AUTO_APPROVE（0.90），減少人工瓶頸
      - 所有操作記錄至 CloudWatch（boto3 logging）
    """

    def __init__(self, region: str = REGION):
        sess           = boto3.Session(region_name=region)
        self.sm        = sess.client("sagemaker")
        self.sns       = sess.client("sns") if SNS_ALERT_ARN else None
        self.region    = region
        self.account   = boto3.client("sts", region_name=region) \
                               .get_caller_identity()["Account"]

    # ── 公開方法 ──────────────────────────────────────────────────────────────

    def ensure_package_group(
        self,
        group_name: str = MODEL_PKG_GROUP,
    ) -> str:
        """
        建立 Model Package Group（已存在則略過）。
        回傳 Group ARN。
        """
        try:
            self.sm.create_model_package_group(
                ModelPackageGroupName=group_name,
                ModelPackageGroupDescription=MODEL_PKG_DESC,
                Tags=[
                    {"Key": "Project",     "Value": "bito-mule-detection"},
                    {"Key": "Environment", "Value": "hackathon"},
                    {"Key": "ManagedBy",   "Value": "train_sagemaker.py"},
                ],
            )
            log.info(f"Model Package Group 已建立：{group_name}")
        except self.sm.exceptions.ConflictException:
            log.info(f"Model Package Group 已存在（略過）：{group_name}")

        arn = (
            f"arn:aws:sagemaker:{self.region}:{self.account}"
            f":model-package-group/{group_name}"
        )
        return arn

    def register(
        self,
        model_data:   str,
        job_name:     str,
        hyperparams:  dict,
        train_uri:    str,
        f1_score:     float,
        precision:    float = 0.0,
        recall:       float = 0.0,
        group_name:   str = MODEL_PKG_GROUP,
    ) -> dict | None:
        """
        將模型版本注入 SageMaker Model Registry。

        Parameters
        ----------
        model_data  : s3://bucket/.../model.tar.gz
        job_name    : SageMaker Training Job 名稱
        hyperparams : 訓練時使用的超參數 dict
        train_uri   : 訓練資料 S3 URI（供稽核追蹤）
        f1_score    : 驗證集 F1（決定核准狀態）
        precision   : 驗證集 Precision（可選，供 Registry 展示）
        recall      : 驗證集 Recall（可選）
        group_name  : Model Package Group 名稱

        Returns
        -------
        dict  — create_model_package 的完整回應，或 None（F1 不足）
        """
        if f1_score < F1_REGISTER_MIN:
            msg = (
                f"F1={f1_score:.4f} 低於最低門檻 {F1_REGISTER_MIN}，"
                f"模型版本不予註冊。Training Job: {job_name}"
            )
            log.warning(msg)
            self._notify_low_f1(f1_score, job_name, msg)
            return None

        # 核准狀態：F1 ≥ 0.90 自動核准，否則等待人工審查
        approval_status = (
            "Approved" if f1_score >= F1_AUTO_APPROVE
            else "PendingManualApproval"
        )

        # 整理超參數快照（確保可序列化）
        hp_snapshot = {k: str(v) for k, v in hyperparams.items()}

        pkg_input = {
            "ModelPackageGroupName": group_name,
            "ModelPackageDescription": (
                f"XGBoost 人頭戶偵測模型 | Job: {job_name} | "
                f"F1={f1_score:.4f} | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
            ),
            "ModelApprovalStatus": approval_status,

            # ── 模型工件 ──────────────────────────────────────────────────
            "InferenceSpecification": {
                "Containers": [
                    {
                        "Image": self._xgboost_image_uri(),
                        "ModelDataUrl": model_data,
                        "Environment": {
                            "SAGEMAKER_PROGRAM":      "train_xgboost_script.py",
                            "SAGEMAKER_SUBMIT_DIRECTORY": model_data,
                        },
                    }
                ],
                "SupportedContentTypes":      ["text/csv"],
                "SupportedResponseMIMETypes":  ["text/csv"],
                "SupportedTransformInstanceTypes":   [INSTANCE_TYPE],
                "SupportedRealtimeInferenceInstanceTypes": [INSTANCE_TYPE],
            },

            # ── 模型品質指標（供 Model Registry 比較版本） ─────────────────
            "ModelMetrics": {
                "ModelQuality": {
                    "Statistics": {
                        "ContentType": "application/json",
                        "S3Uri": self._upload_metrics_json(
                            job_name, f1_score, precision, recall,
                        ),
                    }
                }
            },

            # ── 可追溯性元資料（MLOps 合規要求） ───────────────────────────
            "MetadataProperties": {
                "GeneratedBy":  job_name,
                "ProjectId":    "bito-mule-detection",
                "Repository":   "github.com/team/bito-hackathon",
                "CommitId":     os.environ.get("GIT_COMMIT", "unknown"),
            },

            "CustomerMetadataProperties": {
                "training_job":        job_name,
                "training_data_uri":   train_uri,
                "f1_score":            f"{f1_score:.6f}",
                "precision":           f"{precision:.6f}",
                "recall":              f"{recall:.6f}",
                "scale_pos_weight":    str(hyperparams.get("scale_pos_weight", "")),
                "max_depth":           str(hyperparams.get("max_depth", "")),
                "eta":                 str(hyperparams.get("eta", "")),
                "gamma":               str(hyperparams.get("gamma", "")),
                "registered_at":       datetime.now(timezone.utc).isoformat(),
            },

            "Tags": [
                {"Key": "F1Score",      "Value": f"{f1_score:.4f}"},
                {"Key": "ApprovalStatus", "Value": approval_status},
                {"Key": "TrainingJob",  "Value": job_name},
                {"Key": "Project",      "Value": "bito-mule-detection"},
            ],
        }

        resp = self.sm.create_model_package(**pkg_input)
        pkg_arn = resp["ModelPackageArn"]

        log.info("═" * 58)
        log.info("  Model Registry 版本已註冊")
        log.info(f"  ModelPackageArn  : {pkg_arn}")
        log.info(f"  ApprovalStatus   : {approval_status}")
        log.info(f"  F1-score         : {f1_score:.4f}"
                 f"  (門檻: auto-approve={F1_AUTO_APPROVE}, min={F1_REGISTER_MIN})")
        log.info(f"  Training Job     : {job_name}")
        if approval_status == "Approved":
            log.info("  ✓ F1 ≥ 0.90，模型已自動核准，可直接部署至 Endpoint。")
        else:
            log.info("  ⏳ F1 在 0.80–0.90 之間，等待風控主管於 Model Registry 手動核准。")
        log.info("═" * 58)

        return resp

    def get_latest_approved(self, group_name: str = MODEL_PKG_GROUP) -> dict | None:
        """
        查詢 Model Package Group 中最新一個 Approved 版本。
        供 Endpoint 部署腳本使用。
        """
        resp = self.sm.list_model_packages(
            ModelPackageGroupName=group_name,
            ModelApprovalStatus="Approved",
            SortBy="CreationTime",
            SortOrder="Descending",
            MaxResults=1,
        )
        pkgs = resp.get("ModelPackageSummaryList", [])
        if not pkgs:
            log.info(f"Registry 中尚無 Approved 版本（Group: {group_name}）")
            return None
        return pkgs[0]

    # ── 私有工具 ──────────────────────────────────────────────────────────────

    def _xgboost_image_uri(self) -> str:
        """取得當前 Region 的 XGBoost 1.7 inference image URI。"""
        from sagemaker import image_uris
        return image_uris.retrieve(
            framework="xgboost",
            region=self.region,
            version="1.7-1",
            image_scope="inference",
        )

    def _upload_metrics_json(
        self,
        job_name:  str,
        f1_score:  float,
        precision: float,
        recall:    float,
    ) -> str:
        """
        將模型品質指標上傳至 S3，回傳 URI 供 ModelMetrics.Statistics 引用。

        格式符合 SageMaker Model Quality Report Schema：
        https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-model-quality-metrics.html
        """
        metrics = {
            "multiclass_classification_metrics": {
                "f1":        {"value": round(f1_score,  6), "standard_deviation": "NaN"},
                "precision": {"value": round(precision, 6), "standard_deviation": "NaN"},
                "recall":    {"value": round(recall,    6), "standard_deviation": "NaN"},
            }
        }
        s3    = boto3.client("s3", region_name=self.region)
        key   = f"{S3_PREFIX}/model-metrics/{job_name}/metrics.json"
        body  = json.dumps(metrics, indent=2).encode()
        try:
            s3.put_object(
                Bucket=S3_BUCKET, Key=key, Body=body,
                ContentType="application/json",
                ServerSideEncryption="AES256",
            )
        except ClientError as e:
            log.warning(f"指標 JSON 上傳失敗（不影響主流程）: {e}")
            # 若上傳失敗，回傳一個不存在的 URI（Registry 仍可建立）
        return f"s3://{S3_BUCKET}/{key}"

    def _notify_low_f1(self, f1: float, job_name: str, message: str) -> None:
        """F1 低於門檻時，透過 SNS 發送告警（SNS ARN 未設定則略過）。"""
        if not self.sns or not SNS_ALERT_ARN:
            return
        try:
            self.sns.publish(
                TopicArn=SNS_ALERT_ARN,
                Subject=f"[WARN] 模型 F1 不達標 — {job_name}",
                Message=json.dumps({
                    "event":    "MODEL_REGISTRY_SKIP",
                    "job":      job_name,
                    "f1_score": f1,
                    "threshold": F1_REGISTER_MIN,
                    "message":  message,
                }, ensure_ascii=False),
            )
        except ClientError:
            pass


def extract_f1_from_job(sm_client, job_name: str) -> float:
    """
    從 SageMaker Training Job 的 FinalMetricDataList 擷取 F1-score。

    train_xgboost_script.py 會以以下格式輸出指標至 stdout，
    SageMaker 自動解析並存入 FinalMetricDataList：
        [CV] f1_score: 0.8432

    若找不到指標（例如 fixed 模式未開啟 CV），回傳 0.0。
    """
    try:
        desc    = sm_client.describe_training_job(TrainingJobName=job_name)
        metrics = desc.get("FinalMetricDataList", [])
        for m in metrics:
            if m["MetricName"] in (HPO_METRIC_NAME, "validation:f1_score", "f1_score"):
                return float(m["Value"])
    except ClientError as e:
        log.warning(f"無法取得 Training Job 指標: {e}")
    return 0.0


# ── 6. 主流程 ────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["hpo", "fixed"], default="hpo",
        help="hpo=HyperparameterTuner（預設）  fixed=固定超參數快速訓練",
    )
    parser.add_argument(
        "--hpo-max-jobs", type=int, default=HPO_MAX_JOBS,
        help=f"HPO 總作業數（預設 {HPO_MAX_JOBS}）",
    )
    parser.add_argument(
        "--hpo-parallel", type=int, default=HPO_MAX_PARALLEL,
        help=f"HPO 同時並行 Job 數（預設 {HPO_MAX_PARALLEL}）",
    )
    parser.add_argument(
        "--skip-registry", action="store_true",
        help="訓練完成後略過 Model Registry 註冊",
    )
    parser.add_argument(
        "--registry-group", default=MODEL_PKG_GROUP,
        help=f"Model Package Group 名稱（預設：{MODEL_PKG_GROUP}）",
    )
    args = parser.parse_args()

    # ── 1. 初始化 AWS 客戶端 ──────────────────────────────────────────────
    session    = sagemaker.Session()
    boto_sess  = boto3.Session(region_name=REGION)
    s3         = boto_sess.client("s3")
    sm_client  = boto_sess.client("sagemaker")
    registry   = ModelRegistryManager(region=REGION)

    # ── 2. 優先載入標籤表（必須在 build_features 之前） ──────────────────
    # 關鍵修正：known_blacklist 須在特徵提取前就緒，否則圖跳轉特徵全為 0。
    print("[1/6] 載入標籤表與黑名單...")
    manager       = BitoDataManager()
    train_label   = manager._load_raw("train_label")
    predict_label = manager._load_raw("predict_label")
    train_label["user_id"]   = pd.to_numeric(train_label["user_id"],   errors="coerce")
    predict_label["user_id"] = pd.to_numeric(predict_label["user_id"], errors="coerce")

    # 從 train_label 提取已知黑名單 user_id（status == 1）
    known_blacklist: set[int] = set(
        train_label[train_label["status"] == 1]["user_id"]
        .dropna().astype(int).tolist()
    )
    print(f"  已知黑名單：{len(known_blacklist):,} 位用戶")

    # ── 3. 特徵工程（傳入 known_blacklist 啟用圖跳轉特徵） ───────────────
    print("[2/6] 載入資料與特徵工程...")
    features = build_features(manager, known_blacklist=known_blacklist)

    # ── 4. 合併特徵 & 依 CANONICAL_FEATURE_COLS 固定欄位順序 ──────────────
    # 使用 CANONICAL_FEATURE_COLS 而非動態推導，確保訓練/推論欄位完全一致。
    print("[3/6] 合併特徵...")
    feat_cols = CANONICAL_FEATURE_COLS   # 固定順序，不再動態推導

    train_df = (
        train_label.merge(features[["user_id"] + feat_cols], on="user_id", how="left")
                   .fillna(0)
    )
    predict_df = (
        predict_label.merge(features[["user_id"] + feat_cols], on="user_id", how="left")
                     .fillna(0)
    )

    hyperparams      = build_hyperparams(train_df)
    scale_pos_weight = hyperparams["scale_pos_weight"]

    train_sm = train_df[["status"] + feat_cols]   # label 第一欄，無 header
    infer_sm = predict_df[feat_cols]

    # ── 4. 上傳資料至 S3 ──────────────────────────────────────────────────
    print("[3/6] 上傳訓練資料至 S3...")
    ts         = int(time.time())
    train_key  = f"{S3_PREFIX}/{ts}/train/train.csv"
    infer_key  = f"{S3_PREFIX}/{ts}/infer/infer.csv"
    output_uri = f"s3://{S3_BUCKET}/{S3_PREFIX}/{ts}/output/"

    upload_df_to_s3(train_sm, s3, S3_BUCKET, train_key, header=False)
    upload_df_to_s3(infer_sm, s3, S3_BUCKET, infer_key, header=False)

    train_uri = f"s3://{S3_BUCKET}/{train_key}"
    infer_uri = f"s3://{S3_BUCKET}/{infer_key}"

    # ── 5. Model Package Group 預建（冪等，不影響訓練流程） ──────────────────
    if not args.skip_registry:
        registry.ensure_package_group(args.registry_group)

    # ── 6. 訓練（HPO 或固定參數） ─────────────────────────────────────────
    trained_job_name  = None
    trained_hyperparams = hyperparams.copy()

    if args.mode == "hpo":
        # ── HPO 模式：HyperparameterTuner 搜尋最佳超參數 ─────────────────
        print("[4/7] 啟動 HyperparameterTuner（搜尋 max_depth / eta / gamma）...")
        best_job_name, best_hyperparams = run_hyperparameter_tuning(
            train_uri=train_uri,
            output_uri=output_uri,
            scale_pos_weight=scale_pos_weight,
            session=session,
            max_jobs=args.hpo_max_jobs,
            max_parallel=args.hpo_parallel,
        )

        # 用最佳 Job 的模型 artifacts 做推論
        best_desc          = sm_client.describe_training_job(TrainingJobName=best_job_name)
        model_data         = best_desc["ModelArtifacts"]["S3ModelArtifacts"]
        trained_job_name   = best_job_name
        trained_hyperparams = best_hyperparams

        # 建立 Estimator 以使用 .transformer()
        estimator = XGBoost(
            entry_point="train_xgboost_script.py",
            source_dir=os.path.dirname(os.path.abspath(__file__)),
            framework_version="1.7-1",
            role=ROLE_ARN,
            instance_count=1,
            instance_type=INSTANCE_TYPE,
            output_path=output_uri,
            hyperparameters=best_hyperparams,
            sagemaker_session=session,
        )
        estimator.model_data = model_data

    else:
        # ── Fixed 模式：使用 BASE_HYPERPARAMS 快速訓練 ────────────────────
        print("[4/7] 啟動固定參數訓練（--mode fixed）...")
        fixed_hp = {
            **hyperparams,
            "max_depth": 6,
            "eta":       0.05,
            "gamma":     1.0,
        }
        estimator = XGBoost(
            entry_point="train_xgboost_script.py",
            source_dir=os.path.dirname(os.path.abspath(__file__)),
            framework_version="1.7-1",
            role=ROLE_ARN,
            instance_count=1,
            instance_type=INSTANCE_TYPE,
            output_path=output_uri,
            hyperparameters=fixed_hp,
            sagemaker_session=session,
        )
        # ── ResourceCheck：提交前確認帳號配額充足 ───────────────────────────
        ResourceCheck(sm_client, max_running=ResourceCheck.MAX_RUNNING_DEFAULT) \
            .assert_can_submit(n_new_jobs=1)

        estimator.fit(
            inputs={"train": TrainingInput(train_uri, content_type="text/csv")},
            wait=True,
            logs="All",
        )
        trained_job_name    = estimator.latest_training_job.name
        trained_hyperparams = fixed_hp

    print(f"  model artifacts → {estimator.model_data}")

    # ── 7. Model Registry 自動註冊 ────────────────────────────────────────
    print("[5/7] Model Registry 版本註冊...")
    if not args.skip_registry and trained_job_name:
        # 從 Training Job 取得 F1（由 train_xgboost_script.py 輸出到 stdout）
        f1_from_job = extract_f1_from_job(sm_client, trained_job_name)

        # 若 Training Job 尚未記錄到 F1，本地做一次快速 hold-out 估算
        if f1_from_job == 0.0:
            log.info("Training Job 未記錄 F1，本地 hold-out 估算中...")
            f1_from_job = _estimate_f1_locally(
                train_df, feat_cols, trained_hyperparams
            )

        log.info(f"最終 F1-score：{f1_from_job:.4f}")

        registry.register(
            model_data=estimator.model_data,
            job_name=trained_job_name,
            hyperparams=trained_hyperparams,
            train_uri=train_uri,
            f1_score=f1_from_job,
            group_name=args.registry_group,
        )
    else:
        print("  （Model Registry 已跳過，使用 --skip-registry 旗標）")

    # ── 8. Batch Transform 推論 ───────────────────────────────────────────
    print("[6/7] 執行 Batch Transform 推論...")
    transformer = estimator.transformer(
        instance_count=1,
        instance_type=INSTANCE_TYPE,
        output_path=output_uri,
        strategy="SingleRecord",
        assemble_with="Line",
        accept="text/csv",
    )
    transformer.transform(
        data=infer_uri,
        data_type="S3Prefix",
        content_type="text/csv",
        split_type="Line",
        wait=True,
        logs=True,
    )

    # ── 9. 下載推論結果 & 輸出 CSV ────────────────────────────────────────
    print("[7/7] 下載預測結果並產生提交檔...")
    result_key = infer_key.replace("infer/infer.csv", "output/infer.csv.out")
    result_obj = s3.get_object(Bucket=S3_BUCKET, Key=result_key)
    prob_series = pd.read_csv(
        io.BytesIO(result_obj["Body"].read()), header=None, names=["probability"]
    )

    # 對齊 predict_label 的 user_id
    submission = predict_label[["user_id"]].copy().reset_index(drop=True)
    submission["probability"] = prob_series["probability"].values

    # 決策門檻：預設 0.5，可依 F1 最佳化調整
    threshold = find_best_threshold(train_df, feat_cols, hyperparams, session, s3)
    submission["status"] = (submission["probability"] >= threshold).astype(int)

    print(f"\n  決策門檻: {threshold:.4f}")
    print(f"  預測黑名單數: {submission['status'].sum():,} / {len(submission):,}")

    # 詳細版（含機率）
    submission.to_csv(PROB_CSV, index=False)
    print(f"  詳細版 → {PROB_CSV}")

    # 提交版（僅 user_id + status，符合競賽格式）
    submission[["user_id", "status"]].to_csv(OUTPUT_CSV, index=False)
    print(f"  提交版 → {OUTPUT_CSV}")

    # 預覽
    print("\n=== 提交檔前 5 筆 ===")
    print(submission[["user_id", "status"]].head())


# ── 輔助：本地 hold-out 估算 F1（當 Training Job 未記錄指標時的備援） ────────

def _estimate_f1_locally(
    train_df:    pd.DataFrame,
    feat_cols:   list[str],
    hyperparams: dict,
    test_size:   float = 0.20,
) -> float:
    """
    在 20% hold-out 上訓練本地 XGBoost，回傳最佳 F1。
    此函式僅在 SageMaker Job 未輸出 F1 指標時作為備援。
    """
    try:
        import xgboost as xgb
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import f1_score as sk_f1

        X = train_df[feat_cols].values.astype(float)
        y = train_df["status"].values.astype(int)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        params = {k: v for k, v in hyperparams.items() if k != "num_round"}
        model  = xgb.train(
            params,
            xgb.DMatrix(X_tr, label=y_tr),
            num_boost_round=hyperparams.get("num_round", 300),
        )
        probs  = model.predict(xgb.DMatrix(X_val))
        best_f1 = max(
            sk_f1(y_val, (probs >= t).astype(int), zero_division=0)
            for t in np.arange(0.1, 0.9, 0.02)
        )
        log.info(f"  本地 hold-out F1 估算：{best_f1:.4f}")
        return float(best_f1)
    except Exception as e:
        log.warning(f"  本地 F1 估算失敗（{e}），以 0.0 回傳")
        return 0.0


# ── 輔助：在訓練集上以交叉驗證找 F1 最佳門檻 ─────────────────────────────────

def find_best_threshold(
    train_df: pd.DataFrame,
    feat_cols: list[str],
    hyperparams: dict,
    session,
    s3,
    cv_frac: float = 0.2,
    default_threshold: float = 0.5,
) -> float:
    """
    用 20% 的訓練集做 hold-out，掃描門檻找 F1 最大值。
    若無法執行（例如網路問題）則回傳 default_threshold。
    """
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import f1_score
        import xgboost as xgb

        X = train_df[feat_cols].values
        y = train_df["status"].values
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=cv_frac, random_state=42, stratify=y
        )

        # 本地訓練小模型（與 SageMaker 超參數一致）
        local_params = {k: v for k, v in hyperparams.items() if k != "num_round"}
        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval   = xgb.DMatrix(X_val)
        model  = xgb.train(local_params, dtrain, num_boost_round=hyperparams["num_round"])

        probs = model.predict(dval)
        best_thresh, best_f1 = default_threshold, 0.0
        for t in np.arange(0.1, 0.9, 0.01):
            f1 = f1_score(y_val, (probs >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_thresh = f1, t

        print(f"  [threshold search] best_threshold={best_thresh:.2f}  best_F1={best_f1:.4f}")
        return float(best_thresh)

    except Exception as e:
        print(f"  [threshold search] 跳過（{e}），使用預設門檻 {default_threshold}")
        return default_threshold


if __name__ == "__main__":
    main()
