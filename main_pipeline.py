"""
端到端整合腳本 — BitoGuard AML 人頭戶偵測管線
===============================================

一個指令執行完整流程：
    python main_pipeline.py                    # 本地模式（6 階段）
    python main_pipeline.py --aws-mode         # AWS 完整模式（8 階段）

階段定義
--------
本地模式（6 階段）：
    Stage 1  feature   資料載入 + 特徵工程（含圖跳轉 BFS）
    Stage 2  train     SageMaker XGBoost 訓練 + HPO
    Stage 3  download  從 S3 下載最優模型
    Stage 4  validate  5-Fold CV + 門檻分析 + 驗證報告
    Stage 5  visualize 生成 PR 曲線 / SHAP / 黑名單圖等圖表
    Stage 6  xai       SHAP + Bedrock 風險診斷書（高風險用戶）

AWS 完整模式（在本地 6 階段前後加入 2 個雲端階段）：
    Stage A  aws_ingest         BitoPro API → S3 → Glue Crawler + Glue Job
    Stage 1~6                   同本地模式
    Stage Z  aws_validate_submit 驗證 submission.csv user_id 完整性

斷點續跑
--------
每個 Stage 完成後都會寫入 pipeline_state.json。
中斷後重跑會自動跳過已完成的 Stage：
    python main_pipeline.py --start-from validate

輸出檔案
--------
    submission.csv              競賽提交（user_id, status）
    submission_with_prob.csv    含預測機率的詳細版本
    validation_report.json/txt  驗證報告（F1/P/R/AUC + 門檻掃描）
    xai_reports.json            風險診斷書（高風險用戶）
    plots/                      可視化圖表目錄
    pipeline_state.json         管線執行狀態（斷點續跑用）
    pipeline.log                執行日誌
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

# ── 日誌設定 ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ── 設定 ─────────────────────────────────────────────────────────────────────

STAGES         = ["feature", "train", "download", "validate", "visualize", "xai"]
STATE_FILE     = "pipeline_state.json"
RISK_THRESHOLD = 0.83    # V9 最佳門檻，硬編碼；可被 pipeline_state optimal_thr 覆蓋
FEAT_CACHE  = "feature_cache.parquet"        # 特徵快取（避免重跑 API）
PRED_CACHE  = "prediction_cache.parquet"     # 預測結果快取
PRED_ID_CACHE = "predict_label_ids.parquet"  # predict_label user_id 快取

# Glue Job 預設值（可透過 CLI 覆蓋）
DEFAULT_GLUE_JOB     = "bito-graph-hops"
DEFAULT_GLUE_CRAWLER = "bito-raw-crawler"
DEFAULT_POLL_SEC     = 30    # Glue / SageMaker 輪詢間隔（秒）
GLUE_JOB_TIMEOUT_SEC = 3600  # Glue Job 最長等待時間（1 小時）


# ══════════════════════════════════════════════════════════════════════════════
#  狀態管理（斷點續跑核心）
# ══════════════════════════════════════════════════════════════════════════════

def load_state() -> dict:
    """載入管線執行狀態（不存在時回傳空狀態）。"""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            log.warning("[State] 無法載入 %s：%s，使用空狀態", STATE_FILE, e)
    return {"completed": [], "artifacts": {}, "started_at": datetime.now(timezone.utc).isoformat()}


def save_state(state: dict) -> None:
    """原子寫入管線狀態（防止寫入中斷導致狀態損壞）。"""
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    os.replace(tmp, STATE_FILE)
    log.debug("[State] 已儲存 %s", STATE_FILE)


def mark_done(state: dict, stage: str, artifacts: dict | None = None) -> None:
    """標記指定 Stage 已完成，並記錄產出的檔案路徑。"""
    if stage not in state["completed"]:
        state["completed"].append(stage)
    if artifacts:
        state.setdefault("artifacts", {}).update(artifacts)
    state["last_completed"] = stage
    state["last_updated"]   = datetime.now(timezone.utc).isoformat()
    save_state(state)
    log.info("[State] ✓ Stage '%s' 已完成並記錄", stage)


def is_done(state: dict, stage: str) -> bool:
    """檢查指定 Stage 是否已完成。"""
    return stage in state.get("completed", [])


# ══════════════════════════════════════════════════════════════════════════════
#  Stage A：AWS 資料擷取（aws_ingest）
# ══════════════════════════════════════════════════════════════════════════════

def aws_stage_ingest(state: dict, args: argparse.Namespace) -> None:
    """
    AWS Stage A：BitoPro API → S3 → Glue Crawler → Glue Job（BFS 圖分析）。

    流程：
      1. ingest_to_s3.py：從 BitoPro API 擷取原始資料，寫入 S3 Hive 分區
      2. 觸發 Glue Crawler 掃描新資料，更新 Glue Data Catalog
      3. 輪詢 Crawler 直到 READY 狀態
      4. 觸發 Glue Job（bito-graph-hops）執行分散式 BFS
      5. 輪詢 Job 直到 SUCCEEDED
    """
    if is_done(state, "aws_ingest"):
        log.info("[aws_ingest] 已完成，跳過")
        return

    log.info("=" * 60)
    log.info("[aws_ingest] ▶ 開始 AWS 資料擷取流程")

    import boto3
    from ingest_to_s3 import DataIngester, GlueCatalogSetup

    s3_bucket = os.environ.get("S3_BUCKET", args.s3_bucket)
    s3_prefix = os.environ.get("S3_PREFIX", "bito-mule-detection/raw")
    region    = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

    # ── Step 1：資料擷取 → S3 ──────────────────────────────────────────
    log.info("[aws_ingest] Step 1/4 資料擷取至 S3...")
    ingester = DataIngester(bucket=s3_bucket, prefix=s3_prefix, region=region)
    summary  = ingester.ingest_all()
    log.info("[aws_ingest] 擷取完成：%s", summary)

    # ── Step 2：Glue Crawler ─────────────────────────────────────────
    log.info("[aws_ingest] Step 2/4 啟動 Glue Crawler：%s", args.glue_crawler)
    glue = boto3.client("glue", region_name=region)
    try:
        glue.start_crawler(Name=args.glue_crawler)
    except glue.exceptions.CrawlerRunningException:
        log.info("[aws_ingest] Crawler 已在執行中，等待完成...")

    # ── Step 3：輪詢 Crawler ──────────────────────────────────────────
    log.info("[aws_ingest] Step 3/4 等待 Crawler 完成...")
    deadline = time.time() + 600   # 最多等 10 分鐘
    while time.time() < deadline:
        resp    = glue.get_crawler(Name=args.glue_crawler)
        crawler = resp["Crawler"]
        status  = crawler.get("State", "UNKNOWN")
        if status == "READY":
            log.info("[aws_ingest] Crawler 完成，LastCrawl: %s",
                     crawler.get("LastCrawl", {}).get("Status"))
            break
        log.info("[aws_ingest] Crawler 狀態：%s，等待 %ds...", status, DEFAULT_POLL_SEC)
        time.sleep(DEFAULT_POLL_SEC)
    else:
        raise TimeoutError(f"Glue Crawler '{args.glue_crawler}' 超過 10 分鐘未完成")

    # ── Step 4：啟動 Glue Job（BFS 圖分析） ───────────────────────────
    log.info("[aws_ingest] Step 4/4 啟動 Glue Job：%s", args.glue_job)
    run_resp = glue.start_job_run(
        JobName=args.glue_job,
        Arguments={
            "--s3_input_bucket":   s3_bucket,
            "--s3_input_prefix":   s3_prefix,
            "--s3_output_prefix":  "bito-mule-detection/features",
            "--blacklist_s3_path": state.get("artifacts", {}).get(
                "blacklist_s3_path", f"s3://{s3_bucket}/bito-mule-detection/blacklist.csv"
            ),
            "--max_hops":          "3",
        },
    )
    run_id = run_resp["JobRunId"]
    log.info("[aws_ingest] Glue Job Run ID：%s", run_id)

    # 輪詢 Glue Job
    deadline = time.time() + GLUE_JOB_TIMEOUT_SEC
    while time.time() < deadline:
        run = glue.get_job_run(JobName=args.glue_job, RunId=run_id)["JobRun"]
        job_status = run["JobRunState"]
        if job_status == "SUCCEEDED":
            log.info("[aws_ingest] ✓ Glue Job 完成")
            break
        if job_status in ("FAILED", "ERROR", "TIMEOUT", "STOPPED"):
            raise RuntimeError(
                f"Glue Job '{args.glue_job}' 失敗（狀態：{job_status}）\n"
                f"錯誤訊息：{run.get('ErrorMessage', '無')}"
            )
        log.info("[aws_ingest] Glue Job 狀態：%s，等待 %ds...", job_status, DEFAULT_POLL_SEC)
        time.sleep(DEFAULT_POLL_SEC)
    else:
        raise TimeoutError(f"Glue Job '{args.glue_job}' 超過 {GLUE_JOB_TIMEOUT_SEC}s 未完成")

    mark_done(state, "aws_ingest", {
        "s3_bucket":   s3_bucket,
        "s3_prefix":   s3_prefix,
        "glue_run_id": run_id,
    })
    log.info("[aws_ingest] ✓ AWS 資料擷取全流程完成")


# ══════════════════════════════════════════════════════════════════════════════
#  Stage Z：驗證提交集完整性（aws_validate_submit）
# ══════════════════════════════════════════════════════════════════════════════

def aws_stage_validate_submission(state: dict) -> None:
    """
    AWS Stage Z：驗證 submission.csv 的 user_id 是否與 predict_label 完全一致。

    競賽提交規則：
      - submission.csv 的 user_id 集合 == predict_label 的 user_id 集合
      - status 欄位只能為 0 或 1
      - 不允許重複的 user_id

    若驗證失敗，程式以退出碼 1 終止（CI/CD 阻擋提交）。
    """
    if is_done(state, "aws_validate_submit"):
        log.info("[aws_validate_submit] 已完成，跳過")
        return

    log.info("=" * 60)
    log.info("[aws_validate_submit] ▶ 驗證提交集完整性")

    submission_path = "submission.csv"
    if not os.path.exists(submission_path):
        raise FileNotFoundError(
            f"找不到 {submission_path}，請先完成 Stage 2（train）"
        )

    sub = pd.read_csv(submission_path)

    # ── 必要欄位 ──────────────────────────────────────────────────────────
    missing_cols = [c for c in ("user_id", "status") if c not in sub.columns]
    if missing_cols:
        raise ValueError(f"submission.csv 缺少欄位：{missing_cols}")

    # ── 重複 user_id ──────────────────────────────────────────────────────
    dup_count = sub["user_id"].duplicated().sum()
    if dup_count > 0:
        raise ValueError(f"submission.csv 有 {dup_count} 筆重複 user_id")

    # ── status 合法值 ─────────────────────────────────────────────────────
    invalid_status = sub[~sub["status"].isin([0, 1])]
    if not invalid_status.empty:
        raise ValueError(
            f"submission.csv 中 {len(invalid_status)} 筆 status 非 0/1：\n"
            f"{invalid_status.head()}"
        )

    # ── 與 predict_label 對齊 ─────────────────────────────────────────────
    if os.path.exists(PRED_ID_CACHE):
        pred_ids = pd.read_parquet(PRED_ID_CACHE)["user_id"].astype(int)
    else:
        from bito_data_manager import BitoDataManager
        mgr      = BitoDataManager()
        pred_raw = mgr._load_raw("predict_label")
        pred_ids = pd.to_numeric(pred_raw["user_id"], errors="coerce").dropna().astype(int)

    sub_ids      = set(sub["user_id"].astype(int))
    pred_id_set  = set(pred_ids)
    missing_pred = pred_id_set - sub_ids
    extra_sub    = sub_ids - pred_id_set

    if missing_pred:
        raise ValueError(
            f"submission.csv 缺少 {len(missing_pred)} 個 predict_label 中的 user_id\n"
            f"  前 10 個：{sorted(missing_pred)[:10]}"
        )
    if extra_sub:
        log.warning(
            "[aws_validate_submit] ⚠ submission.csv 多出 %d 個不在 predict_label 的 user_id",
            len(extra_sub),
        )

    pos_count = (sub["status"] == 1).sum()
    neg_count = (sub["status"] == 0).sum()
    log.info("[aws_validate_submit] ✓ 驗證通過")
    log.info("  總筆數：%d  黑名單：%d  正常：%d  預測比例：%.2f%%",
             len(sub), pos_count, neg_count, 100 * pos_count / max(len(sub), 1))

    mark_done(state, "aws_validate_submit", {"submission_rows": len(sub)})


# ══════════════════════════════════════════════════════════════════════════════
#  Stage 1：特徵工程（feature）
# ══════════════════════════════════════════════════════════════════════════════

def stage_feature(state: dict, args: argparse.Namespace) -> None:
    """
    Stage 1：載入資料、計算四大人頭戶特徵，輸出特徵寬表。

    特徵：
      ① 資金滯留時間（快進快出 < 10 分鐘）
      ② IP 異常跳動（同用戶多 IP）
      ③ 量能不對稱（KYC L1 用戶卻有百萬交易）
      ④ 圖跳轉（距黑名單的 BFS 最短跳數 + in_blacklist_network）

    輸出快取到 feature_cache.parquet，供後續 Stage 讀取。
    """
    if is_done(state, "feature") and os.path.exists(FEAT_CACHE):
        log.info("[feature] 已完成且快取存在，跳過")
        return

    log.info("=" * 60)
    log.info("[feature] ▶ 開始特徵工程")

    from bito_data_manager import BitoDataManager
    from train_sagemaker   import build_features, CANONICAL_FEATURE_COLS

    csv_dir = args.csv_dir if hasattr(args, "csv_dir") and args.csv_dir else None
    mgr     = BitoDataManager(csv_dir=csv_dir)

    # ── 優先載入標籤，提取黑名單（圖特徵依賴） ──────────────────────────
    log.info("[feature] 載入 train_label / predict_label...")
    try:
        train_label   = mgr._load_raw("train_label")
        predict_label = mgr._load_raw("predict_label")
    except Exception as e:
        raise RuntimeError(f"無法載入標籤表：{e}\n請確認資料來源（--csv-dir 或 API）") from e

    train_label["user_id"]   = pd.to_numeric(train_label["user_id"],   errors="coerce")
    predict_label["user_id"] = pd.to_numeric(predict_label["user_id"], errors="coerce")

    # 快取 predict_label 的 user_id（供 Stage Z 驗證）
    predict_label[["user_id"]].astype({"user_id": int}).to_parquet(PRED_ID_CACHE, index=False)

    known_blacklist: set[int] = set(
        train_label[train_label["status"] == 1]["user_id"]
        .dropna().astype(int).tolist()
    )
    log.info("[feature] 已知黑名單：%d 位用戶（佔訓練集 %.2f%%）",
             len(known_blacklist),
             100 * len(known_blacklist) / max(len(train_label), 1))

    # ── 特徵計算 ────────────────────────────────────────────────────────
    log.info("[feature] 執行特徵提取（含圖 BFS）...")
    t0       = time.time()
    features = build_features(mgr, known_blacklist=known_blacklist)
    elapsed  = time.time() - t0
    log.info("[feature] 特徵提取完成，耗時 %.1fs，共 %d 用戶 × %d 特徵",
             elapsed, len(features), len(CANONICAL_FEATURE_COLS))

    # ── 合併標籤（並驗證覆蓋率） ─────────────────────────────────────
    feat_cols = CANONICAL_FEATURE_COLS
    all_ids   = pd.concat([
        train_label[["user_id", "status"]],
        predict_label[["user_id"]].assign(status=np.nan),
    ], ignore_index=True)

    merged = all_ids.merge(features[["user_id"] + feat_cols], on="user_id", how="left")
    merged[feat_cols] = merged[feat_cols].fillna(0)

    coverage = merged["user_id"].isin(features["user_id"]).mean()
    log.info("[feature] 特徵覆蓋率：%.1f%%（低覆蓋可能是 API 資料缺失）", 100 * coverage)

    # ── 寫出快取 ─────────────────────────────────────────────────────────
    merged.to_parquet(FEAT_CACHE, index=False)
    log.info("[feature] 已快取至 %s", FEAT_CACHE)

    mark_done(state, "feature", {
        "feat_cache":      FEAT_CACHE,
        "n_users":         len(merged),
        "n_features":      len(feat_cols),
        "blacklist_count": len(known_blacklist),
        "coverage":        round(coverage, 4),
    })
    log.info("[feature] ✓ 特徵工程完成")


# ══════════════════════════════════════════════════════════════════════════════
#  Stage 2：SageMaker 訓練（train）
# ══════════════════════════════════════════════════════════════════════════════

def stage_train(state: dict, args: argparse.Namespace) -> None:
    """
    Stage 2：上傳特徵至 S3，啟動 SageMaker HyperparameterTuner，
    等待訓練完成後向 Model Registry 註冊。

    依賴：Stage 1（feature_cache.parquet 存在）
    """
    if is_done(state, "train"):
        log.info("[train] 已完成，跳過")
        return

    log.info("=" * 60)
    log.info("[train] ▶ 開始 SageMaker 訓練")

    if not os.path.exists(FEAT_CACHE):
        raise FileNotFoundError(
            f"找不到特徵快取 {FEAT_CACHE}，請先執行 Stage 1（feature）"
        )

    import boto3
    import sagemaker
    from train_sagemaker import (
        build_hyperparams, run_hyperparameter_tuning,
        upload_df_to_s3, ModelRegistryManager,
        CANONICAL_FEATURE_COLS, S3_BUCKET, S3_PREFIX, REGION,
    )

    # ── 載入特徵快取 ──────────────────────────────────────────────────────
    merged    = pd.read_parquet(FEAT_CACHE)
    feat_cols = CANONICAL_FEATURE_COLS

    train_df   = merged[merged["status"].notna()].copy()
    predict_df = merged[merged["status"].isna()].copy()

    if len(train_df) == 0:
        raise ValueError("訓練集為空，請確認 train_label 已載入")

    log.info("[train] 訓練集：%d 筆  推論集：%d 筆", len(train_df), len(predict_df))

    # ── 計算 scale_pos_weight ─────────────────────────────────────────────
    hyperparams      = build_hyperparams(train_df)
    scale_pos_weight = hyperparams["scale_pos_weight"]

    # ── 組裝 CSV（SageMaker 格式：label 第一欄，無 header） ───────────────
    train_sm = train_df[["status"] + feat_cols].copy()
    train_sm["status"] = train_sm["status"].astype(int)
    infer_sm = predict_df[feat_cols].copy()

    # ── 上傳至 S3 ─────────────────────────────────────────────────────────
    session   = sagemaker.Session()
    boto_sess = boto3.Session(region_name=REGION)
    s3        = boto_sess.client("s3")

    ts         = int(time.time())
    train_key  = f"{S3_PREFIX}/{ts}/train/train.csv"
    infer_key  = f"{S3_PREFIX}/{ts}/infer/infer.csv"
    output_uri = f"s3://{S3_BUCKET}/{S3_PREFIX}/{ts}/output/"

    log.info("[train] 上傳訓練資料至 S3...")
    import io
    for df, key, hdr in [(train_sm, train_key, False), (infer_sm, infer_key, False)]:
        buf = io.StringIO()
        df.to_csv(buf, index=False, header=hdr)
        s3.put_object(Bucket=S3_BUCKET, Key=key, Body=buf.getvalue().encode())
        log.info("[train] 已上傳 s3://%s/%s", S3_BUCKET, key)

    train_uri = f"s3://{S3_BUCKET}/{train_key}"
    infer_uri = f"s3://{S3_BUCKET}/{infer_key}"

    # ── HPO ────────────────────────────────────────────────────────────────
    registry = ModelRegistryManager(region=REGION)
    if not args.skip_registry:
        registry.ensure_package_group()

    mode = getattr(args, "train_mode", "hpo")
    if mode == "hpo":
        log.info("[train] 啟動 HyperparameterTuner（max_jobs=%d）...",
                 getattr(args, "hpo_max_jobs", 10))
        best_job, best_hparams = run_hyperparameter_tuning(
            train_uri        = train_uri,
            output_uri       = output_uri,
            scale_pos_weight = scale_pos_weight,
            session          = session,
            max_jobs         = getattr(args, "hpo_max_jobs",    10),
            max_parallel     = getattr(args, "hpo_parallel",     2),
            wait             = True,
        )
    else:
        raise ValueError(f"不支援的訓練模式：{mode}（目前只支援 hpo）")

    # ── 取得 F1 並決定是否進入 Registry ──────────────────────────────────
    from train_sagemaker import extract_f1_from_job
    sm_client = boto_sess.client("sagemaker")
    f1_score  = extract_f1_from_job(sm_client, best_job)
    log.info("[train] 最佳 Job：%s  F1=%.4f", best_job, f1_score)

    # 取得模型輸出 URI
    job_desc   = sm_client.describe_training_job(TrainingJobName=best_job)
    model_data = job_desc["ModelArtifacts"]["S3ModelArtifacts"]

    if not args.skip_registry:
        registry.register(
            model_data  = model_data,
            job_name    = best_job,
            hyperparams = best_hparams,
            train_uri   = train_uri,
            f1_score    = f1_score,
        )

    mark_done(state, "train", {
        "best_job":    best_job,
        "model_data":  model_data,
        "train_uri":   train_uri,
        "infer_uri":   infer_uri,
        "f1_score":    round(f1_score, 4),
        "output_uri":  output_uri,
    })
    log.info("[train] ✓ 訓練完成，F1=%.4f", f1_score)


# ══════════════════════════════════════════════════════════════════════════════
#  Stage 3：下載模型（download）
# ══════════════════════════════════════════════════════════════════════════════

def stage_download(state: dict, args: argparse.Namespace) -> None:
    """
    Stage 3：從 S3 下載最優訓練模型（model.tar.gz → model.json）。

    依賴：Stage 2（state["artifacts"]["best_job"] 存在）
    """
    if is_done(state, "download") and os.path.exists("model.json"):
        log.info("[download] 已完成且 model.json 存在，跳過")
        return

    log.info("=" * 60)
    log.info("[download] ▶ 下載模型")

    from download_model import find_model_key_by_job, download_and_extract, verify_model
    from train_sagemaker import S3_BUCKET, REGION

    best_job = state.get("artifacts", {}).get("best_job")
    if not best_job and hasattr(args, "job_name") and args.job_name:
        best_job = args.job_name

    if not best_job:
        raise ValueError(
            "未找到最佳 Job 名稱。請確認 Stage 2 已完成，"
            "或使用 --job-name 參數指定。"
        )

    log.info("[download] 從 Job '%s' 找模型...", best_job)
    model_key = find_model_key_by_job(best_job, S3_BUCKET, region=REGION)
    local_path = download_and_extract(
        bucket   = S3_BUCKET,
        key      = model_key,
        out_dir  = ".",
        region   = REGION,
    )
    verify_model(local_path)
    log.info("[download] ✓ 模型已下載至 %s", local_path)

    mark_done(state, "download", {
        "model_path": local_path,
        "model_key":  model_key,
    })


# ══════════════════════════════════════════════════════════════════════════════
#  Stage 4：驗證報告（validate）
# ══════════════════════════════════════════════════════════════════════════════

def run_cv_report(state: dict, args: argparse.Namespace) -> dict:
    """
    執行 5-Fold CV + 門檻掃描，回傳指標摘要 dict。
    供 stage_validate() 呼叫。
    """
    from validation_report import ValidationReport
    from train_sagemaker   import CANONICAL_FEATURE_COLS

    feat_df   = pd.read_parquet(FEAT_CACHE)
    train_df  = feat_df[feat_df["status"].notna()].copy()
    feat_cols = CANONICAL_FEATURE_COLS

    X = train_df[feat_cols].values.astype(float)
    y = train_df["status"].astype(int).values

    model_path = state.get("artifacts", {}).get("model_path", "model.json")
    booster    = xgb.Booster()
    booster.load_model(model_path)

    report = ValidationReport()
    cv_results, threshold_analysis = report.run_cv(X, y, booster)
    sweep   = report.sweep_thresholds(X, y, booster)
    fi      = report.extract_feature_importance(booster, feat_cols)

    report_obj = report.generate_report(
        cv_results=cv_results,
        threshold_analysis=threshold_analysis,
        sweep=sweep,
        feature_importance=fi,
    )
    report.save_report(report_obj)
    report.render_text_report(report_obj)

    return {
        "cv_f1_mean":   round(float(np.mean([r.f1 for r in cv_results])), 4),
        "optimal_f1":   round(threshold_analysis.optimal_f1, 4),
        "optimal_thr":  round(threshold_analysis.optimal_threshold, 4),
    }


def stage_validate(state: dict, args: argparse.Namespace) -> None:
    """
    Stage 4：5-Fold CV 驗證 + 門檻分析 + 生成 validation_report.json。

    依賴：Stage 1（feature_cache.parquet）+ Stage 3（model.json）
    """
    if is_done(state, "validate"):
        log.info("[validate] 已完成，跳過")
        return

    log.info("=" * 60)
    log.info("[validate] ▶ 開始交叉驗證與報告生成")

    for dep, dep_path in [("feature", FEAT_CACHE), ("download", "model.json")]:
        if not os.path.exists(dep_path):
            raise FileNotFoundError(
                f"找不到 {dep_path}，請先完成 Stage {dep}"
            )

    metrics = run_cv_report(state, args)
    log.info("[validate] CV F1（mean）= %.4f  最佳門檻 F1 = %.4f  門檻 = %.2f",
             metrics["cv_f1_mean"], metrics["optimal_f1"], metrics["optimal_thr"])

    mark_done(state, "validate", metrics)
    log.info("[validate] ✓ 驗證完成，報告已寫入 validation_report.json/.txt")


# ══════════════════════════════════════════════════════════════════════════════
#  Stage 5：視覺化（visualize）
# ══════════════════════════════════════════════════════════════════════════════

def stage_visualize(state: dict, args: argparse.Namespace) -> None:
    """
    Stage 5：生成四種圖表：PR 曲線、門檻掃描、SHAP 蜂群圖、黑名單關聯網絡圖。

    依賴：Stage 1（feature_cache.parquet）+ Stage 3（model.json）
    """
    if is_done(state, "visualize"):
        log.info("[visualize] 已完成，跳過")
        return

    log.info("=" * 60)
    log.info("[visualize] ▶ 生成視覺化圖表")

    if not os.path.exists(FEAT_CACHE):
        raise FileNotFoundError(f"找不到 {FEAT_CACHE}，請先完成 Stage 1（feature）")
    if not os.path.exists("model.json"):
        raise FileNotFoundError("找不到 model.json，請先完成 Stage 3（download）")

    from visualize          import generate_all_plots
    from train_sagemaker    import CANONICAL_FEATURE_COLS

    feat_df   = pd.read_parquet(FEAT_CACHE)
    train_df  = feat_df[feat_df["status"].notna()].copy()
    feat_cols = CANONICAL_FEATURE_COLS

    X = train_df[feat_cols].values.astype(float)
    y = train_df["status"].astype(int).values

    booster = xgb.Booster()
    booster.load_model("model.json")

    Path("plots").mkdir(exist_ok=True)
    threshold = float(state.get("artifacts", {}).get("optimal_thr", RISK_THRESHOLD))

    plot_paths = generate_all_plots(
        booster         = booster,
        X               = X,
        y               = y,
        feature_names   = feat_cols,
        threshold       = float(threshold),
        output_dir      = "plots",
        hop_df          = feat_df,
        target_user_id  = None,
    )
    log.info("[visualize] 已生成 %d 張圖表：%s", len(plot_paths), list(plot_paths.keys()))

    mark_done(state, "visualize", {"plot_paths": plot_paths})
    log.info("[visualize] ✓ 視覺化完成，圖表輸出至 plots/")


# ══════════════════════════════════════════════════════════════════════════════
#  Stage 6：XAI 風險診斷（xai）
# ══════════════════════════════════════════════════════════════════════════════

def stage_xai(state: dict, args: argparse.Namespace) -> None:
    """
    Stage 6：對高風險用戶執行 SHAP 分析 + Bedrock 風險診斷書生成。

    機率路由：
      EXTREME  P > 0.90  → Haiku（制式凍結通知）
      HIGH     0.75~0.90 → Sonnet（標準分析）
      BOUNDARY 0.65~0.75 → Sonnet（深度診斷）
      MEDIUM   0.50~0.65 → Rule-Based Only
      LOW      P < 0.50  → 跳過

    依賴：Stage 3（model.json）+ Stage 1（feature_cache.parquet）
    """
    if is_done(state, "xai"):
        log.info("[xai] 已完成，跳過")
        return

    log.info("=" * 60)
    log.info("[xai] ▶ 開始 SHAP + Bedrock 風險診斷書生成")

    for dep_path in [FEAT_CACHE, "model.json"]:
        if not os.path.exists(dep_path):
            raise FileNotFoundError(f"找不到 {dep_path}，請先完成前置 Stage")

    from xai_bedrock      import XAIReportGenerator
    from train_sagemaker  import CANONICAL_FEATURE_COLS

    feat_df    = pd.read_parquet(FEAT_CACHE)
    feat_cols  = CANONICAL_FEATURE_COLS
    threshold  = float(state.get("artifacts", {}).get("optimal_thr", RISK_THRESHOLD))

    booster = xgb.Booster()
    booster.load_model("model.json")

    # 產生預測機率
    import xgboost as xgb
    predict_df = feat_df[feat_df["status"].isna()].copy()
    if predict_df.empty:
        log.warning("[xai] predict_label 為空，嘗試使用完整特徵集")
        predict_df = feat_df.copy()

    X_pred = predict_df[feat_cols].values.astype(float)
    dmat   = xgb.DMatrix(X_pred, feature_names=feat_cols)
    probs  = booster.predict(dmat)
    predict_df = predict_df.copy()
    predict_df["probability"] = probs

    # 寫出含機率的預測結果（供 Lambda 觸發）
    # user_id 統一轉 int，避免 float("12345.0") 導致黑名單計數為 0
    predict_df["user_id"] = pd.to_numeric(predict_df["user_id"], errors="coerce").astype(int)
    predict_df["status"]  = (probs >= threshold).astype(int)
    predict_df[["user_id", "status"]].to_csv("submission.csv", index=False)
    predict_df[["user_id", "probability", "status"] + feat_cols].to_csv(
        "submission_with_prob.csv", index=False
    )
    log.info("[xai] submission.csv / submission_with_prob.csv 已寫出")

    # 只對高風險用戶生成診斷書
    high_risk = predict_df[predict_df["probability"] >= 0.50].copy()
    log.info("[xai] 高風險用戶（P≥0.50）：%d 位，開始生成診斷書...", len(high_risk))

    if len(high_risk) == 0:
        log.warning("[xai] 無高風險用戶，跳過 Bedrock 診斷書生成")
        mark_done(state, "xai", {"xai_reports": 0})
        return

    generator = XAIReportGenerator(booster=booster, feature_names=feat_cols)
    reports   = generator.explain_batch(
        df              = high_risk,
        feature_cols    = feat_cols,
        user_id_col     = "user_id",
        probability_col = "probability",
    )

    # 儲存診斷書
    with open("xai_reports.json", "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2, ensure_ascii=False, default=str)

    log.info("[xai] ✓ 已生成 %d 份診斷書 → xai_reports.json", len(reports))
    mark_done(state, "xai", {
        "xai_reports":     len(reports),
        "submission_rows": len(predict_df),
        "high_risk_count": len(high_risk),
    })


# ══════════════════════════════════════════════════════════════════════════════
#  主控邏輯
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BitoGuard AML 人頭戶偵測管線",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例：
  python main_pipeline.py                          # 本地完整流程
  python main_pipeline.py --aws-mode               # AWS 完整模式
  python main_pipeline.py --start-from validate    # 從驗證 Stage 繼續
  python main_pipeline.py --only feature train     # 只跑指定 Stage
  python main_pipeline.py --csv-dir ./data         # 指定本地 CSV 目錄
  python main_pipeline.py --skip-registry          # 略過 Model Registry
        """,
    )

    # ── 執行控制 ────────────────────────────────────────────────────────
    parser.add_argument(
        "--start-from", metavar="STAGE",
        choices=STAGES,
        help="從指定 Stage 開始（跳過已標記完成的前置 Stage）",
    )
    parser.add_argument(
        "--only", metavar="STAGE", nargs="+",
        choices=STAGES + ["aws_ingest", "aws_validate_submit"],
        help="只執行指定的一個或多個 Stage",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="強制重跑所有 Stage（忽略 pipeline_state.json）",
    )

    # ── AWS 模式 ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--aws-mode", action="store_true",
        help="啟用 AWS 完整模式（Stage A + Stage Z）",
    )
    parser.add_argument("--s3-bucket",  default=os.environ.get("S3_BUCKET", ""),
                        help="S3 Bucket 名稱")
    parser.add_argument("--glue-job",     default=DEFAULT_GLUE_JOB,
                        help=f"Glue Job 名稱（預設：{DEFAULT_GLUE_JOB}）")
    parser.add_argument("--glue-crawler", default=DEFAULT_GLUE_CRAWLER,
                        help=f"Glue Crawler 名稱（預設：{DEFAULT_GLUE_CRAWLER}）")

    # ── 資料來源 ────────────────────────────────────────────────────────
    parser.add_argument(
        "--csv-dir", default=None, metavar="PATH",
        help="本地 CSV 資料目錄（若未指定，從 BitoPro API 讀取）",
    )

    # ── 訓練設定 ────────────────────────────────────────────────────────
    parser.add_argument("--train-mode", choices=["hpo"], default="hpo",
                        help="訓練模式：hpo=HyperparameterTuner（預設）")
    parser.add_argument("--hpo-max-jobs",  type=int, default=10,
                        help="HPO 總 Job 數（預設 10）")
    parser.add_argument("--hpo-parallel",  type=int, default=2,
                        help="HPO 並行 Job 數（預設 2）")
    parser.add_argument("--job-name", default=None,
                        help="指定 Training Job 名稱（Stage 3 用，跳過 Stage 2 時）")
    parser.add_argument("--skip-registry", action="store_true",
                        help="略過 Model Registry 版本註冊")

    return parser.parse_args()


def main() -> int:
    """
    管線主控函數。

    回傳：
        0 = 成功完成
        1 = 發生錯誤
    """
    args  = _parse_args()
    state = load_state()

    if args.force:
        log.info("[main] --force 模式：清除所有 Stage 完成記錄")
        state["completed"] = []
        save_state(state)

    # ── 決定要執行哪些 Stage ─────────────────────────────────────────────
    if args.only:
        stages_to_run = args.only
        log.info("[main] --only 模式：只執行 %s", stages_to_run)
    else:
        # AWS 模式加入前後兩個雲端 Stage
        all_stages = (
            ["aws_ingest"] + STAGES + ["aws_validate_submit"]
            if args.aws_mode else STAGES
        )
        if args.start_from:
            try:
                start_idx   = all_stages.index(args.start_from)
                stages_to_run = all_stages[start_idx:]
                log.info("[main] --start-from %s：將執行 %s",
                         args.start_from, stages_to_run)
            except ValueError:
                log.error("[main] 無效的 --start-from Stage：%s", args.start_from)
                return 1
        else:
            stages_to_run = all_stages

    # ── Stage 函數映射 ────────────────────────────────────────────────────
    stage_fn = {
        "aws_ingest":           lambda: aws_stage_ingest(state, args),
        "feature":              lambda: stage_feature(state, args),
        "train":                lambda: stage_train(state, args),
        "download":             lambda: stage_download(state, args),
        "validate":             lambda: stage_validate(state, args),
        "visualize":            lambda: stage_visualize(state, args),
        "xai":                  lambda: stage_xai(state, args),
        "aws_validate_submit":  lambda: aws_stage_validate_submission(state),
    }

    # ── 依序執行 ─────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("[main] BitoGuard AML 管線啟動")
    log.info("[main] 模式：%s", "AWS" if args.aws_mode else "本地")
    log.info("[main] 待執行 Stage：%s", stages_to_run)
    log.info("=" * 60)

    overall_start = time.time()

    for stage in stages_to_run:
        fn = stage_fn.get(stage)
        if fn is None:
            log.warning("[main] 未知 Stage '%s'，跳過", stage)
            continue

        stage_start = time.time()
        try:
            fn()
            elapsed = time.time() - stage_start
            log.info("[main] ✓ Stage '%s' 完成（耗時 %.1fs）", stage, elapsed)
        except KeyboardInterrupt:
            log.warning("[main] 使用者中斷（Ctrl+C）")
            log.info("[main] 目前進度已儲存至 %s，可使用 --start-from %s 繼續",
                     STATE_FILE, stage)
            return 1
        except Exception:
            elapsed = time.time() - stage_start
            log.error("[main] ✗ Stage '%s' 失敗（耗時 %.1fs）", stage, elapsed)
            log.error(traceback.format_exc())
            log.info("[main] 目前進度已儲存，可使用 --start-from %s 重試", stage)
            return 1

    total_elapsed = time.time() - overall_start
    log.info("=" * 60)
    log.info("[main] ✓ 所有 Stage 完成，總耗時 %.1fs（%.1f 分鐘）",
             total_elapsed, total_elapsed / 60)
    log.info("[main] 輸出檔案：")
    for artifact, path in state.get("artifacts", {}).items():
        if isinstance(path, str) and os.path.exists(path):
            log.info("  %-25s %s", artifact + ":", path)
    log.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
