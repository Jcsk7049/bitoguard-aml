"""
S3 模型下載腳本

SageMaker 訓練完成後，模型存為 s3://{bucket}/{prefix}/{timestamp}/output/model.tar.gz
此腳本負責：
  1. 列出該 prefix 下最新的訓練輸出
  2. 下載 model.tar.gz
  3. 解壓取出 xgboost-model（XGBoost Booster 原生格式）
  4. 儲存為本地 model.json，供 xai_bedrock.py 載入

使用方式：
    python download_model.py                          # 自動找最新
    python download_model.py --job my-xgboost-job    # 指定訓練 job 名稱
    python download_model.py --s3-uri s3://bucket/prefix/ts/output/model.tar.gz
"""

import io
import os
import sys
import tarfile
import argparse
import boto3
import xgboost as xgb
from datetime import datetime

# ══════════════════════════════════════════════════════════════════════════════
#  Region 合規檢核（競賽規定：僅允許 us-east-1 / us-west-2）
# ══════════════════════════════════════════════════════════════════════════════

_ALLOWED_REGIONS: frozenset[str] = frozenset({"us-east-1", "us-west-2"})

def _validate_region(region: str) -> str:
    """
    Region 合規閘門：若 region 不在核准清單中，立即拋出 RuntimeError 並終止程序。

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


# ── 預設值（與 train_sagemaker.py 一致） ────────────────────────────────────

S3_BUCKET  = os.environ.get("S3_BUCKET", "your-hackathon-bucket")
S3_PREFIX  = "bito-mule-detection"
REGION     = _validate_region(os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
LOCAL_PATH = "model.json"


# ── 工具函式 ─────────────────────────────────────────────────────────────────

def list_training_outputs(s3, bucket: str, prefix: str) -> list[dict]:
    """
    列出 s3://bucket/prefix/*/output/model.tar.gz，依 LastModified 排序。
    回傳 [{"key": ..., "last_modified": ..., "size": ...}, ...]
    """
    paginator = s3.get_paginator("list_objects_v2")
    results   = []

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix + "/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("output/model.tar.gz"):
                results.append({
                    "key":           key,
                    "last_modified": obj["LastModified"],
                    "size_mb":       round(obj["Size"] / 1024 / 1024, 2),
                })

    return sorted(results, key=lambda x: x["last_modified"], reverse=True)


def find_latest_model_key(s3, bucket: str, prefix: str) -> str:
    """回傳最新一次訓練的 model.tar.gz S3 key。"""
    outputs = list_training_outputs(s3, bucket, prefix)
    if not outputs:
        raise FileNotFoundError(
            f"在 s3://{bucket}/{prefix} 下找不到任何 model.tar.gz。\n"
            "請確認訓練已完成，或使用 --s3-uri 手動指定路徑。"
        )

    latest = outputs[0]
    print(f"[找到 {len(outputs)} 個訓練輸出，使用最新：]")
    for i, o in enumerate(outputs[:5]):
        marker = " ← 使用此版本" if i == 0 else ""
        print(f"  {o['last_modified'].strftime('%Y-%m-%d %H:%M:%S')}  "
              f"{o['size_mb']:>6.1f} MB  {o['key']}{marker}")

    return latest["key"]


def find_model_key_by_job(sm, job_name: str) -> str:
    """透過 SageMaker Training Job 名稱取得 model artifact S3 URI。"""
    desc = sm.describe_training_job(TrainingJobName=job_name)
    uri  = desc["ModelArtifacts"]["S3ModelArtifacts"]
    # s3://bucket/key → bucket, key
    without_scheme = uri[len("s3://"):]
    bucket, key    = without_scheme.split("/", 1)
    print(f"[Training Job] {job_name}")
    print(f"  ModelArtifacts → {uri}")
    return bucket, key


def download_and_extract(
    s3,
    bucket: str,
    key: str,
    local_model_path: str = LOCAL_PATH,
) -> str:
    """
    下載 model.tar.gz，解壓後取出 XGBoost 模型檔，轉存為 model.json。

    SageMaker XGBoost 內建容器產出的 tar.gz 結構：
        model.tar.gz
        └── xgboost-model      ← XGBoost Booster 原生二進位格式
    """
    print(f"\n[下載] s3://{bucket}/{key}")
    obj      = s3.get_object(Bucket=bucket, Key=key)
    tar_bytes = obj["Body"].read()
    size_mb   = round(len(tar_bytes) / 1024 / 1024, 2)
    print(f"  大小：{size_mb} MB")

    # 解壓縮（in-memory）
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
        members = tar.getnames()
        print(f"  tar 內容：{members}")

        # 找模型檔（可能叫 xgboost-model 或 model）
        model_member = next(
            (m for m in members if "xgboost-model" in m or m == "model"),
            members[0],
        )
        raw = tar.extractfile(model_member).read()

    # 載入為 Booster，存成 JSON（跨版本相容性更好）
    booster = xgb.Booster()
    booster.load_model(bytearray(raw))
    booster.save_model(local_model_path)

    print(f"  模型已儲存 → {local_model_path}")
    return local_model_path


def verify_model(path: str, n_features: int | None = None) -> None:
    """載入驗證：確認 Booster 可正常運作。"""
    booster = xgb.Booster()
    booster.load_model(path)

    feature_names = booster.feature_names
    num_trees     = booster.num_boosted_rounds()

    print(f"\n[驗證] {path}")
    print(f"  樹的數量      : {num_trees}")
    print(f"  特徵數量      : {len(feature_names) if feature_names else '未知'}")
    if feature_names:
        print(f"  前 5 個特徵  : {feature_names[:5]}")

    if n_features and feature_names and len(feature_names) != n_features:
        print(f"  ⚠ 警告：期望 {n_features} 個特徵，模型有 {len(feature_names)} 個")
    else:
        print("  ✓ 模型載入正常")


# ── 主程式 ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="從 S3 下載 SageMaker XGBoost 模型")
    parser.add_argument("--s3-uri",   default=None,
                        help="直接指定 s3://bucket/key/model.tar.gz")
    parser.add_argument("--job",      default=None,
                        help="SageMaker Training Job 名稱")
    parser.add_argument("--bucket",   default=S3_BUCKET)
    parser.add_argument("--prefix",   default=S3_PREFIX)
    parser.add_argument("--output",   default=LOCAL_PATH,
                        help=f"本地模型輸出路徑（預設：{LOCAL_PATH}）")
    parser.add_argument("--list",     action="store_true",
                        help="只列出可用的訓練輸出，不下載")
    args = parser.parse_args()

    boto_sess = boto3.Session(region_name=REGION)
    s3        = boto_sess.client("s3")
    sm        = boto_sess.client("sagemaker")

    # ── 僅列出 ────────────────────────────────────────────────────────────
    if args.list:
        outputs = list_training_outputs(s3, args.bucket, args.prefix)
        if not outputs:
            print("找不到任何訓練輸出。")
        for o in outputs:
            print(f"  {o['last_modified'].strftime('%Y-%m-%d %H:%M:%S')}"
                  f"  {o['size_mb']:>6.1f} MB  s3://{args.bucket}/{o['key']}")
        return

    # ── 決定 bucket / key ─────────────────────────────────────────────────
    if args.s3_uri:
        without_scheme   = args.s3_uri[len("s3://"):]
        bucket, key      = without_scheme.split("/", 1)
    elif args.job:
        bucket, key = find_model_key_by_job(sm, args.job)
    else:
        key    = find_latest_model_key(s3, args.bucket, args.prefix)
        bucket = args.bucket

    # ── 下載 & 儲存 ───────────────────────────────────────────────────────
    local_path = download_and_extract(s3, bucket, key, args.output)
    verify_model(local_path)

    print(f"\n完成。後續使用方式：")
    print(f"  import xgboost as xgb")
    print(f"  model = xgb.Booster()")
    print(f"  model.load_model('{local_path}')")


if __name__ == "__main__":
    main()
