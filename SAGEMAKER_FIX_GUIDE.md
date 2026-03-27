# SageMaker IAM 權限修復指南

## 問題概述

之前的錯誤：
```
botocore.exceptions.ClientError: An error occurred (ValidationException) when calling the 
CreateTrainingJob operation: Could not assume role 
arn:aws:iam::945349301775:role/AmazonSageMakerServiceCatalogProductsUseRole. 
Please ensure that the role exists and allows principal 'sagemaker.amazonaws.com' to assume the role.
```

**根本原因**：`AmazonSageMakerServiceCatalogProductsUseRole` 缺少正確的 trust relationship，無法被 SageMaker 服務假扮（assume）。

## 解決方案

已實施兩個關鍵修改：

### 1. `train_sagemaker.py` - 新增自動角色偵測函數

```python
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
```

**優勢**：
- ✅ 優先使用 Notebook 本身的 IAM 角色（已被 SageMaker 信任）
- ✅ 避免 trust relationship 問題
- ✅ 自動降級到環境變數（若有設定）
- ✅ 完全向後相容

### 2. `main_pipeline.py` - 恢復 SageMaker 訓練流程

`stage_train()` 函數已從本地訓練恢復為 SageMaker 訓練，並使用新的角色偵測機制：

```python
# 自動取得 IAM 角色（優先使用 Notebook 角色）
role_arn = _get_sagemaker_role()
log.info(f"[IAM] 使用角色：{role_arn}")

# 建立 SageMaker XGBoost Estimator
estimator = XGBoost(
    entry_point="train_xgboost_script.py",
    source_dir=os.path.dirname(os.path.abspath(__file__)),
    framework_version="1.7-1",
    role=role_arn,  # ← 使用自動偵測的角色
    instance_count=1,
    instance_type=INSTANCE_TYPE,
    output_path=output_uri,
    hyperparameters={...},
    sagemaker_session=session,
)
```

## 測試步驟

### 步驟 1：清除舊的管線狀態

```bash
rm -f pipeline_state.json
```

### 步驟 2：執行訓練階段

```bash
python main_pipeline.py --start-from train --csv-dir ./data
```

**預期輸出**：
```
2026-03-26 XX:XX:XX,XXX [INFO] ===============================================
2026-03-26 XX:XX:XX,XXX [INFO] Stage 2 / 6  SageMaker XGBoost 訓練
2026-03-26 XX:XX:XX,XXX [INFO] ===============================================
2026-03-26 XX:XX:XX,XXX [INFO] [SageMaker] 準備訓練資料上傳至 S3...
2026-03-26 XX:XX:XX,XXX [INFO] [IAM] 使用 Notebook 的 IAM 角色：arn:aws:iam::945349301775:role/BaseNotebookInstanceEc2InstanceRole
2026-03-26 XX:XX:XX,XXX [INFO] [S3] 訓練資料已上傳 → s3://sagemaker-us-west-2-945349301775/bito-mule-detection/...
2026-03-26 XX:XX:XX,XXX [INFO] [SageMaker] 建立 XGBoost Estimator...
2026-03-26 XX:XX:XX,XXX [INFO] [SageMaker] 啟動訓練作業...
2026-03-26 XX:XX:XX,XXX [INFO] [SageMaker] 訓練完成 → Job: sagemaker-xgboost-2026-03-26-XX-XX-XX-XXXX
2026-03-26 XX:XX:XX,XXX [INFO] [SageMaker] 模型位置 → s3://sagemaker-us-west-2-945349301775/bito-mule-detection/.../output/model.tar.gz
```

### 步驟 3：驗證訓練成功

檢查以下檔案是否生成：
- ✅ `cv_report.json` - 5-Fold CV 報告
- ✅ `pipeline_state.json` - 更新的管線狀態（train 標記為 completed）
- ✅ S3 中的模型檔案

### 步驟 4：繼續後續階段

```bash
# 下載模型
python main_pipeline.py --start-from download --csv-dir ./data

# 驗證與視覺化
python main_pipeline.py --start-from validate --csv-dir ./data

# 生成診斷書
python main_pipeline.py --start-from xai --csv-dir ./data
```

## 故障排除

### 問題 1：仍然出現 IAM 錯誤

**症狀**：
```
Could not assume role arn:aws:iam::945349301775:role/AmazonSageMakerServiceCatalogProductsUseRole
```

**解決方案**：
1. 確認 Notebook 已重新啟動（清除 Python 快取）
2. 檢查 Notebook 的 IAM 角色是否有 SageMaker 權限：
   ```bash
   aws sts get-caller-identity
   ```
   應該顯示 `BaseNotebookInstanceEc2InstanceRole`

3. 若仍失敗，手動設定環境變數（不推薦）：
   ```bash
   export SAGEMAKER_ROLE_ARN=$(aws sts get-caller-identity --query Arn --output text | sed 's/:sts:/:iam:/;s/assumed-role.*/role\/SageMakerRole/')
   ```

### 問題 2：訓練超時

**症狀**：訓練作業卡住超過 10 分鐘

**解決方案**：
1. 檢查 SageMaker 主控台是否有訓練作業在執行
2. 檢查 S3 是否有訓練資料上傳
3. 檢查 CloudWatch Logs 中的訓練作業日誌

### 問題 3：S3 上傳失敗

**症狀**：
```
[S3] 訓練資料已上傳 → ... 未出現
```

**解決方案**：
1. 確認 S3 bucket 名稱正確：
   ```bash
   echo $S3_BUCKET
   ```
2. 確認 Notebook IAM 角色有 S3 PutObject 權限
3. 檢查 bucket 是否存在：
   ```bash
   aws s3 ls s3://$S3_BUCKET/
   ```

## 技術細節

### 為什麼使用 Notebook 角色？

1. **信任關係已建立**：Notebook 角色由 SageMaker 建立，已有正確的 trust relationship
2. **權限充足**：Notebook 角色通常有 SageMaker 完整權限
3. **無需額外配置**：自動從 SageMaker Session 取得，無需手動設定

### 角色優先順序

```
1. 環境變數 SAGEMAKER_ROLE_ARN（若非預設值）
   ↓
2. SageMaker Session.get_execution_role()（Notebook 角色）
   ↓
3. 環境變數 SAGEMAKER_ROLE_ARN（預設值）
   ↓
4. 硬編碼預設值（最後手段）
```

## 相關檔案修改

| 檔案 | 修改內容 |
|---|---|
| `train_sagemaker.py` | 新增 `_get_sagemaker_role()` 函數 |
| `main_pipeline.py` | 恢復 `stage_train()` 使用 SageMaker 訓練 |

## 驗證修改

```bash
# 檢查語法
python -m py_compile main_pipeline.py train_sagemaker.py

# 檢查導入
python -c "from train_sagemaker import _get_sagemaker_role; print('✓ 導入成功')"
```

## 下一步

1. ✅ 執行訓練：`python main_pipeline.py --start-from train --csv-dir ./data`
2. ✅ 驗證成功後，繼續完整管線
3. ✅ 生成最終提交檔：`submission.csv`
