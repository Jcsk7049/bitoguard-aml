# BitoGuard AML 系統 — 實作任務追蹤

> 最後更新：2026-03-25

---

## 已完成任務

- [x] **T-01** 建立 BitoPro API 擷取器（`bito_api_ingester.py`）
  - Cursor 分頁、Checkpoint 斷點續傳、tenacity 動態退避
- [x] **T-02** S3 原始資料注入流程（`ingest_to_s3.py`）
  - Hive 分區、Glue Catalog、Region 合規驗證
- [x] **T-03** Glue PySpark 圖分析（`glue_graph_hops.py`）
  - BFS hop 計算、超級節點加鹽、EXCLUDED_NODES 過濾
- [x] **T-04** Athena SQL 圖分析（`athena_graph_hops.sql`）
  - 手動 Hop 展開、ip_shared_user_count、has_high_speed_risk
- [x] **T-05** SageMaker 訓練流程（`train_sagemaker.py` + `train_xgboost_script.py`）
  - HPO Tuner、代價敏感損失、對抗增強、Model Registry
- [x] **T-06** SHAP + Bedrock 診斷書生成（`xai_bedrock.py`）
  - 三層 PII 過濾、Rate Limiter（1.1s）、ThrottlingException 退避
  - compliance_summary / graph_risk_narrative / risk_factors 輸出
- [x] **T-07** Lambda 自動診斷（`lambda_diagnosis.py`）
  - S3 觸發、模型路由、DynamoDB 寫入、Rate Limiter
- [x] **T-08** 事件回應工作流（`incident_response_workflow.py`）
  - Incident 建立、Feedback Loop、增量重訓觸發
- [x] **T-09** 合規自動化檢核（`check_compliance.py`）
  - C-1 憑證掃描、C-2 S3 ACL、C-3 PII 過濾器驗證
- [x] **T-10** SAM 部署模板（`template.yaml`）
  - PublicAccessBlock、SSE、SQS SSE、Region 部署規則
- [x] **T-11** Kiro 專案整合（`.kiro/`）
  - steering（product / structure / tech）
  - specs（requirements / design / tasks）
  - hooks（compliance / pre-deploy）

---

- [x] **T-12** ML 邏輯優化（樣本不平衡 + 解釋性增強 + 圖譜視覺化）
  - `train_xgboost_script.py`：`_compute_scale_pos_weight()` 動態 SPW + `IsotonicCalibrator` 機率校正；`train_fold()` 回傳 `best_thresh`；`training_summary.json` 新增 `imbalance` / `calibration` 區塊
  - `xai_bedrock.py`：新增 `FEATURE_INTERPRETATION`（22 特徵自然語言模板）+ `_shap_to_narrative()` + `_feature_table()` 加入「風險解讀」欄
  - `visualize.py`：新增 `plot_blacklist_network()`（同心環佈局 + hop/風險等級側面板）；CLI 新增 `--hop-df-path` / `--target-user`

---

## 待辦任務

- [ ] **T-13** CloudWatch Dashboard 前端整合
  - 接入 DynamoDB 診斷書資料
  - 建立合規人員操作介面

- [ ] **T-14** SAM 部署驗證
  - `sam validate --template template.yaml`
  - `sam deploy --region us-east-1 --guided`

- [ ] **T-15** 端到端 E2E 測試
  - 以 BitoPro 測試資料執行完整 8 階段管線
  - 驗證 F1 ≥ 0.80（Model Registry 核准門檻）
