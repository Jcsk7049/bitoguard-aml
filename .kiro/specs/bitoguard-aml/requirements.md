# BitoGuard AML 系統 — 功能需求規格

> 格式：EARS（Easy Approach to Requirements Syntax）
> 狀態：v1.0（2026-03-25）

---

## 1. 資料擷取（Data Ingestion）

**REQ-1.1**
WHEN BitoPro API 回應包含新交易紀錄
THE SYSTEM SHALL 以 Cursor 分頁拉取所有交易資料，並以 Parquet 格式上傳至 S3（Hive 分區）。

**REQ-1.2**
IF 資料擷取過程中斷
THEN THE SYSTEM SHALL 從最後一次成功的 Part Flush 位置繼續，不得重頭抓取。

**REQ-1.3**
WHERE HTTP 429 Too Many Requests
THE SYSTEM SHALL 依據 Retry-After header 或指數退避（最長 5 分鐘）後重試。

---

## 2. 圖分析（Graph Analysis）

**REQ-2.1**
THE SYSTEM SHALL 對全量交易圖執行 BFS，計算每位用戶到已知黑名單的最短跳轉數（min_hops_to_blacklist）。

**REQ-2.2**
WHERE 單一節點的 Combined Degree 超過 10,000
THE SYSTEM SHALL 對該節點的所有關聯邊施加隨機鹽值（Salting），以分散 Spark 數據傾斜。

**REQ-2.3**
THE SYSTEM SHALL 計算以下衍生特徵：
- `ip_shared_user_count`：同一 IP 被幾個帳號使用（> 3 為機房訊號）
- `has_high_speed_risk`：是否存在 < 10 分鐘完成的交易（自動化腳本特徵）
- `weighted_risk_label`：HIGH_WEIGHTED / BLACKLIST / HIGH / MEDIUM / LOW

---

## 3. 模型訓練（Model Training）

**REQ-3.1**
THE SYSTEM SHALL 使用 XGBoost 訓練二元分類模型，目標變數為 `status`（0=正常 / 1=黑名單）。

**REQ-3.2**
THE SYSTEM SHALL 動態計算 `scale_pos_weight = count_0 / count_1`，不得硬編碼。

**REQ-3.3**
IF 訓練集中無正樣本（count_1 == 0）
THEN THE SYSTEM SHALL 記錄警告並使用預設值 1.0，不得拋出例外。

**REQ-3.4**
WHERE SageMaker Training Job 啟動前
THE SYSTEM SHALL 驗證當前 AWS Account 運行中的 Training Job 數量不超過 2 個。

---

## 4. 風險診斷（Risk Diagnosis）

**REQ-4.1**
THE SYSTEM SHALL 依風險機率路由至對應模型：
- P > 0.90（EXTREME）→ Claude 3 Haiku（制式凍結通知）
- 0.75 < P ≤ 0.90（HIGH）→ Claude 3.5 Sonnet（標準分析）
- 0.65 ≤ P ≤ 0.75（BOUNDARY）→ Claude 3.5 Sonnet（深度診斷）
- P < 0.65 → Rule-Based Only，不呼叫 LLM

**REQ-4.2**
THE SYSTEM SHALL 在診斷書中包含以下欄位：
- `compliance_summary`：白話摘要（供非技術合規人員閱讀）
- `graph_risk_narrative`：資金流向圖結論
- `risk_factors`：分項風險因子清單
- `risk_diagnosis`：技術風險成因分析
- `recommended_actions`：機器可執行的合規指令

**REQ-4.3**
THE SYSTEM SHALL 在每次 Bedrock 呼叫之間強制等待至少 1.1 秒（< 1 RPS）。

**REQ-4.4**
WHERE Bedrock ThrottlingException 發生
THE SYSTEM SHALL 以指數退避（2s, 4s, 8s, 16s, 32s）重試，最多 5 次；超過上限後降級至 Rule-Based。

---

## 5. PII 防護（Privacy Protection）

**REQ-5.1**
THE SYSTEM SHALL 在傳送至 Bedrock 前，強制過濾以下個人識別資訊：
- 姓名、電子郵件、電話號碼、身份證字號
- 完整 IP 位址（僅傳送數值特徵如 `ip_shared_user_count`）
- 錢包地址、銀行帳號

**REQ-5.2**
THE SYSTEM SHALL 僅允許數值型特徵與標籤（`user_id` 除外）進入 Prompt。

**REQ-5.3**
THE SYSTEM SHALL 以正則引擎（`scan_prompt_for_pii()`）在送出前掃描 Prompt，遮罩任何殘留 PII。

---

## 6. 安全與合規（Security & Compliance）

**REQ-6.1**
THE SYSTEM SHALL 僅部署至 `us-east-1` 或 `us-west-2`；違規 Region 在模組載入時立即拋出例外。

**REQ-6.2**
THE SYSTEM SHALL 不得在程式碼中硬編碼任何 AWS Access Key、Secret Key 或密碼。

**REQ-6.3**
THE SYSTEM SHALL 確保所有 S3 Bucket 啟用 PublicAccessBlockConfiguration（四項均為 true）。

**REQ-6.4**
THE SYSTEM SHALL 僅呼叫白名單內的 Bedrock 模型（`_ALLOWED_MODELS`），拒絕任何其他模型的初始化。
