# BitoGuard AML 系統 — 技術設計文件

> 版本：v1.0（2026-03-25）

---

## 架構概覽

```
┌─────────────────────────────────────────────────────────────────────┐
│                     BitoGuard 系統架構                               │
│                                                                     │
│  [BitoPro API]                                                      │
│       │ bito_api_ingester.py（Checkpoint / 退避重試）               │
│       ▼                                                             │
│  [S3 Raw Parquet]  ← Hive 分區（year/month/day）                    │
│       │                                                             │
│       ├─ [Athena]  ← athena_graph_hops.sql（BFS 輕量版）            │
│       │                                                             │
│       └─ [Glue PySpark] ← glue_graph_hops.py（BFS 完整版 + 加鹽）  │
│               │                                                     │
│       [Feature Store] ← feature_store.py                           │
│               │                                                     │
│       [SageMaker Training] ← train_sagemaker.py + train_xgboost_script.py
│               │ HPO Tuner（Bayesian，10 jobs，並行 2）              │
│               ▼                                                     │
│       [Model Registry]（F1 ≥ 0.90 自動核准）                       │
│               │                                                     │
│       [Batch Transform] → submission_with_prob.csv → S3            │
│               │                                                     │
│               ▼（S3 Event Trigger）                                 │
│       [Lambda] ← lambda_diagnosis.py                               │
│               │ _classify() → EXTREME/HIGH/BOUNDARY/MEDIUM/LOW     │
│               ├─ EXTREME  → Haiku（制式報告）                       │
│               ├─ HIGH     → Sonnet（標準分析）                      │
│               ├─ BOUNDARY → Sonnet（深度診斷）                      │
│               └─ MEDIUM+  → Rule-Based                             │
│                       │                                             │
│               [Bedrock]（1.1s 間隔 / 指數退避）                     │
│                       │                                             │
│               [DynamoDB]（bito-diagnoses / 90d TTL）               │
│                       │                                             │
│               [SNS] → 合規主管告警                                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 關鍵設計決策

### 1. 超級節點加鹽策略（Salting for Data Skew）

```
Combined Degree > 10,000 → 隨機後綴 (F.rand() × 16)
正常節點              → salt = 0（單一分區）
BFS crossJoin 時遍歷全部 salt 值（0–15），確保語義正確
```

### 2. Bedrock 三層速率防護

```
L1: _BedrockRateLimiter.acquire()  → min_interval = 1.1s（mutex-based）
L2: ThrottlingException 指數退避   → 2, 4, 8, 16, 32s（最多 5 次）
L3: 超過重試上限                   → 降級 Rule-Based（不中斷批次）
```

### 3. PII 三層過濾

```
L1: pii_filter(record) → 欄位層：移除 source_ip, email, full_name 等
L2: _top5_by_shap()    → 特徵層：僅傳送 Top 5 SHAP 數值特徵
L3: scan_prompt_for_pii() → 正則層：IP / Email / Phone / ID 遮罩
```

### 4. CheckPoint 一致性設計

```
僅在 Part Flush 後（非每頁後）儲存 checkpoint，
確保恢復時 Parquet 檔案 + cursor 狀態完全一致。
原子寫入：tmp 檔案 → os.replace()（防止寫入中斷產生損壞的 checkpoint）
```

---

## 資料模型

### DynamoDB Schema（bito-diagnoses）

```
PK: user_id (String)
SK: generated_at (String, ISO-8601)

屬性：
  risk_level          String    HIGH / MEDIUM / LOW
  scoring_tier        String    EXTREME / HIGH / BOUNDARY / MEDIUM
  probability         Number    0.0 – 1.0
  threat_pattern      String    MULE_ACCOUNT / LAYERING / CLUSTER_FRAUD / …
  compliance_summary  String    白話摘要（供合規主管）
  risk_factors        List      分項風險因子
  primary_action      String    FREEZE_ACCOUNT / LIMIT_WITHDRAWAL / …
  auto_executable     Boolean
  str_required        Boolean
  expire_at           Number    Unix timestamp（90 天 TTL）

GSI-1: risk_level-generated_at-index（高風險列表查詢）
GSI-2: scoring_tier-probability-index（同層最高機率排序）
```
