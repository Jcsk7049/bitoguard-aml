-- ============================================================
-- Amazon Athena SQL — 資金關聯深度 BFS（兩種實作）
-- ============================================================
--
-- 前置條件：
--   1. Glue Data Catalog 中已建立以下表格（對應 S3 Parquet）：
--        bito_raw.twd_transfer
--        bito_raw.crypto_transfer
--        bito_raw.train_label       (user_id, status)
--   2. 結果寫入 bito_features.hop_features（透過 CTAS）
--
-- 執行方式：
--   aws athena start-query-execution \
--       --query-string "file://athena_graph_hops.sql" \
--       --result-configuration "OutputLocation=s3://your-bucket/athena-results/"
--
-- Athena 引擎版本：3（Trino 395）
-- 注意：Athena 的 WITH RECURSIVE 最大遞迴深度約 10 層，
--       大圖建議使用「方案二：手動 Hop 展開」。
-- ============================================================


-- ════════════════════════════════════════════════════════════
--  前置：建立邊列表視圖（雙向，去重）
--  同時合併 twd_transfer 與 crypto_transfer 的關聯邊
-- ════════════════════════════════════════════════════════════

CREATE OR REPLACE VIEW bito_features.v_transaction_edges AS

-- 1. 明確交易對象邊（twd_transfer）
SELECT
    CAST(user_id            AS BIGINT) AS src,
    CAST(relation_user_id   AS BIGINT) AS dst
FROM bito_raw.twd_transfer
WHERE relation_user_id IS NOT NULL
  AND user_id != relation_user_id

UNION ALL

-- 反向（無向圖）
SELECT
    CAST(relation_user_id   AS BIGINT) AS src,
    CAST(user_id            AS BIGINT) AS dst
FROM bito_raw.twd_transfer
WHERE relation_user_id IS NOT NULL
  AND user_id != relation_user_id

UNION ALL

-- 2. 明確交易對象邊（crypto_transfer）
SELECT
    CAST(user_id            AS BIGINT) AS src,
    CAST(relation_user_id   AS BIGINT) AS dst
FROM bito_raw.crypto_transfer
WHERE relation_user_id IS NOT NULL
  AND user_id != relation_user_id

UNION ALL

SELECT
    CAST(relation_user_id   AS BIGINT) AS src,
    CAST(user_id            AS BIGINT) AS dst
FROM bito_raw.crypto_transfer
WHERE relation_user_id IS NOT NULL
  AND user_id != relation_user_id

UNION ALL

-- 3. 錢包地址共用邊（共用 to_wallet → 隱含關聯）
SELECT
    CAST(a.user_id AS BIGINT) AS src,
    CAST(b.user_id AS BIGINT) AS dst
FROM bito_raw.crypto_transfer a
JOIN bito_raw.crypto_transfer b
  ON a.to_wallet = b.to_wallet
 AND a.user_id  != b.user_id
WHERE a.to_wallet IS NOT NULL

UNION ALL

-- 4. User-to-IP 邊（正向：user_id → ip_node）
--    IP 節點 ID = -(ABS(xxhash64(ip)) + 1)，負數空間與正數 user_id 不衝突
--    用途：BFS 可穿越 IP 中繼節點，2 跳即可連結「同 IP 的不同用戶」
SELECT
    CAST(user_id AS BIGINT)                              AS src,
    -(ABS(xxhash64(to_utf8(source_ip))) + 1)            AS dst
FROM bito_raw.twd_transfer
WHERE source_ip IS NOT NULL
  AND user_id   IS NOT NULL

UNION ALL

-- 反向（ip_node → user_id）
SELECT
    -(ABS(xxhash64(to_utf8(source_ip))) + 1)            AS src,
    CAST(user_id AS BIGINT)                              AS dst
FROM bito_raw.twd_transfer
WHERE source_ip IS NOT NULL
  AND user_id   IS NOT NULL

UNION ALL

SELECT
    CAST(user_id AS BIGINT)                              AS src,
    -(ABS(xxhash64(to_utf8(source_ip))) + 1)            AS dst
FROM bito_raw.crypto_transfer
WHERE source_ip IS NOT NULL
  AND user_id   IS NOT NULL

UNION ALL

SELECT
    -(ABS(xxhash64(to_utf8(source_ip))) + 1)            AS src,
    CAST(user_id AS BIGINT)                              AS dst
FROM bito_raw.crypto_transfer
WHERE source_ip IS NOT NULL
  AND user_id   IS NOT NULL
;


-- ════════════════════════════════════════════════════════════
--  前置：黑名單節點視圖
-- ════════════════════════════════════════════════════════════

CREATE OR REPLACE VIEW bito_features.v_blacklist_nodes AS
SELECT DISTINCT CAST(user_id AS BIGINT) AS node
FROM   bito_raw.train_label
WHERE  status = '1'
;


-- ════════════════════════════════════════════════════════════
--  【方案一】WITH RECURSIVE BFS（適合 max_hops <= 5）
--
--  原理：Presto/Trino 的遞迴 CTE 在 Athena Engine v3 支援。
--  限制：每次遞迴展開皆產生全域掃描，超過 ~5 層效能急劇下降。
--        實際資料若邊數超過 1 億，建議改用方案二。
-- ════════════════════════════════════════════════════════════

WITH RECURSIVE

-- ① 去重後的邊列表（本 SQL 中使用 view 取代）
edges AS (
    SELECT DISTINCT src, dst
    FROM   bito_features.v_transaction_edges
),

-- ② BFS：從黑名單出發，每輪擴展一跳
bfs(node, hops) AS (

    -- Anchor：黑名單節點，距離 = 0
    SELECT node, 0 AS hops
    FROM   bito_features.v_blacklist_nodes

    UNION

    -- Recursive：展開到鄰居，距離 +1
    SELECT
        e.dst           AS node,
        b.hops + 1      AS hops
    FROM   bfs b
    JOIN   edges e ON e.src = b.node
    WHERE  b.hops < 3          -- ← max_hops，依需求調整
),

-- ③ 每個節點取最小跳轉數（去除遞迴中的重複路徑）
min_hops AS (
    SELECT
        node,
        MIN(hops) AS min_hops_to_blacklist
    FROM   bfs
    GROUP BY node
),

-- ④ 所有出現在邊中的節點（含孤立節點需另行補全）
all_nodes AS (
    SELECT DISTINCT src AS node FROM edges
    UNION
    SELECT DISTINCT dst         FROM edges
    UNION
    SELECT DISTINCT node        FROM bito_features.v_blacklist_nodes
),

-- ⑤ 直接相連的黑名單節點計數
blacklist_neighbor_cnt AS (
    SELECT
        e.src                               AS node,
        COUNT(DISTINCT e.dst)               AS blacklist_neighbor_count
    FROM   edges e
    JOIN   bito_features.v_blacklist_nodes bl ON e.dst = bl.node
    GROUP BY e.src
)

-- ⑥ 組裝最終輸出
SELECT
    an.node                                                         AS user_id,
    COALESCE(mh.min_hops_to_blacklist, 4)                          AS min_hops_to_blacklist,
    COALESCE(mh.min_hops_to_blacklist, 4) = 1                      AS is_direct_neighbor,
    COALESCE(bnc.blacklist_neighbor_count, 0)                      AS blacklist_neighbor_count,
    CASE
        WHEN COALESCE(mh.min_hops_to_blacklist, 4) = 0 THEN 'blacklist'
        WHEN COALESCE(mh.min_hops_to_blacklist, 4) = 1 THEN 'direct'
        WHEN COALESCE(mh.min_hops_to_blacklist, 4) = 2 THEN 'indirect_2'
        WHEN COALESCE(mh.min_hops_to_blacklist, 4) = 3 THEN 'indirect_3'
        ELSE 'isolated'
    END                                                             AS hop_risk_level,
    CURRENT_TIMESTAMP                                               AS event_time
FROM   all_nodes an
LEFT JOIN min_hops            mh  ON an.node = mh.node
LEFT JOIN blacklist_neighbor_cnt bnc ON an.node = bnc.node
;


-- ════════════════════════════════════════════════════════════
--  【方案二】手動 Hop 展開 CTE（推薦：數億列穩定執行）
--
--  原理：不使用遞迴，逐跳用顯式 CTE 展開 BFS，
--        Athena/Presto 對每個 CTE 獨立優化，可充分利用
--        MPP（大規模平行處理）能力。
--
--  優勢：
--    - 每跳 CTE 可獨立被優化器優化（索引、分區裁剪）
--    - 無遞迴深度限制
--    - 可在 CTAS 中物化中間結果，避免重複掃描
--    - 與 Athena Partition Projection 完全相容
--
--  適用場景：max_hops = 3（標準設定），邊數 1 億+ 列
-- ════════════════════════════════════════════════════════════

CREATE TABLE bito_features.hop_features
WITH (
    format           = 'PARQUET',
    write_compression = 'SNAPPY',
    external_location = 's3://your-bucket/features/hop_features/',
    partitioned_by   = ARRAY['hop_risk_level']
)
AS

WITH

-- ──────────────────────────────────────────────────
--  S0：去重邊列表（物化以避免重複掃描）
--      在 Athena 中用 CTAS 代替 CREATE VIEW 效能更佳
-- ──────────────────────────────────────────────────
edges AS (
    SELECT DISTINCT
        CAST(src AS BIGINT) AS src,
        CAST(dst AS BIGINT) AS dst
    FROM   bito_features.v_transaction_edges
),

-- ──────────────────────────────────────────────────
--  S1：黑名單節點（Hop 0）
-- ──────────────────────────────────────────────────
hop0 AS (
    SELECT DISTINCT
        node    AS node,
        0       AS hops
    FROM bito_features.v_blacklist_nodes
),

-- ──────────────────────────────────────────────────
--  S2：Hop 1（直接鄰居）
--      = 與黑名單直接有邊的節點，排除黑名單本身
-- ──────────────────────────────────────────────────
hop1_raw AS (
    SELECT e.dst AS node, 1 AS hops
    FROM   edges e
    JOIN   hop0  h ON e.src = h.node
),
hop1 AS (
    SELECT node, hops
    FROM   hop1_raw
    WHERE  node NOT IN (SELECT node FROM hop0)
),

-- ──────────────────────────────────────────────────
--  S3：Hop 2（透過一個中間人）
--      = hop1 節點的鄰居，排除 hop0 和 hop1
-- ──────────────────────────────────────────────────
hop2_raw AS (
    SELECT e.dst AS node, 2 AS hops
    FROM   edges e
    JOIN   hop1  h ON e.src = h.node
),
hop2 AS (
    SELECT node, hops
    FROM   hop2_raw
    -- 使用 NOT EXISTS 在大表上效能優於 NOT IN
    WHERE  NOT EXISTS (SELECT 1 FROM hop0 h0 WHERE h0.node = hop2_raw.node)
      AND  NOT EXISTS (SELECT 1 FROM hop1 h1 WHERE h1.node = hop2_raw.node)
),

-- ──────────────────────────────────────────────────
--  S4：Hop 3（透過兩個中間人）
-- ──────────────────────────────────────────────────
hop3_raw AS (
    SELECT e.dst AS node, 3 AS hops
    FROM   edges e
    JOIN   hop2  h ON e.src = h.node
),
hop3 AS (
    SELECT node, hops
    FROM   hop3_raw
    WHERE  NOT EXISTS (SELECT 1 FROM hop0 h0 WHERE h0.node = hop3_raw.node)
      AND  NOT EXISTS (SELECT 1 FROM hop1 h1 WHERE h1.node = hop3_raw.node)
      AND  NOT EXISTS (SELECT 1 FROM hop2 h2 WHERE h2.node = hop3_raw.node)
),

-- ──────────────────────────────────────────────────
--  S5：合併所有 Hop，每個節點取最小跳轉數
--      （因同一節點可能從多個黑名單節點可達）
-- ──────────────────────────────────────────────────
all_reached AS (
    SELECT node, hops FROM hop0
    UNION ALL SELECT node, hops FROM hop1
    UNION ALL SELECT node, hops FROM hop2
    UNION ALL SELECT node, hops FROM hop3
),

min_hops AS (
    SELECT
        node,
        MIN(hops) AS min_hops_to_blacklist
    FROM   all_reached
    GROUP BY node
),

-- ──────────────────────────────────────────────────
--  S6：孤立節點補全
--      = 出現在邊列表但 BFS 沒有到達的節點 → Hop = 4 (ISOLATED)
--      注意：過濾 node > 0，排除 IP 中繼節點（負數 ID）
-- ──────────────────────────────────────────────────
all_nodes AS (
    SELECT DISTINCT src AS node FROM edges WHERE src > 0
    UNION
    SELECT DISTINCT dst         FROM edges WHERE dst > 0
    UNION
    SELECT DISTINCT node        FROM bito_features.v_blacklist_nodes
),

all_nodes_with_hops AS (
    SELECT
        an.node,
        COALESCE(mh.min_hops_to_blacklist, 4) AS min_hops_to_blacklist
    FROM   all_nodes an
    LEFT JOIN min_hops mh ON an.node = mh.node
),

-- ──────────────────────────────────────────────────
--  S7：直接黑名單鄰居計數（衍生特徵）
-- ──────────────────────────────────────────────────
bl_neighbor_cnt AS (
    SELECT
        e.src                       AS node,
        COUNT(DISTINCT e.dst)       AS blacklist_neighbor_count
    FROM   edges e
    JOIN   bito_features.v_blacklist_nodes bl
        ON e.dst = bl.node
    GROUP BY e.src
),

-- ──────────────────────────────────────────────────
--  S8：Union-Find 近似（Athena 版）
--      原理：找出每個節點所在的「最小節點 ID 連通分量」
--            透過兩跳 JOIN 做一次 Small-Star 近似（不完整，僅供輔助）
--      完整 Union-Find 請用 Glue PySpark（glue_graph_hops.py）
-- ──────────────────────────────────────────────────
one_hop_min AS (
    SELECT
        e.src                           AS node,
        MIN(e.dst)                      AS min_neighbor
    FROM   edges e
    GROUP BY e.src
),

-- 若鄰居中有黑名單成員，該節點被標記為「在黑名單網路中」
in_bl_network AS (
    SELECT DISTINCT
        e.src                           AS node
    FROM   edges e
    WHERE  EXISTS (
        SELECT 1 FROM bito_features.v_blacklist_nodes bl
        WHERE  e.dst = bl.node
    )
),

-- ──────────────────────────────────────────────────
--  S9：IP 共用用戶數（ip_shared_user_count）
--      同一 source_ip 被幾個不同 user_id 使用。
--      > 3 → 機房 / 代理伺服器集體詐騙的強力訊號。
-- ──────────────────────────────────────────────────
ip_raw_map AS (
    SELECT
        CAST(user_id   AS BIGINT)   AS user_id,
        source_ip
    FROM bito_raw.twd_transfer
    WHERE source_ip IS NOT NULL AND user_id IS NOT NULL

    UNION ALL

    SELECT
        CAST(user_id   AS BIGINT)   AS user_id,
        source_ip
    FROM bito_raw.crypto_transfer
    WHERE source_ip IS NOT NULL AND user_id IS NOT NULL
),

ip_shared_count AS (
    SELECT
        source_ip,
        COUNT(DISTINCT user_id)     AS ip_shared_user_count
    FROM   ip_raw_map
    GROUP BY source_ip
),

-- 每位用戶取其使用的所有 IP 中，共用人數最高的那個（最危險情境）
user_ip_features AS (
    SELECT
        m.user_id,
        MAX(c.ip_shared_user_count) AS ip_shared_user_count
    FROM   ip_raw_map     m
    JOIN   ip_shared_count c ON m.source_ip = c.source_ip
    GROUP BY m.user_id
),

-- ──────────────────────────────────────────────────
--  S10：高速交易用戶（has_high_speed_risk）
--       交易完成時間 - 建立時間 < 600 秒（10 分鐘）
--       → 自動化腳本 / 機房大量刷單特徵
--
--  注意：請依實際欄位名稱調整 created_at / completed_at
--        若欄位為 epoch 整數（秒），改用 (completed_at - created_at) < 600
-- ──────────────────────────────────────────────────
high_speed_users AS (
    SELECT DISTINCT CAST(user_id AS BIGINT) AS user_id
    FROM bito_raw.twd_transfer
    WHERE created_at   IS NOT NULL
      AND completed_at IS NOT NULL
      AND date_diff(
              'second',
              from_iso8601_timestamp(CAST(created_at   AS VARCHAR)),
              from_iso8601_timestamp(CAST(completed_at AS VARCHAR))
          ) < 600

    UNION

    SELECT DISTINCT CAST(user_id AS BIGINT) AS user_id
    FROM bito_raw.crypto_transfer
    WHERE created_at   IS NOT NULL
      AND completed_at IS NOT NULL
      AND date_diff(
              'second',
              from_iso8601_timestamp(CAST(created_at   AS VARCHAR)),
              from_iso8601_timestamp(CAST(completed_at AS VARCHAR))
          ) < 600
)

-- ──────────────────────────────────────────────────
--  最終輸出
-- ──────────────────────────────────────────────────
SELECT
    anh.node                                            AS user_id,
    anh.min_hops_to_blacklist,
    (anh.min_hops_to_blacklist = 1)                     AS is_direct_neighbor,
    COALESCE(bnc.blacklist_neighbor_count, 0)           AS blacklist_neighbor_count,
    (ibn.node IS NOT NULL)                              AS in_blacklist_network,

    -- ── 新增：IP 層級特徵 ─────────────────────────────────────────────
    COALESCE(uip.ip_shared_user_count, 0)               AS ip_shared_user_count,
    (hsu.user_id IS NOT NULL)                           AS has_high_speed_risk,

    CURRENT_TIMESTAMP                                   AS event_time,

    -- ── 新增：複合風險加權標籤（機房集體詐騙核心判斷）─────────────────
    --   HIGH_WEIGHTED 條件：
    --     ① 與黑名單在 2 跳之內（直接鄰居或中間人連接）
    --     ② 其 IP 被超過 3 個帳號共用（機房 / 代理訊號）
    --     ③ 存在高速交易（< 10 分鐘，自動化腳本行為）
    CASE
        WHEN anh.min_hops_to_blacklist <= 2
         AND COALESCE(uip.ip_shared_user_count, 0) > 3
         AND hsu.user_id IS NOT NULL
        THEN 'HIGH_WEIGHTED'    -- 機房集體詐騙：最高優先處置

        WHEN anh.min_hops_to_blacklist = 0 THEN 'BLACKLIST'
        WHEN anh.min_hops_to_blacklist = 1 THEN 'HIGH'
        WHEN anh.min_hops_to_blacklist IN (2, 3) THEN 'MEDIUM'
        ELSE 'LOW'
    END                                                 AS weighted_risk_label,

    -- 分區欄位（Hive 分區必須放最後）
    CASE
        WHEN anh.min_hops_to_blacklist = 0 THEN 'blacklist'
        WHEN anh.min_hops_to_blacklist = 1 THEN 'direct'
        WHEN anh.min_hops_to_blacklist = 2 THEN 'indirect_2'
        WHEN anh.min_hops_to_blacklist = 3 THEN 'indirect_3'
        ELSE 'isolated'
    END                                                 AS hop_risk_level

FROM   all_nodes_with_hops anh
LEFT JOIN bl_neighbor_cnt   bnc ON anh.node = bnc.node
LEFT JOIN in_bl_network     ibn ON anh.node = ibn.node
LEFT JOIN user_ip_features  uip ON anh.node = uip.user_id
LEFT JOIN high_speed_users  hsu ON anh.node = hsu.user_id
;


-- ════════════════════════════════════════════════════════════
--  驗證查詢：分布統計
-- ════════════════════════════════════════════════════════════

-- Hop 風險分布
SELECT
    hop_risk_level,
    COUNT(*)                                                AS user_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2)    AS pct,
    AVG(blacklist_neighbor_count)                           AS avg_bl_neighbors,
    MAX(blacklist_neighbor_count)                           AS max_bl_neighbors,
    AVG(ip_shared_user_count)                              AS avg_ip_shared,
    ROUND(SUM(CAST(has_high_speed_risk AS INTEGER)) * 100.0
        / COUNT(*), 2)                                     AS high_speed_pct
FROM   bito_features.hop_features
GROUP BY hop_risk_level
ORDER BY
    CASE hop_risk_level
        WHEN 'blacklist'   THEN 0
        WHEN 'direct'      THEN 1
        WHEN 'indirect_2'  THEN 2
        WHEN 'indirect_3'  THEN 3
        ELSE 4
    END
;

-- 機房集體詐騙加權分布（weighted_risk_label）
SELECT
    weighted_risk_label,
    COUNT(*)                                                AS user_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2)    AS pct,
    AVG(ip_shared_user_count)                              AS avg_ip_shared,
    AVG(min_hops_to_blacklist)                             AS avg_hops
FROM   bito_features.hop_features
GROUP BY weighted_risk_label
ORDER BY
    CASE weighted_risk_label
        WHEN 'HIGH_WEIGHTED' THEN 0
        WHEN 'BLACKLIST'     THEN 1
        WHEN 'HIGH'          THEN 2
        WHEN 'MEDIUM'        THEN 3
        ELSE 4
    END
;


-- ════════════════════════════════════════════════════════════
--  即時查詢：取得特定用戶的跳轉數（供推論服務呼叫）
-- ════════════════════════════════════════════════════════════

-- 以 user_id 清單查詢（推論時用此格式）
SELECT
    user_id,
    min_hops_to_blacklist,
    is_direct_neighbor,
    blacklist_neighbor_count,
    in_blacklist_network,
    ip_shared_user_count,
    has_high_speed_risk,
    hop_risk_level,
    weighted_risk_label
FROM   bito_features.hop_features
WHERE  user_id IN (12345, 67890, 111222)  -- 替換為實際 user_id
;
