"""
AWS Glue PySpark Job — 分散式資金關聯深度 (BFS Hops) 計算
==========================================================

設計目標：在幣託全量歷史交易（數億列）上，計算每位用戶距已知黑名單的
最短交易跳轉數（min_hops_to_blacklist），並將結果寫入 S3（供 Athena 查詢）
以及 SageMaker Feature Store（供即時推論）。

演算法選型
----------
                    | 複雜度         | 適用規模      | 產出
---- BFS 迭代 Join  | O(k(V+E))     | 數億邊       | 精確跳數 ✓
---- Union-Find     | O(V α(V))     | 數十億節點   | 連通分量 ID（無跳數）

本腳本同時實作兩者：
  ① 多源 BFS 迭代 Join：精確計算 1~N 跳距離（主特徵）
  ② 圖連通分量 Union-Find：快速判斷是否與黑名單同屬一個連通子圖（輔助特徵）

Glue Job 參數（--job-bookmark-option=job-bookmark-enable 建議開啟）
------------------------------------------------------------------
  --s3_input_bucket      : 輸入資料桶（存放 twd/crypto_transfer parquet）
  --s3_input_prefix      : 輸入路徑前綴（e.g. bito-mule-detection/raw/）
  --s3_output_prefix     : 輸出路徑前綴（e.g. bito-mule-detection/features/）
  --blacklist_s3_path    : 黑名單 CSV 路徑（user_id,status）
  --max_hops             : BFS 最大層數（預設 3）
  --feature_group_name   : SageMaker Feature Store 群組名稱
  --aws_region           : AWS Region（預設 ap-northeast-1）

執行方式（Glue Console 或 CLI）
-------------------------------
  aws glue start-job-run --job-name bito-graph-hops \\
      --arguments '{"--max_hops":"3","--feature_group_name":"bito-mule-hops"}'
"""

import sys
import time
import logging
from datetime import datetime, timezone

import boto3
from awsglue.utils         import getResolvedOptions
from awsglue.context       import GlueContext
from awsglue.job           import Job
from pyspark.context        import SparkContext
from pyspark.sql            import SparkSession, DataFrame
from pyspark.sql            import functions as F
from pyspark.sql.types      import (
    StructType, StructField, LongType, IntegerType,
    BooleanType, StringType, FloatType, TimestampType,
)
from pyspark.storagelevel   import StorageLevel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ── Glue 環境初始化 ──────────────────────────────────────────────────────────

args = getResolvedOptions(sys.argv, [
    "JOB_NAME",
    "s3_input_bucket",
    "s3_input_prefix",
    "s3_output_prefix",
    "blacklist_s3_path",
    "--max_hops",
    "--feature_group_name",
    "--aws_region",
], defaults={
    "--max_hops":             "3",
    "--feature_group_name":   "bito-mule-hops",
    "--aws_region":           "us-east-1",
})

sc          = SparkContext()
glueContext = GlueContext(sc)
spark       = glueContext.spark_session
job         = Job(glueContext)
job.init(args["JOB_NAME"], args)

# 設定 Spark 參數（針對大規模圖計算優化）
spark.conf.set("spark.sql.shuffle.partitions",        "800")   # 大量 shuffle 時的分區數
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "50MB") # 黑名單小表廣播門檻
spark.conf.set("spark.sql.adaptive.enabled",           "true") # AQE 自適應優化
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")

# ── 常數 ─────────────────────────────────────────────────────────────────────

S3_BUCKET    = args["s3_input_bucket"]
INPUT_PFX    = args["s3_input_prefix"].rstrip("/")
OUTPUT_PFX   = args["s3_output_prefix"].rstrip("/")
BLACKLIST_S3 = args["blacklist_s3_path"]
MAX_HOPS     = int(args["--max_hops"])
FG_NAME      = args["--feature_group_name"]
REGION       = args["--aws_region"]
_ALLOWED_REGIONS = {"us-east-1", "us-west-2"}
if REGION not in _ALLOWED_REGIONS:
    raise ValueError(
        f"[Region 合規] --aws_region='{REGION}' 不符合競賽規定。"
        f"僅允許 {sorted(_ALLOWED_REGIONS)}。"
    )
RUN_TS       = datetime.now(timezone.utc).isoformat()

ISOLATED_HOPS = MAX_HOPS + 1    # 未在圖中出現的節點，填此值

IP_SHARED_THRESHOLD   = 3       # ip_shared_user_count 超過此值視為機房風險
SKEW_DEGREE_THRESHOLD     = 1_000   # 警示門檻：超過此值記入 Hot Spot 日誌供稽核
SUPER_NODE_SALT_THRESHOLD = 10_000  # 隨機加鹽門檻：超過此值 → 隨機 salt（保留於圖中）
SALT_FACTOR               = 16      # 加鹽分片數（2 的冪次方；BFS crossJoin 複製份數）
MAX_BFS_HOPS              = 4       # BFS 迭代硬性上限（第 4 跳後強制停止）
HOT_SPOT_LOG_TOP_N        = 20      # 熱點稽核日誌：記錄度數最高的前 N 個節點

# ── 已知非個人節點排除清單（優先過濾，不論動態度數） ──────────────────────
# 格式說明：
#   正整數 → 幣託平台帳號 ID（官方出金帳號、系統帳號、OTC 對手方等）
#   負整數 → _ip_to_node_id 轉換後的已知機房 / CDN / VPN 共用 IP 節點
#
# 新增方式：將節點 ID 加入此集合後重新部署 Glue Job 即可生效。
# 排除原則：只排除「確認為非個人節點」的系統帳號；對「可疑高度數帳號」
#           應先稽核再加入，以免漏判共謀集群。
# 設為 set() 可停用排除邏輯（僅依度數加鹽，不做靜態過濾）。
EXCLUDED_NODES: set = {
    # 999_999_999,     # [範例] BitoPro 官方出金熱錢包帳號
    # 888_888_888,     # [範例] BitoPro 法幣銀行中轉系統帳號
    # 777_777_777,     # [範例] OTC 流動性做市商帳號
    # -9_999_999_999,  # [範例] 已知 VPN / CDN 機房 IP（負整數）
}


def _compute_combined_degree(edges: DataFrame) -> DataFrame:
    """
    計算每個節點的合併度數（出現為 src 的次數 + 出現為 dst 的次數）。

    使用合併度數而非單純出度（out-degree），原因：
    ① 無向圖中 src/dst 角色可互換，單看出度低估了 IP 節點（多為 dst）的真實度數。
    ② 高 in-degree 節點（如出金目標錢包）同樣會在 BFS join 時造成 Data Skew。

    Returns
    -------
    DataFrame  columns: (node BIGINT, degree LONG)
    """
    src_deg = edges.groupBy("src").agg(F.count("*").alias("deg"))
    dst_deg = edges.groupBy("dst").agg(F.count("*").alias("deg"))
    return (
        src_deg.select(F.col("src").alias("node"), "deg")
        .union(dst_deg.select(F.col("dst").alias("node"), "deg"))
        .groupBy("node")
        .agg(F.sum("deg").alias("degree"))
    )


def _filter_excluded_nodes(
    edges:    DataFrame,
    spark:    SparkSession,
    excluded: set = EXCLUDED_NODES,
) -> DataFrame:
    """
    從邊列表中移除 EXCLUDED_NODES 清單內的所有已知非個人節點及其相連邊。

    設計原則
    ─────────
    ① 靜態清單優先於動態度數：已確認為系統帳號（官方出金、OTC 做市）的節點
       無論度數高低均應排除，避免 BFS 穿越後大量誤報。
    ② 排除是雙向的：若 A 在清單中，邊 (A→B) 與 (B→A) 均刪除，
       防止透過反向邊「繞回」排除節點。
    ③ 使用 broadcast join：清單通常 < 1,000 個節點，broadcast 開銷極低。

    Parameters
    ----------
    excluded : set
        節點 ID 集合。正整數為用戶帳號；負整數為 IP 節點（_ip_to_node_id 轉換後）。
        傳入空集合時直接回傳原 edges，跳過所有 join。

    Returns
    -------
    DataFrame  — 過濾後的邊列表（schema 不變：src BIGINT, dst BIGINT）
    """
    if not excluded:
        log.info("[Excluded Nodes] EXCLUDED_NODES 為空，跳過靜態節點過濾")
        return edges

    excl_df = spark.createDataFrame(
        [(int(n),) for n in excluded],
        schema=StructType([StructField("node", LongType(), False)]),
    )
    excl_bc = F.broadcast(excl_df)

    filtered = (
        edges
        .join(excl_bc.select(F.col("node").alias("_es")),
              edges["src"] == F.col("_es"), how="left_anti")
        .join(excl_bc.select(F.col("node").alias("_ed")),
              edges["dst"] == F.col("_ed"), how="left_anti")
    )
    log.info(
        f"[Excluded Nodes] 已從邊列表中排除 {len(excluded):,} 個已知非個人節點"
        f"（含官方出金帳號、系統帳號、已知機房 IP）"
    )
    return filtered


def detect_hot_spots(
    edges:    DataFrame,
    spark:    SparkSession,
    excluded: set = EXCLUDED_NODES,
) -> tuple:
    """
    熱點偵測管線：靜態排除 → 合併度數計算 → 分層日誌記錄。

    執行步驟
    ─────────
    1. 呼叫 _filter_excluded_nodes：移除 EXCLUDED_NODES 靜態清單
    2. 呼叫 _compute_combined_degree：計算合併度數（in + out）
    3. 將節點按度數分三層記錄：
       ① 超級節點（> SUPER_NODE_SALT_THRESHOLD）→ 隨機加鹽、不過濾
       ② 警示節點（> SKEW_DEGREE_THRESHOLD）     → 建議稽核後加入 EXCLUDED_NODES
       ③ 前 HOT_SPOT_LOG_TOP_N 名                → 詳列供 Ops 追蹤
    4. 回傳 (filtered_edges, degree_df) 供後續加鹽步驟使用

    Returns
    -------
    tuple  (filtered_edges: DataFrame, degree_df: DataFrame)
           degree_df columns: (node BIGINT, degree LONG)，已 persist
    """
    # ── 1. 靜態排除 ─────────────────────────────────────────────────────────
    filtered_edges = _filter_excluded_nodes(edges, spark, excluded)

    # ── 2. 合併度數（in + out） ───────────────────────────────────────────────
    degree_df = _compute_combined_degree(filtered_edges).persist(StorageLevel.MEMORY_AND_DISK)

    # ── 3. 分層日誌 ──────────────────────────────────────────────────────────
    total_super = degree_df.filter(F.col("degree") > SUPER_NODE_SALT_THRESHOLD).count()
    total_warn  = degree_df.filter(
        (F.col("degree") > SKEW_DEGREE_THRESHOLD) &
        (F.col("degree") <= SUPER_NODE_SALT_THRESHOLD)
    ).count()

    log.info(
        f"[Hot Spot] 超級節點（度數 > {SUPER_NODE_SALT_THRESHOLD:,}）：{total_super:,} 個"
        f" → 隨機加鹽分散至 {SALT_FACTOR} 個 Bucket"
    )
    log.info(
        f"[Hot Spot] 警示節點（{SKEW_DEGREE_THRESHOLD:,} < 度數 ≤ {SUPER_NODE_SALT_THRESHOLD:,}）："
        f"{total_warn:,} 個 → 建議稽核後加入 EXCLUDED_NODES"
    )

    top_rows = (
        degree_df.orderBy(F.col("degree").desc())
                 .limit(HOT_SPOT_LOG_TOP_N)
                 .collect()
    )
    log.info(f"[Hot Spot] 度數前 {HOT_SPOT_LOG_TOP_N} 名節點（供 Ops 稽核 / 加入 EXCLUDED_NODES）：")
    for row in top_rows:
        node_type = "IP節點" if row.node < 0 else "用戶帳號"
        tier_tag  = (
            " ★ 超級節點→加鹽" if row.degree > SUPER_NODE_SALT_THRESHOLD
            else " ⚠ 警示節點"   if row.degree > SKEW_DEGREE_THRESHOLD
            else ""
        )
        log.info(f"  [{node_type}] id={row.node:>20,}  degree={row.degree:>10,}{tier_tag}")

    return filtered_edges, degree_df


def add_edge_salt(
    edges:     DataFrame,
    degree_df: DataFrame,
    salt_n:    int = SALT_FACTOR,
) -> DataFrame:
    """
    兩段式加鹽策略（依合併度數區分）：

    ① 超級節點邊（src 或 dst 的合併度數 > SUPER_NODE_SALT_THRESHOLD）：
       使用 F.rand() × salt_n 產生隨機整數後綴（0 ~ salt_n-1）。
       效果：同一超級節點的所有出邊均勻散落在 salt_n 個 Bucket，
             BFS join 時每個 Reducer 僅處理 1/salt_n 的邊。
       注意：F.rand() 在 DataFrame persist 時固化（同一 Glue Job 內穩定），
             迭代 BFS 重複使用同一份 persisted edges，salt 不會改變。

    ② 一般邊（兩端點度數均 ≤ SUPER_NODE_SALT_THRESHOLD）：
       salt = 0，所有一般邊集中於 Bucket 0。
       效果：一般節點的 BFS 僅由 frontier(salt=0) 處理，無額外複製開銷。

    BFS join 相容性（_bfs_salt_join）
    ──────────────────────────────────
    Frontier 以 crossJoin(salt_range[0..salt_n-1]) 複製 salt_n 份：
      - 超級節點邊（隨機 salt=k）：由 frontier 中 salt=k 的副本命中 ✓
      - 一般邊（salt=0）          ：由 frontier 中 salt=0 的副本命中 ✓
      - 一般節點 → 超級節點的邊    ：邊 salt = 隨機 k；
        frontier(一般節點, salt=k) 命中該邊，回傳超級節點為鄰居 ✓

    Parameters
    ----------
    degree_df : DataFrame  來自 detect_hot_spots() 的合併度數表 (node, degree)

    Returns
    -------
    DataFrame  columns: (src BIGINT, dst BIGINT, salt INT)
    """
    super_nodes = (
        degree_df.filter(F.col("degree") > SUPER_NODE_SALT_THRESHOLD)
                 .select("node")
                 .persist(StorageLevel.MEMORY_AND_DISK)
    )
    super_bc = F.broadcast(super_nodes)

    salted = (
        edges
        # 左連接：src 端是否為超級節點
        .join(super_bc.select(F.col("node").alias("_sn_src")),
              edges["src"] == F.col("_sn_src"), how="left")
        # 左連接：dst 端是否為超級節點
        .join(super_bc.select(F.col("node").alias("_sn_dst")),
              edges["dst"] == F.col("_sn_dst"), how="left")
        .withColumn(
            "salt",
            F.when(
                F.col("_sn_src").isNotNull() | F.col("_sn_dst").isNotNull(),
                # 隨機後綴：0 ~ salt_n-1（F.rand 在 persist 時固化）
                (F.rand() * F.lit(salt_n)).cast(IntegerType()),
            ).otherwise(F.lit(0))   # 一般邊集中於 Bucket 0
        )
        .drop("_sn_src", "_sn_dst")
    )

    super_count = super_nodes.count()
    log.info(
        f"[Edge Salt] 超級節點 {super_count:,} 個 → 相關邊隨機分散至 {salt_n} Bucket；"
        f"其餘邊 salt=0"
    )
    super_nodes.unpersist()
    return salted


def _dynamic_part_count(
    row_count:            int,
    target_rows_per_part: int = 200_000,
) -> int:
    """
    根據資料列數動態計算最佳分區數，避免單一分區過大導致 OOM。

    公式：clamp(ceil(row_count / target_rows_per_part), floor=200, cap=2_000)
      - floor 200 ：最低保留分區數，確保 Executor 充分並行
      - cap   2_000：上限防止 Driver Task 管理開銷過大
      - target 200,000 列 / 分區：GiB 級資料集的典型安全值

    Args:
        row_count:           資料列數（由 .count() 取得）
        target_rows_per_part: 每個分區目標列數（預設 200,000）

    Returns:
        建議分區數（200 ~ 2,000）
    """
    raw = max(1, row_count) // max(1, target_rows_per_part)
    return int(max(200, min(2_000, raw + 1)))


def _bfs_salt_join(
    edges:    DataFrame,     # (src, dst, salt)
    frontier: DataFrame,     # (node)
    hop:      int,
    spark:    SparkSession,
    salt_n:   int = SALT_FACTOR,
) -> DataFrame:
    """
    加鹽 BFS 擴散 Join：
      左表（交易邊 edges）已帶預分配 salt（add_edge_salt 產生，0 ~ salt_n-1）；
      右表（節點狀態 frontier）複製 salt_n 份後以 F.broadcast() 廣播至全部 Executor。

    廣播策略說明
    ────────────
    Frontier 僅含「本跳剛發現的節點」，規模遠小於 edges（通常 < 百萬列）。
    廣播右表可消除 Shuffle Stage，每個 Executor 就地對位命中 salt Bucket，
    比 persist(MEMORY_AND_DISK) 更快且無需 unpersist 管理。

    流程
    ────
    1. 建立 salt_range（0 ~ salt_n-1 的 DataFrame）
    2. 右表：frontier × salt_range → crossJoin 複製 salt_n 份 → F.broadcast()
    3. 左表：edges（已含 salt 欄位）直接 Join broadcast 右表
       Join 條件：edges.src == frontier.node  AND  edges.salt == frontier.salt
    4. 回傳 (node=dst, dist=hop)，不含 salt 欄位

    Returns
    -------
    DataFrame  (node BIGINT, dist INT)
    """
    salt_range = (
        spark.range(salt_n)
             .select(F.col("id").cast(IntegerType()).alias("salt"))
    )
    # 右表（節點狀態）：複製 salt_n 份後廣播——消除 Shuffle，就地 salt 對位命中
    frontier_replicated = frontier.crossJoin(salt_range)

    neighbors = (
        edges
        .join(
            F.broadcast(frontier_replicated),
            (edges["src"] == frontier_replicated["node"]) &
            (edges["salt"] == frontier_replicated["salt"]),
            how="inner",
        )
        .select(
            edges["dst"].alias("node"),
            F.lit(hop).cast(IntegerType()).alias("dist"),
        )
    )

    return neighbors


def _ip_to_node_id(ip_col: F.Column) -> F.Column:
    """
    將 source_ip 字串轉換為唯一負數 BIGINT 節點 ID。
    負數空間（< 0）與正數 user_id 永不衝突，BFS 可自然穿越 IP 中繼節點。

    原理：SHA-256 前 15 hex 字元 → 十進制 → 取負數
          範圍：-1,152,921,504,606,846,975 ~ -1（60-bit 負整數）
    """
    return (
        F.conv(F.substring(F.sha2(ip_col, 256), 1, 15), 16, 10)
         .cast(LongType())
         * F.lit(-1)
    )


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1  讀取原始交易資料，建立分散式邊列表
# ══════════════════════════════════════════════════════════════════════════════

def load_and_build_edges(spark: SparkSession) -> DataFrame:
    """
    從 S3 讀取 twd_transfer / crypto_transfer parquet，
    建立無向二元邊列表 (src BIGINT, dst BIGINT, salt INT)。

    執行步驟
    ─────────
    1A  讀取 twd_transfer / crypto_transfer，建立明確交易邊
    1B  建立錢包共用邊 + User-to-IP 邊，合併對稱化、去重、去自環
    1C  detect_hot_spots：
        ① 靜態排除 EXCLUDED_NODES（官方帳號、系統節點，不論度數）
        ② 計算合併度數（in + out），分層記錄熱點日誌供 Ops 稽核
    1D  add_edge_salt（兩段式加鹽）：
        ① 超級節點邊（合併度數 > SUPER_NODE_SALT_THRESHOLD）→ F.rand() 隨機後綴
        ② 一般邊                                            → salt = 0
    1E  repartition(800, "src") + persist(DISK_ONLY)

    Data Skew 防護層次
    ──────────────────
    L1 EXCLUDED_NODES（靜態）：已知官方/系統節點直接排除，消除「假超級節點」
    L2 加鹽（動態）           ：剩餘超級節點的相關邊隨機散落 SALT_FACTOR 個 Bucket
    L3 BFS 硬性跳數限制       ：MAX_BFS_HOPS = 4，避免長路徑膨脹

    若資料儲存在 Glue Data Catalog，可改用 glueContext.create_dynamic_frame
    """
    twd    = spark.read.parquet(f"s3://{S3_BUCKET}/{INPUT_PFX}/twd_transfer/")
    crypto = spark.read.parquet(f"s3://{S3_BUCKET}/{INPUT_PFX}/crypto_transfer/")

    # ── 明確交易邊（user_id ↔ relation_user_id） ──────────────────────────
    twd_edges = (
        twd.filter(F.col("relation_user_id").isNotNull())
           .select(
               F.col("user_id").cast(LongType()).alias("src"),
               F.col("relation_user_id").cast(LongType()).alias("dst"),
           )
    )
    crypto_edges = (
        crypto.filter(F.col("relation_user_id").isNotNull())
              .select(
                  F.col("user_id").cast(LongType()).alias("src"),
                  F.col("relation_user_id").cast(LongType()).alias("dst"),
              )
    )

    # ── 錢包地址共用邊（user_i 與 user_j 曾使用相同 to_wallet） ──────────
    #  原理：若 A → wallet_X 且 B ← wallet_X，則 A 與 B 透過錢包形成隱含關聯
    wallet_edges = (
        crypto.filter(F.col("to_wallet").isNotNull() & F.col("relate_stype").isNull())
              .select(
                  F.col("user_id").cast(LongType()).alias("uid"),
                  F.col("to_wallet").alias("wallet"),
              )
    )
    # 自 join：相同錢包地址的不同用戶形成一條邊
    wallet_link = (
        wallet_edges.alias("a")
        .join(
            wallet_edges.alias("b"),
            on=(F.col("a.wallet") == F.col("b.wallet")) &
               (F.col("a.uid")    != F.col("b.uid")),
            how="inner",
        )
        .select(
            F.col("a.uid").alias("src"),
            F.col("b.uid").alias("dst"),
        )
    )

    # ── User-to-IP 雙向邊（source_ip_hash 作為圖節點）────────────────
    #  IP 節點 ID = 負數 BIGINT（見 _ip_to_node_id）
    #  用途：BFS 穿越 IP 中繼節點，偵測「共用同一 IP 的用戶群」
    #        例：user_A → IP_X → user_B （2 跳），代表 A/B 從同一機房登入
    ip_edges_twd = (
        twd.filter(F.col("source_ip").isNotNull())
           .select(
               F.col("user_id").cast(LongType()).alias("src"),
               _ip_to_node_id(F.col("source_ip")).alias("dst"),
           )
    )
    ip_edges_crypto = (
        crypto.filter(F.col("source_ip").isNotNull())
              .select(
                  F.col("user_id").cast(LongType()).alias("src"),
                  _ip_to_node_id(F.col("source_ip")).alias("dst"),
              )
    )

    # ── 合併所有邊 → 對稱化 → 去重 ──────────────────────────────────────
    directed_edges = (
        twd_edges.union(crypto_edges)
                 .union(wallet_link)
                 .union(ip_edges_twd)
                 .union(ip_edges_crypto)
    )

    reversed_edges = directed_edges.select(
        F.col("dst").alias("src"),
        F.col("src").alias("dst"),
    )

    undirected_edges = (
        directed_edges.union(reversed_edges)
        .filter(F.col("src") != F.col("dst"))           # 去除自環
        .filter(F.col("src").isNotNull() & F.col("dst").isNotNull())
        .dropDuplicates(["src", "dst"])
    )

    # ── Step 1C：靜態排除 + 熱點偵測 ────────────────────────────────────────
    #  ① 移除 EXCLUDED_NODES（已知官方帳號、系統節點）— 不論其度數
    #  ② 計算合併度數（in + out），分層記錄熱點供 Ops 稽核
    #  degree_df 保持 persist 至 Step 1D 加鹽完成後釋放
    undirected_filtered, degree_df = detect_hot_spots(undirected_edges, spark)

    # ── Step 1D：兩段式加鹽 ──────────────────────────────────────────────────
    #  超級節點邊（度數 > SUPER_NODE_SALT_THRESHOLD）→ F.rand() 隨機後綴
    #  一般邊                                        → salt = 0（集中 Bucket 0）
    #  BFS crossJoin(salt_range) 確保所有 salt 值均能命中（見 _bfs_salt_join 說明）
    edges_salted = add_edge_salt(undirected_filtered, degree_df)
    degree_df.unpersist()   # Step 1D 完成後釋放，不再需要

    # ── Step 1E：動態分區 + DISK_ONLY Persist ───────────────────────────────
    # 先以 MEMORY_AND_DISK 暫時持久化取得精確列數，
    # 再依 _dynamic_part_count 計算分區數後落盤，避免硬編碼 800 分區 OOM。
    # target_rows_per_part=500_000：邊列（3 欄位）寬度低，每分區可承載更多列。
    edges_tmp  = edges_salted.persist(StorageLevel.MEMORY_AND_DISK)
    edge_count = edges_tmp.count()
    n_parts    = _dynamic_part_count(edge_count, target_rows_per_part=500_000)

    edges = (
        edges_tmp
        .repartition(n_parts, "src")
        .persist(StorageLevel.DISK_ONLY)    # 數億邊，DISK_ONLY 避免 OOM
    )
    edges_tmp.unpersist()

    log.info(f"[Step 1] 邊總數（含 salt 欄位）：{edge_count:,}  分區數：{n_parts}")
    return edges   # columns: (src BIGINT, dst BIGINT, salt INT)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1B  計算 IP 層級特徵（ip_shared_user_count、has_high_speed_risk）
# ══════════════════════════════════════════════════════════════════════════════

def compute_ip_features(spark: SparkSession) -> DataFrame:
    """
    計算兩個 IP 相關特徵，用於複合風險加權：

    ip_shared_user_count : 該用戶所用 IP 中，最多有幾個不同 user_id 共用
                           > 3 → 高機率機房 / 代理伺服器集體詐騙
    has_high_speed_risk  : 該用戶是否存在 < 10 分鐘完成的交易
                           True → 自動化腳本行為特徵

    回傳欄位：
      user_id              BIGINT
      ip_shared_user_count INT
      has_high_speed_risk  BOOLEAN
    """
    twd    = spark.read.parquet(f"s3://{S3_BUCKET}/{INPUT_PFX}/twd_transfer/")
    crypto = spark.read.parquet(f"s3://{S3_BUCKET}/{INPUT_PFX}/crypto_transfer/")

    # ── 建立 user_id ↔ source_ip 對應表 ──────────────────────────────────
    user_ip = (
        twd.filter(F.col("source_ip").isNotNull())
           .select(F.col("user_id").cast(LongType()), F.col("source_ip"))
        .union(
            crypto.filter(F.col("source_ip").isNotNull())
                  .select(F.col("user_id").cast(LongType()), F.col("source_ip"))
        )
        .dropDuplicates(["user_id", "source_ip"])
        .persist(StorageLevel.MEMORY_AND_DISK)
    )

    # ── 每個 IP 的共用 user 數 ────────────────────────────────────────────
    ip_user_count = (
        user_ip
        .groupBy("source_ip")
        .agg(F.countDistinct("user_id").cast(IntegerType()).alias("ip_shared_user_count"))
    )

    # ── 每位用戶取其使用的所有 IP 中最大的共用人數 ────────────────────────
    #    （一個用戶可能使用多個 IP，取最危險的那個）
    user_max_ip_shared = (
        user_ip
        .join(ip_user_count, on="source_ip", how="inner")
        .groupBy("user_id")
        .agg(F.max("ip_shared_user_count").alias("ip_shared_user_count"))
    )

    # ── High Speed Risk：完成時間 - 建立時間 < 600 秒 ─────────────────────
    #    以 unix_timestamp 差值計算（相容 ISO8601 字串或 Timestamp 型別）
    def _speed_filter(df: DataFrame) -> DataFrame:
        return (
            df.filter(
                F.col("created_at").isNotNull() &
                F.col("completed_at").isNotNull() &
                (
                    (F.unix_timestamp(F.col("completed_at").cast(TimestampType())) -
                     F.unix_timestamp(F.col("created_at").cast(TimestampType()))) < 600
                )
            )
            .select(F.col("user_id").cast(LongType()).alias("user_id"))
        )

    high_speed_users = (
        _speed_filter(twd)
        .union(_speed_filter(crypto))
        .dropDuplicates()
        .withColumn("has_high_speed_risk", F.lit(True).cast(BooleanType()))
    )

    # ── 合併 IP 特徵 ──────────────────────────────────────────────────────
    ip_features = (
        user_max_ip_shared
        .join(high_speed_users, on="user_id", how="left")
        .withColumn(
            "has_high_speed_risk",
            F.coalesce(F.col("has_high_speed_risk"), F.lit(False))
        )
        .persist(StorageLevel.MEMORY_AND_DISK)
    )

    count = ip_features.count()
    log.info(f"[Step 1B] IP 特徵計算完成：{count:,} 位用戶")
    log.info(f"[Step 1B] 高共用 IP 用戶（>{IP_SHARED_THRESHOLD}）：")
    ip_features.filter(
        F.col("ip_shared_user_count") > IP_SHARED_THRESHOLD
    ).agg(F.count("*")).show()

    user_ip.unpersist()
    return ip_features   # (user_id BIGINT, ip_shared_user_count INT, has_high_speed_risk BOOLEAN)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2  讀取黑名單，初始化 BFS 起始距離
# ══════════════════════════════════════════════════════════════════════════════

def load_blacklist(spark: SparkSession) -> DataFrame:
    """
    讀取黑名單 CSV（user_id, status），回傳 status=1 的節點 DataFrame。
    若黑名單數量 < 50MB，PySpark 會自動廣播到各 Executor。
    """
    bl = (
        spark.read.option("header", "true")
             .csv(BLACKLIST_S3)
             .filter(F.col("status") == "1")
             .select(F.col("user_id").cast(LongType()).alias("node"))
             .dropDuplicates()
    )
    count = bl.count()
    log.info(f"[Step 2] 黑名單節點數：{count:,}")
    return bl


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3A  多源 BFS 迭代 Join（精確跳數計算）
# ══════════════════════════════════════════════════════════════════════════════

def multi_source_bfs(
    edges: DataFrame,
    blacklist: DataFrame,
    max_hops: int = MAX_HOPS,
) -> DataFrame:
    """
    分散式多源 BFS：從所有黑名單節點同時出發，逐跳擴散至整張圖。

    實作策略（Pregel-style 迭代 Join）
    ------------------------------------
    第 0 跳：distances = {blacklist → 0}
    第 k 跳：
      1. frontier  = distances WHERE dist == k-1          （上一輪新發現節點）
      2. neighbors = _bfs_salt_join(edges, frontier)      （廣播右表加鹽展開）
      3. new_nodes = neighbors SALTED-ANTI-JOIN distances  （加鹽排除已標記節點）
      4. distances = distances UNION (new_nodes → k)       （動態分區數更新）

    性能優化四層策略
    -----------------
    ① Super Node Salting（加鹽緩解 Data Skew）
       edges 於 load_and_build_edges 階段預分配 salt（0 ~ SALT_FACTOR-1）；
       超級節點邊（度數 > SUPER_NODE_SALT_THRESHOLD）使用 rand() 隨機 salt，
       一般邊 salt=0，確保高度數節點的出邊均勻落入 SALT_FACTOR 個 Reducer。

    ② Broadcast Right Table（廣播右表消除 Shuffle）
       Expansion Join：_bfs_salt_join 對 frontier（右表/節點狀態）廣播，
         消除 edges ⟶ frontier 的 Shuffle Stage。
       Anti-Join：對左表（neighbors）加隨機桶標 rand() % SALT_FACTOR；
         對右表（distances / 節點狀態）crossJoin 複製 SALT_FACTOR 份後廣播，
         確保高度數節點在 Anti-Join 時均勻分散至 SALT_FACTOR 個 Reducer。
         Anti-Join 語義正確性：節點 X 若在 distances 中，其 16 個 salt 副本
         均存在；任意 neighbors(X, k) 都能在 distances_bc(X, k) 命中 → 正確排除。

    ③ Dynamic Partition Count（動態分區防 OOM）
       每輪以 _dynamic_part_count(new_count) 計算新節點分區數，
       以 _dynamic_part_count(dist_count) 計算累積 distances 分區數，
       替代硬編碼固定值，避免單一分區列數過大導致 Executor OOM。

    ④ DISK_ONLY Persist + Checkpoint（防 RDD Lineage 過長）
       distances 與 new_nodes 均以 DISK_ONLY 持久化，放棄 JVM Heap 佔用；
       每偶數跳呼叫 checkpoint() 截斷血統（Lineage），防止 Driver 端
       執行計劃遞迴深度無限增長。

    Parameters
    ----------
    edges     : 無向邊列表 (src BIGINT, dst BIGINT, salt INT)
    blacklist : 黑名單節點 (node BIGINT)
    max_hops  : 最大擴散層數

    Returns
    -------
    DataFrame (node BIGINT, dist INT)
    """
    sc.setCheckpointDir(f"s3://{S3_BUCKET}/tmp/glue-checkpoints/")

    # ── 初始化：黑名單節點距離 = 0 ───────────────────────────────────────────
    distances = (
        blacklist
        .select(F.col("node"), F.lit(0).cast(IntegerType()).alias("dist"))
        .persist(StorageLevel.DISK_ONLY)       # ④ DISK_ONLY：避免 JVM Heap 壓力
    )

    # ── Anti-Join 加鹽資源（迭代共用） ───────────────────────────────────────
    # 對左表 neighbors 加 rand() % SALT_FACTOR 桶標，
    # 對右表 distances 複製 SALT_FACTOR 份廣播，兩側在 (node, _anti_salt) 對齊。
    anti_salt_range = (
        spark.range(SALT_FACTOR)
             .select(F.col("id").cast(IntegerType()).alias("_anti_salt"))
    )

    effective_max = min(max_hops, MAX_BFS_HOPS)
    log.info(
        f"[BFS] 有效迭代上限：{effective_max} 跳"
        f"（max_hops={max_hops}, 硬性上限={MAX_BFS_HOPS}）"
    )
    init_count = distances.count()
    dist_count = init_count          # 追蹤累積節點數，避免每輪重複 .count()
    log.info(f"[BFS] 初始節點數（黑名單）：{init_count:,}")

    for hop in range(1, effective_max + 1):
        t0 = time.time()

        # ── Step A：提取 Frontier（上一跳新發現節點） ────────────────────
        frontier = (
            distances.filter(F.col("dist") == hop - 1)
                     .select(F.col("node"))
                     .persist(StorageLevel.MEMORY_AND_DISK)   # Frontier 小，允許記憶體快取
        )

        # ── Step B：Expansion Join（左表加鹽邊 × 廣播右表 frontier） ─────
        # edges（左表）已帶預分配 salt；
        # frontier（右表/節點狀態）在 _bfs_salt_join 內複製後廣播。
        neighbors = _bfs_salt_join(edges, frontier, hop, spark)

        # ── Step C：Anti-Join（加鹽版，防超級節點在排除步驟造成 Skew） ───
        # 左表：neighbors 加 rand() % SALT_FACTOR 隨機桶標
        neighbors_salted = neighbors.withColumn(
            "_anti_salt",
            (F.rand() * F.lit(SALT_FACTOR)).cast(IntegerType()),
        )
        # 右表：distances（節點狀態）複製 SALT_FACTOR 份後廣播
        # 說明：distances 可能在後期增長至數百萬列；廣播 hint 為建議，
        #       若超過 autoBroadcastJoinThreshold Spark 會自動降為 Sort-Merge Join。
        distances_bc = F.broadcast(
            distances.select("node").crossJoin(anti_salt_range)
        )
        new_nodes = (
            neighbors_salted
            .join(distances_bc, on=["node", "_anti_salt"], how="left_anti")
            .drop("_anti_salt")
            .dropDuplicates(["node"])
            .persist(StorageLevel.DISK_ONLY)   # ④ DISK_ONLY：中間結果不佔 Heap
        )

        new_count = new_nodes.count()
        elapsed   = round(time.time() - t0, 1)

        # ③ 動態分區數：依本輪新增節點規模計算
        n_new_parts = _dynamic_part_count(new_count)
        log.info(
            f"[BFS] 第 {hop} 跳：新增節點 {new_count:,}"
            f"  分區數 {n_new_parts}  耗時 {elapsed}s"
        )

        if new_count == 0:
            log.info("[BFS] 圖已完全遍歷，提前結束。")
            frontier.unpersist()
            new_nodes.unpersist()
            break

        # ── Step D：更新 distances（動態分區 + DISK_ONLY / Checkpoint） ──
        prev_distances  = distances
        dist_count     += new_count
        n_dist_parts    = _dynamic_part_count(dist_count)   # ③ 動態分區

        distances = (
            distances.union(new_nodes)
                     .repartition(n_dist_parts, "node")
        )

        # ④ 偶數跳：checkpoint() 截斷 Lineage（防 Driver 記憶體溢出）
        #    奇數跳：DISK_ONLY persist（縮短重算路徑，不佔 Heap）
        if hop % 2 == 0:
            distances = distances.checkpoint()
        else:
            distances = distances.persist(StorageLevel.DISK_ONLY)

        prev_distances.unpersist()
        frontier.unpersist()

    total_nodes = distances.count()
    log.info(f"[BFS] 完成。覆蓋節點總數：{total_nodes:,}")
    return distances    # (node BIGINT, dist INT)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3B  圖連通分量 Union-Find（Small-Star / Large-Star 演算法）
# ══════════════════════════════════════════════════════════════════════════════

def distributed_union_find(
    edges: DataFrame,
    blacklist: DataFrame,
    max_iter: int = 20,
) -> DataFrame:
    """
    使用 Small-Star / Large-Star 迭代演算法（Cohen 2009）
    計算圖的連通分量（Connected Components），判斷每個節點是否
    與任何黑名單節點屬於同一個連通子圖。

    輸出欄位：
      node         BIGINT
      component_id BIGINT  — 連通分量代表元（最小 node_id）
      in_blacklist_component BOOLEAN — 該分量是否包含黑名單節點

    注意：此演算法不計算跳轉距離，只判斷「是否在黑名單的網路圈子內」，
    適合作為 BFS 跳轉數的快速前置過濾。

    時間複雜度：O(log(V) × (V + E))，比 BFS 迭代 Join 更快，
    適合超大規模圖（數十億節點）。
    """
    log.info("[Union-Find] 開始計算連通分量...")

    # 初始化：每個節點的代表元 = 自身（node, node）
    all_nodes = (
        edges.select(F.col("src").alias("node"))
        .union(edges.select(F.col("dst").alias("node")))
        .dropDuplicates()
        .select(F.col("node"), F.col("node").alias("root"))
        .persist(StorageLevel.DISK_ONLY)
    )

    components = all_nodes  # (node BIGINT, root BIGINT)

    for i in range(max_iter):
        t0 = time.time()

        # ── Large-Star：節點接收所有鄰居中最小的 root ────────────────
        #    原理：若 root(neighbor) < root(self)，則更新 root(self)
        neighbor_roots = (
            edges
            .join(components.alias("c1"), edges["src"] == F.col("c1.node"), "inner")
            .join(components.alias("c2"), edges["dst"] == F.col("c2.node"), "inner")
            .select(
                edges["src"].alias("node"),
                F.least(F.col("c1.root"), F.col("c2.root")).alias("min_root"),
            )
            .union(
                edges
                .join(components.alias("c3"), edges["dst"] == F.col("c3.node"), "inner")
                .join(components.alias("c4"), edges["src"] == F.col("c4.node"), "inner")
                .select(
                    edges["dst"].alias("node"),
                    F.least(F.col("c3.root"), F.col("c4.root")).alias("min_root"),
                )
            )
        )

        new_components = (
            components
            .join(neighbor_roots, on="node", how="left")
            .select(
                F.col("node"),
                F.least(
                    F.col("root"),
                    F.coalesce(F.col("min_root"), F.col("root"))
                ).alias("root"),
            )
            .persist(StorageLevel.DISK_ONLY)
        )

        # ── 收斂判斷：若沒有任何節點更新則停止 ──────────────────────
        changed = (
            components.join(new_components.alias("n"), on="node", how="inner")
                      .filter(F.col("root") != F.col("n.root"))
                      .count()
        )

        elapsed = round(time.time() - t0, 1)
        log.info(f"[Union-Find] 第 {i+1} 輪：{changed:,} 個節點更新  耗時 {elapsed}s")

        components.unpersist()
        components = new_components

        if changed == 0:
            log.info("[Union-Find] 收斂完成。")
            break

    # ── 標記含黑名單的連通分量 ────────────────────────────────────────
    blacklist_components = (
        blacklist
        .join(components, blacklist["node"] == components["node"], "inner")
        .select(F.col("root").alias("blacklist_root"))
        .distinct()
    )

    result = (
        components
        .join(blacklist_components,
              components["root"] == blacklist_components["blacklist_root"],
              how="left")
        .select(
            F.col("node"),
            F.col("root").alias("component_id"),
            F.col("blacklist_root").isNotNull().alias("in_blacklist_component"),
        )
    )

    log.info(f"[Union-Find] 完成。總連通分量數：{result.select('component_id').distinct().count():,}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4  組裝最終特徵 DataFrame
# ══════════════════════════════════════════════════════════════════════════════

def assemble_hop_features(
    edges:       DataFrame,
    bfs_result:  DataFrame,     # (node BIGINT, dist INT)  ← 含 IP 中繼節點（負數）
    uf_result:   DataFrame,     # (node BIGINT, component_id BIGINT, in_blacklist_component BOOL)
    blacklist:   DataFrame,
    ip_features: DataFrame,     # (user_id BIGINT, ip_shared_user_count INT, has_high_speed_risk BOOL)
) -> DataFrame:
    """
    結合 BFS + Union-Find + IP 特徵，輸出最終特徵表。

    輸出欄位（對應 Feature Group Schema）：
      user_id                    BIGINT
      min_hops_to_blacklist      INT        0=黑名單 1=直接鄰居 ... ISOLATED_HOPS=孤立
      is_direct_neighbor         BOOLEAN
      blacklist_neighbor_count   INT        直接相連的黑名單節點數
      in_blacklist_network       BOOLEAN    是否在黑名單連通分量中
      ip_shared_user_count       INT        用戶所用 IP 最多被幾人共用（機房指標）
      has_high_speed_risk        BOOLEAN    是否有 <10 分鐘的高速交易
      hop_risk_level             STRING     blacklist/direct/indirect_2/indirect_3/isolated
      weighted_risk_label        STRING     HIGH_WEIGHTED（機房集體詐騙）/ HIGH / MEDIUM / LOW
      event_time                 TIMESTAMP  特徵計算時間（Feature Store 必填）

    IP 節點過濾策略：
      edges 與 bfs_result 包含負數 IP 節點（_ip_to_node_id 產生）。
      all_nodes / bfs_user / uf_user 只保留 node > 0（真實 user_id）。
    """
    # ── 過濾 IP 節點：只保留正數 user_id 節點 ───────────────────────────
    bfs_user = bfs_result.filter(F.col("node") > 0)
    uf_user  = uf_result.filter(F.col("node") > 0)

    # 所有真實用戶節點（排除負數 IP 中繼節點）
    all_nodes = (
        edges.filter(F.col("src") > 0).select(F.col("src").alias("node"))
        .union(edges.filter(F.col("dst") > 0).select(F.col("dst").alias("node")))
        .union(blacklist.select("node"))
        .dropDuplicates()
    )

    # ── 直接鄰居黑名單計數（User-to-User 邊，排除 IP 中繼） ──────────────
    blacklist_ids = blacklist.select(F.col("node").alias("bl_node"))

    bl_neighbor_count = (
        edges.filter(F.col("src") > 0)          # 只計算 User 發出的邊
        .join(blacklist_ids, edges["dst"] == blacklist_ids["bl_node"], "inner")
        .groupBy(edges["src"].alias("node"))
        .agg(F.countDistinct("dst").alias("blacklist_neighbor_count"))
    )

    # ── 合併結果 ────────────────────────────────────────────────────────
    result = (
        all_nodes
        .join(bfs_user.alias("bfs"),                          on="node", how="left")
        .join(uf_user.select("node", "in_blacklist_component"), on="node", how="left")
        .join(bl_neighbor_count,                               on="node", how="left")
        .join(ip_features, all_nodes["node"] == ip_features["user_id"], how="left")
        .select(
            F.col("node").alias("user_id"),
            F.coalesce(
                F.col("dist"), F.lit(ISOLATED_HOPS)
            ).cast(IntegerType()).alias("min_hops_to_blacklist"),
            F.coalesce(
                F.col("blacklist_neighbor_count"), F.lit(0)
            ).cast(IntegerType()).alias("blacklist_neighbor_count"),
            F.coalesce(
                F.col("in_blacklist_component"), F.lit(False)
            ).alias("in_blacklist_network"),
            F.coalesce(
                F.col("ip_shared_user_count"), F.lit(0)
            ).cast(IntegerType()).alias("ip_shared_user_count"),
            F.coalesce(
                F.col("has_high_speed_risk"), F.lit(False)
            ).alias("has_high_speed_risk"),
        )
    )

    # ── 衍生欄位 ────────────────────────────────────────────────────────
    result = result.withColumn(
        "is_direct_neighbor",
        F.col("min_hops_to_blacklist") == 1
    ).withColumn(
        "hop_risk_level",
        F.when(F.col("min_hops_to_blacklist") == 0, "blacklist")
         .when(F.col("min_hops_to_blacklist") == 1, "direct")
         .when(F.col("min_hops_to_blacklist") == 2, "indirect_2")
         .when(F.col("min_hops_to_blacklist") == 3, "indirect_3")
         .otherwise("isolated")
    ).withColumn(
        # ── 複合風險加權標籤（機房集體詐騙核心判斷） ─────────────────────
        #   條件：2 跳內 ＋ IP 被超過 3 帳號共用 ＋ 有高速交易 → 機房詐騙
        "weighted_risk_label",
        F.when(
            (F.col("min_hops_to_blacklist") <= 2) &
            (F.col("ip_shared_user_count") > IP_SHARED_THRESHOLD) &
            F.col("has_high_speed_risk"),
            "HIGH_WEIGHTED"     # 機房集體詐騙：最高優先處置
        ).when(
            F.col("hop_risk_level") == "blacklist",
            "BLACKLIST"
        ).when(
            F.col("hop_risk_level") == "direct",
            "HIGH"
        ).when(
            F.col("hop_risk_level").isin("indirect_2", "indirect_3"),
            "MEDIUM"
        ).otherwise("LOW")
    ).withColumn(
        "event_time",
        F.lit(RUN_TS).cast(TimestampType())
    )

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5  寫出：S3 Parquet + SageMaker Feature Store 批次攝取
# ══════════════════════════════════════════════════════════════════════════════

def write_to_s3(features: DataFrame, output_prefix: str) -> str:
    """
    將特徵寫出為 S3 Parquet（Hive 分區格式，供 Athena 查詢）。

    分區策略：依 hop_risk_level 分區
      s3://bucket/prefix/hop_risk_level=direct/part-00001.parquet
    """
    output_path = f"s3://{S3_BUCKET}/{output_prefix}/hop_features/"

    (
        features
        .repartition(200, "hop_risk_level")
        .write
        .mode("overwrite")
        .option("compression", "snappy")        # 明確指定 Snappy（Glue/Athena 最佳相容性）
        .partitionBy("hop_risk_level")
        .parquet(output_path)
    )

    log.info(f"[Step 5] 特徵已寫出 → {output_path}")
    return output_path


def ingest_to_feature_store(
    features: DataFrame,
    feature_group_name: str,
    region: str,
    batch_size: int = 500,
) -> None:
    """
    將 hop 特徵批次攝取到 SageMaker Feature Store（Online + Offline Store）。

    策略：將 DataFrame 以 foreachPartition 分批呼叫 PutRecord API，
    利用 Spark 的分散式 Executor 並行上傳，避免 Driver 成為瓶頸。

    注意：SageMaker Feature Store PutRecord 每次限 1 筆，
         不支援 Batch API，因此用 foreachPartition 分批控制並發。

    Parameters
    ----------
    features           : 包含 user_id, event_time 等欄位的 DataFrame
    feature_group_name : SageMaker Feature Group 名稱
    region             : AWS Region
    batch_size         : 每個 partition 最多保留幾筆上傳（測試時可縮小）
    """
    def put_records_partition(rows):
        """在 Executor 上執行，每個 partition 建立自己的 boto3 client。"""
        import boto3
        client = boto3.client("sagemaker-featurestore-runtime", region_name=region)

        for row in rows:
            record = [
                {"FeatureName": "user_id",                    "ValueAsString": str(int(row.user_id))},
                {"FeatureName": "min_hops_to_blacklist",      "ValueAsString": str(int(row.min_hops_to_blacklist))},
                {"FeatureName": "is_direct_neighbor",         "ValueAsString": "1" if row.is_direct_neighbor else "0"},
                {"FeatureName": "blacklist_neighbor_count",   "ValueAsString": str(int(row.blacklist_neighbor_count))},
                {"FeatureName": "in_blacklist_network",       "ValueAsString": "1" if row.in_blacklist_network else "0"},
                {"FeatureName": "ip_shared_user_count",       "ValueAsString": str(int(row.ip_shared_user_count))},
                {"FeatureName": "has_high_speed_risk",        "ValueAsString": "1" if row.has_high_speed_risk else "0"},
                {"FeatureName": "hop_risk_level",             "ValueAsString": str(row.hop_risk_level)},
                {"FeatureName": "weighted_risk_label",        "ValueAsString": str(row.weighted_risk_label)},
                {"FeatureName": "event_time",                 "ValueAsString": row.event_time.isoformat()},
            ]
            try:
                client.put_record(
                    FeatureGroupName=feature_group_name,
                    Record=record,
                )
            except Exception as e:
                # 單筆失敗不中斷整批（記錄錯誤供後續補傳）
                log.warning(f"PutRecord 失敗 user_id={row.user_id}: {e}")

    log.info(f"[Step 5] 開始攝取 Feature Store: {feature_group_name}")
    (
        features
        .repartition(200)   # 200 個 Partition → 200 個並行 boto3 client
        .foreachPartition(put_records_partition)
    )
    log.info("[Step 5] Feature Store 攝取完成。")


# ══════════════════════════════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 60)
    log.info("幣託 Glue Graph BFS Job 啟動")
    log.info(f"  max_hops         = {MAX_HOPS}")
    log.info(f"  feature_group    = {FG_NAME}")
    log.info(f"  blacklist_s3     = {BLACKLIST_S3}")
    log.info("=" * 60)

    t_start = time.time()

    # Step 1A：建邊（User-to-User + User-to-IP 雙向邊）
    edges = load_and_build_edges(spark)

    # Step 1B：計算 IP 層級特徵（ip_shared_user_count、has_high_speed_risk）
    ip_features = compute_ip_features(spark)

    # Step 2：讀黑名單
    blacklist = load_blacklist(spark)

    # Step 3A：多源 BFS（精確跳數，含 IP 中繼節點）
    bfs_result = multi_source_bfs(edges, blacklist, MAX_HOPS)

    # Step 3B：Union-Find（連通分量）
    uf_result = distributed_union_find(edges, blacklist)

    # Step 4：組裝特徵（過濾 IP 節點 + 加入複合風險加權標籤）
    features = assemble_hop_features(edges, bfs_result, uf_result, blacklist, ip_features)
    features = features.persist(StorageLevel.MEMORY_AND_DISK)

    total = features.count()
    log.info(f"[Step 4] 特徵總筆數：{total:,}")
    log.info("[Step 4] Hop 風險分布：")
    features.groupBy("hop_risk_level").count().orderBy("hop_risk_level").show()
    log.info("[Step 4] 複合加權風險分布（weighted_risk_label）：")
    features.groupBy("weighted_risk_label").count().orderBy("weighted_risk_label").show()
    log.info("[Step 4] 機房集體詐騙（HIGH_WEIGHTED）數量：")
    features.filter(F.col("weighted_risk_label") == "HIGH_WEIGHTED").count()

    # Step 5：寫出
    write_to_s3(features, OUTPUT_PFX)
    ingest_to_feature_store(features, FG_NAME, REGION)

    elapsed = round((time.time() - t_start) / 60, 1)
    log.info(f"Glue Job 完成。總耗時：{elapsed} 分鐘")

    job.commit()


if __name__ == "__main__":
    main()
