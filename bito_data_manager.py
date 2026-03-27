"""
BitoDataManager — 幣託科技 AWS Hackathon 資料管理工具

資料來源: BitoPro AWS Event API (https://aws-event-api.bitopro.com/)
欄位說明: aws-event 資料表及欄位說明.pdf

金額欄位說明：所有 decimal(65,0) 金額欄位皆需乘以 1e-8 才是真實數值。

人頭戶特徵定義（依命題文件）：
  - 快進快出之滯留時間：法幣入金 → 虛幣出金時間差 < 10 分鐘 → high_speed_risk
  - IP 異常跳動：單一用戶在 login_logs 中出現多個不同 source_ip
  - 量能不對稱：KYC 等級與交易金額不相稱（L1 用戶卻有百萬級交易）
"""

import pandas as pd
import numpy as np
import requests
from collections import deque
from typing import Optional

# ── 常數設定 ──────────────────────────────────────────────────────────────────

API_BASE_URL = "https://aws-event-api.bitopro.com"

# 各資料表中需乘以 1e-8 的金額欄位
AMOUNT_FIELDS: dict[str, list[str]] = {
    "user_info":        [],
    "twd_transfer":     ["ori_samount"],
    "crypto_transfer":  ["ori_samount", "twd_srate"],
    "usdt_twd_trading": ["trade_samount", "twd_srate"],
    "usdt_swap":        ["twd_samount", "currency_samount"],
}

# 各資料表中的時間欄位
DATETIME_FIELDS: dict[str, list[str]] = {
    "user_info":        ["confirmed_at", "level1_finished_at", "level2_finished_at"],
    "twd_transfer":     ["created_at"],
    "crypto_transfer":  ["created_at"],
    "usdt_twd_trading": ["updated_at"],
    "usdt_swap":        ["created_at"],
}

DATE_FIELDS: dict[str, list[str]] = {
    "user_info": ["birthday"],
}

SCALE = 1e-8


# ── 核心類別 ──────────────────────────────────────────────────────────────────

class BitoDataManager:
    """
    讀取、正規化幣託科技各資料表。

    支援兩種資料來源：
      1. CSV 檔案（本地）：傳入 csv_dir 參數指定資料夾路徑。
      2. REST API（遠端）：不傳 csv_dir，改呼叫 API 取得資料。

    主要功能：
      - normalize_amounts()：將所有金額欄位乘以 1e-8
      - parse_datetimes()：統一轉換為 pandas Timestamp（可直接計算時間差）
      - load_*()：載入各單一資料表並完成正規化
    """

    def __init__(self, csv_dir: Optional[str] = None, api_base: str = API_BASE_URL):
        """
        Parameters
        ----------
        csv_dir : str, optional
            本地 CSV 資料夾路徑。若為 None，改從 API 取資料。
        api_base : str
            API 根網址，預設為幣託 AWS Event API。
        """
        self.csv_dir = csv_dir
        self.api_base = api_base.rstrip("/")

    # ── 內部工具 ──────────────────────────────────────────────────────────────

    def _load_raw(self, table: str) -> pd.DataFrame:
        """從 CSV 或 API 取得原始 DataFrame。"""
        if self.csv_dir:
            return self._load_csv(table)
        return self._load_api(table)

    def _load_csv(self, table: str) -> pd.DataFrame:
        import os
        import glob
        
        # 嘗試兩種路徑格式：
        # 1. 簡單格式：{csv_dir}/{table}.csv
        # 2. 分區格式：{csv_dir}/{table}/dt=*/part-*.csv（來自 bito_api_ingester）
        
        simple_path = os.path.join(self.csv_dir, f"{table}.csv")
        if os.path.exists(simple_path):
            return pd.read_csv(simple_path, dtype=str)
        
        # 嘗試分區格式
        partition_pattern = os.path.join(self.csv_dir, table, "dt=*", "part-*.csv")
        partition_files = sorted(glob.glob(partition_pattern))
        
        if partition_files:
            dfs = [pd.read_csv(f, dtype=str) for f in partition_files]
            return pd.concat(dfs, ignore_index=True)
        
        raise FileNotFoundError(
            f"找不到 CSV 檔案：{table}\n"
            f"  嘗試路徑 1：{simple_path}\n"
            f"  嘗試路徑 2：{partition_pattern}"
        )

    def _load_api(self, table: str) -> pd.DataFrame:
        url = f"{self.api_base}/{table}"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        # API 可能回傳 {"data": [...]} 或直接回傳 list
        if isinstance(data, dict):
            data = data.get("data", data)
        return pd.DataFrame(data)

    # ── 正規化函式 ────────────────────────────────────────────────────────────

    def normalize_amounts(self, df: pd.DataFrame, table: str) -> pd.DataFrame:
        """
        將指定資料表的所有金額欄位乘以 1e-8。

        除了根據 AMOUNT_FIELDS 設定的明確欄位外，
        也會自動偵測欄位名稱中包含 'amount' 或 'srate' 的欄位作為保險。

        Parameters
        ----------
        df    : 待處理的 DataFrame
        table : 資料表名稱（用於查詢 AMOUNT_FIELDS 設定）

        Returns
        -------
        已正規化金額的 DataFrame（不修改原始物件）
        """
        df = df.copy()

        # 1. 由設定檔決定的明確欄位
        explicit = set(AMOUNT_FIELDS.get(table, []))

        # 2. 自動偵測（欄位名稱含 amount / srate 的欄位）
        auto_detected = {
            col for col in df.columns
            if any(kw in col.lower() for kw in ("amount", "srate"))
        }

        target_cols = explicit | auto_detected

        for col in target_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce") * SCALE

        return df

    def parse_datetimes(self, df: pd.DataFrame, table: str) -> pd.DataFrame:
        """
        將時間欄位統一轉換為 pandas Timestamp（UTC-naïve）。

        轉換後可直接做時間差運算，例如：
            df['created_at'] - df['confirmed_at']  → Timedelta

        Parameters
        ----------
        df    : 待處理的 DataFrame
        table : 資料表名稱

        Returns
        -------
        已轉換時間欄位的 DataFrame（不修改原始物件）
        """
        df = df.copy()

        for col in DATETIME_FIELDS.get(table, []):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce", format="mixed")

        for col in DATE_FIELDS.get(table, []):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce").dt.date

        return df

    def _process(self, table: str) -> pd.DataFrame:
        """載入 → 金額正規化 → 時間轉換，回傳乾淨的 DataFrame。"""
        df = self._load_raw(table)
        df = self.normalize_amounts(df, table)
        df = self.parse_datetimes(df, table)
        return df

    # ── 各資料表載入介面 ──────────────────────────────────────────────────────

    def load_users(self) -> pd.DataFrame:
        """
        載入 user_info 資料表（用戶基本資料與 KYC 資訊）。

        主要欄位：
          user_id, status, sex, birthday, career, income_source,
          confirmed_at, level1_finished_at, level2_finished_at, user_source
        """
        return self._process("user_info")

    def load_twd_transfer(self) -> pd.DataFrame:
        """
        載入 twd_transfer 資料表（台幣入金／出金）。

        主要欄位：
          created_at, user_id, kind (0=入金, 1=出金),
          ori_samount [TWD, 已 ×1e-8], source_ip
        """
        return self._process("twd_transfer")

    def load_crypto_transfer(self) -> pd.DataFrame:
        """
        載入 crypto_transfer 資料表（虛擬貨幣入出金）。

        主要欄位：
          created_at, user_id, kind, sub_kind,
          ori_samount [幣量, 已 ×1e-8],
          twd_srate [對台幣匯率, 已 ×1e-8],
          currency, protocol, from_wallet, to_wallet,
          relation_user_id, source_ip
        """
        return self._process("crypto_transfer")

    def load_trades(self, include_swap: bool = True) -> pd.DataFrame:
        """
        載入交易訂單（掛單簿 + 一鍵買賣）。

        Parameters
        ----------
        include_swap : bool
            True  → 合併 usdt_twd_trading（掛單簿）與 usdt_swap（一鍵買賣）。
            False → 僅回傳 usdt_twd_trading。

        掛單簿欄位：
          updated_at, user_id, is_buy, trade_samount [已 ×1e-8],
          twd_srate [已 ×1e-8], is_market, source, source_ip

        一鍵買賣欄位：
          created_at, user_id, kind (0=買幣, 1=賣幣),
          twd_samount [已 ×1e-8], currency_samount [已 ×1e-8]
        """
        trading = self._process("usdt_twd_trading")
        trading["_table"] = "usdt_twd_trading"

        if not include_swap:
            return trading

        swap = self._process("usdt_swap")
        swap["_table"] = "usdt_swap"

        # 統一時間欄位名稱為 created_at 方便後續比對
        if "updated_at" in trading.columns and "created_at" not in trading.columns:
            trading = trading.rename(columns={"updated_at": "created_at"})

        return pd.concat([trading, swap], ignore_index=True)

    # ── 人頭戶特徵提取 ────────────────────────────────────────────────────────

    def feature_retention_time(
        self,
        twd_transfer: pd.DataFrame,
        crypto_transfer: pd.DataFrame,
        high_speed_threshold_minutes: float = 10.0,
    ) -> pd.DataFrame:
        """
        特徵①：資金滯留時間 (Retention Time)

        邏輯：
          對每筆法幣入金（twd_transfer.kind == 0），找同一用戶在其之後
          最快發生的虛幣出金（crypto_transfer.kind == 1），計算時間差。
          若最小時間差 < high_speed_threshold_minutes 分鐘，標記 high_speed_risk = True。

        Parameters
        ----------
        twd_transfer      : load_twd_transfer() 的結果（已正規化）
        crypto_transfer   : load_crypto_transfer() 的結果（已正規化）
        high_speed_threshold_minutes : 風險門檻，預設 10 分鐘

        Returns
        -------
        DataFrame，以 user_id 為索引，欄位：
          min_retention_minutes  : 該用戶最短滯留時間（分鐘），無配對則為 NaN
          retention_event_count  : 入金→出金配對次數
          high_speed_risk        : bool，是否有任一筆 < 門檻
        """
        # 取法幣入金、虛幣出金
        # ── 容錯：crypto_transfer 為空或缺欄位時回傳空結果 ──────────────────
        if (
            crypto_transfer is None
            or crypto_transfer.empty
            or "kind" not in crypto_transfer.columns
        ):
            base_users = twd_transfer[["user_id"]].drop_duplicates().copy()
            base_users["min_retention_minutes"] = np.nan
            base_users["retention_event_count"] = 0
            base_users["high_speed_risk"]       = False
            return base_users

        deposits     = twd_transfer[twd_transfer["kind"].astype(str) == "0"][["user_id", "created_at"]].copy()
        withdrawals  = crypto_transfer[crypto_transfer["kind"].astype(str) == "1"][["user_id", "created_at"]].copy()

        deposits["created_at"]    = pd.to_datetime(deposits["created_at"],    errors="coerce")
        withdrawals["created_at"] = pd.to_datetime(withdrawals["created_at"], errors="coerce")

        deposits    = deposits.dropna(subset=["created_at"]).rename(columns={"created_at": "dep_at"})
        withdrawals = withdrawals.dropna(subset=["created_at"]).rename(columns={"created_at": "wit_at"})

        # Cross-join 同一用戶的入金與出金，再篩選出金在入金之後
        merged = deposits.merge(withdrawals, on="user_id", how="inner")
        merged = merged[merged["wit_at"] > merged["dep_at"]].copy()
        merged["retention_minutes"] = (merged["wit_at"] - merged["dep_at"]).dt.total_seconds() / 60

        # 每筆入金只取最快的一次出金（防止同一筆入金重複計算）
        fastest = (
            merged.sort_values("retention_minutes")
                  .drop_duplicates(subset=["user_id", "dep_at"])
        )

        result = (
            fastest.groupby("user_id")
                   .agg(
                       min_retention_minutes=("retention_minutes", "min"),
                       retention_event_count=("retention_minutes", "count"),
                   )
                   .reset_index()
        )
        result["high_speed_risk"] = (
            result["min_retention_minutes"] < high_speed_threshold_minutes
        )
        return result

    def feature_ip_anomaly(
        self,
        login_logs: pd.DataFrame,
        *,
        fallback_tables: Optional[list[pd.DataFrame]] = None,
        unique_ip_threshold: int = 3,
    ) -> pd.DataFrame:
        """
        特徵②：IP 異常跳動

        邏輯：
          統計每個用戶在 login_logs 中出現的不同 source_ip 數量。
          若 login_logs 為空或不存在，自動 fallback 至 twd_transfer /
          crypto_transfer / usdt_twd_trading 的 source_ip 欄位（合併計算）。

        Parameters
        ----------
        login_logs        : login_logs 資料表 DataFrame（需含 user_id, source_ip）
                            若尚無此資料表可傳入空 DataFrame()。
        fallback_tables   : 當 login_logs 無資料時，用於補充 IP 來源的其他資料表列表
                            （例如 [twd_transfer, crypto_transfer]）
        unique_ip_threshold : 不同 IP 超過此數即標記 ip_anomaly = True，預設 3

        Returns
        -------
        DataFrame，以 user_id 為索引，欄位：
          unique_ip_count  : 不同 source_ip 數量
          ip_source        : 資料來源（"login_logs" 或 "fallback"）
          ip_anomaly       : bool
        """
        def _extract_ip_records(df: pd.DataFrame) -> pd.DataFrame:
            if df.empty or "source_ip" not in df.columns or "user_id" not in df.columns:
                return pd.DataFrame(columns=["user_id", "source_ip"])
            return df[["user_id", "source_ip"]].dropna()

        use_login = not login_logs.empty and "source_ip" in login_logs.columns
        if use_login:
            ip_records = _extract_ip_records(login_logs)
            source_label = "login_logs"
        else:
            parts = [_extract_ip_records(t) for t in (fallback_tables or [])]
            ip_records = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["user_id", "source_ip"])
            source_label = "fallback"

        ip_records["source_ip"] = ip_records["source_ip"].astype(str)

        result = (
            ip_records.groupby("user_id")["source_ip"]
                      .nunique()
                      .reset_index(name="unique_ip_count")
        )
        result["ip_source"]  = source_label
        result["ip_anomaly"] = result["unique_ip_count"] > unique_ip_threshold
        return result

    def feature_volume_asymmetry(
        self,
        users: pd.DataFrame,
        twd_transfer: pd.DataFrame,
        crypto_transfer: pd.DataFrame,
        trades: Optional[pd.DataFrame] = None,
        *,
        l1_threshold_twd: float = 1_000_000.0,
        l0_threshold_twd: float = 100_000.0,
    ) -> pd.DataFrame:
        """
        特徵③：量能不對稱 (KYC Level vs. Trading Volume)

        KYC 等級定義（依命題文件欄位）：
          L0 : level1_finished_at 為空            → 未完成手機驗證
          L1 : level1_finished_at 有值，level2 為空 → 僅手機驗證，無法幣交易資格
          L2 : level2_finished_at 有值            → 完整身份驗證

        偏離度計算：
          total_twd_volume = 所有台幣入出金 + 虛幣折台幣 + 掛單交易折台幣 的加總。
          volume_zscore    = (total_twd_volume - group_mean) / group_std
                             （以相同 KYC 等級的用戶為母體）
          asymmetry_flag   = L0 用戶超過 l0_threshold_twd，或
                             L1 用戶超過 l1_threshold_twd

        Parameters
        ----------
        users           : load_users() 的結果
        twd_transfer    : load_twd_transfer() 的結果
        crypto_transfer : load_crypto_transfer() 的結果
        trades          : load_trades() 的結果（可選，None 則略過掛單部分）
        l1_threshold_twd : L1 用戶的百萬級門檻，預設 1,000,000 TWD
        l0_threshold_twd : L0 用戶的門檻，預設 100,000 TWD

        Returns
        -------
        DataFrame，以 user_id 為索引，欄位：
          kyc_level           : 0 / 1 / 2
          total_twd_volume    : 估算總交易量（TWD）
          volume_zscore       : 同 KYC 群組內的 Z-score
          asymmetry_flag      : bool，量能明顯不對稱
          asymmetry_reason    : 說明字串
        """
        # ── 1. 計算 KYC level ────────────────────────────────────────────────
        u = users[["user_id", "level1_finished_at", "level2_finished_at"]].copy()
        u["level1_finished_at"] = pd.to_datetime(u["level1_finished_at"], errors="coerce")
        u["level2_finished_at"] = pd.to_datetime(u["level2_finished_at"], errors="coerce")

        u["kyc_level"] = 0
        u.loc[u["level1_finished_at"].notna(), "kyc_level"] = 1
        u.loc[u["level2_finished_at"].notna(), "kyc_level"] = 2

        # ── 2. 匯總各來源的 TWD 交易量 ───────────────────────────────────────
        # 法幣入出金（已是 TWD）
        twd_vol = (
            twd_transfer.groupby("user_id")["ori_samount"]
                        .sum()
                        .reset_index(name="twd_from_transfer")
        )

        # 虛幣折台幣：ori_samount × twd_srate
        # ── 容錯：crypto_transfer 為空或缺欄位時跳過 ──────────────────────
        if (
            crypto_transfer is not None
            and not crypto_transfer.empty
            and "ori_samount" in crypto_transfer.columns
            and "twd_srate" in crypto_transfer.columns
        ):
            ct = crypto_transfer.copy()
            ct["twd_equiv"] = (
                pd.to_numeric(ct["ori_samount"], errors="coerce") *
                pd.to_numeric(ct["twd_srate"],   errors="coerce")
            )
            crypto_vol = (
                ct.groupby("user_id")["twd_equiv"]
                  .sum()
                  .reset_index(name="twd_from_crypto")
            )
        else:
            crypto_vol = pd.DataFrame(columns=["user_id", "twd_from_crypto"])

        # 掛單交易折台幣（可選）
        if trades is not None:
            tr = trades.copy()
            # usdt_twd_trading: trade_samount × twd_srate
            # usdt_swap: twd_samount 直接就是 TWD
            if "trade_samount" in tr.columns and "twd_srate" in tr.columns:
                tr["twd_equiv"] = (
                    pd.to_numeric(tr.get("trade_samount", 0), errors="coerce") *
                    pd.to_numeric(tr.get("twd_srate", 1),    errors="coerce")
                )
            elif "twd_samount" in tr.columns:
                tr["twd_equiv"] = pd.to_numeric(tr["twd_samount"], errors="coerce")
            else:
                tr["twd_equiv"] = 0.0

            trades_vol = (
                tr.groupby("user_id")["twd_equiv"]
                  .sum()
                  .reset_index(name="twd_from_trades")
            )
        else:
            trades_vol = pd.DataFrame(columns=["user_id", "twd_from_trades"])

        # ── 3. 合併 & 計算總量 ────────────────────────────────────────────────
        vol = u.merge(twd_vol,    on="user_id", how="left")
        vol = vol.merge(crypto_vol, on="user_id", how="left")
        vol = vol.merge(trades_vol, on="user_id", how="left")

        vol = vol.fillna({"twd_from_transfer": 0, "twd_from_crypto": 0, "twd_from_trades": 0})
        vol["total_twd_volume"] = (
            vol["twd_from_transfer"] + vol["twd_from_crypto"] + vol["twd_from_trades"]
        )

        # ── 4. 同 KYC 群組 Z-score ───────────────────────────────────────────
        group_stats = (
            vol.groupby("kyc_level")["total_twd_volume"]
               .agg(g_mean="mean", g_std="std")
               .reset_index()
        )
        vol = vol.merge(group_stats, on="kyc_level", how="left")
        vol["volume_zscore"] = (
            (vol["total_twd_volume"] - vol["g_mean"]) /
            vol["g_std"].replace(0, np.nan)
        )

        # ── 5. 標記偏離 ───────────────────────────────────────────────────────
        def _flag(row: pd.Series):
            lvl, vol_twd = row["kyc_level"], row["total_twd_volume"]
            if lvl == 0 and vol_twd > l0_threshold_twd:
                return True, f"L0 用戶交易量 {vol_twd:,.0f} TWD 超過門檻 {l0_threshold_twd:,.0f}"
            if lvl == 1 and vol_twd > l1_threshold_twd:
                return True, f"L1 用戶交易量 {vol_twd:,.0f} TWD 超過百萬門檻 {l1_threshold_twd:,.0f}"
            if abs(row.get("volume_zscore", 0) or 0) > 3:
                return True, f"KYC L{lvl} 群組 Z-score={row['volume_zscore']:.2f} (>3σ)"
            return False, ""

        flags = vol.apply(_flag, axis=1, result_type="expand")
        vol["asymmetry_flag"]   = flags[0]
        vol["asymmetry_reason"] = flags[1]

        return vol[[
            "user_id", "kyc_level", "total_twd_volume",
            "volume_zscore", "asymmetry_flag", "asymmetry_reason",
        ]]

    # ── 特徵④：資金關聯深度（BFS 跳轉數） ────────────────────────────────────

    def _build_transaction_graph(
        self,
        twd_transfer: pd.DataFrame,
        crypto_transfer: pd.DataFrame,
        *,
        use_wallet_edges: bool = True,
        use_ip_edges: bool = False,
    ) -> dict[int, set[int]]:
        """
        從交易資料建立用戶間的無向鄰接表（adjacency dict）。

        邊的來源（三層，由明確到隱含）：
          1. 【明確邊】crypto_transfer.relation_user_id
               平台內部用戶間的加密貨幣直接轉帳，最可靠的關聯依據。

          2. 【錢包邊】crypto_transfer 的 from_wallet / to_wallet 複用
               若同一個錢包地址同時出現在不同用戶的交易紀錄中，
               代表他們共用錢包，視為隱含關聯。
               （use_wallet_edges=True 啟用）

          3. 【IP 邊】twd_transfer 的相同 source_ip 於同一小時共現
               同 IP 同時段操作視為隱含關聯，較弱的訊號。
               （use_ip_edges=True 啟用，預設關閉，避免 NAT/VPN 誤報）

        Parameters
        ----------
        use_wallet_edges : 啟用錢包地址複用邊（推薦開啟）
        use_ip_edges     : 啟用 IP 共現邊（謹慎使用）

        Returns
        -------
        dict[int, set[int]] — 無向鄰接表，key 為 user_id（int）
        """
        graph: dict[int, set[int]] = {}

        def _add_edge(u: int, v: int) -> None:
            """將無向邊 (u, v) 加入鄰接表，跳過自環與無效節點。"""
            if u == v or u <= 0 or v <= 0:
                return
            graph.setdefault(u, set()).add(v)
            graph.setdefault(v, set()).add(u)

        # ── 邊類型 1：crypto_transfer.relation_user_id（明確用戶對用戶） ──────
        ct = crypto_transfer.copy()
        ct["user_id"]          = pd.to_numeric(ct["user_id"],          errors="coerce")
        ct["relation_user_id"] = pd.to_numeric(ct.get("relation_user_id", np.nan), errors="coerce")

        explicit = ct.dropna(subset=["user_id", "relation_user_id"])
        explicit = explicit[explicit["relation_user_id"] > 0]
        for uid, rid in zip(
            explicit["user_id"].astype(int),
            explicit["relation_user_id"].astype(int),
        ):
            _add_edge(uid, rid)

        # ── 邊類型 2：錢包地址複用（多個 user 共用同一錢包） ─────────────────
        if use_wallet_edges and not ct.empty:
            # API 實際欄位為 from_wallet_hash / to_wallet_hash
            # 同時支援無 _hash 後綴的舊版 CSV 格式
            _wallet_candidates = (
                "from_wallet_hash", "to_wallet_hash",
                "from_wallet",      "to_wallet",
            )
            _present_wallets = [c for c in _wallet_candidates if c in ct.columns]
            for wallet_col in _present_wallets:
                if wallet_col not in ct.columns:
                    continue

                wallet_df = (
                    ct[["user_id", wallet_col]]
                    .dropna()
                    .rename(columns={wallet_col: "wallet"})
                )
                wallet_df = wallet_df[wallet_df["wallet"].astype(str).str.len() > 5]
                wallet_df["user_id"] = wallet_df["user_id"].astype(int)

                # 同一 wallet 下，找所有 user_id 對
                wallet_users = (
                    wallet_df.groupby("wallet")["user_id"]
                              .apply(lambda ids: list(set(ids)))
                              .reset_index(name="users")
                )
                for users_in_wallet in wallet_users["users"]:
                    if len(users_in_wallet) < 2:
                        continue
                    # 只對最多 50 個用戶建邊（防止超大公共錢包造成圖爆炸）
                    sample = users_in_wallet[:50]
                    for i in range(len(sample)):
                        for j in range(i + 1, len(sample)):
                            _add_edge(sample[i], sample[j])

        # ── 邊類型 3：相同 IP 同小時共現（可選，較弱訊號） ────────────────────
        if use_ip_edges:
            twd = twd_transfer.copy()
            twd["user_id"]    = pd.to_numeric(twd["user_id"],    errors="coerce")
            twd["created_at"] = pd.to_datetime(twd["created_at"], errors="coerce")
            twd = twd.dropna(subset=["user_id", "created_at", "source_ip"])
            twd["hour_bucket"] = twd["created_at"].dt.floor("H")
            twd["user_id"]     = twd["user_id"].astype(int)

            ip_hour = (
                twd.groupby(["source_ip", "hour_bucket"])["user_id"]
                   .apply(lambda ids: list(set(ids)))
                   .reset_index(name="users")
            )
            for users_in_bucket in ip_hour["users"]:
                if len(users_in_bucket) < 2:
                    continue
                sample = users_in_bucket[:20]   # 同 IP 最多取 20 人，避免 VPN 誤爆
                for i in range(len(sample)):
                    for j in range(i + 1, len(sample)):
                        _add_edge(sample[i], sample[j])

        return graph

    @staticmethod
    def _multi_source_bfs(
        graph: dict[int, set[int]],
        sources: set[int],
        max_hops: int,
    ) -> dict[int, int]:
        """
        多源 BFS（Multi-Source BFS）：同時從所有黑名單節點出發，
        向外擴散，計算每個節點距最近黑名單節點的最短跳轉數。

        複雜度：O(V + E)，與黑名單大小無關。

        Parameters
        ----------
        graph     : 無向鄰接表
        sources   : 已知黑名單 user_id 集合（起始節點，距離=0）
        max_hops  : 搜尋上限，超過則停止（避免遍歷整張圖）

        Returns
        -------
        dict[user_id → min_hops]
          - 黑名單本身      : 0
          - 可到達的節點    : 1 ~ max_hops
          - 不在圖中的節點 : 不在回傳 dict 中（呼叫端填 max_hops+1）
        """
        dist: dict[int, int] = {}
        queue: deque[int]    = deque()

        # 初始化：所有黑名單節點距離設為 0
        for s in sources:
            if s in graph:          # 只將有邊的黑名單節點加入（孤立節點不擴散）
                dist[s] = 0
                queue.append(s)

        while queue:
            node = queue.popleft()
            current_dist = dist[node]

            if current_dist >= max_hops:
                continue            # 已達上限，不再擴散

            for neighbor in graph.get(node, set()):
                if neighbor not in dist:
                    dist[neighbor] = current_dist + 1
                    queue.append(neighbor)

        return dist

    def feature_graph_hops(
        self,
        twd_transfer: pd.DataFrame,
        crypto_transfer: pd.DataFrame,
        known_blacklist: set,
        *,
        max_hops: int = 3,
        use_wallet_edges: bool = True,
        use_ip_edges: bool = False,
    ) -> pd.DataFrame:
        """
        特徵④：資金關聯深度（Graph Hops to Blacklist）

        計算每個 user_id 距離最近已知黑名單用戶的最短交易跳轉數，
        作為「共謀程度」的代理指標。

        跳轉數語義：
          0 → 本身即黑名單（訓練集正樣本）
          1 → 直接與黑名單有交易（高風險）
          2 → 透過一個中間人與黑名單有交易（中風險）
          3 → 三跳內可達（低風險但需關注）
          max_hops+1（預設 4） → 在 N 層內無關聯（孤立/正常）

        邊的建構（詳見 _build_transaction_graph）：
          - 明確邊：crypto_transfer.relation_user_id
          - 錢包邊：共用 from_wallet / to_wallet（可選）
          - IP 邊 ：同 IP 同小時出現（可選，預設關閉）

        Parameters
        ----------
        twd_transfer     : load_twd_transfer() 的結果
        crypto_transfer  : load_crypto_transfer() 的結果
        known_blacklist  : 已知黑名單 user_id 的集合（int 或可轉換型態）
        max_hops         : BFS 搜尋上限，預設 3
        use_wallet_edges : 啟用錢包地址複用邊（推薦）
        use_ip_edges     : 啟用 IP 共現邊（謹慎使用）

        Returns
        -------
        DataFrame，欄位：
          user_id                   : int
          min_hops_to_blacklist     : int，0~max_hops+1
          is_direct_neighbor        : bool（hops == 1）
          blacklist_neighbor_count  : int，直接相連的黑名單節點數
          hop_risk_level            : str，"blacklist" / "direct" /
                                           "indirect_2" / "indirect_3" /
                                           "isolated"
        """
        # 正規化 blacklist 為 int set
        blacklist_int: set[int] = {
            int(x) for x in known_blacklist
            if pd.notna(x) and str(x).strip() not in ("", "nan")
        }

        # 建圖
        graph = self._build_transaction_graph(
            twd_transfer, crypto_transfer,
            use_wallet_edges=use_wallet_edges,
            use_ip_edges=use_ip_edges,
        )

        # Multi-Source BFS
        dist_map = self._multi_source_bfs(graph, blacklist_int, max_hops)

        # 收集所有出現在圖中的節點 + 黑名單本身
        all_nodes: set[int] = set(graph.keys()) | blacklist_int

        # 額外計算每個用戶「直接相連的黑名單鄰居數」
        blacklist_neighbor_count: dict[int, int] = {}
        for node in all_nodes:
            neighbors = graph.get(node, set())
            blacklist_neighbor_count[node] = len(neighbors & blacklist_int)

        # ── 計算 in_blacklist_network（連通分量旗標） ────────────────────────
        # 定義：節點只要在 BFS 可達範圍內（dist < ISOLATED）即視為
        # 與黑名單同屬同一個連通子圖，旗標設為 True。
        # 這與 glue_graph_hops.py 的 Union-Find 語義一致：
        #   glue 版本使用完整 Union-Find 確定連通分量，
        #   本地版本以「BFS 可達 = 距離 ≤ max_hops」作為近似。
        # 兩者差異：glue 版本可跨越 max_hops 的長鏈；本地版本受 max_hops 截斷。
        # 在 max_hops=3 下，實際漏判率極低（> 3 跳的黑名單關聯風險已很低）。

        # 組裝結果 DataFrame
        records = []
        ISOLATED = max_hops + 1
        for node in all_nodes:
            hops = dist_map.get(node, ISOLATED)
            bn   = blacklist_neighbor_count.get(node, 0)

            if hops == 0:
                level = "blacklist"
            elif hops == 1:
                level = "direct"
            elif hops == 2:
                level = "indirect_2"
            elif hops == 3:
                level = "indirect_3"
            else:
                level = "isolated"

            records.append({
                "user_id":                  node,
                "min_hops_to_blacklist":    hops,
                "is_direct_neighbor":       hops == 1,
                "blacklist_neighbor_count": bn,
                "hop_risk_level":           level,
                # 連通分量旗標：BFS 可達（含黑名單本身）即為 True
                "in_blacklist_network":     hops <= max_hops,
            })

        result = pd.DataFrame(records)
        result["user_id"] = result["user_id"].astype(int)
        return result.sort_values("min_hops_to_blacklist").reset_index(drop=True)

    def extract_mule_features(
        self,
        users: pd.DataFrame,
        twd_transfer: pd.DataFrame,
        crypto_transfer: pd.DataFrame,
        trades: Optional[pd.DataFrame] = None,
        login_logs: Optional[pd.DataFrame] = None,
        known_blacklist: Optional[set] = None,
        *,
        graph_max_hops: int = 3,
        use_wallet_edges: bool = True,
        use_ip_edges: bool = False,
    ) -> pd.DataFrame:
        """
        整合四項人頭戶特徵，回傳單一寬表（以 user_id 為主鍵）。

        Columns
        -------
        user_id, kyc_level,
        min_retention_minutes, retention_event_count, high_speed_risk,   ← 特徵①
        unique_ip_count, ip_source, ip_anomaly,                          ← 特徵②
        total_twd_volume, volume_zscore, asymmetry_flag, asymmetry_reason, ← 特徵③
        min_hops_to_blacklist, is_direct_neighbor,                       ← 特徵④
        blacklist_neighbor_count, hop_risk_level,
        mule_risk_score  : 0–4，命中幾項風險特徵（含跳轉風險）

        Parameters
        ----------
        known_blacklist : 已知黑名單 user_id 集合。
                          若為 None，略過特徵④（圖跳轉）計算。
        graph_max_hops  : BFS 搜尋層數上限，預設 3
        use_wallet_edges: 啟用錢包複用邊（推薦）
        use_ip_edges    : 啟用 IP 共現邊（謹慎使用）
        """
        f1 = self.feature_retention_time(twd_transfer, crypto_transfer)
        f2 = self.feature_ip_anomaly(
            login_logs if login_logs is not None else pd.DataFrame(),
            fallback_tables=[twd_transfer, crypto_transfer],
        )
        f3 = self.feature_volume_asymmetry(users, twd_transfer, crypto_transfer, trades)

        result = (
            f3.merge(f1, on="user_id", how="left")
              .merge(f2, on="user_id", how="left")
        )

        # ── 特徵④：資金關聯深度（BFS 跳轉數） ──────────────────────────────
        if known_blacklist is not None:
            f4 = self.feature_graph_hops(
                twd_transfer, crypto_transfer,
                known_blacklist=known_blacklist,
                max_hops=graph_max_hops,
                use_wallet_edges=use_wallet_edges,
                use_ip_edges=use_ip_edges,
            )
            result = result.merge(f4, on="user_id", how="left")

            # 孤立節點（不在圖中）填入最大跳轉數，代表無關聯
            result["min_hops_to_blacklist"]    = result["min_hops_to_blacklist"].fillna(graph_max_hops + 1).astype(int)
            result["is_direct_neighbor"]       = result["is_direct_neighbor"].fillna(False)
            result["blacklist_neighbor_count"] = result["blacklist_neighbor_count"].fillna(0).astype(int)
            result["hop_risk_level"]           = result["hop_risk_level"].fillna("isolated")
            # in_blacklist_network：孤立節點（不在圖中）= False
            result["in_blacklist_network"]     = result["in_blacklist_network"].fillna(False)

            # 跳轉風險：1跳=高風險，2跳=中風險，其餘=低風險
            hop_risk_flag = result["min_hops_to_blacklist"].isin([1, 2])
        else:
            hop_risk_flag = pd.Series(False, index=result.index)

        # ── 綜合風險評分（0–4） ───────────────────────────────────────────────
        result["mule_risk_score"] = (
            result["high_speed_risk"].fillna(False).astype(int) +
            result["ip_anomaly"].fillna(False).astype(int) +
            result["asymmetry_flag"].fillna(False).astype(int) +
            hop_risk_flag.astype(int)
        )
        return result

    # ── 便利方法 ──────────────────────────────────────────────────────────────

    def load_all(self) -> dict[str, pd.DataFrame]:
        """
        一次載入全部資料表，回傳 dict。

        若某表無法載入（如 CSV 不存在），會記錄警告並跳過該表。

        Returns
        -------
        {
            "users":           DataFrame,
            "twd_transfer":    DataFrame,
            "crypto_transfer": DataFrame,
            "trades":          DataFrame,
        }
        """
        import logging
        log = logging.getLogger(__name__)
        
        tables = {}
        loaders = {
            "users":           self.load_users,
            "twd_transfer":    self.load_twd_transfer,
            "crypto_transfer": self.load_crypto_transfer,
            "trades":          self.load_trades,
        }
        
        for name, loader in loaders.items():
            try:
                tables[name] = loader()
            except FileNotFoundError as e:
                log.warning(f"跳過表 {name}：{e}")
            except Exception as e:
                log.warning(f"載入表 {name} 失敗：{e}")
        
        return tables


# ── 使用範例 ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    manager = BitoDataManager()          # 從 API 讀取；或傳 csv_dir="./data"

    tables          = manager.load_all()
    users           = tables["users"]
    twd_transfer    = tables["twd_transfer"]
    crypto_transfer = tables["crypto_transfer"]
    trades          = tables["trades"]

    # ── 特徵① 資金滯留時間 ────────────────────────────────────────────────
    f1 = manager.feature_retention_time(twd_transfer, crypto_transfer)
    print("=== 特徵① 資金滯留時間（前 10 筆高風險）===")
    print(f1[f1["high_speed_risk"]].head(10))

    # ── 特徵② IP 異常跳動 ─────────────────────────────────────────────────
    f2 = manager.feature_ip_anomaly(
        pd.DataFrame(),                                     # login_logs 尚未取得時傳空
        fallback_tables=[twd_transfer, crypto_transfer],
    )
    print("\n=== 特徵② IP 異常跳動（前 10 筆異常）===")
    print(f2[f2["ip_anomaly"]].head(10))

    # ── 特徵③ 量能不對稱 ─────────────────────────────────────────────────
    f3 = manager.feature_volume_asymmetry(users, twd_transfer, crypto_transfer, trades)
    print("\n=== 特徵③ 量能不對稱（前 10 筆異常）===")
    print(f3[f3["asymmetry_flag"]][["user_id", "kyc_level", "total_twd_volume",
                                    "volume_zscore", "asymmetry_reason"]].head(10))

    # ── 特徵④ 資金關聯深度（需先有黑名單標籤） ──────────────────────────
    # 假設已載入 train_label（含 user_id, status 欄位）
    # train_label = manager._load_raw("train_label")
    # blacklist = set(train_label[train_label["status"] == 1]["user_id"].astype(int))
    blacklist = set()  # 示範用空集合；實際使用時替換為上方兩行

    f4 = manager.feature_graph_hops(twd_transfer, crypto_transfer, blacklist)
    print("\n=== 特徵④ 資金關聯深度（前 10 筆直接鄰居）===")
    print(f4[f4["is_direct_neighbor"]].head(10))

    # ── 整合特徵：人頭戶風險評分 ─────────────────────────────────────────
    mule_df = manager.extract_mule_features(
        users, twd_transfer, crypto_transfer, trades,
        known_blacklist=blacklist,   # 傳入黑名單以啟用特徵④
    )
    print("\n=== 人頭戶風險評分（mule_risk_score >= 2，前 20 筆）===")
    high_risk = mule_df[mule_df["mule_risk_score"] >= 2].sort_values(
        "mule_risk_score", ascending=False
    )
    print(high_risk[[
        "user_id", "kyc_level", "mule_risk_score",
        "min_retention_minutes", "high_speed_risk",
        "unique_ip_count", "ip_anomaly",
        "total_twd_volume", "asymmetry_flag",
        "min_hops_to_blacklist", "hop_risk_level",
    ]].head(20))
