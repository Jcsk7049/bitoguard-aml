"""
BitoGuard AML — LightGBM 最佳化管線 (本地版)
==============================================
執行方式：
    python lgb_pipeline.py

輸出：
    submission.csv          — 競賽提交 (user_id, status)
    submission_with_prob.csv — 含預測機率版本
    cv_report_lgb.json      — 交叉驗證報告
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix,
    accuracy_score
)
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

# ── 路徑自動偵測（本機 Windows / SageMaker Linux）────────────────────────────
_SAGEMAKER = Path("/home/ec2-user/SageMaker")
_LOCAL     = Path("C:/AWS")
BASE_DIR   = _SAGEMAKER if _SAGEMAKER.exists() else _LOCAL
DATA_DIR   = BASE_DIR / "data"
OUT_DIR    = BASE_DIR          # 輸出檔案放在 BASE_DIR

def _find_csv(table: str) -> Path:
    """支援 dt=YYYY-MM-DD 子目錄或直接放在 table/ 下的 CSV。"""
    # 嘗試最新日期子目錄
    for sub in sorted((DATA_DIR / table).glob("dt=*/part-*.csv"), reverse=True):
        return sub
    # fallback: 直接檔案
    direct = DATA_DIR / table / "part-00000.csv"
    if direct.exists():
        return direct
    raise FileNotFoundError(f"找不到資料表：{DATA_DIR / table}")

def load_csv(table: str) -> pd.DataFrame:
    return pd.read_csv(_find_csv(table), low_memory=False)

# ── 特徵工程 ──────────────────────────────────────────────────────────────────
def build_features(train_label: pd.DataFrame,
                   predict_label: pd.DataFrame) -> pd.DataFrame:
    print("Loading tables...")
    user_info    = load_csv("user_info")
    twd          = load_csv("twd_transfer")
    crypto       = load_csv("crypto_transfer")   # 239,958 筆真實加密貨幣轉帳
    usdt_trading = load_csv("usdt_twd_trading")

    all_users = pd.concat([
        train_label[["user_id"]],
        predict_label[["user_id"]]
    ]).drop_duplicates()

    # ── 1. 用戶基本資訊 ──────────────────────────────────────────────────────
    print("  [1/6] user_info features...")
    ui = user_info[["user_id", "kyc_level", "user_source", "birthday", "confirmed_at"]].copy()
    ui["kyc_level"]   = pd.to_numeric(ui["kyc_level"], errors="coerce").fillna(0)
    ui["user_source"] = pd.to_numeric(ui["user_source"], errors="coerce").fillna(0)
    # 帳號年齡（天）
    ui["confirmed_at"] = pd.to_datetime(ui["confirmed_at"], errors="coerce", utc=True)
    ref_date = pd.Timestamp("2026-03-26", tz="UTC")
    ui["age"] = (ref_date - ui["confirmed_at"]).dt.days.fillna(-1)

    # ── 2. TWD 入/出金特徵 ───────────────────────────────────────────────────
    print("  [2/6] twd_transfer features...")
    twd["ori_samount"] = pd.to_numeric(twd.get("ori_samount", pd.Series(0, index=twd.index)),
                                        errors="coerce").fillna(0)
    twd["kind"] = pd.to_numeric(twd.get("kind", pd.Series(0, index=twd.index)),
                                 errors="coerce").fillna(0)
    twd["created_at"] = pd.to_datetime(twd["created_at"], errors="coerce", utc=True)
    twd["hour"] = twd["created_at"].dt.hour

    twd_g = twd.groupby("user_id").agg(
        twd_deposit_count  = ("ori_samount", lambda x: (twd.loc[x.index, "kind"] == 0).sum()),
        twd_withdraw_count = ("ori_samount", lambda x: (twd.loc[x.index, "kind"] == 1).sum()),
        total_twd_volume   = ("ori_samount", "sum"),
        avg_twd_amount     = ("ori_samount", "mean"),
        night_tx_count     = ("hour", lambda x: ((x >= 22) | (x <= 5)).sum()),
        total_tx_count     = ("ori_samount", "count"),
    ).reset_index()
    twd_g["night_tx_ratio"] = (
        twd_g["night_tx_count"] / twd_g["total_tx_count"].replace(0, 1)
    )

    # 高速交易風險：同一用戶在1分鐘內多筆交易
    twd_sorted = twd.sort_values(["user_id", "created_at"])
    twd_sorted["prev_time"] = twd_sorted.groupby("user_id")["created_at"].shift(1)
    twd_sorted["gap_sec"] = (twd_sorted["created_at"] - twd_sorted["prev_time"]).dt.total_seconds()
    high_speed = (
        twd_sorted[twd_sorted["gap_sec"] < 60]
        .groupby("user_id").size().reset_index(name="high_speed_risk")
    )
    twd_g = twd_g.merge(high_speed, on="user_id", how="left")
    twd_g["high_speed_risk"] = twd_g["high_speed_risk"].fillna(0)

    # IP 特徵
    if "source_ip" in twd.columns:
        ip_df = twd[twd["source_ip"].notna() & (twd["source_ip"] != "")]
        ip_user = ip_df.groupby("user_id")["source_ip"].nunique().reset_index(name="unique_ip_count")
        ip_shared = (
            ip_df.groupby("source_ip")["user_id"].nunique()
            .reset_index(name="users_per_ip")
        )
        ip_df2 = ip_df.merge(ip_shared, on="source_ip")
        max_ip_shared = (
            ip_df2.groupby("user_id")["users_per_ip"].max()
            .reset_index(name="max_ip_shared_users")
        )
        ip_anomaly = (
            ip_df2[ip_df2["users_per_ip"] > 1]
            .groupby("user_id").size().reset_index(name="ip_anomaly")
        )
        twd_g = (twd_g
                 .merge(ip_user, on="user_id", how="left")
                 .merge(max_ip_shared, on="user_id", how="left")
                 .merge(ip_anomaly, on="user_id", how="left"))
    else:
        twd_g["unique_ip_count"]    = 0
        twd_g["max_ip_shared_users"] = 0
        twd_g["ip_anomaly"]          = 0

    for c in ["unique_ip_count", "max_ip_shared_users", "ip_anomaly"]:
        twd_g[c] = twd_g[c].fillna(0)

    # ── 3. Crypto 交易特徵（使用 crypto_transfer：239,958 筆）────────────────
    print("  [3/6] crypto_transfer features...")
    cs = crypto.copy()
    cs["kind"] = pd.to_numeric(cs.get("kind", pd.Series(0, index=cs.index)),
                                errors="coerce").fillna(0)
    cs["ori_samount"] = pd.to_numeric(cs.get("ori_samount", pd.Series(0, index=cs.index)),
                                       errors="coerce").fillna(0)
    cs["currency"] = cs.get("currency", pd.Series("", index=cs.index)).fillna("").astype(str)
    cs["created_at"] = pd.to_datetime(cs["created_at"], errors="coerce", utc=True)

    # 入/出計數、幣種數
    cs_g = cs.groupby("user_id").agg(
        crypto_deposit_count  = ("kind", lambda x: (x == 0).sum()),
        crypto_withdraw_count = ("kind", lambda x: (x == 1).sum()),
        crypto_currency_count = ("currency", "nunique"),
    ).reset_index()

    # 資金留存時間（最早到最晚交易的分鐘差）
    cs_time = cs.groupby("user_id")["created_at"].agg(["min", "max"]).reset_index()
    cs_time.columns = ["user_id", "first_tx", "last_tx"]
    cs_time["min_retention_minutes"] = (
        (cs_time["last_tx"] - cs_time["first_tx"]).dt.total_seconds() / 60
    ).fillna(0)
    cs_time["retention_event_count"] = cs.groupby("user_id").size().values
    cs_g = cs_g.merge(cs_time[["user_id", "min_retention_minutes", "retention_event_count"]],
                      on="user_id", how="left")

    # ── 4. USDT/TWD 交易特徵 ────────────────────────────────────────────────
    print("  [4/6] usdt_twd_trading + usdt_swap features...")
    ut = usdt_trading.copy()
    # 入/出金不對稱：大量入金但少量出金 → 洗錢嫌疑
    if "trade_samount" in ut.columns and "kind" in ut.columns:
        ut["trade_samount"] = pd.to_numeric(ut["trade_samount"], errors="coerce").fillna(0)
        ut["kind"] = pd.to_numeric(ut["kind"], errors="coerce").fillna(0)
        ut_g = ut.groupby("user_id").agg(
            usdt_deposit_vol  = ("trade_samount", lambda x: x[ut.loc[x.index, "kind"] == 0].sum()),
            usdt_withdraw_vol = ("trade_samount", lambda x: x[ut.loc[x.index, "kind"] == 1].sum()),
            usdt_tx_count     = ("trade_samount", "count"),
        ).reset_index()
        ut_g["asymmetry_flag"] = (
            (ut_g["usdt_deposit_vol"] > 0) &
            (ut_g["usdt_withdraw_vol"] / ut_g["usdt_deposit_vol"].replace(0, 1) > 3)
        ).astype(int)
    else:
        ut_g = ut[["user_id"]].drop_duplicates() if "user_id" in ut.columns else pd.DataFrame(columns=["user_id"])
        ut_g["asymmetry_flag"] = 0
        ut_g["usdt_tx_count"] = 0

    # usdt_swap 特徵（法幣快速換加密貨幣 → 洗錢訊號）
    try:
        swap = load_csv("usdt_swap")
        swap["twd_samount"] = pd.to_numeric(swap.get("twd_samount", pd.Series(0, index=swap.index)),
                                             errors="coerce").fillna(0)
        swap["kind"] = pd.to_numeric(swap.get("kind", pd.Series(0, index=swap.index)),
                                      errors="coerce").fillna(0)
        swap["created_at"] = pd.to_datetime(swap["created_at"], errors="coerce", utc=True)
        swap_g = swap.groupby("user_id").agg(
            swap_count       = ("twd_samount", "count"),
            swap_twd_volume  = ("twd_samount", "sum"),
            swap_buy_count   = ("kind", lambda x: (x == 0).sum()),
            swap_sell_count  = ("kind", lambda x: (x == 1).sum()),
        ).reset_index()
        swap_g["swap_buy_ratio"] = (
            swap_g["swap_buy_count"] / swap_g["swap_count"].replace(0, 1)
        )
        print(f"    usdt_swap: {len(swap):,} 筆")
    except FileNotFoundError:
        swap_g = pd.DataFrame(columns=["user_id", "swap_count", "swap_twd_volume",
                                        "swap_buy_count", "swap_sell_count", "swap_buy_ratio"])

    # ── 5. 圖特徵：黑名單鄰居 ─────────────────────────────────────────────
    print("  [5/6] graph features (blacklist neighbors)...")
    blacklist_set = set(
        train_label.loc[train_label["status"] == 1, "user_id"]
    )

    neighbor_count = {}
    direct_neighbor = {}

    # 來源①：crypto_transfer.relation_user_id（直接交易對手，最可靠）
    if "relation_user_id" in cs.columns:
        ct_edges = cs[cs["relation_user_id"].notna()][["user_id", "relation_user_id"]].copy()
        ct_edges["relation_user_id"] = ct_edges["relation_user_id"].astype(float).astype("Int64")
        ct_edges = ct_edges.dropna(subset=["relation_user_id"])
        for _, row in ct_edges.iterrows():
            u, v = int(row["user_id"]), int(row["relation_user_id"])
            for src, dst in [(u, v), (v, u)]:   # 無向
                if dst in blacklist_set and src != dst:
                    neighbor_count[src] = neighbor_count.get(src, 0) + 1
                    direct_neighbor[src] = 1

    # 來源②：crypto_transfer.source_ip_hash（共用 IP 節點）
    if "source_ip_hash" in cs.columns:
        ip_col = "source_ip_hash"
        ip_df = cs[cs[ip_col].notna() & (cs[ip_col] != "")]
        ip_groups = ip_df.groupby(ip_col)["user_id"].apply(set)
        for ip, users in ip_groups.items():
            bl_in_group = users & blacklist_set
            for u in users:
                if bl_in_group - {u}:
                    neighbor_count[u] = neighbor_count.get(u, 0) + len(bl_in_group - {u})
                    direct_neighbor[u] = 1

    # 來源③：twd_transfer.source_ip_hash（TWD 交易共用 IP）
    ip_col_twd = "source_ip_hash" if "source_ip_hash" in twd.columns else (
                 "source_ip" if "source_ip" in twd.columns else None)
    if ip_col_twd:
        ip_df_t = twd[twd[ip_col_twd].notna() & (twd[ip_col_twd] != "")]
        for ip, users in ip_df_t.groupby(ip_col_twd)["user_id"].apply(set).items():
            bl_in_group = users & blacklist_set
            for u in users:
                if bl_in_group - {u}:
                    neighbor_count[u] = neighbor_count.get(u, 0) + len(bl_in_group - {u})
                    direct_neighbor[u] = 1

    graph_df = pd.DataFrame({
        "user_id": all_users["user_id"].values,
        "blacklist_neighbor_count": [neighbor_count.get(u, 0) for u in all_users["user_id"]],
        "is_direct_neighbor":       [direct_neighbor.get(u, 0) for u in all_users["user_id"]],
    })

    # ── 6. 合併所有特徵 ──────────────────────────────────────────────────────
    print("  [6/6] merging features + derived features...")
    feat = all_users.merge(ui[["user_id", "age", "kyc_level", "user_source"]], on="user_id", how="left")
    feat = feat.merge(twd_g[[
        "user_id", "twd_deposit_count", "twd_withdraw_count",
        "total_twd_volume", "avg_twd_amount",
        "night_tx_ratio", "high_speed_risk",
        "unique_ip_count", "max_ip_shared_users", "ip_anomaly"
    ]], on="user_id", how="left")
    feat = feat.merge(cs_g[[
        "user_id", "crypto_deposit_count", "crypto_withdraw_count",
        "crypto_currency_count", "min_retention_minutes", "retention_event_count"
    ]], on="user_id", how="left")
    feat = feat.merge(ut_g[["user_id", "asymmetry_flag", "usdt_tx_count"]], on="user_id", how="left")
    feat = feat.merge(swap_g[["user_id", "swap_count", "swap_twd_volume",
                               "swap_buy_count", "swap_sell_count", "swap_buy_ratio"]],
                      on="user_id", how="left")
    feat = feat.merge(graph_df[["user_id", "blacklist_neighbor_count", "is_direct_neighbor"]],
                      on="user_id", how="left")

    feat = feat.fillna(0)

    # ── 衍生特徵（比率 & 密度）──────────────────────────────────────────────
    feat["deposit_only_flag"] = (
        (feat["twd_deposit_count"] > 0) & (feat["twd_withdraw_count"] == 0)
    ).astype(int)
    feat["twd_deposit_ratio"] = feat["twd_deposit_count"] / (
        feat["twd_deposit_count"] + feat["twd_withdraw_count"] + 1
    )
    feat["crypto_net_flow"] = feat["crypto_deposit_count"] - feat["crypto_withdraw_count"]
    feat["tx_per_day"] = (feat["twd_deposit_count"] + feat["twd_withdraw_count"]) / (
        feat["age"].clip(lower=1)
    )
    feat["total_volume"] = feat["total_twd_volume"] + feat["swap_twd_volume"]
    feat["swap_to_twd_ratio"] = feat["swap_twd_volume"] / (feat["total_twd_volume"] + 1)

    print(f"  Feature matrix: {feat.shape[0]} rows × {feat.shape[1]} cols")
    return feat


# ── 訓練特徵列 ────────────────────────────────────────────────────────────────
FEATS = [
    # 用戶基本資訊
    "age", "kyc_level", "user_source",
    # TWD 交易行為
    "twd_deposit_count", "twd_withdraw_count", "total_twd_volume", "avg_twd_amount",
    "night_tx_ratio", "high_speed_risk",
    "unique_ip_count", "max_ip_shared_users", "ip_anomaly",
    # 加密貨幣行為
    "crypto_deposit_count", "crypto_withdraw_count", "crypto_currency_count",
    "min_retention_minutes", "retention_event_count",
    # USDT 交易
    "asymmetry_flag", "usdt_tx_count",
    # usdt_swap（法幣快速換加密貨幣）
    "swap_count", "swap_twd_volume", "swap_buy_count", "swap_sell_count", "swap_buy_ratio",
    # 圖特徵
    "blacklist_neighbor_count", "is_direct_neighbor",
    # 衍生特徵
    "deposit_only_flag", "twd_deposit_ratio", "crypto_net_flow",
    "tx_per_day", "total_volume", "swap_to_twd_ratio",
]

# ── LightGBM 基礎超參數 ───────────────────────────────────────────────────────
LGB_BASE_PARAMS = dict(
    objective="binary",
    metric="auc",
    num_leaves=63,
    learning_rate=0.01,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    min_child_samples=10,
    seed=42,
    verbosity=-1,
    n_jobs=-1,
)
NUM_ROUNDS = 1000

# scale_pos_weight 候選值（neg/pos ≈ 30；從保守到積極掃描）
SPW_GRID = [1, 3, 5, 10, 20, 30]

# 複合評分權重（AUC 排序能力 + F1 綜合 + 準確度）
W_AUC, W_F1, W_ACC = 0.40, 0.35, 0.25


def composite_score(auc: float, f1: float, acc: float) -> float:
    return W_AUC * auc + W_F1 * f1 + W_ACC * acc


def best_threshold_composite(y_true, y_prob, auc: float):
    """找讓複合分數最高的閾值。"""
    best_t, best_score = 0.5, -1.0
    for t in np.arange(0.05, 0.95, 0.01):
        pred = (y_prob >= t).astype(int)
        f1  = f1_score(y_true, pred, zero_division=0)
        acc = accuracy_score(y_true, pred)
        s   = composite_score(auc, f1, acc)
        if s > best_score:
            best_score, best_t = s, t
    return best_t, best_score


# ── 主流程 ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  BitoGuard AML — LightGBM 管線")
    print("=" * 60)

    # 載入標籤
    train_label   = load_csv("train_label")
    predict_label = load_csv("predict_label")
    train_label["status"] = pd.to_numeric(train_label["status"], errors="coerce").fillna(0).astype(int)

    print(f"\n訓練集: {len(train_label):,} 筆  "
          f"(黑名單: {train_label['status'].sum():,} / 正常: {(train_label['status']==0).sum():,})")
    print(f"預測集: {len(predict_label):,} 筆")

    # 特徵工程
    print("\n[Stage 1] 特徵工程...")
    feat = build_features(train_label, predict_label)

    # 訓練集 / 預測集分離
    train_ids = set(train_label["user_id"])
    pred_ids  = set(predict_label["user_id"])

    train_feat = feat[feat["user_id"].isin(train_ids)].merge(
        train_label[["user_id", "status"]], on="user_id", how="left"
    )
    pred_feat = feat[feat["user_id"].isin(pred_ids)]

    X = train_feat[FEATS].values.astype(np.float32)
    y = train_feat["status"].values.astype(int)
    X_pred = pred_feat[FEATS].values.astype(np.float32)
    pred_user_ids = pred_feat["user_id"].values

    print(f"\n訓練特徵矩陣: {X.shape}  |  預測集: {X_pred.shape}")
    print(f"正類比率: {y.mean():.4f}  ({y.sum()}/{len(y)})")

    # ── Stage 2a：scale_pos_weight 網格搜尋（3-Fold 快速估計）────────────────
    print("\n[Stage 2a] scale_pos_weight 網格搜尋...")
    skf3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    spw_results = {}
    for spw in SPW_GRID:
        params = {**LGB_BASE_PARAMS, "scale_pos_weight": spw}
        oof_tmp = np.zeros(len(y), dtype=np.float32)
        for tr_idx, val_idx in skf3.split(X, y):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]
            dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=FEATS)
            dval   = lgb.Dataset(X_val, label=y_val, feature_name=FEATS, reference=dtrain)
            m = lgb.train(params, dtrain, num_boost_round=NUM_ROUNDS,
                          valid_sets=[dval],
                          callbacks=[lgb.log_evaluation(period=-1),
                                     lgb.early_stopping(50, verbose=False)])
            oof_tmp[val_idx] = m.predict(X_val)
        auc_tmp = roc_auc_score(y, oof_tmp)
        best_t_tmp, comp_tmp = best_threshold_composite(y, oof_tmp, auc_tmp)
        pred_tmp  = (oof_tmp >= best_t_tmp).astype(int)
        f1_tmp    = f1_score(y, pred_tmp, zero_division=0)
        acc_tmp   = accuracy_score(y, pred_tmp)
        spw_results[spw] = {"composite": comp_tmp, "auc": auc_tmp,
                             "f1": f1_tmp, "acc": acc_tmp, "thr": best_t_tmp}
        print(f"  SPW={spw:>2}  composite={comp_tmp:.4f}  AUC={auc_tmp:.4f}"
              f"  F1={f1_tmp:.4f}  Acc={acc_tmp:.4f}  thr={best_t_tmp:.2f}")

    best_spw = max(spw_results, key=lambda k: spw_results[k]["composite"])
    print(f"\n  >>> 最佳 scale_pos_weight = {best_spw} "
          f"(composite={spw_results[best_spw]['composite']:.4f})")
    LGB_PARAMS = {**LGB_BASE_PARAMS, "scale_pos_weight": best_spw}

    # ── Stage 2b：5-Fold 完整交叉驗證（使用最佳 SPW）────────────────────────
    print("\n[Stage 2b] 5-Fold StratifiedKFold 交叉驗證...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros(len(y), dtype=np.float32)
    test_probs = np.zeros(len(X_pred), dtype=np.float32)
    fold_results = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=FEATS)
        dval   = lgb.Dataset(X_val, label=y_val, feature_name=FEATS, reference=dtrain)

        model = lgb.train(
            LGB_PARAMS,
            dtrain,
            num_boost_round=NUM_ROUNDS,
            valid_sets=[dval],
            callbacks=[
                lgb.log_evaluation(period=-1),       # 靜默
                lgb.early_stopping(stopping_rounds=50, verbose=False),  # 50 輪無改善則停止
            ],
        )

        val_prob  = model.predict(X_val)
        test_prob = model.predict(X_pred)

        oof_probs[val_idx] = val_prob
        test_probs += test_prob / 5

        # 找複合分數最佳閾值
        auc  = roc_auc_score(y_val, val_prob)
        best_t, _ = best_threshold_composite(y_val, val_prob, auc)

        val_pred = (val_prob >= best_t).astype(int)
        best_f1  = f1_score(y_val, val_pred, zero_division=0)
        prec = precision_score(y_val, val_pred, zero_division=0)
        rec  = recall_score(y_val, val_pred, zero_division=0)
        acc  = accuracy_score(y_val, val_pred)

        fold_results.append({
            "fold": fold,
            "threshold": round(float(best_t), 2),
            "precision": round(float(prec), 4),
            "recall":    round(float(rec), 4),
            "f1":        round(float(best_f1), 4),
            "auc":       round(float(auc), 4),
            "accuracy":  round(float(acc), 4),
            "val_pos":   int(y_val.sum()),
        })
        # 保存最後一折的特徵重要度
        if fold == 5:
            fi = pd.DataFrame({
                "feature": FEATS,
                "importance": model.feature_importance(importance_type="gain"),
            }).sort_values("importance", ascending=False)
            fi.to_csv(OUT_DIR / "feature_importance.csv", index=False)
        print(f"  Fold {fold}: AUC={auc:.4f}  F1={best_f1:.4f}  "
              f"P={prec:.4f}  R={rec:.4f}  Acc={acc:.4f}  thr={best_t:.2f}")

    # ── OOF 整體評估 ─────────────────────────────────────────────────────────
    print("\n[Stage 3] OOF 全局評估...")
    oof_auc  = roc_auc_score(y, oof_probs)
    best_t, best_comp = best_threshold_composite(y, oof_probs, oof_auc)

    oof_pred = (oof_probs >= best_t).astype(int)
    best_f1  = f1_score(y, oof_pred, zero_division=0)
    oof_prec = precision_score(y, oof_pred, zero_division=0)
    oof_rec  = recall_score(y, oof_pred, zero_division=0)
    oof_acc  = accuracy_score(y, oof_pred)
    tn, fp, fn, tp = confusion_matrix(y, oof_pred, labels=[0, 1]).ravel()

    print("\n" + "=" * 60)
    print("  ★ 判斷成功率 (OOF 驗證集表現) ★")
    print("=" * 60)
    print(f"  AUC       : {oof_auc:.4f}  → 排序能力")
    print(f"  F1-Score  : {best_f1:.4f}  → 綜合指標")
    print(f"  Precision : {oof_prec:.4f}  → 預測黑名單中 {oof_prec*100:.1f}% 真的是黑名單")
    print(f"  Recall    : {oof_rec:.4f}  → 實際黑名單抓到 {oof_rec*100:.1f}%")
    print(f"  Accuracy  : {oof_acc:.4f}")
    print(f"  Composite : {best_comp:.4f}  (0.4×AUC + 0.35×F1 + 0.25×Acc)")
    print(f"  Threshold : {best_t:.2f}  (複合分數最佳)")
    print(f"  SPW used  : {best_spw}")
    print(f"\n  混淆矩陣:")
    print(f"    TP={tp:,}  FP={fp:,}")
    print(f"    FN={fn:,}  TN={tn:,}")
    print("=" * 60)

    # ── 儲存 OOF 機率（供圖表生成使用）─────────────────────────────────────
    oof_df = pd.DataFrame({"true_label": y, "oof_prob": oof_probs})
    oof_df.to_csv(OUT_DIR / "oof_predictions.csv", index=False)

    # ── 生成提交檔 ────────────────────────────────────────────────────────────
    print("\n[Stage 4] 生成 submission.csv...")
    pred_labels = (test_probs >= best_t).astype(int)
    n_black = pred_labels.sum()
    n_normal = len(pred_labels) - n_black

    submission = pd.DataFrame({
        "user_id": pred_user_ids,
        "status":  pred_labels,
    })
    submission.to_csv(OUT_DIR / "submission.csv", index=False)

    submission_prob = submission.copy()
    submission_prob["probability"] = test_probs
    submission_prob.to_csv(OUT_DIR / "submission_with_prob.csv", index=False)

    print(f"  submission.csv 已儲存")
    print(f"  總筆數:{len(submission):,}  黑名單:{n_black:,}  正常:{n_normal:,}")

    # ── 儲存報告 ─────────────────────────────────────────────────────────────
    report = {
        "model": "LightGBM",
        "params": LGB_PARAMS,
        "num_rounds": NUM_ROUNDS,
        "features": FEATS,
        "oof_metrics": {
            "auc":       round(float(oof_auc), 4),
            "f1":        round(float(best_f1), 4),
            "precision": round(float(oof_prec), 4),
            "recall":    round(float(oof_rec), 4),
            "threshold": round(float(best_t), 2),
            "accuracy":  round(float(oof_acc), 4),
            "composite": round(float(best_comp), 4),
        },
        "best_spw": best_spw,
        "spw_search": {str(k): v for k, v in spw_results.items()},
        "submission": {
            "total": int(len(submission)),
            "blacklist": int(n_black),
            "normal": int(n_normal),
        },
        "folds": fold_results,
    }
    with open(OUT_DIR / "cv_report_lgb.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("  cv_report_lgb.json 已儲存")

    print("\n[DONE] 完成！最終 submission.csv 已準備好提交。")


if __name__ == "__main__":
    main()
