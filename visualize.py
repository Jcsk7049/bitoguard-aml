"""
視覺化模組

產出四張圖表（存為 PNG），供簡報與驗證報告使用：

  1. plots/pr_curve.png          — Precision-Recall 曲線 + 最佳門檻標記
  2. plots/threshold_sweep.png   — 門檻 vs F1/Precision/Recall/FPR 折線圖
  3. plots/shap_beeswarm.png     — SHAP Beeswarm 圖（特徵對個別預測的貢獻分布）
  4. plots/feature_importance.png — XGBoost Gain 特徵重要度橫條圖

使用方式：
    # 完整模式（需要已訓練模型 + 驗證資料）
    python visualize.py

    # 僅重繪門檻掃描圖（從 validation_report.json 讀取）
    python visualize.py --from-report
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import xgboost as xgb
import shap
from pathlib import Path
from typing import Optional
from sklearn.metrics import precision_recall_curve, auc
from sklearn.model_selection import train_test_split

matplotlib.rcParams["font.family"]    = ["Microsoft JhengHei", "Arial Unicode MS", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

PLOT_DIR   = Path("plots")
DPI        = 150
FIGSIZE_SM = (7, 5)
FIGSIZE_LG = (10, 6)

# 品牌配色
C_BLUE   = "#2563EB"
C_ORANGE = "#F97316"
C_GREEN  = "#16A34A"
C_RED    = "#DC2626"
C_GRAY   = "#6B7280"
C_BG     = "#F9FAFB"


def _ensure_plot_dir():
    PLOT_DIR.mkdir(exist_ok=True)


# ── 1. Precision-Recall 曲線 ─────────────────────────────────────────────────

def plot_pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    optimal_threshold: float,
    save_path: Path = PLOT_DIR / "pr_curve.png",
) -> None:
    """
    繪製 PR 曲線，標記：
      - 最佳 F1 門檻點（橙色星形）
      - 保守模式門檻點（FPR<5%，綠色菱形）
      - 隨機猜測基準線（灰色虛線）
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)

    # F1 最佳點
    f1_scores     = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-9)
    best_idx      = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_p, best_r = precision[best_idx], recall[best_idx]

    # 保守點（precision >= 0.8）
    conservative_mask = precision[:-1] >= 0.8
    if conservative_mask.any():
        c_idx = np.where(conservative_mask)[0][np.argmax(recall[:-1][conservative_mask])]
        c_p, c_r = precision[c_idx], recall[c_idx]
        c_thresh  = thresholds[c_idx]
        show_conservative = True
    else:
        show_conservative = False

    fig, ax = plt.subplots(figsize=FIGSIZE_SM, facecolor=C_BG)
    ax.set_facecolor(C_BG)

    # 曲線本體
    ax.plot(recall, precision, color=C_BLUE, lw=2.2, label=f"PR 曲線 (AUC = {pr_auc:.4f})")

    # 填色
    ax.fill_between(recall, precision, alpha=0.08, color=C_BLUE)

    # 隨機基準
    pos_rate = y_true.mean()
    ax.axhline(pos_rate, color=C_GRAY, lw=1, ls="--", label=f"隨機基準 (P = {pos_rate:.3f})")

    # 最佳 F1 門檻
    ax.scatter(best_r, best_p, s=180, zorder=5, color=C_ORANGE, marker="*",
               label=f"最佳 F1 門檻 {best_threshold:.2f}  (P={best_p:.3f}, R={best_r:.3f})")
    ax.annotate(f"  thr={best_threshold:.2f}", (best_r, best_p),
                fontsize=8, color=C_ORANGE)

    # 保守門檻
    if show_conservative:
        ax.scatter(c_r, c_p, s=140, zorder=5, color=C_GREEN, marker="D",
                   label=f"保守門檻 {c_thresh:.2f}  (P={c_p:.3f}, R={c_r:.3f})")
        ax.annotate(f"  thr={c_thresh:.2f}", (c_r, c_p), fontsize=8, color=C_GREEN)

    ax.set_xlabel("Recall（召回率）", fontsize=11)
    ax.set_ylabel("Precision（精確率）", fontsize=11)
    ax.set_title("Precision-Recall 曲線", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [儲存] {save_path}")


# ── 2. 門檻掃描折線圖 ────────────────────────────────────────────────────────

def plot_threshold_sweep(
    sweep_table: list[dict],
    optimal_threshold: float,
    conservative_threshold: float,
    save_path: Path = PLOT_DIR / "threshold_sweep.png",
) -> None:
    """
    繪製門檻 vs F1 / Precision / Recall / FPR 的折線圖，
    用垂直線標記最佳門檻與保守門檻。
    """
    df = pd.DataFrame(sweep_table)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True, facecolor=C_BG)
    ax1.set_facecolor(C_BG)
    ax2.set_facecolor(C_BG)

    # 上圖：F1 / Precision / Recall
    ax1.plot(df["threshold"], df["f1"],        color=C_BLUE,   lw=2,   label="F1-score")
    ax1.plot(df["threshold"], df["precision"],  color=C_GREEN,  lw=1.8, label="Precision", ls="--")
    ax1.plot(df["threshold"], df["recall"],     color=C_ORANGE, lw=1.8, label="Recall",    ls="-.")

    # 下圖：FPR（誤判率）
    ax2.plot(df["threshold"], df["fpr"], color=C_RED, lw=2, label="FPR（誤判率）")
    ax2.axhline(0.05, color=C_GRAY, lw=1, ls=":", label="FPR = 5% 警戒線")
    ax2.fill_between(df["threshold"], df["fpr"], 0.05,
                     where=df["fpr"] > 0.05, alpha=0.15, color=C_RED, label="超標區間")

    # 最佳門檻垂直線
    for ax in (ax1, ax2):
        ax.axvline(optimal_threshold,     color=C_ORANGE, lw=1.5, ls="--",
                   label=f"最佳 F1 門檻 ({optimal_threshold:.2f})")
        ax.axvline(conservative_threshold, color=C_GREEN,  lw=1.5, ls=":",
                   label=f"保守門檻 ({conservative_threshold:.2f})")
        ax.grid(alpha=0.3)

    ax1.set_ylabel("指標值", fontsize=11)
    ax1.set_title("決策門檻掃描（0.10 → 0.90）", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=8, loc="center left")
    ax1.set_ylim([-0.02, 1.05])

    ax2.set_xlabel("決策門檻 (threshold)", fontsize=11)
    ax2.set_ylabel("FPR（誤判率）", fontsize=11)
    ax2.legend(fontsize=8, loc="upper left")
    ax2.set_ylim([-0.01, min(df["fpr"].max() * 1.3, 1.0)])

    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [儲存] {save_path}")


# ── 3. SHAP Beeswarm 圖 ──────────────────────────────────────────────────────

def plot_shap_beeswarm(
    model: xgb.Booster,
    X: np.ndarray,
    feature_names: list[str],
    feature_labels: dict[str, str],
    max_samples: int = 500,
    save_path: Path = PLOT_DIR / "shap_beeswarm.png",
) -> None:
    """
    使用 shap.TreeExplainer 計算全樣本 SHAP 值，
    繪製 Beeswarm 圖呈現每個特徵對所有樣本預測的貢獻分布。

    Beeswarm 解讀：
      - 每個點 = 一個用戶的一個特徵 SHAP 值
      - 橫軸 = SHAP 值（正=推向黑名單，負=推向正常）
      - 顏色 = 該特徵的原始數值（紅=高，藍=低）
    """
    # 抽樣（節省計算時間）
    if len(X) > max_samples:
        idx = np.random.RandomState(42).choice(len(X), max_samples, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    explainer  = shap.TreeExplainer(model)
    dmat       = xgb.DMatrix(X_sample, feature_names=feature_names)
    shap_vals  = explainer.shap_values(dmat)   # shape: (n_samples, n_features)

    # 替換特徵名為中文標籤
    display_names = [feature_labels.get(n, n) for n in feature_names]

    # 建立 shap.Explanation 物件（shap >= 0.40 推薦用法）
    explanation = shap.Explanation(
        values=shap_vals,
        data=X_sample,
        feature_names=display_names,
    )

    fig, ax = plt.subplots(figsize=(9, max(5, len(feature_names) * 0.42)), facecolor=C_BG)
    ax.set_facecolor(C_BG)

    shap.plots.beeswarm(explanation, max_display=15, show=False, color_bar=True)

    plt.title("SHAP Beeswarm — 特徵對人頭戶判定的貢獻分布", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  [儲存] {save_path}")


# ── 4. 特徵重要度橫條圖 ──────────────────────────────────────────────────────

def plot_feature_importance(
    model: xgb.Booster,
    feature_labels: dict[str, str],
    top_n: int = 15,
    save_path: Path = PLOT_DIR / "feature_importance.png",
) -> None:
    """
    XGBoost Gain 特徵重要度橫條圖（Gain = 該特徵在所有樹中提供的平均資訊增益）。
    """
    scores = model.get_score(importance_type="gain")
    total  = sum(scores.values()) or 1.0
    ranked = sorted(scores.items(), key=lambda x: x[1])[-top_n:]

    labels = [feature_labels.get(k, k) for k, _ in ranked]
    values = [v / total * 100 for _, v in ranked]

    # 根據貢獻度深淺著色
    norm    = plt.Normalize(min(values), max(values))
    colors  = plt.cm.Blues(norm(values))

    fig, ax = plt.subplots(figsize=FIGSIZE_LG, facecolor=C_BG)
    ax.set_facecolor(C_BG)

    bars = ax.barh(labels, values, color=colors, edgecolor="white", linewidth=0.5)

    # 數值標籤
    for bar, val in zip(bars, values):
        ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=8, color="#374151")

    ax.set_xlabel("Gain 貢獻佔比 (%)", fontsize=11)
    ax.set_title(f"XGBoost 特徵重要度 Top {top_n}（Gain）", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim([0, max(values) * 1.15])

    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [儲存] {save_path}")


# ── 5. 黑名單關聯圖譜（Athena BFS 結果視覺化） ──────────────────────────────

def plot_blacklist_network(
    hop_df: pd.DataFrame,
    target_user_id: Optional[int] = None,
    max_per_ring: int = 30,
    save_path: Path = PLOT_DIR / "blacklist_network.png",
) -> None:
    """
    將 athena_graph_hops.sql 的 BFS 查詢結果繪製為同心環關聯圖。

    佈局規則
    --------
      中心環   hop == 0（黑名單自身）— 深紅色節點
      第一環   hop == 1（直接鄰居）  — 橘色節點
      第二環   hop == 2（二度關聯）  — 金黃色節點
      目標用戶 target_user_id       — 藍星標記（可選）

    節點大小：正比於 blacklist_neighbor_count（最少 80，最多 400）。
    節點標籤：user_id（若 > max_per_ring 則省略標籤避免重疊）。

    側面板：hop 分布長條圖 + weighted_risk_label 分布長條圖。

    Parameters
    ----------
    hop_df         : athena_graph_hops.sql 的查詢結果 DataFrame，必要欄位：
                       user_id, min_hops_to_blacklist, blacklist_neighbor_count,
                       weighted_risk_label
    target_user_id : 本次診斷的目標用戶（以藍星標記）
    max_per_ring   : 每環最多繪製的節點數（超過則取風險最高的前 N 個）
    save_path      : 圖檔輸出路徑
    """
    required_cols = {"user_id", "min_hops_to_blacklist", "blacklist_neighbor_count",
                     "weighted_risk_label"}
    missing = required_cols - set(hop_df.columns)
    if missing:
        raise ValueError(f"[plot_blacklist_network] hop_df 缺少欄位：{missing}")

    # ── 資料準備 ────────────────────────────────────────────────────────────
    df = hop_df.copy()
    df["min_hops_to_blacklist"] = pd.to_numeric(
        df["min_hops_to_blacklist"], errors="coerce"
    ).fillna(4).clip(0, 4).astype(int)
    df["blacklist_neighbor_count"] = pd.to_numeric(
        df["blacklist_neighbor_count"], errors="coerce"
    ).fillna(0).clip(0)

    # 各層設定：(hop值, 顏色, 環半徑, 圖層標籤)
    ring_cfg = [
        (0, "#7F1D1D", 0.0,  "黑名單（Hop 0）"),
        (1, "#EA580C", 0.38, "一度鄰居（Hop 1）"),
        (2, "#CA8A04", 0.72, "二度關聯（Hop 2）"),
    ]

    _ensure_plot_dir()

    fig = plt.figure(figsize=(14, 7), facecolor=C_BG)
    # 左：圖譜（70%），右：統計面板（30%）
    gs  = fig.add_gridspec(2, 2, width_ratios=[7, 3], hspace=0.45, wspace=0.35)
    ax_net  = fig.add_subplot(gs[:, 0])   # 跨兩列，圖譜主圖
    ax_hop  = fig.add_subplot(gs[0, 1])   # 上：hop 分布
    ax_risk = fig.add_subplot(gs[1, 1])   # 下：風險等級分布

    ax_net.set_facecolor(C_BG)
    ax_net.set_aspect("equal")
    ax_net.axis("off")

    legend_patches = []

    for hop_val, color, radius, ring_label in ring_cfg:
        ring_df = df[df["min_hops_to_blacklist"] == hop_val].copy()
        if ring_df.empty:
            continue

        # 超過 max_per_ring 時，優先保留 blacklist_neighbor_count 最高的節點
        if len(ring_df) > max_per_ring:
            ring_df = ring_df.nlargest(max_per_ring, "blacklist_neighbor_count")

        n = len(ring_df)

        if radius == 0.0:
            # 中心環：單點放置
            xs = np.zeros(n)
            ys = np.zeros(n)
            if n > 1:
                angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
                xs = np.cos(angles) * 0.06
                ys = np.sin(angles) * 0.06
        else:
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            xs = np.cos(angles) * radius
            ys = np.sin(angles) * radius

        # 節點大小：正比於黑名單鄰居數，範圍 [80, 400]
        raw_sz = ring_df["blacklist_neighbor_count"].values.astype(float)
        if raw_sz.max() > raw_sz.min():
            norm_sz = (raw_sz - raw_sz.min()) / (raw_sz.max() - raw_sz.min())
        else:
            norm_sz = np.ones(n) * 0.5
        sizes = 80 + norm_sz * 320

        ax_net.scatter(xs, ys, s=sizes, c=color, alpha=0.80,
                       edgecolors="white", linewidths=0.8, zorder=3)

        # 節點標籤（節點少於 max_per_ring 時才顯示）
        if n <= max_per_ring:
            for x, y, uid in zip(xs, ys, ring_df["user_id"].values):
                ax_net.text(x, y, str(uid), ha="center", va="center",
                            fontsize=5.5, color="white", fontweight="bold", zorder=4)

        legend_patches.append(mpatches.Patch(color=color, label=f"{ring_label}（n={n}）"))

    # ── 目標用戶標記 ────────────────────────────────────────────────────────
    if target_user_id is not None:
        target_row = df[df["user_id"] == target_user_id]
        if not target_row.empty:
            hop = int(target_row["min_hops_to_blacklist"].iloc[0])
            # 找出其在對應環的位置
            ring_df = df[df["min_hops_to_blacklist"] == hop]
            if len(ring_df) > max_per_ring:
                ring_df = ring_df.nlargest(max_per_ring, "blacklist_neighbor_count")
            if target_user_id in ring_df["user_id"].values:
                idx    = list(ring_df["user_id"].values).index(target_user_id)
                n      = len(ring_df)
                radius = [0.0, 0.38, 0.72][min(hop, 2)]
                if radius == 0.0 and n > 1:
                    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
                    tx, ty = np.cos(angles[idx]) * 0.06, np.sin(angles[idx]) * 0.06
                elif radius > 0:
                    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
                    tx, ty = np.cos(angles[idx]) * radius, np.sin(angles[idx]) * radius
                else:
                    tx, ty = 0.0, 0.0
                ax_net.scatter([tx], [ty], s=350, marker="*", c=C_BLUE,
                               edgecolors="white", linewidths=1.0, zorder=5)
                legend_patches.append(
                    mpatches.Patch(color=C_BLUE, label=f"目標用戶 {target_user_id}")
                )

    # 環形輔助線
    for _, _, r, _ in ring_cfg:
        if r > 0:
            circle = plt.Circle((0, 0), r, fill=False, linestyle="--",
                                 linewidth=0.6, color=C_GRAY, alpha=0.4)
            ax_net.add_patch(circle)

    ax_net.set_xlim(-1.05, 1.05)
    ax_net.set_ylim(-1.05, 1.05)
    ax_net.set_title("黑名單關聯圖譜（2-Hop BFS）", fontsize=13, fontweight="bold", pad=10)
    ax_net.legend(handles=legend_patches, loc="lower left", fontsize=8,
                  framealpha=0.85, edgecolor=C_GRAY)

    # ── 側面板 1：hop 分布長條圖 ────────────────────────────────────────────
    hop_counts = df["min_hops_to_blacklist"].value_counts().sort_index()
    hop_colors = {0: "#7F1D1D", 1: "#EA580C", 2: "#CA8A04", 3: C_GRAY, 4: C_GRAY}
    bar_colors = [hop_colors.get(h, C_GRAY) for h in hop_counts.index]

    ax_hop.bar(hop_counts.index.astype(str), hop_counts.values,
               color=bar_colors, edgecolor="white", linewidth=0.5)
    ax_hop.set_facecolor(C_BG)
    ax_hop.set_xlabel("距黑名單跳轉數", fontsize=9)
    ax_hop.set_ylabel("用戶數", fontsize=9)
    ax_hop.set_title("Hop 距離分布", fontsize=10, fontweight="bold")
    ax_hop.spines[["top", "right"]].set_visible(False)
    ax_hop.grid(axis="y", alpha=0.3)
    for bar, val in zip(ax_hop.patches, hop_counts.values):
        ax_hop.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    str(val), ha="center", va="bottom", fontsize=8)

    # ── 側面板 2：weighted_risk_label 分布 ──────────────────────────────────
    risk_order  = ["BLACKLIST", "HIGH_WEIGHTED", "HIGH", "MEDIUM", "LOW"]
    risk_colors = {
        "BLACKLIST":     "#7F1D1D",
        "HIGH_WEIGHTED": "#DC2626",
        "HIGH":          "#F97316",
        "MEDIUM":        "#FBBF24",
        "LOW":           "#16A34A",
    }
    risk_counts = df["weighted_risk_label"].value_counts()
    # 保持固定順序
    risk_labels = [r for r in risk_order if r in risk_counts.index]
    risk_vals   = [risk_counts[r] for r in risk_labels]
    bar_rc      = [risk_colors.get(r, C_GRAY) for r in risk_labels]

    ax_risk.barh(risk_labels, risk_vals, color=bar_rc, edgecolor="white", linewidth=0.5)
    ax_risk.set_facecolor(C_BG)
    ax_risk.set_xlabel("用戶數", fontsize=9)
    ax_risk.set_title("風險等級分布", fontsize=10, fontweight="bold")
    ax_risk.spines[["top", "right"]].set_visible(False)
    ax_risk.grid(axis="x", alpha=0.3)
    for bar, val in zip(ax_risk.patches, risk_vals):
        ax_risk.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                     str(val), va="center", fontsize=8)

    fig.suptitle(
        "BitoGuard 黑名單網絡分析｜Athena BFS 查詢結果",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [儲存] {save_path}")


# ── 整合入口 ─────────────────────────────────────────────────────────────────

def generate_all_plots(
    model: xgb.Booster,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
    feature_labels: dict[str, str],
    report: dict | None = None,
    hop_df: Optional[pd.DataFrame] = None,
    target_user_id: Optional[int] = None,
) -> None:
    """
    一次生成全部圖表（4 張基礎圖 + 選配第 5 張黑名單圖譜）。

    Parameters
    ----------
    model          : 已訓練的 XGBoost Booster
    X_val          : validation set 特徵矩陣
    y_val          : validation set 標籤
    feature_names  : 特徵欄位名稱（英文，與模型一致）
    feature_labels : 特徵英文名 → 中文對照 dict
    report         : validation_report.json 的 dict，若有則直接用其門檻資訊
    hop_df         : athena_graph_hops.sql 查詢結果（若提供則額外繪製圖譜）
    target_user_id : 圖譜中標記目標用戶（藍星）
    """
    _ensure_plot_dir()

    dmat   = xgb.DMatrix(X_val, feature_names=feature_names)
    y_prob = model.predict(dmat)

    # 從報告取門檻，否則用預設
    if report and "threshold_analysis" in report:
        ta                     = report["threshold_analysis"]
        optimal_threshold      = ta["optimal_threshold"]
        conservative_threshold = ta["conservative_threshold"]
        sweep_table            = ta["sweep_table"]
    else:
        from validation_report import sweep_thresholds
        ta                     = sweep_thresholds(y_val, y_prob)
        optimal_threshold      = ta.optimal_threshold
        conservative_threshold = ta.conservative_threshold
        sweep_table            = ta.sweep_table

    total = 5 if hop_df is not None else 4

    print(f"\n[1/{total}] PR 曲線...")
    plot_pr_curve(y_val, y_prob, optimal_threshold)

    print(f"[2/{total}] 門檻掃描圖...")
    plot_threshold_sweep(sweep_table, optimal_threshold, conservative_threshold)

    print(f"[3/{total}] SHAP Beeswarm 圖...")
    plot_shap_beeswarm(model, X_val, feature_names, feature_labels)

    print(f"[4/{total}] 特徵重要度圖...")
    plot_feature_importance(model, feature_labels)

    if hop_df is not None:
        print(f"[5/{total}] 黑名單關聯圖譜...")
        plot_blacklist_network(hop_df, target_user_id=target_user_id)

    print(f"\n全部圖表已儲存至 {PLOT_DIR}/")


# ── CLI 入口 ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="生成 XGBoost + SHAP 視覺化圖表")
    parser.add_argument("--from-report",  action="store_true",
                        help="僅用 validation_report.json 重繪門檻掃描圖（不需要模型）")
    parser.add_argument("--model-path",   default="model.json")
    parser.add_argument("--report-path",  default="validation_report.json")
    parser.add_argument("--hop-df-path",  default=None,
                        help="athena_graph_hops.sql 查詢結果 CSV 路徑（選填，用於繪製黑名單圖譜）")
    parser.add_argument("--target-user",  type=int, default=None,
                        help="在黑名單圖譜中標記目標用戶的 user_id（選填）")
    args = parser.parse_args()

    from xai_bedrock import FEATURE_LABELS

    _ensure_plot_dir()

    # ── 僅重繪模式 ────────────────────────────────────────────────────────
    if args.from_report:
        if not os.path.exists(args.report_path):
            print(f"找不到 {args.report_path}，請先執行 validation_report.py。")
            return
        with open(args.report_path, encoding="utf-8") as f:
            report = json.load(f)
        ta = report["threshold_analysis"]
        print("[重繪] 門檻掃描圖...")
        plot_threshold_sweep(ta["sweep_table"], ta["optimal_threshold"],
                             ta["conservative_threshold"])
        return

    # ── 完整模式 ─────────────────────────────────────────────────────────
    if not os.path.exists(args.model_path):
        print(f"找不到模型檔 {args.model_path}。請先執行：")
        print("  python download_model.py")
        return

    from bito_data_manager import BitoDataManager
    from train_sagemaker   import build_features, build_hyperparams

    print("[載入] 資料與特徵...")
    manager      = BitoDataManager()
    tables       = manager.load_all()
    features     = build_features(manager)
    train_label  = manager._load_raw("train_label")
    train_label["user_id"] = pd.to_numeric(train_label["user_id"], errors="coerce")

    FEAT_COLS = [c for c in features.columns if c not in (
        "user_id", "asymmetry_reason", "ip_source"
    )]
    merged = (
        train_label.merge(features[["user_id"] + FEAT_COLS], on="user_id", how="left")
                   .fillna(0)
    )
    X = merged[FEAT_COLS].values.astype(float)
    y = merged["status"].values.astype(int)

    _, X_val, _, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = xgb.Booster()
    model.load_model(args.model_path)

    report = None
    if os.path.exists(args.report_path):
        with open(args.report_path, encoding="utf-8") as f:
            report = json.load(f)

    # ── 黑名單圖譜（選填） ────────────────────────────────────────────────
    hop_df = None
    if args.hop_df_path:
        if os.path.exists(args.hop_df_path):
            hop_df = pd.read_csv(args.hop_df_path)
            print(f"[載入] Athena hop 資料：{len(hop_df):,} 筆 → {args.hop_df_path}")
        else:
            print(f"[警告] hop-df-path 找不到檔案 {args.hop_df_path}，跳過圖譜繪製。")

    generate_all_plots(
        model, X_val, y_val, FEAT_COLS, FEATURE_LABELS, report,
        hop_df=hop_df, target_user_id=args.target_user,
    )


if __name__ == "__main__":
    main()
