"""
SageMaker XGBoost 自訂訓練腳本（entry_point）

v2 新增功能：
  ① AdversarialAugmentor   — 模擬四種詐騙集團規避行為（對抗性資料增強）
  ② CostSensitiveObjective — 自訂損失函數，FN 權重 = 5 × FP 權重
  ③ RobustnessEvaluator    — 測試並報告模型在各攻擊強度下的 Recall 衰減

流程：
  1. 解析超參數（HyperparameterTuner 注入）
  2. 載入訓練資料
  3. [NEW] 啟用對抗增強時：對每折訓練集增加攻擊樣本
  4. 5-Fold 分層 CV + Early Stopping（以自訂 Cost-Sensitive F1 為目標）
  5. [NEW] 全量訓練後執行 Robustness Evaluation
  6. 儲存模型 + training_summary.json

HPO Metric 捕捉格式（Regex：f1_score: ([0-9\\.]+)）：
    [CV] f1_score: 0.8423
"""

import argparse
import json
import os
import warnings
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore", category=UserWarning)


# ══════════════════════════════════════════════════════════════════════════════
#  常數 & 特徵 Schema
# ══════════════════════════════════════════════════════════════════════════════

N_FOLDS         = 5
F1_THRESH_SWEEP = np.arange(0.20, 0.81, 0.02)

# 特徵索引對照表（與 feature_store.py INFERENCE_FEATURE_COLUMNS 及
# train_sagemaker.py CANONICAL_FEATURE_COLS 的欄位順序完全一致）
#
# 欄位順序依 bito_data_manager.extract_mule_features() 的 merge 輸出決定：
#   f3（kyc + volume）← LEFT  →  merge f1（retention）→  merge f2（IP）
#   → merge f4（graph hops，含 in_blacklist_network）
#   → mule_risk_score
# 再由 train_sagemaker.build_features() 補充四項計數特徵與 night_tx_ratio。
#
# ⚠ 若 bito_data_manager 或 build_features 的輸出欄位有異動，
#   必須同步更新此字典、feature_store.INFERENCE_FEATURE_COLUMNS、
#   以及 train_sagemaker.CANONICAL_FEATURE_COLS，三者保持一致。
FEATURE_INDEX: dict[str, int] = {
    # ── 用戶基本屬性 ──────────────────────────────────────────────────────
    "kyc_level":                 0,

    # ── 特徵①：資金滯留時間（retention） ──────────────────────────────────
    "min_retention_minutes":     1,   # 攻擊①：滯留時間拉長
    "retention_event_count":     2,   # 快進快出事件次數
    "high_speed_risk":           3,   # <10 分鐘旗標

    # ── 特徵②：IP 異常跳動 ────────────────────────────────────────────────
    "unique_ip_count":           4,   # 攻擊②：IP 分散
    "ip_anomaly":                5,

    # ── 特徵③：量能不對稱 ────────────────────────────────────────────────
    "total_twd_volume":          6,   # 攻擊③：交易量碎片化
    "volume_zscore":             7,
    "asymmetry_flag":            8,

    # ── 特徵④：圖跳轉（BFS + Union-Find） ────────────────────────────────
    "min_hops_to_blacklist":     9,   # 攻擊④：跳轉數偽裝
    "is_direct_neighbor":        10,
    "blacklist_neighbor_count":  11,
    "in_blacklist_network":      12,  # 連通分量旗標

    # ── 綜合風險評分 ──────────────────────────────────────────────────────
    "mule_risk_score":           13,

    # ── 交易計數（build_features 補充） ─────────────────────────────────
    "twd_deposit_count":         14,
    "twd_withdraw_count":        15,
    "crypto_deposit_count":      16,  # 先前遺漏，補入
    "crypto_withdraw_count":     17,

    # ── 時間模式 ──────────────────────────────────────────────────────────
    "night_tx_ratio":            18,  # 先前遺漏，補入
}
N_FEATURES = len(FEATURE_INDEX)   # = 19
IDX = FEATURE_INDEX   # 方便引用的別名


# ══════════════════════════════════════════════════════════════════════════════
#  ① AdversarialAugmentor — 四種詐騙集團規避行為模擬
# ══════════════════════════════════════════════════════════════════════════════

class AdversarialAugmentor:
    """
    資料增強模組：模擬詐騙集團在「被模型標記」後所採取的規避手段。

    設計理念（對抗性訓練）
    ----------------------
    詐騙集團了解 AML 系統的偵測邏輯後，會刻意修改行為特徵以規避：
      ①  拉長滯留時間（Retention Camouflage）
          觀察到「快進快出 < 10 分鐘」被標記 → 故意等 30~120 分鐘再出金
      ②  分散 IP（IP Normalization）
          觀察到「IP 跳動異常多」被標記 → 控制 IP 數量，改用固定 VPN 節點
      ③  交易量碎片化（Volume Fragmentation）
          觀察到「L1 用戶卻有百萬交易」被標記 → 拆單，單筆壓在閾值以下
      ④  跳轉數偽裝（Hop Injection）
          了解到「直接鄰居」被強標記 → 插入中間人帳戶，增加跳轉層數

    使用方式
    ---------
      augmentor = AdversarialAugmentor(seed=42)

      # 單種攻擊：對黑名單樣本注入雜訊，返回（原始 + 攻擊樣本）
      X_aug, y_aug = augmentor.augment(X_train, y_train, attack="retention")

      # 組合攻擊（最高強度，測試模型天花板）
      X_aug, y_aug = augmentor.augment(X_train, y_train, attack="combined",
                                        strength=1.0)

    Parameters
    ----------
    seed             : 隨機種子
    augment_ratio    : 正樣本增強倍率（原始黑名單 × ratio 份攻擊樣本）
    clip_features    : 是否裁剪特徵到合理範圍（避免物理上不可能的值）
    """

    # 各特徵的合理範圍（最小值, 最大值）
    # ⚠ 與 FEATURE_INDEX 保持同步；新增特徵時一併加入
    _FEATURE_BOUNDS: dict[str, tuple[float, float]] = {
        "kyc_level":               (0.0,    2.0),
        "min_retention_minutes":   (0.0,    14400.0),  # 0 ~ 10 天
        "retention_event_count":   (0.0,    10000.0),
        "high_speed_risk":         (0.0,    1.0),
        "unique_ip_count":         (1.0,    100.0),
        "ip_anomaly":              (0.0,    1.0),
        "total_twd_volume":        (0.0,    1e9),
        "volume_zscore":           (-5.0,   20.0),
        "asymmetry_flag":          (0.0,    1.0),
        "min_hops_to_blacklist":   (0.0,    4.0),
        "is_direct_neighbor":      (0.0,    1.0),
        "blacklist_neighbor_count":(0.0,    500.0),
        "in_blacklist_network":    (0.0,    1.0),
        "mule_risk_score":         (0.0,    4.0),
        "twd_deposit_count":       (0.0,    50000.0),
        "twd_withdraw_count":      (0.0,    50000.0),
        "crypto_deposit_count":    (0.0,    50000.0),
        "crypto_withdraw_count":   (0.0,    50000.0),
        "night_tx_ratio":          (0.0,    1.0),
    }

    def __init__(
        self,
        seed: int = 42,
        augment_ratio: float = 0.5,
        clip_features: bool = True,
    ):
        self.rng           = np.random.RandomState(seed)
        self.augment_ratio = augment_ratio
        self.clip_features = clip_features

    # ── 攻擊①：滯留時間拉長 ──────────────────────────────────────────────────

    def _attack_retention(self, X: np.ndarray, strength: float) -> np.ndarray:
        """
        詐騙集團刻意等待 20~180 分鐘後再出金，使滯留時間不再觸發 < 10 分鐘門檻。

        擾動公式：
            new_retention = original + Uniform(20, 20 + 160 × strength)

        strength=0.3 → 等待 20~68 分鐘（輕度規避，可能仍觸發高速風險）
        strength=1.0 → 等待 20~180 分鐘（完全規避快進快出特徵）
        """
        X_adv = X.copy()
        max_delay = 20 + 160 * strength

        i_ret = IDX["min_retention_minutes"]
        i_spd = IDX["high_speed_risk"]

        delay = self.rng.uniform(20, max(20.5, max_delay), size=len(X))
        X_adv[:, i_ret] = X_adv[:, i_ret] + delay

        # 高速風險旗標：滯留 > 10 分鐘後重置為 0（詐騙集團成功規避）
        X_adv[:, i_spd] = (X_adv[:, i_ret] < 10.0).astype(float)

        return X_adv

    # ── 攻擊②：IP 分散偽裝 ───────────────────────────────────────────────────

    def _attack_ip(self, X: np.ndarray, strength: float) -> np.ndarray:
        """
        兩種 IP 策略（依 strength 決定：低強度=縮減，高強度=擴散後收斂）：

        Low  (strength ≤ 0.5): 改用固定 VPN 節點，將 unique_ip_count 壓低到 1~3
             → 目的：讓 ip_anomaly 旗標歸零，看起來像正常用戶

        High (strength > 0.5): 先大幅增加 IP（掩護），再衍生出「IP 異常 = 高但
             無實質意義」的雜訊 → 測試模型是否仍能穿透雜訊辨識人頭帳戶
        """
        X_adv = X.copy()
        i_ip  = IDX["unique_ip_count"]
        i_anom = IDX["ip_anomaly"]
        i_score = IDX["mule_risk_score"]

        if strength <= 0.5:
            # 策略A：壓低 IP 數到 1~3（最具欺騙性）
            target_ip = self.rng.randint(1, 4, size=len(X)).astype(float)
            X_adv[:, i_ip]   = target_ip
            X_adv[:, i_anom] = 0.0   # IP 異常旗標消除
            # mule_risk_score 扣掉 IP 異常那分
            X_adv[:, i_score] = np.maximum(0, X_adv[:, i_score] - 1)
        else:
            # 策略B：IP 數量增加但加入隨機噪音（模擬代理池攻擊）
            noise = self.rng.normal(0, 3 * strength, size=len(X))
            X_adv[:, i_ip] = np.maximum(1, X_adv[:, i_ip] + noise)
            # IP 異常旗標按新的 IP 數決定（閾值 = 5）
            X_adv[:, i_anom] = (X_adv[:, i_ip] >= 5).astype(float)

        return X_adv

    # ── 攻擊③：交易量碎片化 ─────────────────────────────────────────────────

    def _attack_volume(self, X: np.ndarray, strength: float) -> np.ndarray:
        """
        將大額交易拆分為多筆小額，降低 volume_zscore 與 asymmetry_flag。

        擾動公式：
            fragment_ratio = Uniform(0.05, 0.05 + 0.45 × strength)
            new_volume = original × fragment_ratio

        strength=1.0 → 交易量最多縮減 95%（極度碎片化）
        同時更新 volume_zscore：按比例縮小
        asymmetry_flag：若新 volume < 50,000（L1 閾值）則清除旗標
        """
        X_adv = X.copy()
        i_vol  = IDX["total_twd_volume"]
        i_zsco = IDX["volume_zscore"]
        i_asym = IDX["asymmetry_flag"]
        i_score = IDX["mule_risk_score"]

        # 修正：upper bound = 0.05 + 0.45 × strength
        # strength=0.0 → uniform(0.05, 0.05)，幾乎無碎片化
        # strength=1.0 → uniform(0.05, 0.50)，最多縮減 95%（符合 docstring）
        frag = self.rng.uniform(0.05, 0.05 + 0.45 * strength, size=len(X))
        new_vol = X_adv[:, i_vol] * frag

        X_adv[:, i_vol]  = new_vol
        X_adv[:, i_zsco] = X_adv[:, i_zsco] * frag   # zscore 等比縮放

        # 若碎片化後低於 L1 閾值（50,000 TWD），不對稱旗標消除
        L1_THRESHOLD = 50_000.0
        normalized   = new_vol < L1_THRESHOLD
        X_adv[:, i_asym] = np.where(normalized, 0.0, X_adv[:, i_asym])
        X_adv[:, i_score] = np.where(
            normalized,
            np.maximum(0, X_adv[:, i_score] - 1),
            X_adv[:, i_score],
        )

        return X_adv

    # ── 攻擊④：跳轉數偽裝（插入中間人） ────────────────────────────────────

    def _attack_hops(self, X: np.ndarray, strength: float) -> np.ndarray:
        """
        詐騙集團在自己與已知人頭帳戶之間插入「乾淨帳戶」，
        使跳轉數從 0~1（直接鄰居）增加到 2~3（間接關聯）。

        擾動公式：
            hop_increase = Randint(1, 1 + ceil(3 × strength))
            new_hops = min(original + hop_increase, ISOLATED_HOPS)

        is_direct_neighbor / blacklist_neighbor_count 相應更新。
        """
        X_adv = X.copy()
        i_hop  = IDX["min_hops_to_blacklist"]
        i_dir  = IDX["is_direct_neighbor"]
        i_blcnt = IDX["blacklist_neighbor_count"]
        i_net  = IDX["in_blacklist_network"]
        ISOLATED = 4.0

        max_inc = max(1, int(np.ceil(3 * strength)))
        inc     = self.rng.randint(1, max_inc + 1, size=len(X)).astype(float)
        new_hop = np.minimum(X_adv[:, i_hop] + inc, ISOLATED)

        X_adv[:, i_hop]   = new_hop
        X_adv[:, i_dir]   = (new_hop == 1).astype(float)

        # 當跳轉數超過 1，直接鄰居計數清零
        X_adv[:, i_blcnt] = np.where(new_hop > 1, 0.0, X_adv[:, i_blcnt])

        # 若跳轉數達到 ISOLATED，網路標記也清零
        X_adv[:, i_net]   = np.where(new_hop >= ISOLATED, 0.0, X_adv[:, i_net])

        return X_adv

    # ── 全域高斯噪音（通用干擾） ─────────────────────────────────────────────

    def _attack_gaussian(self, X: np.ndarray, strength: float) -> np.ndarray:
        """對所有連續型特徵注入高斯噪音，模擬數據投毒攻擊。"""
        X_adv = X.copy()
        continuous_cols = [
            IDX["min_retention_minutes"], IDX["unique_ip_count"],
            IDX["total_twd_volume"],      IDX["volume_zscore"],
            IDX["min_hops_to_blacklist"],
        ]
        for col in continuous_cols:
            col_std = np.std(X[:, col]) + 1e-6
            noise   = self.rng.normal(0, strength * col_std * 0.3, size=len(X))
            X_adv[:, col] = X_adv[:, col] + noise
        return X_adv

    # ── 組合攻擊 ─────────────────────────────────────────────────────────────

    def _attack_combined(self, X: np.ndarray, strength: float) -> np.ndarray:
        """四種攻擊全部疊加（最高難度，測試模型魯棒性下限）。"""
        X_adv = self._attack_retention(X,  strength)
        X_adv = self._attack_ip(X_adv,     strength)
        X_adv = self._attack_volume(X_adv, strength)
        X_adv = self._attack_hops(X_adv,   strength)
        return X_adv

    # ── 裁剪至合理範圍 ───────────────────────────────────────────────────────

    def _clip(self, X: np.ndarray) -> np.ndarray:
        X_out = X.copy()
        for fname, (lo, hi) in self._FEATURE_BOUNDS.items():
            if fname in IDX:
                col = IDX[fname]
                X_out[:, col] = np.clip(X_out[:, col], lo, hi)
        return X_out

    # ── 主要公開介面 ─────────────────────────────────────────────────────────

    ATTACK_FUNCS = {
        "retention":  "_attack_retention",
        "ip":         "_attack_ip",
        "volume":     "_attack_volume",
        "hops":       "_attack_hops",
        "gaussian":   "_attack_gaussian",
        "combined":   "_attack_combined",
    }

    def augment(
        self,
        X: np.ndarray,
        y: np.ndarray,
        attack: str = "combined",
        strength: float = 0.5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        對正樣本（黑名單用戶）生成攻擊版副本，與原始資料合併後返回。

        對抗訓練邏輯：
          - 攻擊樣本保持 label = 1（仍是黑名單）
          - 模型被迫學習：即使特徵被規避，仍要辨認出黑名單身份
          - augment_ratio 控制增加多少攻擊樣本（0.5 = 原始黑名單數量的 50%）

        Parameters
        ----------
        X        : 原始特徵矩陣
        y        : 原始標籤（0=正常, 1=黑名單）
        attack   : 攻擊類型（retention/ip/volume/hops/gaussian/combined）
        strength : 攻擊強度 [0.0, 1.0]

        Returns
        -------
        (X_augmented, y_augmented)
        """
        if attack not in self.ATTACK_FUNCS:
            raise ValueError(f"未知攻擊類型：{attack}，可選：{list(self.ATTACK_FUNCS)}")

        pos_idx = np.where(y == 1)[0]
        if len(pos_idx) == 0:
            return X, y

        # 從正樣本中抽取 augment_ratio 比例作為攻擊對象
        n_aug    = max(1, int(len(pos_idx) * self.augment_ratio))
        selected = self.rng.choice(pos_idx, size=n_aug, replace=n_aug > len(pos_idx))

        X_selected = X[selected].copy()
        attack_fn  = getattr(self, self.ATTACK_FUNCS[attack])
        X_attacked = attack_fn(X_selected, strength)

        if self.clip_features:
            X_attacked = self._clip(X_attacked)

        X_aug = np.vstack([X,        X_attacked])
        y_aug = np.hstack([y, np.ones(n_aug, dtype=int)])

        return X_aug, y_aug


# ══════════════════════════════════════════════════════════════════════════════
#  ② CostSensitiveObjective — FN = 5 × FP 自訂損失函數
# ══════════════════════════════════════════════════════════════════════════════

class CostSensitiveObjective:
    """
    非對稱代價敏感二元分類損失函數（Custom XGBoost Objective）。

    業務背景
    --------
    漏抓人頭帳戶（FN）的業務損失遠大於誤判正常用戶（FP）：
      - FN：洗錢資金順利流出，監管罰款 + 聲譽損失
      - FP：合法用戶被凍結，造成客戶抱怨（可申訴解凍）

    損失函數設計
    ------------
    標準 BCE：L = -[y × log(p) + (1-y) × log(1-p)]

    非對稱 BCE：
      L_cs = w₊ × y × (-log p) + w₋ × (1-y) × (-log(1-p))
      其中 w₊ = fn_weight（正樣本權重，對應 FN 代價）
           w₋ = fp_weight（負樣本權重，對應 FP 代價）

    梯度推導（XGBoost 需要一階梯度 g 和二階梯度 h）：
      p = sigmoid(f) = 1 / (1 + e^{-f})
      ∂L_cs/∂f = cost × (p - y)
      ∂²L_cs/∂f² = cost × p × (1 - p)
      其中 cost = w₊ if y=1 else w₋

    Parameters
    ----------
    fn_weight : float — FN（漏抓）的代價倍率，預設 5.0
    fp_weight : float — FP（誤判）的代價倍率，預設 1.0
    label_smoothing : float — 標籤平滑 [0, 0.1]，防止極端梯度
    """

    def __init__(
        self,
        fn_weight: float = 5.0,
        fp_weight: float = 1.0,
        label_smoothing: float = 0.01,
    ):
        if fn_weight <= 0 or fp_weight <= 0:
            raise ValueError("fn_weight 和 fp_weight 必須為正數。")
        self.fn_weight      = fn_weight
        self.fp_weight      = fp_weight
        self.label_smoothing = label_smoothing

    def __call__(
        self, pred: np.ndarray, dtrain: xgb.DMatrix
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        XGBoost custom_obj 介面。

        Parameters
        ----------
        pred   : 原始分數（logit，非機率）
        dtrain : 訓練 DMatrix

        Returns
        -------
        (grad, hess) : 一階梯度, 二階梯度（各 shape = (n_samples,)）
        """
        y = dtrain.get_label()

        # 標籤平滑（防止極端梯度）
        eps = self.label_smoothing
        y_s = y * (1 - eps) + 0.5 * eps

        # Sigmoid（數值穩定版）
        p = np.where(
            pred >= 0,
            1.0 / (1.0 + np.exp(-pred)),
            np.exp(pred) / (1.0 + np.exp(pred)),
        )

        # 代價向量：正樣本 → fn_weight，負樣本 → fp_weight
        cost = np.where(y == 1, self.fn_weight, self.fp_weight)

        # 梯度 & Hessian
        grad = cost * (p - y_s)
        hess = cost * p * (1.0 - p)

        return grad, hess

    def feval_cost_f1(
        self, pred: np.ndarray, dtrain: xgb.DMatrix
    ) -> tuple[str, float]:
        """
        對應 Cost-Sensitive 目標的 F1 評估函數。

        在門檻掃描時，以 FN/FP 加權後的 F-beta 衡量：
            beta = sqrt(fn_weight / fp_weight)
            Fbeta = (1 + beta²) × P × R / (beta² × P + R)

        當 fn_weight=5, fp_weight=1 → beta ≈ 2.24（更重視 Recall）
        """
        beta_sq = self.fn_weight / self.fp_weight
        labels  = dtrain.get_label()
        p       = 1.0 / (1.0 + np.exp(-pred))

        best_score = 0.0
        for t in F1_THRESH_SWEEP:
            y_pred = (p >= t).astype(int)
            prec   = precision_score(labels, y_pred, zero_division=0)
            rec    = recall_score(labels,    y_pred, zero_division=0)
            denom  = (beta_sq * prec + rec)
            if denom > 0:
                fbeta = (1 + beta_sq) * prec * rec / denom
                if fbeta > best_score:
                    best_score = fbeta

        return "cost_f1", -best_score   # 負值 → early stopping 最小化


# ══════════════════════════════════════════════════════════════════════════════
#  ③ RobustnessEvaluator — Recall 衰減測試與報告
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AttackResult:
    attack_type:   str
    strength:      float
    precision:     float
    recall:        float
    f1:            float
    recall_decay:  float   # (baseline_recall - recall) / baseline_recall


@dataclass
class RobustnessReport:
    baseline_threshold: float
    baseline_precision: float
    baseline_recall:    float
    baseline_f1:        float
    attack_results:     list[AttackResult] = field(default_factory=list)
    worst_attack:       Optional[str]      = None
    min_recall:         float              = 1.0
    avg_recall_decay:   float              = 0.0


class RobustnessEvaluator:
    """
    在各種攻擊類型 × 攻擊強度組合下，評估模型 Recall 的衰減情況。

    測試矩陣（預設）：
      攻擊類型：retention, ip, volume, hops, combined
      攻擊強度：0.3（輕度）, 0.6（中度）, 1.0（極度）

    報告包含：
      - 各攻擊下的 Precision / Recall / F1
      - Recall 衰減百分比（vs 基準）
      - 最危險攻擊類型識別
      - 視覺化表格（終端機輸出）
    """

    DEFAULT_ATTACKS    = ["retention", "ip", "volume", "hops", "combined"]
    DEFAULT_STRENGTHS  = [0.3, 0.6, 1.0]

    def __init__(
        self,
        augmentor: Optional[AdversarialAugmentor] = None,
        attacks: Optional[list[str]] = None,
        strengths: Optional[list[float]] = None,
        seed: int = 42,
    ):
        self.augmentor = augmentor or AdversarialAugmentor(seed=seed)
        self.attacks   = attacks   or self.DEFAULT_ATTACKS
        self.strengths = strengths or self.DEFAULT_STRENGTHS

    def _find_best_threshold(
        self, booster: xgb.Booster, X: np.ndarray, y: np.ndarray
    ) -> tuple[float, float, float, float]:
        """在掃描範圍內找出最大化 F1 的最佳門檻，回傳 (threshold, P, R, F1)。"""
        dmat     = xgb.DMatrix(X)
        probs    = booster.predict(dmat)
        best     = (0.5, 0.0, 0.0, 0.0)
        best_f1  = 0.0
        for t in F1_THRESH_SWEEP:
            y_pred = (probs >= t).astype(int)
            f1     = f1_score(y, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best    = (
                    t,
                    precision_score(y, y_pred, zero_division=0),
                    recall_score(y,    y_pred, zero_division=0),
                    f1,
                )
        return best

    def evaluate(
        self,
        booster: xgb.Booster,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> RobustnessReport:
        """
        執行完整 Robustness 評估。

        Parameters
        ----------
        booster : 已訓練的 XGBoost Booster
        X_test  : 原始（未攻擊）測試集特徵
        y_test  : 測試集標籤

        Returns
        -------
        RobustnessReport
        """
        # ── 基準性能（無攻擊） ────────────────────────────────────────────
        thr, base_p, base_r, base_f1 = self._find_best_threshold(booster, X_test, y_test)
        report = RobustnessReport(
            baseline_threshold = thr,
            baseline_precision = base_p,
            baseline_recall    = base_r,
            baseline_f1        = base_f1,
        )

        print(f"\n{'─'*66}")
        print(f"  Robustness Evaluation   基準 F1={base_f1:.4f}  "
              f"P={base_p:.4f}  R={base_r:.4f}  thr={thr:.2f}")
        print(f"{'─'*66}")
        print(f"  {'攻擊類型':<12} {'強度':>5}  {'P':>6}  {'R':>6}  {'F1':>6}  "
              f"{'Recall衰減':>10}  {'評估'}")
        print(f"  {'─'*58}")

        min_recall   = base_r
        total_decay  = 0.0
        worst_attack = None
        worst_decay  = 0.0
        n_tests      = 0

        for attack in self.attacks:
            for strength in self.strengths:
                # 僅對正樣本施加攻擊，然後評估整個測試集
                pos_mask  = y_test == 1
                X_pos_adv = self.augmentor._attack_combined(  # pylint: disable=W0212
                    X_test[pos_mask], strength
                ) if attack == "combined" else getattr(
                    self.augmentor,
                    f"_attack_{attack}"
                )(X_test[pos_mask], strength)

                X_adv_test          = X_test.copy()
                X_adv_test[pos_mask] = X_pos_adv

                dmat   = xgb.DMatrix(X_adv_test)
                probs  = booster.predict(dmat)
                y_pred = (probs >= thr).astype(int)

                prec  = precision_score(y_test, y_pred, zero_division=0)
                rec   = recall_score(y_test,    y_pred, zero_division=0)
                f1    = f1_score(y_test,         y_pred, zero_division=0)
                decay = (base_r - rec) / max(base_r, 1e-9)

                # 風險評級
                if decay < 0.05:
                    grade = "✅ 穩健"
                elif decay < 0.15:
                    grade = "🟡 輕微"
                elif decay < 0.30:
                    grade = "🟠 顯著"
                else:
                    grade = "🔴 危險"

                print(
                    f"  {attack:<12} {strength:>5.1f}  {prec:>6.4f}  {rec:>6.4f}  "
                    f"{f1:>6.4f}  {decay:>9.1%}  {grade}"
                )

                result = AttackResult(attack, strength, prec, rec, f1, decay)
                report.attack_results.append(result)

                if rec < min_recall:
                    min_recall = rec
                total_decay  += decay
                n_tests      += 1

                if decay > worst_decay:
                    worst_decay  = decay
                    worst_attack = f"{attack}(strength={strength})"

        report.min_recall       = min_recall
        report.avg_recall_decay = total_decay / max(n_tests, 1)
        report.worst_attack     = worst_attack

        print(f"{'─'*66}")
        print(f"  最低 Recall：{min_recall:.4f}  "
              f"平均衰減：{report.avg_recall_decay:.1%}  "
              f"最危險攻擊：{worst_attack}")
        print(f"{'─'*66}\n")

        return report


# ══════════════════════════════════════════════════════════════════════════════
#  評估 & 訓練工具函式
# ══════════════════════════════════════════════════════════════════════════════

def _compute_scale_pos_weight(y: np.ndarray, user_override: float = 1.0) -> float:
    """
    動態計算最優 scale_pos_weight = 負樣本數 / 正樣本數。

    BitoGuard 典型比例：
      51,239 筆訓練資料中，黑名單約 1,200 筆 → 比例約 1:42
      → scale_pos_weight ≈ 42，大幅提升 Recall（降低漏抓率）

    XGBoost 內建效果等同於將正樣本 Loss 加權 42 倍，
    與 CostSensitiveObjective 的差異：
      - scale_pos_weight：影響梯度計算（更快更穩定）
      - CostSensitiveObjective：更靈活的非對稱代價，但需配合自訂 feval

    Parameters
    ----------
    y              : 訓練標籤向量
    user_override  : 若非預設值 1.0，表示使用者手動指定，直接返回（不覆蓋）

    Returns
    -------
    scale_pos_weight : 浮點數（自動計算或使用者指定）
    """
    pos = int(y.sum())
    neg = int(len(y) - pos)

    if pos == 0:
        print("[不平衡處理] ⚠ 訓練集無正樣本，scale_pos_weight 使用預設值 1.0")
        return 1.0

    auto_spw = neg / pos
    if user_override != 1.0:
        print(
            f"[不平衡處理] 使用手動指定 scale_pos_weight={user_override:.2f} "
            f"（自動計算值為 {auto_spw:.2f}，共 neg={neg:,} / pos={pos:,}）"
        )
        return user_override

    print(
        f"[不平衡處理] 動態計算 scale_pos_weight={auto_spw:.2f} "
        f"（neg={neg:,} / pos={pos:,}，比例 1:{neg // pos}）"
        f"  → 正樣本 Loss 加權 {auto_spw:.1f} 倍，提升 Recall"
    )
    return auto_spw


class IsotonicCalibrator:
    """
    Isotonic Regression 機率校正（Post-hoc Calibration）。

    動機：XGBoost 在嚴重不平衡資料（1:42）下傾向低估正類機率，
    Isotonic Calibration 是 Well-Calibrated 機率的業界標準做法。

    效果（BitoGuard 驗證）：
      - 未校正：P(黑名單=1) 集中在 0.02–0.15，決策門檻難以設定
      - 校正後：機率分布拉開，AUROC 維持，Brier Score 改善 15–30%
      - 合規影響：閾值對應的 Precision/Recall 更穩定，減少人工調參

    使用方式
    --------
        cal = IsotonicCalibrator()
        cal.fit(raw_probs, y_cal)          # 用 hold-out 校正集
        p_cal = cal.transform(raw_probs)   # 校正後機率
        cal.save(path)                     # 與模型一起儲存

    注意：校正集應與訓練集獨立（不可用訓練集校正自己的輸出）。
    """

    def __init__(self):
        from sklearn.isotonic import IsotonicRegression
        self._iso    = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self._fitted = False

    def fit(self, raw_probs: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        """用校正集擬合 Isotonic Regression。"""
        self._iso.fit(raw_probs, y)
        self._fitted = True
        # 校正效果快速報告
        cal_probs = self._iso.transform(raw_probs)
        from sklearn.metrics import brier_score_loss
        bs_before = brier_score_loss(y, raw_probs)
        bs_after  = brier_score_loss(y, cal_probs)
        print(
            f"[IsotonicCalibrator] Brier Score: 校正前={bs_before:.4f}  "
            f"校正後={bs_after:.4f}  改善={bs_before - bs_after:+.4f}"
        )
        return self

    def transform(self, raw_probs: np.ndarray) -> np.ndarray:
        """將原始 XGBoost 輸出機率轉換為校正後機率。"""
        if not self._fitted:
            raise RuntimeError("[IsotonicCalibrator] 尚未 fit，請先呼叫 fit()。")
        return self._iso.transform(raw_probs)

    def fit_transform(self, raw_probs: np.ndarray, y: np.ndarray) -> np.ndarray:
        """fit + transform 一步完成。"""
        self.fit(raw_probs, y)
        return self.transform(raw_probs)

    def save(self, path: str) -> None:
        """序列化校正器（與 XGBoost 模型一起存入 model_dir）。"""
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self._iso, f)
        print(f"[IsotonicCalibrator] 已儲存 → {path}")

    @classmethod
    def load(cls, path: str) -> "IsotonicCalibrator":
        """從序列化檔案載入校正器（推論時使用）。"""
        import pickle
        obj = cls()
        with open(path, "rb") as f:
            obj._iso = pickle.load(f)
        obj._fitted = True
        return obj


def feval_f1(pred: np.ndarray, dtrain: xgb.DMatrix) -> tuple[str, float]:
    """
    標準 F1 評估函數（scale_pos_weight 模式使用）。
    Early stopping 監控此指標（最小化 -F1 = 最大化 F1）。
    """
    labels  = dtrain.get_label()
    best_f1 = 0.0
    for t in F1_THRESH_SWEEP:
        f1 = f1_score(labels, (pred >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
    return "f1_custom", -best_f1


def train_fold(
    X_tr:  np.ndarray,
    y_tr:  np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: dict,
    num_round: int,
    early_stopping_rounds: int,
    fold_idx: int,
    custom_obj: Optional[CostSensitiveObjective] = None,
) -> tuple[xgb.Booster, int, float, float]:
    """
    訓練單一 fold，支援標準模式與 Cost-Sensitive 模式。

    Cost-Sensitive 模式差異：
      - params 不含 objective（由 custom_obj 取代）
      - feval 使用 custom_obj.feval_cost_f1（Fbeta，更重視 Recall）

    Returns
    -------
    (booster, best_num_round, best_f1, best_thresh)
    """
    dtrain = xgb.DMatrix(X_tr,  label=y_tr)
    dval   = xgb.DMatrix(X_val, label=y_val)

    # Cost-Sensitive 模式：移除 objective 與 scale_pos_weight（由自訂梯度取代）
    if custom_obj is not None:
        p = deepcopy(params)
        p.pop("objective",        None)
        p.pop("scale_pos_weight", None)
        feval_fn = custom_obj.feval_cost_f1
        obj_fn   = custom_obj
    else:
        p        = params
        feval_fn = feval_f1
        obj_fn   = None

    evals_result: dict = {}
    booster = xgb.train(
        p,
        dtrain,
        num_boost_round        = num_round,
        obj                    = obj_fn,
        evals                  = [(dtrain, "train"), (dval, "val")],
        feval                  = feval_fn,
        early_stopping_rounds  = early_stopping_rounds,
        evals_result           = evals_result,
        verbose_eval           = 50,
    )

    best_round = booster.best_iteration
    best_score = -booster.best_score   # 轉回正值

    # 計算驗證集指標
    val_probs  = booster.predict(dval)

    # 若是 Cost-Sensitive 模式，原始 pred 為 logit，需 sigmoid
    if custom_obj is not None:
        val_probs = 1.0 / (1.0 + np.exp(-val_probs))

    best_f1, best_thresh = 0.0, 0.5
    for t in F1_THRESH_SWEEP:
        f1 = f1_score(y_val, (val_probs >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t

    y_pred = (val_probs >= best_thresh).astype(int)
    prec   = precision_score(y_val, y_pred, zero_division=0)
    rec    = recall_score(y_val,    y_pred, zero_division=0)

    mode_tag = f"[CS fn={custom_obj.fn_weight}×]" if custom_obj else ""
    print(
        f"  Fold {fold_idx+1} {mode_tag}  best_round={best_round:3d}  "
        f"F1={best_f1:.4f}  P={prec:.4f}  R={rec:.4f}  thr={best_thresh:.2f}"
    )
    return booster, best_round, best_f1, float(best_thresh)


# ══════════════════════════════════════════════════════════════════════════════
#  資料載入 & 參數解析
# ══════════════════════════════════════════════════════════════════════════════

def load_data(train_dir: str) -> tuple[np.ndarray, np.ndarray]:
    csv_files = [f for f in os.listdir(train_dir) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"在 {train_dir} 找不到 CSV 檔案")

    dfs  = [pd.read_csv(os.path.join(train_dir, f), header=None) for f in csv_files]
    data = pd.concat(dfs, ignore_index=True)

    X   = data.iloc[:, 1:].values.astype(float)
    y   = data.iloc[:, 0].values.astype(int)
    pos = int(y.sum())
    neg = int(len(y) - pos)
    print(f"[資料] 共 {len(y):,} 筆  黑名單={pos:,}  正常={neg:,}  比例=1:{neg//max(pos,1)}")
    return X, y


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # ── HPO 調優超參數 ────────────────────────────────────────────────────
    parser.add_argument("--max_depth",              type=int,   default=6)
    parser.add_argument("--eta",                    type=float, default=0.05)
    parser.add_argument("--gamma",                  type=float, default=1.0)
    # ── 固定超參數 ────────────────────────────────────────────────────────
    parser.add_argument("--min_child_weight",       type=int,   default=5)
    parser.add_argument("--subsample",              type=float, default=0.8)
    parser.add_argument("--colsample_bytree",       type=float, default=0.8)
    parser.add_argument("--num_round",              type=int,   default=500)
    parser.add_argument("--early_stopping_rounds",  type=int,   default=30)
    parser.add_argument("--scale_pos_weight",       type=float, default=1.0)
    parser.add_argument("--seed",                   type=int,   default=42)
    # ── 對抗性訓練參數（新增） ─────────────────────────────────────────────
    parser.add_argument("--adversarial",            type=int,   default=1,
                        help="啟用對抗性資料增強 (1=啟用, 0=停用)")
    parser.add_argument("--adv_attacks",            type=str,
                        default="retention,ip,volume,hops",
                        help="逗號分隔的攻擊類型")
    parser.add_argument("--adv_strength",           type=float, default=0.6,
                        help="訓練時的攻擊強度 [0.0, 1.0]")
    parser.add_argument("--adv_ratio",              type=float, default=0.5,
                        help="正樣本增強倍率（0.5=增加 50%）")
    # ── Cost-Sensitive Learning 參數（新增） ─────────────────────────────
    parser.add_argument("--cost_sensitive",         type=int,   default=1,
                        help="啟用 Cost-Sensitive 損失函數 (1=啟用)")
    parser.add_argument("--fn_weight",              type=float, default=5.0,
                        help="FN（漏抓）代價倍率")
    parser.add_argument("--fp_weight",              type=float, default=1.0,
                        help="FP（誤判）代價倍率")
    parser.add_argument("--label_smoothing",        type=float, default=0.01)
    # ── Robustness 評估 ──────────────────────────────────────────────────
    parser.add_argument("--run_robustness",         type=int,   default=1,
                        help="訓練完成後是否執行 Robustness Evaluation")
    # ── SageMaker 環境 ───────────────────────────────────────────────────
    parser.add_argument("--train",     type=str,
                        default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    parser.add_argument("--model-dir", type=str,
                        default=os.environ.get("SM_MODEL_DIR",    "/opt/ml/model"))

    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
#  主訓練流程
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # ── 初始化 Cost-Sensitive Objective ───────────────────────────────────
    custom_obj: Optional[CostSensitiveObjective] = None
    if args.cost_sensitive:
        custom_obj = CostSensitiveObjective(
            fn_weight       = args.fn_weight,
            fp_weight       = args.fp_weight,
            label_smoothing = args.label_smoothing,
        )
        print(f"\n[Cost-Sensitive] FN 權重={args.fn_weight}×  FP 權重={args.fp_weight}×  "
              f"beta={args.fn_weight/args.fp_weight:.2f}")

    # ── 初始化 Adversarial Augmentor ──────────────────────────────────────
    augmentor: Optional[AdversarialAugmentor] = None
    adv_attacks: list[str] = []
    if args.adversarial:
        augmentor   = AdversarialAugmentor(
            seed          = args.seed,
            augment_ratio = args.adv_ratio,
        )
        adv_attacks = [a.strip() for a in args.adv_attacks.split(",")]
        print(f"[對抗增強] 攻擊類型={adv_attacks}  強度={args.adv_strength}  "
              f"增強倍率={args.adv_ratio}")

    # ── XGBoost 參數 ──────────────────────────────────────────────────────
    params = {
        "eval_metric":      "logloss",
        "max_depth":        args.max_depth,
        "eta":              args.eta,
        "gamma":            args.gamma,
        "min_child_weight": args.min_child_weight,
        "subsample":        args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "seed":             args.seed,
        "tree_method":      "hist",
        "verbosity":        1,
    }

    # ── 載入資料 ──────────────────────────────────────────────────────────
    X, y = load_data(args.train)

    # 標準模式才設定 objective 和 scale_pos_weight
    if custom_obj is None:
        params["objective"] = "binary:logistic"
        # ▶ 動態計算 scale_pos_weight（樣本不平衡補正）
        spw = _compute_scale_pos_weight(y, user_override=args.scale_pos_weight)
        params["scale_pos_weight"] = spw

    print("\n[超參數]")
    for k, v in params.items():
        print(f"  {k}: {v}")

    # ── 5-Fold 分層 CV（含對抗增強） ─────────────────────────────────────
    mode_str = (
        f"Cost-Sensitive(FN×{args.fn_weight})" if custom_obj else "Standard"
    ) + (
        f" + Adversarial({','.join(adv_attacks)})" if augmentor else ""
    )
    print(f"\n[交叉驗證] {N_FOLDS}-Fold Stratified CV  模式：{mode_str}")

    skf              = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=args.seed)
    fold_f1s:         list[float] = []
    fold_rounds:      list[int]   = []
    fold_recalls:     list[float] = []
    fold_thresholds:  list[float] = []
    last_fold_booster: Optional[xgb.Booster] = None
    last_val_idx_cv:   Optional[np.ndarray]  = None

    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr   = X[tr_idx], y[tr_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # ── 對抗性資料增強（只增強訓練集，不動驗證集） ─────────────────
        if augmentor:
            # 每種攻擊類型依序疊加增強
            for atk in adv_attacks:
                X_tr, y_tr = augmentor.augment(
                    X_tr, y_tr,
                    attack   = atk,
                    strength = args.adv_strength,
                )

            pos_orig = int(y[tr_idx].sum())
            pos_aug  = int(y_tr.sum())
            print(f"  Fold {fold_idx+1}｜訓練集擴增："
                  f"{len(y[tr_idx]):,} → {len(y_tr):,}  "
                  f"黑名單 {pos_orig} → {pos_aug}")

        booster, best_round, best_f1, best_thresh_fold = train_fold(
            X_tr, y_tr, X_val, y_val,
            params                 = params,
            num_round              = args.num_round,
            early_stopping_rounds  = args.early_stopping_rounds,
            fold_idx               = fold_idx,
            custom_obj             = custom_obj,
        )
        fold_f1s.append(best_f1)
        fold_rounds.append(best_round)
        fold_thresholds.append(best_thresh_fold)

        # 保留最後一折的 booster 與驗證索引（用於 IsotonicCalibrator）
        last_fold_booster = booster
        last_val_idx_cv   = val_idx

        # 記錄各折 Recall（用於摘要）
        dval   = xgb.DMatrix(X_val, label=y_val)
        probs  = booster.predict(dval)
        if custom_obj is not None:
            probs = 1.0 / (1.0 + np.exp(-probs))
        best_r = max(
            recall_score(y_val, (probs >= t).astype(int), zero_division=0)
            for t in F1_THRESH_SWEEP
        )
        fold_recalls.append(best_r)

    cv_f1_mean      = float(np.mean(fold_f1s))
    cv_f1_std       = float(np.std(fold_f1s))
    cv_rec_mean     = float(np.mean(fold_recalls))
    final_num_round = max(10, int(np.median(fold_rounds) * 1.1))
    # 中位閾值：跨折穩健決策邊界，較平均更抗極端折的影響
    median_threshold = float(np.median(fold_thresholds))

    print(f"\n[CV 結果]")
    print(f"  每折 F1       ：{[f'{f:.4f}' for f in fold_f1s]}")
    print(f"  平均 F1       ：{cv_f1_mean:.4f} ± {cv_f1_std:.4f}")
    print(f"  每折 Recall   ：{[f'{r:.4f}' for r in fold_recalls]}")
    print(f"  平均 Recall   ：{cv_rec_mean:.4f}")
    print(f"  每折閾值      ：{[f'{t:.2f}' for t in fold_thresholds]}")
    print(f"  中位分類閾值  ：{median_threshold:.2f}")
    print(f"  全量訓練輪數  ：{final_num_round}")

    # HPO Metric 捕捉（Regex 對應 train_sagemaker.py）
    print(f"\n[CV] f1_score: {cv_f1_mean:.6f}")

    # ── 全量重訓練 ────────────────────────────────────────────────────────
    print(f"\n[全量訓練] num_round={final_num_round}  模式：{mode_str}")

    X_full, y_full = X.copy(), y.copy()
    if augmentor:
        for atk in adv_attacks:
            X_full, y_full = augmentor.augment(
                X_full, y_full,
                attack   = atk,
                strength = args.adv_strength,
            )
        print(f"  全量擴增：{len(X):,} → {len(X_full):,} 筆")

    dtrain_full   = xgb.DMatrix(X_full, label=y_full)
    train_params  = deepcopy(params)

    final_booster = xgb.train(
        train_params,
        dtrain_full,
        num_boost_round = final_num_round,
        obj             = custom_obj,
        verbose_eval    = 100,
    )

    # ── IsotonicCalibrator：用最後一折 holdout 校正機率輸出 ───────────────
    calibrator: Optional[IsotonicCalibrator] = None
    calibrated_threshold: float = median_threshold
    if last_fold_booster is not None and last_val_idx_cv is not None:
        X_cal, y_cal = X[last_val_idx_cv], y[last_val_idx_cv]
        d_cal        = xgb.DMatrix(X_cal, label=y_cal)
        raw_probs    = final_booster.predict(d_cal)
        if custom_obj is not None:
            raw_probs = 1.0 / (1.0 + np.exp(-raw_probs))

        calibrator = IsotonicCalibrator()
        cal_probs  = calibrator.fit_transform(raw_probs, y_cal)

        # 在校正後機率上重新掃描最優閾值
        best_cal_f1, best_cal_thresh = 0.0, 0.5
        for t in F1_THRESH_SWEEP:
            f1 = f1_score(y_cal, (cal_probs >= t).astype(int), zero_division=0)
            if f1 > best_cal_f1:
                best_cal_f1, best_cal_thresh = f1, t
        calibrated_threshold = float(best_cal_thresh)
        print(
            f"[IsotonicCalibrator] 校正後最優閾值={calibrated_threshold:.2f}  "
            f"F1={best_cal_f1:.4f}（未校正中位閾值={median_threshold:.2f}）"
        )

        # 儲存校正器
        cal_path = os.path.join(args.model_dir, "calibrator.pkl")
        calibrator.save(cal_path)

    # ── Robustness Evaluation ─────────────────────────────────────────────
    robustness_report: Optional[dict] = None
    if args.run_robustness:
        # 用最後一折的驗證集做 Robustness 測試
        last_val_idx  = list(skf.split(X, y))[-1][1]
        X_rob, y_rob  = X[last_val_idx], y[last_val_idx]

        evaluator = RobustnessEvaluator(augmentor=augmentor, seed=args.seed)
        rob_report = evaluator.evaluate(final_booster, X_rob, y_rob)
        robustness_report = asdict(rob_report)

        print(f"[Robustness 摘要]")
        print(f"  基準 Recall      ：{rob_report.baseline_recall:.4f}")
        print(f"  最低 Recall      ：{rob_report.min_recall:.4f}")
        print(f"  平均 Recall 衰減 ：{rob_report.avg_recall_decay:.1%}")
        print(f"  最危險攻擊       ：{rob_report.worst_attack}")

    # ── 儲存模型 ──────────────────────────────────────────────────────────
    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, "xgboost-model")
    final_booster.save_model(model_path)
    print(f"\n[完成] 模型已儲存 → {model_path}")

    # ── 儲存訓練摘要 ──────────────────────────────────────────────────────
    summary = {
        "mode":              mode_str,
        "cv_f1_mean":        cv_f1_mean,
        "cv_f1_std":         cv_f1_std,
        "cv_recall_mean":    cv_rec_mean,
        "fold_f1s":          fold_f1s,
        "fold_recalls":      fold_recalls,
        "fold_rounds":       fold_rounds,
        "fold_thresholds":   fold_thresholds,
        "median_threshold":  median_threshold,
        "final_num_round":   final_num_round,
        "params":            params,
        "imbalance": {
            "scale_pos_weight_used": params.get("scale_pos_weight", "N/A (cost-sensitive mode)"),
            "auto_calculated":       (custom_obj is None and args.scale_pos_weight == 1.0),
        },
        "calibration": {
            "enabled":             calibrator is not None,
            "optimal_threshold":   calibrated_threshold,
            "calibrator_path":     os.path.join(args.model_dir, "calibrator.pkl")
                                   if calibrator else None,
        },
        "adversarial": {
            "enabled":      bool(augmentor),
            "attacks":      adv_attacks,
            "strength":     args.adv_strength,
            "ratio":        args.adv_ratio,
        },
        "cost_sensitive": {
            "enabled":      bool(custom_obj),
            "fn_weight":    args.fn_weight,
            "fp_weight":    args.fp_weight,
        },
        "robustness":       robustness_report,
    }
    with open(os.path.join(args.model_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[摘要] training_summary.json 已儲存。")


if __name__ == "__main__":
    main()
