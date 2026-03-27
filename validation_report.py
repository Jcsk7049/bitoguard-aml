"""
技術驗證報告生成器
2026 去偽存真：AI 全民偵查黑客松 — 幣託科技

執行後自動計算：
  - Hold-out 集上的 F1 / Precision / Recall / AUC
  - 門檻掃描曲線（Precision-Recall trade-off）
  - AWS 架構紀錄
  - 輸出 validation_report.json 與 validation_report.txt
"""

import json
import textwrap
import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime
from dataclasses import dataclass, asdict, field
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, precision_recall_curve,
)

# ── 常數 ─────────────────────────────────────────────────────────────────────

REPORT_VERSION = "1.0.0"
PROJECT_NAME   = "BitoMuleDetector"
GENERATED_AT   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ── 資料結構 ─────────────────────────────────────────────────────────────────

@dataclass
class MetricSnapshot:
    threshold:  float
    f1:         float
    precision:  float
    recall:     float
    fpr:        float   # False Positive Rate（誤判率）
    tn: int; fp: int; fn: int; tp: int


@dataclass
class CVFoldResult:
    fold:      int
    f1:        float
    precision: float
    recall:    float
    auc:       float


@dataclass
class ThresholdAnalysis:
    optimal_threshold:   float
    optimal_f1:          float
    optimal_precision:   float
    optimal_recall:      float
    optimal_fpr:         float
    conservative_threshold: float   # 低誤判版本（FPR < 5%）
    conservative_f1:        float
    conservative_precision: float
    conservative_recall:    float
    sweep_table: list[dict]         # 門檻掃描完整表格


@dataclass
class ValidationReport:
    project:        str
    version:        str
    generated_at:   str
    dataset_stats:  dict
    cv_results:     list[dict]
    cv_summary:     dict
    threshold_analysis: dict
    aws_architecture:   dict
    feature_importance: list[dict]
    technical_highlights: list[str]
    risk_design_notes:    list[str]


# ── 1. 指標計算核心 ───────────────────────────────────────────────────────────

def evaluate_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> MetricSnapshot:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return MetricSnapshot(
        threshold = round(threshold, 4),
        f1        = round(f1_score(y_true, y_pred, zero_division=0), 4),
        precision = round(precision_score(y_true, y_pred, zero_division=0), 4),
        recall    = round(recall_score(y_true, y_pred, zero_division=0), 4),
        fpr       = round(fpr, 4),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
    )


def run_cv(
    X: np.ndarray,
    y: np.ndarray,
    hyperparams: dict,
    n_splits: int = 5,
) -> list[CVFoldResult]:
    """5-fold 分層交叉驗證，回傳每折指標。"""
    skf     = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval   = xgb.DMatrix(X_val)
        model  = xgb.train(
            {k: v for k, v in hyperparams.items() if k != "num_round"},
            dtrain,
            num_boost_round=hyperparams.get("num_round", 300),
            verbose_eval=False,
        )
        probs  = model.predict(dval)
        preds  = (probs >= 0.5).astype(int)

        results.append(CVFoldResult(
            fold=fold,
            f1        = round(f1_score(y_val, preds, zero_division=0), 4),
            precision = round(precision_score(y_val, preds, zero_division=0), 4),
            recall    = round(recall_score(y_val, preds, zero_division=0), 4),
            auc       = round(roc_auc_score(y_val, probs), 4),
        ))
        print(f"  Fold {fold}: F1={results[-1].f1:.4f}  P={results[-1].precision:.4f}"
              f"  R={results[-1].recall:.4f}  AUC={results[-1].auc:.4f}")

    return results


def sweep_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    steps: int = 81,
) -> ThresholdAnalysis:
    """掃描 0.1–0.9 門檻，找最佳 F1 與低誤判版本。"""
    thresholds  = np.linspace(0.1, 0.9, steps)
    sweep_table = []
    best_f1_snap: MetricSnapshot | None = None
    low_fpr_snap: MetricSnapshot | None = None   # FPR < 5%，F1 最大

    for t in thresholds:
        snap = evaluate_at_threshold(y_true, y_prob, t)
        sweep_table.append(asdict(snap))

        if best_f1_snap is None or snap.f1 > best_f1_snap.f1:
            best_f1_snap = snap

        if snap.fpr < 0.05:
            if low_fpr_snap is None or snap.f1 > low_fpr_snap.f1:
                low_fpr_snap = snap

    # fallback：若 FPR 全部 >= 5%，取誤判率最低的
    if low_fpr_snap is None:
        low_fpr_snap = min(sweep_table, key=lambda d: d["fpr"])
        low_fpr_snap = evaluate_at_threshold(y_true, y_prob, low_fpr_snap["threshold"])

    return ThresholdAnalysis(
        optimal_threshold   = best_f1_snap.threshold,
        optimal_f1          = best_f1_snap.f1,
        optimal_precision   = best_f1_snap.precision,
        optimal_recall      = best_f1_snap.recall,
        optimal_fpr         = best_f1_snap.fpr,
        conservative_threshold = low_fpr_snap.threshold,
        conservative_f1        = low_fpr_snap.f1,
        conservative_precision = low_fpr_snap.precision,
        conservative_recall    = low_fpr_snap.recall,
        sweep_table = sweep_table,
    )


# ── 2. 特徵重要度 ────────────────────────────────────────────────────────────

def extract_feature_importance(
    model: xgb.Booster,
    feature_names: list[str],
    top_n: int = 15,
) -> list[dict]:
    scores = model.get_score(importance_type="gain")
    total  = sum(scores.values()) or 1.0
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    from xai_bedrock import FEATURE_LABELS
    return [
        {
            "rank":         i + 1,
            "feature":      name,
            "label":        FEATURE_LABELS.get(name, name),
            "gain":         round(score, 2),
            "gain_pct":     round(score / total * 100, 2),
        }
        for i, (name, score) in enumerate(ranked)
    ]


# ── 3. AWS 架構紀錄 ──────────────────────────────────────────────────────────

AWS_ARCHITECTURE = {
    "services": [
        {
            "service":  "Amazon SageMaker",
            "role":     "模型訓練 & 批次推論",
            "config": {
                "framework":       "XGBoost 1.7-1（內建容器）",
                "instance_train":  "ml.m5.xlarge × 1",
                "instance_infer":  "ml.m5.xlarge × 1（Batch Transform）",
                "region":          "us-east-1",
            },
            "optimization_notes": [
                "使用 SageMaker 內建 XGBoost 容器，省去自行封裝 Docker 成本",
                "Batch Transform 策略設為 SingleRecord + Line 拆分，確保預測結果與輸入行一一對齊",
                "model artifacts 自動存至 S3 output_path，便於版本管理與回溯",
                "訓練資料上傳前以 dtype=str 讀取 CSV 再轉 float，防止大整數（ori_samount 原始值）精度損失",
            ],
        },
        {
            "service":  "Amazon S3",
            "role":     "資料湖（訓練集、推論集、模型檔、輸出結果）",
            "config": {
                "prefix_structure": (
                    "bito-mule-detection/"
                    "{timestamp}/train/  ← 訓練資料\n"
                    "                   /infer/ ← 推論資料\n"
                    "                   /output/ ← 模型 & 預測"
                ),
            },
            "optimization_notes": [
                "以 Unix timestamp 為 prefix，確保每次實驗互不覆蓋",
                "使用 put_object() 串流上傳（io.StringIO），避免寫入本地暫存檔",
                "SageMaker 訓練格式要求：label 置首欄、無 header，於上傳函式集中處理",
            ],
        },
        {
            "service":  "Amazon Bedrock",
            "role":     "XAI 自然語言風險診斷書生成",
            "config": {
                "model":    "anthropic.claude-3-5-sonnet-20241022-v2:0",
                "region":   "us-east-1",
                "protocol": "bedrock-2023-05-31 / invoke_model API",
                "max_tokens": 1024,
            },
            "optimization_notes": [
                "僅對機率 >= 0.5 的高風險用戶呼叫 Bedrock，節省約 50~80% Token 費用",
                "System Prompt 固定角色為「AML/KYC 金融詐騙分析師」，確保輸出語氣一致",
                "User Prompt 內嵌特徵中文對照表，讓 Claude 讀到人類可理解的描述而非技術欄位名",
                "以「---」分隔符切分兩段輸出（風險診斷 / 攔截建議），便於程式解析與 UI 呈現",
                "異常處理：API 失敗時保留 SHAP 數值報告，不中斷整體流程",
            ],
        },
        {
            "service":  "BitoPro REST API",
            "role":     "原始資料來源（替代本地 CSV）",
            "config": {
                "base_url": "https://aws-event-api.bitopro.com",
                "tables":   ["user_info", "twd_transfer", "crypto_transfer",
                             "usdt_twd_trading", "usdt_swap",
                             "train_label", "predict_label"],
            },
            "optimization_notes": [
                "BitoDataManager 支援 csv_dir 與 API 雙模式，方便本地 Debug 切換",
                "全部金額欄位統一在 normalize_amounts() 乘以 1e-8，集中管理避免遺漏",
                "AMOUNT_FIELDS 設定表 + 自動偵測雙保險機制，應對未來新增欄位",
            ],
        },
    ],
    "data_flow": [
        "BitoPro API → BitoDataManager.load_all()",
        "→ build_features()（特徵工程）",
        "→ S3 上傳（train.csv / infer.csv）",
        "→ SageMaker XGBoost Estimator.fit()",
        "→ Batch Transform → predictions.out",
        "→ find_best_threshold()（F1 最佳門檻）",
        "→ submission.csv（競賽提交）",
        "→ XAIReportGenerator（SHAP + Bedrock）",
        "→ xai_reports.json（風險診斷書）",
    ],
}

TECHNICAL_HIGHLIGHTS = [
    "非平衡資料處理：scale_pos_weight = 負樣本數 / 正樣本數，動態計算，無需手動調整",
    "三維人頭戶特徵：滯留時間（時序配對）× IP 異常（集合計數）× 量能不對稱（群組 Z-score），覆蓋命題定義的全部訊號",
    "深夜交易比例：額外捕捉「深夜時段異常大額提現」行為特徵（命題明確列出）",
    "虛幣折台幣：crypto_transfer.ori_samount × twd_srate，統一量綱後計算總交易量，跨幣種可比",
    "F1 最佳門檻：hold-out 20% 掃描 0.1–0.9（步長 0.01），以 F1-score 選最佳，與評分標準一致",
    "保守模式門檻：額外計算 FPR < 5% 約束下的最佳 F1，提供低誤判率方案",
    "XAI 診斷書：SHAP TreeExplainer + Bedrock Claude，輸出可讀性高的中文診斷書，符合可解釋性評比要求",
    "login_logs Fallback：IP 特徵在 login_logs 尚未提供前自動 fallback 至 twd/crypto 的 source_ip",
    "5-fold 分層交叉驗證：保持正負樣本比例，評估模型泛化能力",
    "端到端管線：API → 特徵工程 → SageMaker 訓練 → 門檻最佳化 → 提交 CSV，單一指令完成",
]

RISK_DESIGN_NOTES = [
    "【為何選 F1 而非 Accuracy】訓練集正負樣本比約 1:N（高度不平衡），Accuracy 會因全猜 0 而虛高；F1 同時考量 Precision 與 Recall，與競賽評比一致。",
    "【門檻設計哲學】預設門檻 0.5 對不平衡資料偏保守；透過 hold-out 掃描，通常可將門檻下調至 0.3–0.45 以提升 Recall，但須注意 Precision 下降。保守模式（FPR<5%）適合正式上線前。",
    "【scale_pos_weight 效果】若正樣本佔 2%，則 scale_pos_weight ≈ 49，等效於對每筆正樣本的 loss 放大 49 倍，讓模型學習少數類邊界。",
    "【Z-score 群組化】量能不對稱特徵以「同 KYC 等級」為群組計算 Z-score，避免 L2 用戶高交易量污染 L0/L1 基準，降低誤判率。",
    "【SHAP 貢獻百分比化】原始 SHAP 值為 log-odds 空間，數值範圍不直觀；轉換為佔所有特徵絕對值之百分比後，非技術人員可直接理解「IP 跳動佔 40% 的風險貢獻」。",
]


# ── 4. 主報告生成流程 ────────────────────────────────────────────────────────

def generate_report(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    hyperparams: dict,
) -> ValidationReport:
    """
    完整驗證流程：
      1. 5-fold CV 計算平均指標
      2. Hold-out 門檻掃描
      3. 特徵重要度
      4. 組裝報告
    """
    # ── CV ──────────────────────────────────────────────────────────────────
    print("\n[1/4] 5-Fold 交叉驗證...")
    cv_results = run_cv(X_train, y_train, hyperparams)
    cv_summary = {
        "f1_mean":        round(np.mean([r.f1        for r in cv_results]), 4),
        "f1_std":         round(np.std( [r.f1        for r in cv_results]), 4),
        "precision_mean": round(np.mean([r.precision for r in cv_results]), 4),
        "recall_mean":    round(np.mean([r.recall    for r in cv_results]), 4),
        "auc_mean":       round(np.mean([r.auc       for r in cv_results]), 4),
        "auc_std":        round(np.std( [r.auc       for r in cv_results]), 4),
    }
    print(f"  CV 平均 → F1={cv_summary['f1_mean']:.4f}±{cv_summary['f1_std']:.4f}"
          f"  AUC={cv_summary['auc_mean']:.4f}±{cv_summary['auc_std']:.4f}")

    # ── Hold-out 門檻掃描 ────────────────────────────────────────────────────
    print("\n[2/4] Hold-out 門檻掃描...")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    neg = (y_tr == 0).sum(); pos = (y_tr == 1).sum()
    sp_weight = round(neg / pos, 4)
    params = {**hyperparams, "scale_pos_weight": sp_weight}

    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    model  = xgb.train(
        {k: v for k, v in params.items() if k != "num_round"},
        dtrain,
        num_boost_round=params.get("num_round", 300),
        verbose_eval=False,
    )
    val_probs = model.predict(xgb.DMatrix(X_val))
    threshold_analysis = sweep_thresholds(y_val, val_probs)
    print(f"  最佳 F1 門檻: {threshold_analysis.optimal_threshold}"
          f"  F1={threshold_analysis.optimal_f1}"
          f"  P={threshold_analysis.optimal_precision}"
          f"  R={threshold_analysis.optimal_recall}"
          f"  FPR={threshold_analysis.optimal_fpr}")
    print(f"  保守門檻(FPR<5%): {threshold_analysis.conservative_threshold}"
          f"  F1={threshold_analysis.conservative_f1}"
          f"  P={threshold_analysis.conservative_precision}")

    # ── 特徵重要度 ────────────────────────────────────────────────────────────
    print("\n[3/4] 特徵重要度...")
    feat_importance = extract_feature_importance(model, feature_names)
    for fi in feat_importance[:5]:
        print(f"  #{fi['rank']:2d}  {fi['gain_pct']:5.1f}%  {fi['label']}")

    # ── 資料集統計 ────────────────────────────────────────────────────────────
    pos_total = int(y_train.sum())
    neg_total = int((y_train == 0).sum())
    dataset_stats = {
        "train_total":      len(y_train),
        "positive_mule":    pos_total,
        "negative_normal":  neg_total,
        "imbalance_ratio":  round(neg_total / pos_total, 2),
        "scale_pos_weight": sp_weight,
        "feature_count":    len(feature_names),
        "feature_names":    feature_names,
    }

    print(f"\n[4/4] 組裝報告... (正樣本={pos_total}, 負樣本={neg_total}, 比={sp_weight})")

    return ValidationReport(
        project         = PROJECT_NAME,
        version         = REPORT_VERSION,
        generated_at    = GENERATED_AT,
        dataset_stats   = dataset_stats,
        cv_results      = [asdict(r) for r in cv_results],
        cv_summary      = cv_summary,
        threshold_analysis  = asdict(threshold_analysis),
        aws_architecture    = AWS_ARCHITECTURE,
        feature_importance  = feat_importance,
        technical_highlights = TECHNICAL_HIGHLIGHTS,
        risk_design_notes    = RISK_DESIGN_NOTES,
    )


# ── 5. 文字報告渲染 ──────────────────────────────────────────────────────────

def render_text_report(r: ValidationReport) -> str:
    sep1 = "=" * 68
    sep2 = "-" * 68
    W    = 66

    def wrap(text: str, indent: int = 4) -> str:
        return textwrap.fill(text, width=W, initial_indent=" " * indent,
                             subsequent_indent=" " * (indent + 2))

    lines = [
        sep1,
        f"  {r.project}  技術驗證報告  v{r.version}",
        f"  生成時間：{r.generated_at}",
        sep1,
        "",
        "█ 一、資料集概況",
        sep2,
        f"  訓練集總筆數          : {r.dataset_stats['train_total']:,}",
        f"  正樣本（人頭戶）       : {r.dataset_stats['positive_mule']:,}",
        f"  負樣本（正常用戶）     : {r.dataset_stats['negative_normal']:,}",
        f"  不平衡比（負/正）      : {r.dataset_stats['imbalance_ratio']}",
        f"  scale_pos_weight      : {r.dataset_stats['scale_pos_weight']}",
        f"  特徵維度              : {r.dataset_stats['feature_count']} 維",
        "",
        "█ 二、5-Fold 交叉驗證結果",
        sep2,
        f"  {'Fold':<6} {'F1':>7} {'Precision':>10} {'Recall':>8} {'AUC':>8}",
        "  " + "-" * 42,
    ]

    for cv in r.cv_results:
        lines.append(
            f"  {cv['fold']:<6} {cv['f1']:>7.4f} {cv['precision']:>10.4f}"
            f" {cv['recall']:>8.4f} {cv['auc']:>8.4f}"
        )

    s = r.cv_summary
    lines += [
        "  " + "-" * 42,
        f"  {'平均':<6} {s['f1_mean']:>7.4f} {s['precision_mean']:>10.4f}"
        f" {s['recall_mean']:>8.4f} {s['auc_mean']:>8.4f}",
        f"  {'標準差':<6} {s['f1_std']:>7.4f} {'':>10} {'':>8} {s['auc_std']:>8.4f}",
        "",
        "█ 三、風險門檻分析",
        sep2,
        "  ┌─ 模式A：最佳 F1 門檻（預設推薦）",
    ]

    ta = r.threshold_analysis
    lines += [
        f"  │   門檻值   : {ta['optimal_threshold']}",
        f"  │   F1-score : {ta['optimal_f1']}",
        f"  │   Precision: {ta['optimal_precision']}  ← 預測為黑名單的準確率",
        f"  │   Recall   : {ta['optimal_recall']}  ← 實際黑名單的抓獲率",
        f"  │   FPR      : {ta['optimal_fpr']}  ← 正常用戶被誤判機率",
        "  │",
        "  └─ 模式B：保守模式（FPR < 5%，優先保護用戶體驗）",
        f"      門檻值   : {ta['conservative_threshold']}",
        f"      F1-score : {ta['conservative_f1']}",
        f"      Precision: {ta['conservative_precision']}",
        f"      Recall   : {ta['conservative_recall']}",
        "",
        "  門檻選擇策略：",
        wrap("模式A 以 F1 最大化為目標，適合競賽評分（命題主要指標為 F1）。"
             "模式B 將 FPR 限制在 5% 以內，確保每 100 位正常用戶中最多 5 位被誤標，"
             "適合正式上線時優先維護用戶體驗的場景。兩組門檻均在相同 hold-out 集（20%）"
             "計算，避免 data leakage。", 4),
        "",
        "  門檻掃描摘要（每 10 步取樣）：",
        f"  {'門檻':>6} {'F1':>7} {'Precision':>10} {'Recall':>8} {'FPR':>7}",
        "  " + "-" * 42,
    ]

    sweep = ta["sweep_table"]
    step  = max(1, len(sweep) // 10)
    for row in sweep[::step]:
        lines.append(
            f"  {row['threshold']:>6.2f} {row['f1']:>7.4f} {row['precision']:>10.4f}"
            f" {row['recall']:>8.4f} {row['fpr']:>7.4f}"
        )

    lines += [
        "",
        "█ 四、特徵重要度（Top 10，依 Gain 排序）",
        sep2,
        f"  {'#':>3} {'特徵（中文）':<28} {'Gain%':>6}  圖示",
        "  " + "-" * 55,
    ]

    for fi in r.feature_importance[:10]:
        bar = "▓" * int(fi["gain_pct"] / 2)
        lines.append(
            f"  {fi['rank']:>3}  {fi['label'][:26]:<28} {fi['gain_pct']:>5.1f}%  {bar}"
        )

    lines += [
        "",
        "█ 五、AWS 架構優化紀錄",
        sep2,
    ]

    for svc in r.aws_architecture["services"]:
        lines.append(f"  ▶ {svc['service']}（{svc['role']}）")
        for note in svc["optimization_notes"]:
            lines.append(wrap(f"• {note}", 6))
        lines.append("")

    lines += [
        "  資料流向：",
    ]
    for i, step in enumerate(r.aws_architecture["data_flow"], 1):
        lines.append(f"  {i:>2}. {step}")

    lines += [
        "",
        "█ 六、技術亮點摘要",
        sep2,
    ]
    for i, h in enumerate(r.technical_highlights, 1):
        lines.append(wrap(f"{i:>2}. {h}", 4))

    lines += [
        "",
        "█ 七、風險門檻設計說明",
        sep2,
    ]
    for note in r.risk_design_notes:
        lines.append(wrap(note, 4))
        lines.append("")

    lines.append(sep1)
    return "\n".join(lines)


# ── 6. 儲存 ──────────────────────────────────────────────────────────────────

def save_report(report: ValidationReport) -> None:
    # JSON（結構化，可供 dashboard 讀取）
    with open("validation_report.json", "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, ensure_ascii=False, indent=2)
    print("  → validation_report.json")

    # 純文字（可直接貼入簡報）
    text = render_text_report(report)
    with open("validation_report.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print("  → validation_report.txt")
    print("\n" + text)


# ── 入口 ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from bito_data_manager import BitoDataManager
    from train_sagemaker import build_features, BASE_HYPERPARAMS, build_hyperparams

    # ── 載入資料 ────────────────────────────────────────────────────────────
    print("[init] 載入資料與特徵工程...")
    manager     = BitoDataManager()
    tables      = manager.load_all()
    features    = build_features(manager)

    train_label = manager._load_raw("train_label")
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

    hyperparams = build_hyperparams(merged)

    # ── 生成報告 ─────────────────────────────────────────────────────────────
    report = generate_report(X, y, FEAT_COLS, hyperparams)

    print("\n[save] 儲存報告...")
    save_report(report)
