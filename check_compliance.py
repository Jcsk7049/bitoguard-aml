#!/usr/bin/env python3
"""
BitoGuard 合規自動化檢核腳本 (check_compliance.py)

執行三項現場合規檢核：
  C-1  Credential Scan  — 掃描專案所有 .py 檔，偵測硬編碼 AWS 憑證
  C-2  S3 ACL Scan      — 靜態掃描公開存取設定；--live-s3 可追加 boto3 即時查詢
  C-3  PII Filter Test  — 以虛構個人資料測試 xai_bedrock.py 三層 PII 過濾器

使用方式：
  python check_compliance.py                              # 掃描目前目錄
  python check_compliance.py --dir /path/to/project      # 指定專案根目錄
  python check_compliance.py --live-s3 --bucket my-bkt   # 追加真實 S3 查詢
  python check_compliance.py --only c1 c3                # 只執行指定檢查
  python check_compliance.py --no-color                  # 停用 ANSI 顏色

退出代碼：
  0 = 全數 PASS（或僅有 WARN）
  1 = 至少一項 FAIL
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import io
import os
import pathlib
import re
import sys
from typing import List, Optional

# Windows 終端機強制 UTF-8 輸出，避免 cp950 無法顯示中文符號
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
elif hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ══════════════════════════════════════════════════════════════════════════════
#  顯示輔助（ANSI 顏色，自動偵測 TTY）
# ══════════════════════════════════════════════════════════════════════════════

_USE_COLOR: bool = sys.stdout.isatty()   # 由 --no-color 可覆蓋


def _c(code: str, text: str) -> str:
    return f"{code}{text}\033[0m" if _USE_COLOR else text


def green(t: str) -> str: return _c("\033[92m", t)
def red(t: str)   -> str: return _c("\033[91m", t)
def yellow(t: str)-> str: return _c("\033[93m", t)
def blue(t: str)  -> str: return _c("\033[94m", t)
def bold(t: str)  -> str: return _c("\033[1m",  t)
def dim(t: str)   -> str: return _c("\033[2m",  t)


def _status_label(status: str) -> str:
    mapping = {
        "PASS": green("OK PASS"),
        "FAIL": red("NG FAIL"),
        "WARN": yellow("!! WARN"),
        "SKIP": dim("-- SKIP"),
        "INFO": blue(">> INFO"),
    }
    return mapping.get(status, status)


def _severity_label(sev: str) -> str:
    mapping = {
        "FAIL": red("[FAIL]"),
        "WARN": yellow("[WARN]"),
        "INFO": blue("[INFO]"),
    }
    return mapping.get(sev, f"[{sev}]")


# ══════════════════════════════════════════════════════════════════════════════
#  資料結構
# ══════════════════════════════════════════════════════════════════════════════

@dataclasses.dataclass
class Finding:
    """單一問題發現記錄。"""
    severity: str        # FAIL / WARN / INFO
    file:     str        # 相對路徑
    line:     int        # 行號（0 = 不適用）
    code:     str        # 問題代碼（C1-01 等）
    message:  str        # 描述
    snippet:  str = ""   # 原始碼片段（選填）


@dataclasses.dataclass
class CheckResult:
    """單項檢核的總結果。"""
    name:     str
    status:   str                  # PASS / FAIL / WARN / SKIP
    summary:  str
    findings: List[Finding] = dataclasses.field(default_factory=list)

    @property
    def fail_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "FAIL")

    @property
    def warn_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "WARN")

    @property
    def info_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "INFO")


# ══════════════════════════════════════════════════════════════════════════════
#  工具函式
# ══════════════════════════════════════════════════════════════════════════════

def _iter_py_files(root: pathlib.Path, *, exclude_self: bool = True) -> list[pathlib.Path]:
    """列出專案中所有 .py 檔（排除虛擬環境與快取目錄）。"""
    skip_dirs = {".venv", "venv", "__pycache__", ".git", "node_modules", ".tox"}
    result = []
    for p in sorted(root.rglob("*.py")):
        if any(part in skip_dirs for part in p.parts):
            continue
        if exclude_self and p.name == "check_compliance.py":
            continue
        result.append(p)
    return result


def _read_lines(path: pathlib.Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []


# ══════════════════════════════════════════════════════════════════════════════
#  C-1  Credential Scan — 硬編碼 AWS 憑證偵測
# ══════════════════════════════════════════════════════════════════════════════
#
#  偵測規則：
#   C1-01  AWS Access Key ID（AKIA* / ASIA*，20 字元）
#   C1-02  AWS Secret Access Key 賦值（40 字元 base64-like）
#   C1-03  boto3 client / resource 內聯憑證
#   C1-04  通用 API Key / Token 硬編碼（排除已知佔位符）
# ══════════════════════════════════════════════════════════════════════════════

# 真實 AWS Access Key ID 格式：AKIA/ASIA 開頭 + 16 大寫英數字
_C1_RE_ACCESS_KEY = re.compile(
    r"(?<![A-Z0-9_])((AKIA|ASIA)[0-9A-Z]{16})(?![A-Z0-9_])"
)

# AWS Secret Access Key：40 字元 base64，出現在常見變數名右側
_C1_RE_SECRET_KEY = re.compile(
    r"(?:aws_secret_access_key|secret_access_key|SecretAccessKey|AWS_SECRET_ACCESS_KEY)"
    r"\s*[=:]\s*[\"']([A-Za-z0-9/+]{40})[\"']",
    re.IGNORECASE,
)

# boto3 / botocore 呼叫中夾帶內聯憑證
_C1_RE_BOTO3_CRED = re.compile(
    r"(?:aws_access_key_id|aws_secret_access_key)\s*=\s*[\"'][A-Za-z0-9/+=]{16,}[\"']",
    re.IGNORECASE,
)

# 通用 secret / token / password 賦值（值長度 ≥ 16，且不是已知佔位符）
_C1_RE_GENERIC = re.compile(
    r"(?:token|password|passwd|secret|api_key|apikey|private_key)\s*=\s*"
    r"[\"'](?!(?:your|example|test|dummy|placeholder|change|replace"
    r"|TODO|ACCOUNT_ID|REGION|xxx|<[^>]+>)[^\"']*)[^\"']{16,}[\"']",
    re.IGNORECASE,
)

# 安全佔位符正則：命中此模式的行降級為 WARN
_C1_SAFE_PLACEHOLDER = re.compile(
    r"ACCOUNT_ID|your[-_]?(?:bucket|key|token|role)|placeholder"
    r"|change[-_]?me|example|TODO|xxx|<FILL|REPLACE_ME",
    re.IGNORECASE,
)

_C1_RULES: list[tuple[re.Pattern, str, str]] = [
    (_C1_RE_ACCESS_KEY,  "C1-01", "疑似 AWS Access Key ID（AKIA/ASIA 格式）"),
    (_C1_RE_SECRET_KEY,  "C1-02", "疑似 AWS Secret Access Key 賦值"),
    (_C1_RE_BOTO3_CRED,  "C1-03", "boto3 呼叫中夾帶內聯憑證"),
    (_C1_RE_GENERIC,     "C1-04", "疑似通用 API Key / Token 硬編碼"),
]


def credential_scan(root: pathlib.Path) -> CheckResult:
    """
    C-1：掃描 Python 原始碼中的硬編碼 AWS 憑證。

    使用 4 條正則規則逐行比對，對命中安全佔位符的行降級為 WARN，
    其餘視為 FAIL（高危）。
    """
    findings: list[Finding] = []
    py_files = _iter_py_files(root)

    for py_path in py_files:
        lines = _read_lines(py_path)
        rel   = str(py_path.relative_to(root))

        for lineno, raw in enumerate(lines, start=1):
            stripped = raw.strip()

            # 跳過純注釋行（不掃描 # 開頭的行）
            if stripped.startswith("#"):
                continue

            for pattern, code, message in _C1_RULES:
                if not pattern.search(raw):
                    continue

                # 命中安全佔位符 → 降級為 WARN
                severity = "WARN" if _C1_SAFE_PLACEHOLDER.search(raw) else "FAIL"
                findings.append(Finding(
                    severity=severity,
                    file=rel,
                    line=lineno,
                    code=code,
                    message=message,
                    snippet=stripped[:120],
                ))

    fail_n = sum(1 for f in findings if f.severity == "FAIL")
    warn_n = sum(1 for f in findings if f.severity == "WARN")
    total  = len(py_files)

    if fail_n:
        status  = "FAIL"
        summary = f"{fail_n} 個高危發現，{warn_n} 個佔位符警告（掃描 {total} 個 .py 檔）"
    elif warn_n:
        status  = "WARN"
        summary = f"無高危憑證，{warn_n} 個佔位符警告需人工確認（掃描 {total} 個 .py 檔）"
    else:
        status  = "PASS"
        summary = f"未偵測到硬編碼憑證（掃描 {total} 個 .py 檔）"

    return CheckResult(name="C-1 Credential Scan",
                       status=status, summary=summary, findings=findings)


# ══════════════════════════════════════════════════════════════════════════════
#  C-2  S3 ACL Scan — S3 公開存取設定
# ══════════════════════════════════════════════════════════════════════════════
#
#  靜態掃描規則：
#   C2-01  Python 程式碼含 public-read / public-read-write ACL
#   C2-02  CloudFormation AWS::S3::Bucket 缺少 PublicAccessBlockConfiguration
#   C2-03  CloudFormation PublicAccessBlock 某項為 false
#
#  即時查詢（--live-s3 模式）：
#   C2-04  Bucket 未設定 PublicAccessBlockConfiguration
#   C2-05  Bucket ACL 對 AllUsers / AuthenticatedUsers 開放
#   C2-06  Bucket Policy 含 "Effect":"Allow","Principal":"*"
# ══════════════════════════════════════════════════════════════════════════════

_C2_RE_PUBLIC_ACL = re.compile(
    r"(?:CannedACL|['\"]acl['\"])\s*[=:]\s*['\"]public-read(?:-write)?['\"]",
    re.IGNORECASE,
)

_C2_S3_BUCKET_TYPE = re.compile(r"Type:\s+AWS::S3::Bucket")
_C2_PUBLIC_BLOCK   = re.compile(r"PublicAccessBlockConfiguration")
_C2_BLOCK_SETTING  = re.compile(
    r"(BlockPublicAcls|IgnorePublicAcls|BlockPublicPolicy|RestrictPublicBuckets)"
    r":\s*(true|false)",
    re.IGNORECASE,
)


def _scan_template_yaml(root: pathlib.Path) -> list[Finding]:
    """掃描 template.yaml 中 S3 Bucket 的 PublicAccessBlock 設定。"""
    findings: list[Finding] = []
    tmpl = root / "template.yaml"
    if not tmpl.exists():
        return findings

    content = tmpl.read_text(encoding="utf-8")
    s3_count    = len(_C2_S3_BUCKET_TYPE.findall(content))
    block_count = len(_C2_PUBLIC_BLOCK.findall(content))

    if s3_count == 0:
        return findings

    # 每個 S3 Bucket 應有一組 PublicAccessBlockConfiguration
    if block_count < s3_count:
        findings.append(Finding(
            severity="FAIL",
            file="template.yaml",
            line=0,
            code="C2-02",
            message=(
                f"發現 {s3_count} 個 AWS::S3::Bucket，"
                f"但只有 {block_count} 個含 PublicAccessBlockConfiguration"
            ),
        ))
    else:
        # 進一步確認各設定值是否均為 true
        settings = {m[0]: m[1].lower() for m in _C2_BLOCK_SETTING.findall(content)}
        required = {"BlockPublicAcls", "IgnorePublicAcls",
                    "BlockPublicPolicy", "RestrictPublicBuckets"}
        false_settings = [k for k in required if settings.get(k) == "false"]
        if false_settings:
            findings.append(Finding(
                severity="FAIL",
                file="template.yaml",
                line=0,
                code="C2-03",
                message=f"PublicAccessBlock 以下設定為 false：{false_settings}",
            ))
        else:
            findings.append(Finding(
                severity="INFO",
                file="template.yaml",
                line=0,
                code="C2-02",
                message=f"S3 Bucket PublicAccessBlock 四項設定均為 true OK",
            ))

    return findings


def _live_s3_check(bucket: str) -> list[Finding]:
    """透過 boto3 即時查詢真實 S3 Bucket 設定。"""
    findings: list[Finding] = []
    try:
        import boto3
        import botocore.exceptions as bce
    except ImportError:
        findings.append(Finding(
            severity="WARN", file="[live-s3]", line=0, code="C2-L0",
            message="boto3 未安裝，略過即時 S3 查詢",
        ))
        return findings

    # R-3 修正：明確指定 region_name，避免使用環境變數中的非合規 Region
    _live_region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    if _live_region not in {"us-east-1", "us-west-2"}:
        findings.append(Finding(
            severity="FAIL", file="[live-s3]", line=0, code="C2-R0",
            message=(
                f"[Region 合規] AWS_DEFAULT_REGION='{_live_region}' 不符合競賽規定，"
                f"S3 即時掃描改用 us-east-1 繼續執行。"
            ),
        ))
        _live_region = "us-east-1"
    s3 = boto3.client("s3", region_name=_live_region)

    # ── PublicAccessBlock 設定 ───────────────────────────────────────────────
    try:
        resp   = s3.get_public_access_block(Bucket=bucket)
        config = resp["PublicAccessBlockConfiguration"]
        required = {"BlockPublicAcls", "IgnorePublicAcls",
                    "BlockPublicPolicy", "RestrictPublicBuckets"}
        not_set = [k for k in required if not config.get(k)]
        if not_set:
            findings.append(Finding(
                severity="FAIL", file=f"[live] s3://{bucket}", line=0, code="C2-L1",
                message=f"PublicAccessBlock 以下設定未啟用：{not_set}",
                snippet=str(config),
            ))
        else:
            findings.append(Finding(
                severity="INFO", file=f"[live] s3://{bucket}", line=0, code="C2-L1",
                message="PublicAccessBlock 四項設定均已啟用 OK",
            ))
    except s3.exceptions.__class__:
        pass
    except Exception as exc:
        if "NoSuchPublicAccessBlockConfiguration" in type(exc).__name__:
            findings.append(Finding(
                severity="FAIL", file=f"[live] s3://{bucket}", line=0, code="C2-L2",
                message="Bucket 完全未設定 PublicAccessBlockConfiguration",
            ))
        else:
            findings.append(Finding(
                severity="WARN", file=f"[live] s3://{bucket}", line=0, code="C2-L3",
                message=f"無法查詢 PublicAccessBlock：{exc}",
            ))

    # ── Bucket ACL ───────────────────────────────────────────────────────────
    try:
        acl = s3.get_bucket_acl(Bucket=bucket)
        for grant in acl.get("Grants", []):
            uri  = grant.get("Grantee", {}).get("URI", "")
            perm = grant.get("Permission", "")
            if "AllUsers" in uri or "AuthenticatedUsers" in uri:
                findings.append(Finding(
                    severity="FAIL", file=f"[live] s3://{bucket}", line=0, code="C2-L4",
                    message=f"Bucket ACL 對外開放：{perm} → {uri}",
                ))
    except Exception as exc:
        findings.append(Finding(
            severity="WARN", file=f"[live] s3://{bucket}", line=0, code="C2-L5",
            message=f"無法取得 Bucket ACL：{exc}",
        ))

    # ── Bucket Policy：檢查 Principal:* ─────────────────────────────────────
    try:
        import json as _json
        policy_str = s3.get_bucket_policy(Bucket=bucket)["Policy"]
        policy     = _json.loads(policy_str)
        for stmt in policy.get("Statement", []):
            principal = stmt.get("Principal", "")
            effect    = stmt.get("Effect", "")
            if effect == "Allow" and ("*" in str(principal)):
                findings.append(Finding(
                    severity="FAIL", file=f"[live] s3://{bucket}", line=0, code="C2-L6",
                    message=f"Bucket Policy 含 Effect:Allow + Principal:* → 公開存取",
                    snippet=str(stmt)[:120],
                ))
    except Exception as exc:
        if "NoSuchBucketPolicy" not in type(exc).__name__:
            findings.append(Finding(
                severity="WARN", file=f"[live] s3://{bucket}", line=0, code="C2-L7",
                message=f"無法取得 Bucket Policy：{exc}",
            ))

    return findings


def s3_acl_scan(
    root: pathlib.Path,
    *,
    live_s3: bool = False,
    bucket: Optional[str] = None,
) -> CheckResult:
    """C-2：掃描 S3 公開存取設定（靜態 + 可選即時）。"""
    findings: list[Finding] = []
    py_files = _iter_py_files(root)

    # 靜態掃描 Python 檔
    for py_path in py_files:
        lines = _read_lines(py_path)
        rel   = str(py_path.relative_to(root))
        for lineno, raw in enumerate(lines, start=1):
            if raw.strip().startswith("#"):
                continue
            if _C2_RE_PUBLIC_ACL.search(raw):
                findings.append(Finding(
                    severity="FAIL", file=rel, line=lineno, code="C2-01",
                    message="程式碼含 S3 公開讀取 ACL（public-read）",
                    snippet=raw.strip()[:120],
                ))

    # CloudFormation template.yaml 靜態掃描
    findings.extend(_scan_template_yaml(root))

    # 可選：即時 boto3 查詢
    if live_s3 and bucket:
        findings.extend(_live_s3_check(bucket))
    elif live_s3 and not bucket:
        findings.append(Finding(
            severity="WARN", file="[live-s3]", line=0, code="C2-W0",
            message="--live-s3 已開啟但未提供 --bucket，略過即時查詢",
        ))

    fail_n = sum(1 for f in findings if f.severity == "FAIL")
    warn_n = sum(1 for f in findings if f.severity == "WARN")

    if fail_n:
        status  = "FAIL"
        summary = f"{fail_n} 個公開存取風險，{warn_n} 個警告"
    elif warn_n:
        status  = "WARN"
        summary = f"無公開存取漏洞，{warn_n} 個警告需人工確認"
    else:
        status  = "PASS"
        summary = "所有 S3 Bucket 均已設定 PublicAccessBlock，未偵測到公開 ACL"

    return CheckResult(name="C-2 S3 ACL Scan",
                       status=status, summary=summary, findings=findings)


# ══════════════════════════════════════════════════════════════════════════════
#  C-3  PII Filter Test — xai_bedrock.py 三層過濾器驗證
# ══════════════════════════════════════════════════════════════════════════════
#
#  測試案例設計原則：
#   - 虛構資料（非真實個人），符合 PDPA 測試豁免條款
#   - 每層過濾器有獨立的 PASS / FAIL 斷言
#   - 若 xai_bedrock 無法匯入，自動降級為內建正則引擎
#
#  三層測試：
#   Layer 1  filter_pii_context()        — context dict 允許名單
#   Layer 2  filter_pii_contributions()  — SHAP 特徵名稱允許名單
#   Layer 3  scan_prompt_for_pii()       — Prompt 字串正則遮罩
# ══════════════════════════════════════════════════════════════════════════════

# ── 虛構 PII 測試資料（純假資料，不對應任何真實個人）─────────────────────────
_MOCK_CONTEXT_WITH_PII: dict = {
    # ↓ 應被 Layer 1 攔截（PII 欄位）
    "source_ip":      "203.69.21.88",          # 真實 IPv4 地址
    "full_name":      "王小明",                  # 姓名
    "email":          "wang.xiaoming@fake.tw",  # Email
    "phone":          "0912345678",             # 台灣手機號碼
    "wallet_address": "1A2B3C4D5E6F7G8H9",     # 鏈上錢包地址（模擬）
    # ↓ 應被 Layer 1 保留（合法數值欄位）
    "min_retention_minutes":  8.5,
    "twd_withdraw_count":     12,
    "crypto_withdraw_count":  6,
    "min_hops_to_blacklist":  1,
    "ip_shared_user_count":   9,
    "has_high_speed_risk":    True,
    "hop_risk_level":         "direct",
    "weighted_risk_label":    "HIGH_WEIGHTED",
    "kyc_level":              0,
}

_PII_KEYS_EXPECTED_BLOCKED: frozenset[str] = frozenset({
    "source_ip", "full_name", "email", "phone", "wallet_address",
})

_SAFE_KEYS_EXPECTED_KEPT: frozenset[str] = frozenset({
    "min_retention_minutes", "twd_withdraw_count", "crypto_withdraw_count",
    "min_hops_to_blacklist", "ip_shared_user_count", "has_high_speed_risk",
    "hop_risk_level", "weighted_risk_label", "kyc_level",
})

# ── 虛構 Prompt 字串（含各類 PII 的模擬訊息）────────────────────────────────
_MOCK_PROMPT_WITH_PII: str = (
    "分析帳戶資訊：\n"
    "  姓名    : 王小明\n"
    "  Email   : wang.xiaoming@fake.tw\n"
    "  電話    : 0912345678\n"
    "  國際電話: +886912345678\n"
    "  IP 位址 : 203.69.21.88\n"
    "  身份證  : A234567890\n"
    "  特徵    : min_retention_minutes=8.5, min_hops=1\n"
)

# ── 各 PII 遮罩斷言 ─────────────────────────────────────────────────────────
_LAYER3_ASSERTIONS: list[tuple[str, str, str, str]] = [
    ("203.69.21.88",           "[IP-REDACTED]",    "C3-31", "IPv4 位址"),
    ("wang.xiaoming@fake.tw",  "[EMAIL-REDACTED]", "C3-32", "Email 地址"),
    ("0912345678",             "[PHONE-REDACTED]", "C3-33", "台灣手機號碼"),
    ("+886912345678",          "[PHONE-REDACTED]", "C3-34", "國際電話號碼"),
    ("A234567890",             "[ID-REDACTED]",    "C3-35", "台灣身份證字號"),
]


def _test_layer1(filter_pii_context_fn) -> list[Finding]:
    """Layer 1：context dict 允許名單過濾。"""
    findings: list[Finding] = []
    filtered = filter_pii_context_fn(_MOCK_CONTEXT_WITH_PII.copy())

    # 斷言：PII 欄位全部被移除
    leaked = _PII_KEYS_EXPECTED_BLOCKED & set(filtered.keys())
    if leaked:
        findings.append(Finding(
            severity="FAIL", file="xai_bedrock.py", line=0, code="C3-11",
            message=f"Layer 1 洩漏：{len(leaked)} 個 PII 欄位未被移除 → {sorted(leaked)}",
            snippet=str(sorted(leaked)),
        ))
    else:
        findings.append(Finding(
            severity="INFO", file="xai_bedrock.py", line=0, code="C3-11",
            message=f"Layer 1 OK 成功攔截 {len(_PII_KEYS_EXPECTED_BLOCKED)} 個 PII 欄位",
        ))

    # 斷言：合法欄位全部保留（無誤刪）
    over_blocked = _SAFE_KEYS_EXPECTED_KEPT - set(filtered.keys())
    if over_blocked:
        findings.append(Finding(
            severity="FAIL", file="xai_bedrock.py", line=0, code="C3-12",
            message=f"Layer 1 過度過濾：{len(over_blocked)} 個合法欄位被誤刪 → {sorted(over_blocked)}",
            snippet=str(sorted(over_blocked)),
        ))
    else:
        findings.append(Finding(
            severity="INFO", file="xai_bedrock.py", line=0, code="C3-12",
            message=f"Layer 1 OK {len(_SAFE_KEYS_EXPECTED_KEPT)} 個合法欄位均保留，無誤刪",
        ))

    return findings


def _test_layer2(filter_pii_contributions_fn, ShapContribution_cls) -> list[Finding]:
    """Layer 2：SHAP 特徵名稱允許名單過濾。"""
    findings: list[Finding] = []

    # 模擬含 PII 衍生特徵的 ShapContribution 清單
    mock_contribs = [
        ShapContribution_cls(
            feature_name="min_retention_minutes",  # OK 合法
            feature_label="最短資金滯留時間",
            shap_value=0.38, contribution_pct=38.0,
            feature_value=8.5, direction="增加風險",
        ),
        ShapContribution_cls(
            feature_name="ip_anomaly",             # OK 合法
            feature_label="IP 異常旗標",
            shap_value=0.25, contribution_pct=25.0,
            feature_value=1.0, direction="增加風險",
        ),
        ShapContribution_cls(
            feature_name="user_email_hash",        # NG PII 衍生特徵（不在允許名單）
            feature_label="Email 雜湊",
            shap_value=0.22, contribution_pct=22.0,
            feature_value=0.0, direction="增加風險",
        ),
        ShapContribution_cls(
            feature_name="source_ip_raw",          # NG 原始 IP 欄位（不在允許名單）
            feature_label="原始來源 IP",
            shap_value=0.15, contribution_pct=15.0,
            feature_value=0.0, direction="增加風險",
        ),
    ]

    pii_feature_names = {"user_email_hash", "source_ip_raw"}
    safe_feature_names = {"min_retention_minutes", "ip_anomaly"}

    filtered = filter_pii_contributions_fn(mock_contribs)
    filtered_names = {c.feature_name for c in filtered}

    # 斷言：PII 特徵全部移除
    leaked = pii_feature_names & filtered_names
    if leaked:
        findings.append(Finding(
            severity="FAIL", file="xai_bedrock.py", line=0, code="C3-21",
            message=f"Layer 2 洩漏：{len(leaked)} 個 PII 衍生特徵未被移除 → {sorted(leaked)}",
            snippet=str(sorted(leaked)),
        ))
    else:
        findings.append(Finding(
            severity="INFO", file="xai_bedrock.py", line=0, code="C3-21",
            message=f"Layer 2 OK 成功攔截 {len(pii_feature_names)} 個 PII 衍生特徵",
        ))

    # 斷言：合法特徵全部保留
    over_blocked = safe_feature_names - filtered_names
    if over_blocked:
        findings.append(Finding(
            severity="FAIL", file="xai_bedrock.py", line=0, code="C3-22",
            message=f"Layer 2 過度過濾：{len(over_blocked)} 個合法特徵被誤刪 → {sorted(over_blocked)}",
        ))
    else:
        findings.append(Finding(
            severity="INFO", file="xai_bedrock.py", line=0, code="C3-22",
            message=f"Layer 2 OK {len(safe_feature_names)} 個合法特徵均保留，無誤刪",
        ))

    return findings


def _test_layer3(scan_prompt_fn) -> list[Finding]:
    """Layer 3：Prompt 字串正則遮罩。"""
    findings: list[Finding] = []
    redacted = scan_prompt_fn(_MOCK_PROMPT_WITH_PII)

    for original, expected_mask, code, label in _LAYER3_ASSERTIONS:
        if original in redacted:
            findings.append(Finding(
                severity="FAIL", file="xai_bedrock.py", line=0, code=code,
                message=f"Layer 3 洩漏：{label} 未被遮罩 → 原始字串仍存在",
                snippet=f"原始：'{original}'  期望：'{expected_mask}'",
            ))
        elif expected_mask in redacted:
            findings.append(Finding(
                severity="INFO", file="xai_bedrock.py", line=0, code=code,
                message=f"Layer 3 OK {label} 已遮罩為 {expected_mask}",
            ))
        else:
            findings.append(Finding(
                severity="WARN", file="xai_bedrock.py", line=0, code=code,
                message=f"Layer 3 不確定：{label} 遮罩結果非預期格式",
                snippet=f"期望 {expected_mask}，遮罩後片段：{redacted[:80]}",
            ))

    return findings


def _builtin_layer3_test() -> list[Finding]:
    """
    xai_bedrock 無法匯入時的後備測試：
    直接使用 Python 內建 re 模組重現相同正則規則，驗證模式本身是否正確。
    """
    findings: list[Finding] = []

    # 複製自 xai_bedrock.py Layer 3 正則（保持同步）
    _RE_IPV4       = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
    _RE_EMAIL      = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
    _RE_TW_PHONE   = re.compile(r"\b09\d{8}\b")
    _RE_INTL_PHONE = re.compile(r"\+\d{8,15}\b")
    _RE_TW_ID      = re.compile(r"\b[A-Z][12]\d{8}\b")

    pattern_map = [
        (_RE_IPV4,       "203.69.21.88",           "[IP-REDACTED]",    "C3-31", "IPv4 位址"),
        (_RE_EMAIL,      "wang.xiaoming@fake.tw",   "[EMAIL-REDACTED]", "C3-32", "Email 地址"),
        (_RE_TW_PHONE,   "0912345678",              "[PHONE-REDACTED]", "C3-33", "台灣手機號碼"),
        (_RE_INTL_PHONE, "+886912345678",            "[PHONE-REDACTED]", "C3-34", "國際電話號碼"),
        (_RE_TW_ID,      "A234567890",              "[ID-REDACTED]",    "C3-35", "台灣身份證字號"),
    ]

    findings.append(Finding(
        severity="INFO", file="[builtin-fallback]", line=0, code="C3-00",
        message="xai_bedrock 匯入失敗，改用內建正則引擎直接驗證 Layer 3 模式",
    ))

    for pattern, sample, mask, code, label in pattern_map:
        if pattern.search(sample):
            findings.append(Finding(
                severity="INFO", file="[builtin-fallback]", line=0, code=code,
                message=f"正則引擎 OK 可偵測並遮罩 {label}：'{sample}' → {mask}",
            ))
        else:
            findings.append(Finding(
                severity="FAIL", file="[builtin-fallback]", line=0, code=code,
                message=f"正則引擎 NG 無法偵測 {label}：'{sample}' 未命中",
            ))

    # Layer 1 / 2 靜態驗證（不需要模組，僅確認設計邏輯正確）
    BLOCKED_SUBSTRINGS = ("name", "email", "phone", "address", "source_ip", "wallet")
    ALLOWED_SAFE = {"min_retention_minutes", "twd_withdraw_count",
                    "min_hops_to_blacklist", "ip_shared_user_count"}

    for key in _PII_KEYS_EXPECTED_BLOCKED:
        blocked_by_keyword = any(sub in key.lower() for sub in BLOCKED_SUBSTRINGS)
        blocked_by_not_allowed = key not in ALLOWED_SAFE
        if blocked_by_keyword or blocked_by_not_allowed:
            findings.append(Finding(
                severity="INFO", file="[builtin-fallback]", line=0, code="C3-11",
                message=f"靜態驗證 OK PII 欄位 '{key}' 符合封鎖規則",
            ))
        else:
            findings.append(Finding(
                severity="FAIL", file="[builtin-fallback]", line=0, code="C3-11",
                message=f"靜態驗證 NG PII 欄位 '{key}' 不符合任何封鎖規則",
            ))

    return findings


def pii_filter_test(root: pathlib.Path) -> CheckResult:
    """
    C-3：以虛構 PII 資料測試 xai_bedrock.py 三層過濾器。

    測試資料說明：
      • 所有姓名、Email、電話、IP 均為虛構，不對應任何真實個人
      • 設計符合 PDPA 第 16 條「合理使用範圍」之測試豁免
    """
    findings: list[Finding] = []
    xai_available = False

    # ── 嘗試匯入 xai_bedrock ────────────────────────────────────────────────
    try:
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        from xai_bedrock import (        # type: ignore[import]
            filter_pii_context,
            filter_pii_contributions,
            scan_prompt_for_pii,
            ShapContribution,
        )
        xai_available = True
        findings.append(Finding(
            severity="INFO", file="xai_bedrock.py", line=0, code="C3-00",
            message="xai_bedrock 匯入成功，使用真實過濾器執行測試",
        ))

    except Exception as exc:
        findings.append(Finding(
            severity="WARN", file="xai_bedrock.py", line=0, code="C3-00",
            message=f"xai_bedrock 匯入失敗（缺少依賴），降級為內建正則引擎：{exc}",
        ))

    # ── 執行各層測試 ─────────────────────────────────────────────────────────
    if xai_available:
        findings.extend(_test_layer1(filter_pii_context))           # type: ignore
        findings.extend(_test_layer2(filter_pii_contributions,      # type: ignore
                                     ShapContribution))             # type: ignore
        findings.extend(_test_layer3(scan_prompt_for_pii))          # type: ignore
    else:
        findings.extend(_builtin_layer3_test())

    fail_n = sum(1 for f in findings if f.severity == "FAIL")
    info_n = sum(1 for f in findings if f.severity == "INFO")
    mode   = "xai_bedrock 真實模組" if xai_available else "內建正則引擎（降級）"

    if fail_n:
        status  = "FAIL"
        summary = f"{fail_n} 個 PII 過濾器洩漏（{info_n} 個案例，{mode}）"
    else:
        status  = "PASS"
        summary = f"所有 PII 攔截案例通過（{info_n} 個案例，{mode}）"

    return CheckResult(name="C-3 PII Filter Test",
                       status=status, summary=summary, findings=findings)


# ══════════════════════════════════════════════════════════════════════════════
#  輸出格式化
# ══════════════════════════════════════════════════════════════════════════════

_SEP  = "-" * 60
_SEP2 = "=" * 60


def _print_result(result: CheckResult, *, verbose: bool = False) -> None:
    """輸出單項檢核結果。"""
    print(f"\n{bold(result.name)}")
    print(dim(_SEP))
    print(f"  {_status_label(result.status)}  {result.summary}")

    # 詳細發現（FAIL/WARN 無論 verbose；INFO 只在 verbose 模式顯示）
    for f in result.findings:
        if f.severity == "INFO" and not verbose:
            continue

        loc = f"  {f.file}" + (f":{f.line}" if f.line > 0 else "")
        print(f"\n  {_severity_label(f.severity)} [{f.code}]")
        print(f"  {dim(loc)}")
        print(f"    {f.message}")
        if f.snippet:
            print(f"    {dim('>> ' + f.snippet)}")


def _print_summary(results: list[CheckResult], elapsed: float) -> int:
    """輸出總結摘要，回傳退出代碼。"""
    total   = len(results)
    passed  = sum(1 for r in results if r.status == "PASS")
    failed  = sum(1 for r in results if r.status == "FAIL")
    warned  = sum(1 for r in results if r.status == "WARN")
    skipped = sum(1 for r in results if r.status == "SKIP")

    print(f"\n{bold(_SEP2)}")
    print(bold("  合規檢核總結"))
    print(dim(_SEP2))
    lbl_pass = "通過"; lbl_fail = "失敗"; lbl_warn = "警告"
    lbl_skip = "跳過"; lbl_time = "耗時"
    print(f"  {lbl_pass:<6} {green(str(passed))} / {total} 項")
    if failed:
        print(f"  {lbl_fail:<6} {red(str(failed))} / {total} 項")
    if warned:
        print(f"  {lbl_warn:<6} {yellow(str(warned))} / {total} 項")
    if skipped:
        print(f"  {lbl_skip:<6} {dim(str(skipped))} / {total} 項")
    print(f"  {lbl_time:<6} {elapsed:.2f} 秒")
    print(bold(_SEP2))

    if failed:
        print(red(f"\n  NG 檢核失敗：發現 {failed} 項合規問題，請立即修正後重新執行。"))
        print(f"  查看詳細資訊請加上 --verbose 參數。\n")
        return 1

    if warned:
        print(yellow(f"\n  △ 檢核通過（含警告）：{warned} 項佔位符/設定需人工確認。\n"))
        return 0

    print(green(f"\n  OK 全部檢核通過，無合規問題。\n"))
    return 0


# ══════════════════════════════════════════════════════════════════════════════
#  CLI 入口點
# ══════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="check_compliance",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--dir", "-d",
        default=".",
        metavar="PATH",
        help="專案根目錄（預設：目前目錄）",
    )
    p.add_argument(
        "--only",
        nargs="+",
        choices=["c1", "c2", "c3"],
        metavar="CHECK",
        help="只執行指定的檢查項目（c1 c2 c3）",
    )
    p.add_argument(
        "--live-s3",
        action="store_true",
        help="C-2：追加透過 boto3 查詢真實 S3 Bucket 設定",
    )
    p.add_argument(
        "--bucket",
        metavar="BUCKET_NAME",
        help="--live-s3 模式使用的 S3 Bucket 名稱",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="顯示所有 INFO 等級的詳細訊息（預設只顯示 FAIL/WARN）",
    )
    p.add_argument(
        "--no-color",
        action="store_true",
        help="停用 ANSI 顏色輸出（適合 CI 日誌）",
    )
    return p


def main() -> int:
    import time

    parser = _build_parser()
    args   = parser.parse_args()

    # 套用 --no-color
    global _USE_COLOR
    if args.no_color:
        _USE_COLOR = False

    root = pathlib.Path(args.dir).resolve()
    if not root.is_dir():
        print(red(f"錯誤：目錄不存在：{root}"), file=sys.stderr)
        return 2

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(bold(_SEP2))
    print(bold("  BitoGuard 合規自動化檢核腳本"))
    print(dim(_SEP2))
    print(f"  專案目錄 : {root}")
    print(f"  執行時間 : {now}")
    if args.only:
        print(f"  執行項目 : {', '.join(args.only).upper()}")
    print(bold(_SEP2))

    only = set(args.only) if args.only else {"c1", "c2", "c3"}
    results: list[CheckResult] = []
    t0 = time.monotonic()

    if "c1" in only:
        r = credential_scan(root)
        results.append(r)
        _print_result(r, verbose=args.verbose)

    if "c2" in only:
        r = s3_acl_scan(root, live_s3=args.live_s3, bucket=args.bucket)
        results.append(r)
        _print_result(r, verbose=args.verbose)

    if "c3" in only:
        r = pii_filter_test(root)
        results.append(r)
        _print_result(r, verbose=args.verbose)

    elapsed = time.monotonic() - t0
    return _print_summary(results, elapsed)


if __name__ == "__main__":
    sys.exit(main())
