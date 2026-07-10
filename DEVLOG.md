# 📓 BitoGuard AML — 開發日誌(DEVLOG)

> **這份檔案是本專案的「工作記憶」。** 每次工作前先讀它、每天結束前寫它,
> 確保任何人(或任何 session)接手時都能立刻知道:目前進度到哪、上次做了什麼、
> 為什麼這樣做、還有什麼待辦。

---

## 📌 使用規範(務必遵守)

### 每次 session 開始工作前
1. **先完整閱讀本檔**(至少最近 3 天的紀錄 + 「目前狀態」與「待辦清單」兩節)。
2. 確認目前所在分支與最新進度,再開始動工,避免重複或衝突。

### 每天工作結束前(或每完成一項有意義的變更)
1. 在「每日紀錄」最上方新增當天條目(若當天已有條目則追加)。
2. 條目需包含:**日期、負責人/session、做了什麼、為什麼、影響的檔案、結果/驗證、待辦**。
3. 同步更新「目前狀態」與「待辦清單」兩節。
4. 連同程式碼一起 commit & push(commit message 可註明 `docs: 更新 DEVLOG`)。

### 撰寫原則
- **反時序**:最新的日期放最上面。
- **誠實**:失敗、跳過、未驗證的事也要寫清楚,不要只寫成功的部分。
- **可追溯**:提到變更時附上檔名;提到數據時附上來源檔。

---

## 🎯 專案目的(背景,長期不變)

BitoGuard AML 是**加密貨幣交易所反洗錢(AML)風險偵測系統**,針對 BitoPro 交易所
資料辨識高風險帳戶。核心 pipeline:資料擷取 → 特徵工程 → 圖譜跳數 → LGB+XGB 集成
模型 → 預測 → XAI 診斷 → Streamlit 儀表板。詳見 `README.md`。

---

## 🧭 目前狀態(每次更新請覆寫本節)

- **目前分支**:`claude/jolly-galileo-8a4adr`
- **部署決策**:採用 **Streamlit Community Cloud**(免費、直接連 GitHub)。
  - 已評估並放棄:Cloudflare Tunnel 自架(嫌開 VM 麻煩)、付費 Railway/Render。
  - 已知限制:Streamlit Cloud 免費版**閒置會休眠**,使用者已接受此限制。
- **可直接部署**:主程式 `app.py`,相依套件齊全,結果檔已附於 repo。
- **文件**:`README.md`(含 Mermaid 架構圖)、部署說明兩份、本 DEVLOG 已建立。
- **模型現況**:LGB(0.60)+XGB(0.40) 集成,OOF AUC ≈ 0.832,12,753 帳戶標記 501 個高風險。

---

## ✅ 待辦清單(每次更新請維護本節)

- [ ] 使用者到 <https://share.streamlit.io> 實際完成部署(需本人登入 GitHub 授權,AI 無法代做)。
- [ ] (可選)為 app 建一份精簡 requirements(僅 streamlit/plotly/pandas/numpy/sklearn)以加快建置。
- [ ] (可選)`feature_cache_v2.parquet` 被 gitignore,儀表板該分頁目前無資料;如需展示需另外提供。
- [ ] (未來)若日後仍在意休眠,再評估自架方案(相關設定檔 `Dockerfile`/`docker-compose.yml` 已備好)。

---

## 🗓️ 每日紀錄(最新在最上)

### 2026-06-14

**做了什麼**
- 撰寫並新增專案 `README.md`:專案簡介、功能特色、模型表現表格(數據取自
  `cv_report_lgb.json`)、專案結構、技術棧、本機與 Streamlit Cloud 部署步驟。
- 在 README 加入 **Mermaid 系統架構流程圖**(六階段:資料來源→擷取→特徵/圖譜→
  模型訓練→輸出/XAI→儀表板),各節點標註對應實際檔案。
- 建立本開發日誌 `DEVLOG.md` 與 `CLAUDE.md`(規範每個 session 先讀 DEVLOG、每日回寫)。

**為什麼**
- 專案原本沒有 README,新接手者難以理解;使用者要求補齊文件並建立每日進度追蹤機制。

**影響檔案**
- 新增:`README.md`、`DEVLOG.md`、`CLAUDE.md`

**結果 / 驗證**
- README 數據與 repo 內結果檔一致;Mermaid 圖 GitHub 可自動渲染。
- 已 commit 並 push 至 `claude/jolly-galileo-8a4adr`。

**待辦**
- 見上方「待辦清單」。

---

### 2026-06-13

**做了什麼**
- 釐清部署路線:確認本專案是 Python Streamlit App,**無法**直接部署到
  Cloudflare Pages/Workers。
- 新增 `DEPLOY_STREAMLIT_CLOUD.md`:Streamlit Community Cloud 部署說明。
- 新增自架方案設定檔:`Dockerfile`、`docker-compose.yml`、`.env.example`、
  `.dockerignore`、`DEPLOY_CLOUDFLARE_TUNNEL.md`(容器化 + Cloudflare Tunnel)。

**為什麼**
- 使用者最初希望「推到 Cloudflare」;因 Streamlit 需常駐 Python 伺服器,
  逐步評估後,最終決定改用 Streamlit Community Cloud(見「目前狀態」)。

**影響檔案**
- 新增:`Dockerfile`、`docker-compose.yml`、`.env.example`、`.dockerignore`、
  `DEPLOY_STREAMLIT_CLOUD.md`、`DEPLOY_CLOUDFLARE_TUNNEL.md`

**結果 / 驗證**
- `app.py` 通過 `py_compile` 語法檢查;資料載入檔案皆已 commit 且對缺檔有 fallback。
- 已 push 至 `claude/jolly-galileo-8a4adr`。

**備註**
- 自架方案雖已備好設定檔,但使用者決定暫不採用(嫌開 VM 麻煩)。設定檔保留供未來使用。

---

### 2026-04-16 以前(回填自 git 歷史,概要)

- 2026-04-16:新增 Dev Container 資料夾(`.devcontainer`)。
- 2026-04-03:多次 sidebar 版面調整(固定寬度 260px、position 定位、頁面切換淡入動畫等)。
- 更早:封面過場動畫、AWS/SageMaker pipeline、模型訓練與 XAI 等核心功能開發
  (詳見各 `*_GUIDE.md` 與 `git log`)。

---

<!-- 新增紀錄請複製以下範本,貼到「每日紀錄」最上方
### YYYY-MM-DD

**做了什麼**
-

**為什麼**
-

**影響檔案**
-

**結果 / 驗證**
-

**待辦**
-
-->
