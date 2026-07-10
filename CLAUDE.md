# CLAUDE.md — 給 AI 助手 / 開發者的專案守則

## ⚠️ 最重要:每次工作前先讀 DEVLOG

**開始任何工作之前,務必先完整閱讀 [`DEVLOG.md`](DEVLOG.md)**(開發日誌),
以掌握:目前進度、上次做了什麼、部署決策、待辦清單。這是本專案的「工作記憶」,
不讀就動工容易重複或推翻既有決定。

## ⚠️ 每天結束前必須回寫 DEVLOG

每天(或每完成一項有意義的變更)**都要在 `DEVLOG.md` 的「每日紀錄」最上方新增當天條目**,
並更新其中的「目前狀態」與「待辦清單」兩節,連同程式碼一起 commit & push。
撰寫格式與規範見 `DEVLOG.md` 開頭的「使用規範」。

---

## 專案速覽

- **性質**:加密貨幣交易所反洗錢(AML)風險偵測系統(BitoPro 資料)。
- **主程式**:`app.py`(Streamlit 儀表板,亦為部署進入點)。
- **Pipeline 調度**:`main_pipeline.py`。
- **模型**:LightGBM(0.60)+ XGBoost(0.40)集成。
- **完整說明**:見 [`README.md`](README.md)。

## 開發慣例

- **開發分支**:`claude/jolly-galileo-8a4adr`(除非另有指示,勿直接推 `main`)。
- **部署目標**:Streamlit Community Cloud(見 DEVLOG「目前狀態」)。
- **語言**:文件與 commit message 以繁體中文為主,與既有風格一致。
- **機密**:切勿 commit `.env` 或任何 token(已在 `.gitignore` 排除)。

## 常用指令

```bash
# 本機啟動儀表板
streamlit run app.py

# 執行完整 pipeline
python main_pipeline.py --start-from feature --csv-dir ./data
```
