# 部署到 Streamlit Community Cloud

本專案是 Python **Streamlit** App,**無法**直接部署到 Cloudflare Pages / Workers
(它們不支援常駐的 Python 伺服器)。最簡單的免費託管方式是
**Streamlit Community Cloud**。本 repo 已設定為可直接部署的狀態。

## 部署前確認(已完成)

- 主程式:`app.py`
- 相依套件:`requirements.txt`(已涵蓋 app 所需的 streamlit / plotly / pandas / numpy / scikit-learn)
- 資料檔:`cv_report_lgb.json`、`submission_with_prob.csv`、`oof_predictions.csv`、
  `feature_importance.csv` 皆已 commit 進 repo。
- `feature_cache_v2.parquet` 被 `.gitignore` 排除,但 app 對缺檔有 fallback,不影響執行。

## 部署步驟(在瀏覽器操作,約 2 分鐘)

1. 前往 <https://share.streamlit.io> 並用 GitHub 帳號登入。
2. 點 **Create app** → **Deploy a public app from GitHub**。
3. 填入:
   - **Repository**:`Jcsk7049/bitoguard-aml`
   - **Branch**:要部署的分支(例如 `main`,或本次的開發分支)
   - **Main file path**:`app.py`
4. (可選)在 **Advanced settings** 選擇 Python 版本(建議 3.11)。
5. 點 **Deploy**。首次建置會安裝 `requirements.txt`,完成後即取得一個
   `https://<your-app>.streamlit.app` 的公開網址。

## 之後接 Cloudflare(可選)

拿到 `*.streamlit.app` 網址後,若想用自己的網域:

- Streamlit Community Cloud 免費版**不支援自訂網域 / CNAME**。
- 若一定要用自有網域 + Cloudflare,需改用可自架的平台(Render / Railway /
  自架 VM + Cloudflare Tunnel),再於 Cloudflare DNS 指向該服務。

## 注意

- `requirements.txt` 內含 `lightgbm`、`matplotlib`、`pyarrow`,是給訓練 /
  pipeline 腳本用的;`app.py` 本身不需要,但保留不影響部署(建置時間略長)。
- 若想加快建置,可另建一份精簡的 app 專用 requirements(僅 streamlit / plotly /
  pandas / numpy / scikit-learn)。
