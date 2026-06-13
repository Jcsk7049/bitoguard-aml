# 自架 + Cloudflare Tunnel 部署(免費、永不休眠)

把 Streamlit App 跑在一台免費 VM 上,前面用 **Cloudflare Tunnel** 接到你自己的
網域。沒有公開 IP、不用開防火牆 port、**不會休眠**,且全程走 Cloudflare(含 HTTPS)。

本 repo 已備好 `Dockerfile`、`docker-compose.yml`、`.env.example`,在 VM 上只需
三步:裝 Docker → 建 Tunnel 拿 token → `docker compose up -d`。

---

## 1. 準備一台免費 VM(建議 Oracle Cloud Always Free)

Oracle Cloud **Always Free** 的 ARM (Ampere A1) VM 真正永久免費、且常駐不睡:
最高 4 vCPU / 24 GB RAM。其他選擇:任何 VPS、家裡的機器、樹莓派皆可。

VM 開好後,SSH 進去安裝 Docker:

```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER    # 之後重新登入一次讓群組生效
```

## 2. 取得專案

```bash
git clone https://github.com/Jcsk7049/bitoguard-aml.git
cd bitoguard-aml
```

## 3. 在 Cloudflare 建立 Tunnel(拿 token)

> 前提:你的網域已加入 Cloudflare(DNS 由 Cloudflare 託管)。

1. 進 **Cloudflare Zero Trust** 後台:<https://one.dash.cloudflare.com>
   → **Networks** → **Tunnels** → **Create a tunnel**。
2. 連線方式選 **Cloudflared**,給 Tunnel 取個名字(例如 `bitoguard`),儲存。
3. 在 **Install connector** 那頁,複製指令裡的 **token**(很長的一串,
   `eyJ...` 開頭)。**只要 token,不用真的跑它給的安裝指令**——我們用 Docker 跑。
4. 切到 **Public Hostnames** 分頁 → **Add a public hostname**:
   - **Subdomain**:例如 `aml`
   - **Domain**:選你的網域 → 最終網址會是 `aml.你的網域`
   - **Type**:`HTTP`
   - **URL**:`app:8501`  ← 對應 docker-compose 裡的服務名與 port
   - 儲存。

## 4. 填入 token 並啟動

```bash
cp .env.example .env
nano .env            # 把 TUNNEL_TOKEN= 後面換成第 3 步複製的 token

docker compose up -d --build
```

第一次會建置映像(裝 Python 套件,數分鐘)。完成後:

```bash
docker compose ps          # 兩個容器 app / cloudflared 都應為 running/healthy
docker compose logs -f cloudflared   # 看到 "Registered tunnel connection" 即成功
```

打開 `https://aml.你的網域` 就能看到 App,HTTPS 由 Cloudflare 自動處理。

---

## 維運常用指令

```bash
docker compose logs -f app      # 看 App log
docker compose restart          # 重啟
docker compose down             # 停止
git pull && docker compose up -d --build   # 更新程式後重新部署
```

## 備註

- **WebSocket**:Streamlit 靠 WebSocket,Cloudflare Tunnel 預設支援,免額外設定。
- **不休眠**:VM 常駐 + `restart: unless-stopped`,VM 重開機後容器也會自動拉起。
- **安全**:App 的 8501 只在 Docker 內網(`expose`,非 `ports`),不對公網開放,
  外部唯一入口是 Cloudflare Tunnel。可再到 Zero Trust → **Access** 加一層登入保護。
- **token 機密**:`.env` 已被 `.gitignore` 排除,不會進版控;切勿把 token 貼進程式或 commit。
