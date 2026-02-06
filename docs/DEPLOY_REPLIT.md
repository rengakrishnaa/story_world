# Deploy Main API on Replit

Replit hosts the FastAPI app with **Python 3.10** (no card required).

---

## 1. Import from GitHub

1. Go to [Replit](https://replit.com) and sign in
2. **Create Repl** → **Import from GitHub**
3. Enter: `https://github.com/rengakrishnaa/story_world`
4. Click **Import**
5. Replit will clone the repo and detect `.replit` config

---

## 2. Environment Variables (Secrets)

In Replit: **Tools** (wrench) → **Secrets** (or left sidebar **Secrets**)

Add:

| Key | Value |
|-----|-------|
| `REDIS_URL` | Your Upstash Redis URL |
| `DATABASE_URL` | `sqlite:///./local.db` |
| `S3_ENDPOINT` | R2 endpoint |
| `S3_BUCKET` | Bucket name |
| `S3_ACCESS_KEY` | R2 access key |
| `S3_SECRET_KEY` | R2 secret key |
| `GEMINI_API_KEY` | Gemini API key |
| `JOB_QUEUE` | `storyworld:gpu:jobs` |
| `RESULT_QUEUE` | `storyworld:gpu:results` |

**Do not set** `SERVERLESS` – Replit runs a long-lived process; ResultConsumer loop will run.

---

## 3. Run and Deploy

1. Click **Run** – Replit will `pip install` and start uvicorn
2. Open the **Webview** tab to test
3. Click **Deploy** (top right) → **Deploy**
4. Copy your Replit URL: `https://storyworld-xxx.replit.app` (or your repl name)

---

## 4. Connect Netlify Frontend

In `netlify.toml`, replace the backend URL with your Replit URL:

```toml
to = "https://YOUR_REPL_NAME.YOUR_USERNAME.repl.co/episodes"
```

Example: `https://storyworld-abc123.yourname.repl.co`

Then push and let Netlify redeploy.

---

## 5. Port / Webview

Replit binds the app to port 8000. The **Webview** and **Deploy** URL expose it. No extra config needed.

---

## 6. Build Failed / No Logs

If deployment shows "build failed" with no logs:

1. **Run locally first** – Click **Run** in Replit. Check the **Console** for pip/build errors.
2. **Lighter deps** – We use `requirements-replit.txt` (no CLIP) to avoid git clone and long installs.
3. **Retry deploy** – Push changes, then **Deploy** again.
4. **Replit status** – Check [status.replit.com](https://status.replit.com) for outages.

---

## Stack (all on Replit)

| Component | Service |
|-----------|---------|
| Main API | Replit (this) |
| Frontend | Netlify |
| Bridge | GitHub Actions |
| GPU Worker | RunPod Serverless |
| Redis | Upstash |
| Storage | Cloudflare R2 |
