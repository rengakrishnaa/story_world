# Deploy Main API on Fly.io

Fly.io hosts the FastAPI app with **Python 3.10**. Free tier available.

---

## 1. Install Fly CLI

```powershell
# Windows (PowerShell)
iwr https://fly.io/install.ps1 -useb | iex
```

Or: https://fly.io/docs/hands-on/install-flyctl/

---

## 2. Login & Launch (First Time)

```bash
cd c:\Users\KRISH\Desktop\LEARN\Vidme\story_world

fly auth login
fly launch
```

When prompted:
- **App name:** `story-world` (or accept suggestion)
- **Region:** `ams` (Amsterdam)
- **Postgres:** No (we use SQLite in /tmp; ephemeral)
- **Redis:** No (we use Upstash)

---

## 3. Set Secrets (Required)

```bash
fly secrets set REDIS_URL="rediss://default:YOUR_PASSWORD@YOUR_UPSTASH_HOST:6379"
fly secrets set GEMINI_API_KEY="your-gemini-api-key"
fly secrets set S3_ENDPOINT="https://xxx.r2.cloudflarestorage.com"
fly secrets set S3_BUCKET="your-bucket"
fly secrets set S3_ACCESS_KEY="your-r2-access-key"
fly secrets set S3_SECRET_KEY="your-r2-secret-key"
fly secrets set JOB_QUEUE="storyworld:gpu:jobs"
fly secrets set RESULT_QUEUE="storyworld:gpu:results"
```

Optional: `USE_MOCK_PLANNER=true`, `USE_OBSERVER_IN_PRODUCTION=true`

**Note:** `DATABASE_URL` is set in `fly.toml` to `sqlite:////tmp/storyworld.db`. Override with a secret only if using Postgres.

---

## 4. Deploy

```bash
fly deploy
```

---

## 5. Check Status

```bash
fly status
fly logs
fly open
```

`fly open` opens `https://story-world.fly.dev` in browser.

---

## 6. Netlify

`netlify.toml` is already configured to proxy API calls to `https://story-world.fly.dev`. Push to GitHub; Netlify will redeploy.

---

## 7. Troubleshooting

| Issue | Fix |
|-------|-----|
| OOM / crash | `fly scale memory 512` |
| Build fails | Check `Dockerfile.fly`; ensure `.dockerignore` does NOT exclude `runtime/`, `static/`, `main.py` |
| Redis fail | Verify `REDIS_URL` secret; use `rediss://` for TLS |
