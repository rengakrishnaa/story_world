# Deploy Main API on Fly.io

**For future deployment.** Current setup runs locally: [LOCAL_SETUP.md](LOCAL_SETUP.md)

When scaling: Fly.io hosts the FastAPI app with Python 3.10.

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

## 3. Secrets

```bash
fly secrets set REDIS_URL="rediss://..."
fly secrets set GEMINI_API_KEY="..."
fly secrets set S3_ENDPOINT="https://..."
fly secrets set S3_BUCKET="..."
fly secrets set S3_ACCESS_KEY="..."
fly secrets set S3_SECRET_KEY="..."
fly secrets set JOB_QUEUE="storyworld:gpu:jobs"
fly secrets set RESULT_QUEUE="storyworld:gpu:results"
```

We also set `USE_MOCK_PLANNER` and `USE_OBSERVER_IN_PRODUCTION` when needed.

`DATABASE_URL` is set in `fly.toml` to `sqlite:////tmp/storyworld.db`. We override it with a secret when using Postgres.

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
