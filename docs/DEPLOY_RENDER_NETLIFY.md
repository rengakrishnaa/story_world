# Deploy Main App (Render) + Frontend (Netlify)

**For future deployment.** Current setup runs locally: [LOCAL_SETUP.md](LOCAL_SETUP.md)

When scaling: StoryWorld on Render (free tier) + Netlify (free).

---

## 1. Deploy Main App on Render

### Option A: Blueprint (recommended)

1. Push `render.yaml` to the repo.
2. Go to [Render Dashboard](https://dashboard.render.com/) → **New +** → **Blueprint**.
3. Connect GitHub, select repo.
4. Render detects `render.yaml`. Click **Apply**.
5. Add **Environment Variables** (Dashboard → storyworld-api → Environment):

   | Key | Value |
   |-----|-------|
   | `REDIS_URL` | Upstash Redis URL |
   | `S3_ENDPOINT` | R2 endpoint |
   | `S3_BUCKET` | Bucket name |
   | `S3_ACCESS_KEY` | R2 access key |
   | `S3_SECRET_KEY` | R2 secret key |
   | `GEMINI_API_KEY` | Gemini API key |
   | `JOB_QUEUE` | `storyworld:gpu:jobs` |
   | `RESULT_QUEUE` | `storyworld:gpu:results` |

6. Wait for deploy. URL: `https://storyworld-api.onrender.com` (or `https://storyworld-api-XXXX.onrender.com` if name taken).

### Option B: Manual

1. **New +** → **Web Service**.
2. Connect GitHub, select repo.
3. **Name**: `storyworld-api`
4. **Build Command**: `pip install -r requirements-replit.txt`
5. **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
6. **Instance Type**: Free
7. Add env vars (same as above).

### Notes

- **Deps**: Uses `requirements-replit.txt` (no CLIP/mediapipe/opencv) to avoid build timeout.
- **SQLite**: `/tmp/storyworld.db` – ephemeral (lost on restart).
- **Cold start**: Free tier spins down after ~15 min; first request may take 30–60 s.

---

## 2. Deploy Frontend on Netlify

`netlify.toml` is pre-configured for `https://storyworld-api.onrender.com`.

If the Render URL differs, we update the proxy `to` URLs in `netlify.toml`.

1. [Netlify](https://app.netlify.com/) → **Add new site** → **Import from Git**.
2. Connect GitHub, select repo.
3. **Build command**: `true` (or empty)
4. **Publish directory**: `.`
5. **Deploy**.

### Step 3: Custom Domain (optional)

Netlify → Site settings → Domain management → Add custom domain.

---

## 3. Verify

| Check | How |
|-------|-----|
| API health | `curl https://storyworld-api.onrender.com/phase-status` |
| Frontend | Open Netlify URL |
| Create sim | Frontend → New Run → fill goal → Initialize |
| Bridge | GitHub Actions runs every 3 min; jobs flow Redis → RunPod → results |

---

## 4. Env Vars Summary

### Render (main app)

| Variable | Required | Notes |
|----------|----------|-------|
| `REDIS_URL` | Yes | Upstash Redis URL |
| `S3_ENDPOINT` | Yes | R2 endpoint |
| `S3_BUCKET` | Yes | Bucket name |
| `S3_ACCESS_KEY` | Yes | R2 token |
| `S3_SECRET_KEY` | Yes | R2 token |
| `GEMINI_API_KEY` | Yes | For observer/planner |
| `NANOBANANA_API_KEY` | If using Veo | NanoBanana API |
| `JOB_QUEUE` | No | Default: storyworld:gpu:jobs |
| `RESULT_QUEUE` | No | Default: storyworld:gpu:results |

### Netlify

- No env vars needed if `netlify.toml` has the correct Render URL.

---

## 5. Cost

| Component | Cost |
|-----------|------|
| Render (free tier) | $0 |
| Netlify | $0 |
| Upstash Redis | $0 |
| Cloudflare R2 | $0 |
| RunPod Serverless | Pay per use |
| GitHub Actions (bridge) | $0 |
