# Deploy Main App (Render) + Frontend (Netlify)

Step-by-step deployment for StoryWorld using Render (free) and Netlify (free).

---

## 1. Deploy Main App on Render

### Option A: Blueprint (render.yaml)

1. Push `render.yaml` to your repo.
2. Go to [Render Dashboard](https://dashboard.render.com/) → **New +** → **Blueprint**.
3. Connect your GitHub repo, select the repo.
4. Render will detect `render.yaml`. Click **Apply**.
5. Add **Environment Variables** (Dashboard → your service → Environment):
   - `REDIS_URL` (Upstash URL)
   - `S3_ENDPOINT`, `S3_BUCKET`, `S3_ACCESS_KEY`, `S3_SECRET_KEY`
   - `GEMINI_API_KEY`
   - `NANOBANANA_API_KEY` (if using Veo)
6. Wait for deploy. Note your service URL: `https://storyworld-api.onrender.com` (or similar).

### Option B: Manual

1. **New +** → **Web Service**.
2. Connect GitHub, select repo.
3. Settings:
   - **Name**: `storyworld-api`
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.base.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Instance Type**: Free
4. Add env vars (same as above).
5. Deploy. Copy the service URL.

### Notes

- **SQLite**: Data is ephemeral on Render free tier (lost on restart). Acceptable for demos.
- **Cold start**: Free tier spins down after ~15 min; first request may take 30–60 s.
- **CLIP build**: If `pip install` fails on the CLIP git dependency, ensure the build has git. Render's default Python environment includes git. If it fails, try adding `nixpacksPackages = [pkgs.git]` in a `render.yaml` config or use a Docker deploy.

---

## 2. Deploy Frontend on Netlify

### Step 1: Update netlify.toml

Replace `YOUR_RENDER_URL` with your Render URL (no trailing slash):

```toml
# Example:
to = "https://storyworld-api.onrender.com/episodes"
```

Use Find & Replace: `YOUR_RENDER_URL` → `storyworld-api.onrender.com` (or your actual URL).

### Step 2: Deploy

1. Go to [Netlify](https://app.netlify.com/) → **Add new site** → **Import an existing project**.
2. Connect GitHub, select your repo.
3. Settings:
   - **Build command**: (leave empty or `true`)
   - **Publish directory**: `.` (root)
4. Click **Deploy site**.

### Step 3: Custom Domain (optional)

Netlify → Site settings → Domain management → Add custom domain.

---

## 3. Verify

| Check | How |
|-------|-----|
| API health | `curl https://YOUR_RENDER_URL/phase-status` |
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
| `S3_BUCKET` | Yes | e.g. storyworld-artifacts |
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
