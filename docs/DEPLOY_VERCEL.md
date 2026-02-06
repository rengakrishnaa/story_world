# Deploy Main Server on Vercel

Vercel hosts the main API with **zero config** for FastAPI. No credit card required.

---

## 1. Prerequisites

- [Vercel](https://vercel.com) account (free, no card)
- GitHub repo pushed
- Env vars ready (Redis, R2, Gemini)

---

## 2. Deploy via Dashboard

### Step 1: Import Project

1. Go to [vercel.com/new](https://vercel.com/new)
2. **Import** your GitHub repository
3. Select the `story_world` repo
4. Click **Deploy** (Vercel auto-detects FastAPI)

### Step 2: Environment Variables

Before or after first deploy, add env vars in **Settings → Environment Variables**:

| Variable | Value | Required |
|----------|-------|----------|
| `REDIS_URL` | Upstash Redis URL | Yes |
| `DATABASE_URL` | `sqlite:////tmp/storyworld.db` | Yes (use `/tmp` on serverless) |
| `S3_ENDPOINT` | R2 endpoint | Yes |
| `S3_BUCKET` | Bucket name | Yes |
| `S3_ACCESS_KEY` | R2 access key | Yes |
| `S3_SECRET_KEY` | R2 secret key | Yes |
| `GEMINI_API_KEY` | Gemini API key | Yes |
| `SERVERLESS` | `true` | Yes (enables cron-based result processing) |
| `JOB_QUEUE` | `storyworld:gpu:jobs` | No |
| `RESULT_QUEUE` | `storyworld:gpu:results` | No |

**Important:** `DATABASE_URL=sqlite:////tmp/storyworld.db` – Vercel filesystem is read-only except `/tmp`. Data is ephemeral (reset on cold start).

### Step 3: Redeploy

After adding env vars, trigger a new deploy: **Deployments → ⋮ → Redeploy**.

---

## 3. Cron (Result Processing)

`vercel.json` configures a cron that hits `/internal/process-results` every minute. This processes GPU results from Redis (since serverless cannot run a background loop).

- Cron runs automatically after deploy
- No extra setup needed

---

## 4. Get Your URL

After deploy, Vercel gives you a URL like:
```
https://story-world-xxx.vercel.app
```

Use this as the API base for Netlify proxy (update `netlify.toml`).

---

## 5. Connect Netlify Frontend

In `netlify.toml`, replace `YOUR_RENDER_URL` with your Vercel URL (no trailing slash):

```toml
to = "https://story-world-xxx.vercel.app/episodes"
```

Redeploy Netlify after updating.

---

## 6. Limitations

| Item | Note |
|------|------|
| **Function timeout** | 10s (Hobby) / 60s (Pro). Long `/compose` calls may timeout. |
| **Bundle size** | 250MB max. CLIP + opencv can be large; build may fail. |
| **SQLite** | Ephemeral in `/tmp`. Data lost on cold start. |
| **Cold starts** | First request after idle can take several seconds. |

If the build fails on CLIP or size, consider removing optional deps or using a lighter requirements set.

---

## 7. Local Test

```bash
vercel dev
```

Minimizes local/server differences.
