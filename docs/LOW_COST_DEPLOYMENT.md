# StoryWorld Low-Cost / Free Deployment Guide

**For future deployment.** Current setup runs locally: [LOCAL_SETUP.md](LOCAL_SETUP.md)

When we scale: Deploy StoryWorld with minimal cost. GPU runs only when a simulation is requested (serverless).

---

## Architecture Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────────┐
│  Netlify        │     │  Main Server    │     │  Upstash Redis      │
│  (Frontend)     │────►│  (Render/Railway│────►│  (Free tier)        │
│  FREE           │     │   /Fly.io)      │     │  FREE               │
└─────────────────┘     └────────┬────────┘     └──────────┬──────────┘
                                 │                         │
                                 │                         │ blpop
                                 │                         ▼
                                 │               ┌─────────────────────┐
                                 │               │  Bridge             │
                                 │               │  (Render/Fly.io)    │
                                 │               │  FREE tier          │
                                 │               └──────────┬──────────┘
                                 │                          │ invoke
                                 │                          ▼
                                 │               ┌─────────────────────┐
                                 │               │  RunPod Serverless  │
                                 │               │  (Pay per second)   │
                                 │               │  ~$0.0002/sec T4    │
                                 │               └──────────┬──────────┘
                                 │                          │ upload
                                 ▼                          ▼
                        ┌─────────────────────────────────────────────┐
                        │  Cloudflare R2 (already set)                 │
                        └─────────────────────────────────────────────┘
```

---

## 1. RunPod Serverless Setup (One-Time, Persistent Config)

### Step 1: Create Endpoint

1. Go to [RunPod Console](https://www.runpod.io/console/serverless)
2. **Create Endpoint**
3. Choose GPU (T4 cheapest ~$0.0002/sec, L4 for faster)
4. Use Docker image: build from `Dockerfile.serverless` and push to Docker Hub or use RunPod's build

### Step 2: Set Environment Variables (Persist Forever)

**Critical:** RunPod Console > Endpoint > **Settings** > **Environment Variables**

We add these once. They persist with the endpoint.

| Variable | Value | Required |
|----------|-------|----------|
| `S3_ENDPOINT` | R2 endpoint URL | Yes |
| `S3_BUCKET` | storyworld-artifacts | Yes |
| `S3_ACCESS_KEY` | R2 access key | Yes |
| `S3_SECRET_KEY` | R2 secret key | Yes |
| `S3_REGION` | auto | Yes |
| `DEFAULT_BACKEND` | veo or svd | Yes |
| `VEO_FALLBACK_BACKEND` | svd | No |
| `USE_DIFFUSION` | true | For credit fallback |
| `GEMINI_API_KEY` | Gemini API key | If using Veo |

Copy from `runpod-template.env`.

### Step 3: Note Endpoint ID and API Key

- **Endpoint ID:** From endpoint URL or Settings
- **API Key:** RunPod Console > Settings > API Keys

---

## 2. Bridge Service (Redis → RunPod)

The bridge polls Redis for jobs and invokes RunPod Serverless. No GPU needed—runs on free tier.

### Option A: Render.com (Free Tier)

**Option A1: Background Worker** (may spin down on free tier)

1. Create `render.yaml`:

```yaml
services:
  - type: worker
    name: storyworld-bridge
    env: python
    buildCommand: pip install -r requirements.bridge.txt
    startCommand: python bridge_serverless.py
    envVars:
      - key: REDIS_URL
        sync: false
      - key: RUNPOD_API_KEY
        sync: false
      - key: RUNPOD_ENDPOINT_ID
        sync: false
```

**Option A2: Cron Job** (runs every 2 min, processes one job per run)

1. Create a Cron Job in Render: schedule `*/2 * * * *` (every 2 min)
2. Build: `pip install -r requirements.bridge.txt`
3. Command: `BRIDGE_CRON_MODE=true python bridge_serverless.py`
4. Add env vars. Each cron run processes one job (if any) and exits.

### Option B: Fly.io (Free Tier)

```bash
# Dockerfile.bridge
FROM python:3.11-slim
WORKDIR /app
COPY requirements.bridge.txt .
RUN pip install -r requirements.bridge.txt
COPY bridge_serverless.py .
CMD ["python", "bridge_serverless.py"]
```

```bash
fly launch
fly secrets set REDIS_URL=... RUNPOD_API_KEY=... RUNPOD_ENDPOINT_ID=...
fly deploy
```

### Option C: Railway ($5 Free Credit)

1. New Project > Deploy from GitHub
2. Add `requirements.bridge.txt`, `bridge_serverless.py`
3. Set env vars
4. Deploy

### Bridge Env Vars

| Variable | Description |
|----------|-------------|
| `REDIS_URL` | Upstash Redis URL |
| `JOB_QUEUE` | storyworld:gpu:jobs |
| `RESULT_QUEUE` | storyworld:gpu:results |
| `RUNPOD_API_KEY` | RunPod API key |
| `RUNPOD_ENDPOINT_ID` | RunPod serverless endpoint ID |

---

## 3. Main Server (Render / Railway / Fly.io)

### Render Free Tier

- **Web Service** for FastAPI
- Env: `REDIS_URL`, `GEMINI_API_KEY`, `DATABASE_URL`, `S3_*`
- Free tier: spins down after 15 min; cold start ~30s

### Railway

- Deploy `main.py` with uvicorn
- Add Redis (or use Upstash)
- $5 free credit/month

### Fly.io

```bash
fly launch
fly secrets set REDIS_URL=... GEMINI_API_KEY=... S3_ENDPOINT=... S3_BUCKET=... S3_ACCESS_KEY=... S3_SECRET_KEY=...
fly deploy
```

---

## 4. Frontend (Netlify)

1. Build: none (static)
2. Publish directory: `static/`
3. Rewrite: `/* /index.html` (for SPA routing if needed)
4. Env: `VITE_API_URL` or similar pointing to main server URL

When the frontend calls the API directly, we set `API_BASE_URL` to the main server URL.

---

## 5. Summary: No More Manual Env Setup

| Component | Config Persistence |
|-----------|--------------------|
| **RunPod Serverless** | Env vars set in Console > Endpoint > Settings. Saved with endpoint. **Set once.** |
| **Bridge** | Env vars in Render/Railway/Fly dashboard. Saved with service. |
| **Main Server** | Env vars in hosting dashboard. |
| **Netlify** | Env vars in Netlify dashboard. |

**You never set RunPod env vars "each time before run"—they persist with the endpoint.**

---

## 6. Cost Estimate

| Component | Cost |
|-----------|------|
| Netlify | Free |
| Upstash Redis | Free (10K cmds/day) |
| Cloudflare R2 | Free tier generous |
| Render free | Free (with limits) |
| Fly.io free | Free (limited resources) |
| RunPod Serverless | ~$0.0002/sec (T4). 60s job ≈ $0.012. 100 sims/month ≈ $1.20 |

**Total: ~$0–5/month** for light usage.

---

## 7. Build and Push Docker Image for RunPod

```bash
# Build
docker build -f Dockerfile.serverless -t <dockerhub>/storyworld-worker:serverless .

# Push
docker push <dockerhub>/storyworld-worker:serverless
```

In RunPod we create an endpoint with the image `<dockerhub>/storyworld-worker:serverless`.
