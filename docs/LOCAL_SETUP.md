# Local Setup (Current Architecture)

StoryWorld runs locally with GPU jobs handled by RunPod Serverless. No deployment.

---

## Architecture

| Component | Where | What |
|-----------|-------|------|
| **Main API** | Your machine | `uvicorn main:app` |
| **GPU Worker** | RunPod Serverless | Already configured |
| **Bridge** | GitHub Actions | Redis ↔ RunPod (every 3 min) |
| **Redis** | Upstash | Job and result queues |
| **R2** | Cloudflare | Video storage |

---

## Prerequisites

- Python 3.10+
- `.env` with REDIS_URL, GEMINI_API_KEY, S3_*, etc.

---

## 1. Environment

```bash
cp env.example .env
```

Edit `.env`:

| Variable | Required | Description |
|----------|----------|-------------|
| `REDIS_URL` | Yes | Upstash Redis URL |
| `GEMINI_API_KEY` | Yes | For observer and planner |
| `S3_ENDPOINT` | Yes | R2 endpoint |
| `S3_BUCKET` | Yes | Bucket name |
| `S3_ACCESS_KEY` | Yes | R2 access key |
| `S3_SECRET_KEY` | Yes | R2 secret key |
| `DEFAULT_BACKEND` | No | `veo` or `svd` (default: veo) |
| `JOB_QUEUE` | No | `storyworld:gpu:jobs` |
| `RESULT_QUEUE` | No | `storyworld:gpu:results` |

---

## 2. Install

```bash
pip install -r requirements-replit.txt
```

---

## 3. Run

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000

---

## 4. Bridge (GitHub Actions)

The bridge runs every 3 minutes. Ensure these GitHub secrets are set:

- `REDIS_URL`
- `RUNPOD_API_KEY`
- `RUNPOD_ENDPOINT_ID`

See [bridge-cron.yml](../.github/workflows/bridge-cron.yml).

---

## 5. RunPod Serverless

One-time setup: [SETUP_SERVERLESS.md](SETUP_SERVERLESS.md)

---

## Verification

1. `curl http://localhost:8000/health` → `{"status":"ok"}`
2. Create a simulation in the UI
3. Bridge runs every 3 min; jobs flow to RunPod; results return via Redis; ResultConsumer updates DB

---

## Future Deployment

When we scale and get users, we will deploy the main API (Render, Fly, etc.). See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md).
