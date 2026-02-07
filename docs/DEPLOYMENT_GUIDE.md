# StoryWorld Deployment Guide

**For future use.** Current setup is local: [LOCAL_SETUP.md](LOCAL_SETUP.md)

When we upgrade and get users, deploy the main API to Render, Fly, etc. GPU stays on RunPod Serverless.

---

## Current vs Deployed

| Component | Current | Future (Deployed) |
|-----------|---------|-------------------|
| Main API | Local (uvicorn) | Render / Fly / Replit |
| GPU Worker | RunPod Serverless | RunPod Serverless |
| Bridge | GitHub Actions | GitHub Actions |
| Frontend | Served by main API | Netlify + proxy to API |

---

## Deployment Options (Future)

| Platform | Config File | Doc |
|----------|-------------|-----|
| Render | `render.yaml` | [DEPLOY_RENDER_NETLIFY.md](DEPLOY_RENDER_NETLIFY.md) |
| Fly.io | `fly.toml` | [DEPLOY_FLY.md](DEPLOY_FLY.md) |
| Replit | `.replit` | [DEPLOY_REPLIT.md](DEPLOY_REPLIT.md) |
| Zeabur | `Dockerfile.zeabur`, `zbpack.json` | [DEPLOY_ZEABUR.md](DEPLOY_ZEABUR.md) |
| Vercel | `vercel.json` | [DEPLOY_VERCEL.md](DEPLOY_VERCEL.md) |

---

## Main Server (When Deploying)

- **Build**: `pip install -r requirements-replit.txt`
- **Start**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- **Env vars**: REDIS_URL, GEMINI_API_KEY, S3_*, JOB_QUEUE, RESULT_QUEUE, DATABASE_URL

---

## Queue Names

| Variable | Default |
|----------|---------|
| `JOB_QUEUE` | `storyworld:gpu:jobs` |
| `RESULT_QUEUE` | `storyworld:gpu:results` |

---

## Clearing Old Jobs

```bash
python clear_job_queue.py
```
