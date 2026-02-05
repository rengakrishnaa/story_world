# StoryWorld Deployment Guide

## Overview

StoryWorld requires:
1. **Main Server** — FastAPI, SQLite, ResultConsumer
2. **Redis** — Upstash or compatible (job + result queues)
3. **GPU Worker** — RunPod or similar (renders video, uploads to R2)
4. **Cloudflare R2** — Video storage (or S3-compatible)

---

## Main Server Setup

### 1. Environment

```bash
# Copy env.example to .env
cp env.example .env

# Required:
REDIS_URL=rediss://default:YOUR_PASSWORD@your-upstash-host:6379
DATABASE_URL=sqlite:///./local.db
GEMINI_API_KEY=your_gemini_api_key

# R2 (for observer to access videos; worker uploads)
S3_ENDPOINT=https://xxx.r2.cloudflarestorage.com
S3_BUCKET=storyworld-artifacts
S3_ACCESS_KEY=...
S3_SECRET_KEY=...
S3_REGION=auto
```

### 2. Install Dependencies

```bash
pip install -r requirements.base.txt
```

### 3. Start Server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## GPU Worker Setup

### 1. Environment

Worker needs same `.env` with:
- `REDIS_URL`, `JOB_QUEUE`, `RESULT_QUEUE`
- `S3_ENDPOINT`, `S3_BUCKET`, `S3_ACCESS_KEY`, `S3_SECRET_KEY`
- `DEFAULT_BACKEND=veo` (or svd)
- `VEO_FALLBACK_BACKEND=svd`
- `USE_DIFFUSION=true` (for credit-exhausted fallback)

### 2. Install Dependencies

```bash
pip install -r requirements.gpu.txt
```

### 3. Verify GPU

```bash
nvidia-smi
```

### 4. Start Worker

```bash
python worker.py
```

Worker polls `storyworld:gpu:jobs`, renders via selected backend, uploads to R2, pushes to `storyworld:gpu:results`.

---

## Queue Names

**Critical:** Main server and worker must use the same queue names.

| Variable | Default |
|----------|---------|
| `JOB_QUEUE` | `storyworld:gpu:jobs` |
| `RESULT_QUEUE` | `storyworld:gpu:results` |

---

## Clearing Old Jobs

If jobs have wrong backend (e.g., stub), clear the queue before new simulations:

```bash
python clear_job_queue.py
```

---

## Docker

- **Dockerfile.cpu** — Main server
- **Dockerfile.gpu** — GPU worker

---

## Verification

1. Main server: `curl http://localhost:8000/diagnostics`
2. Redis: `python debug_queue_status.py`
3. Run simulation: `POST /simulate?world_id=default&goal=...`
4. Check worker logs for `backend=veo` or `backend=svd`
