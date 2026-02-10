# StoryWorld — Testing Instructions (Hackathon)

This document explains how to run StoryWorld on your local machine and why the live demo link may not show results.

---

## Why the Live Link May Not Show Results

We provide a deployed link (e.g. `https://storyworld--rengakrishnaa.replit.app`) for convenience, but it often does **not** complete simulations and show results. Here is why:

1. **RunPod GPU workers are unreliable in this setup**
   - Video rendering runs on RunPod Serverless. Workers can fail due to cold starts (downloading ~5GB models), timeouts, or GPU availability.
   - We did not have time to train or tune the RunPod setup for stable demos.

2. **External service limits**
   - Gemini API has rate limits and quotas. When exhausted, the system falls back to local SDXL, which requires model download and can timeout.
   - Redis, R2, and RunPod must all be up and correctly configured. Any failure in this chain means no video or no result.

3. **No persistent infra for the demo**
   - The live deployment is not kept warm or monitored 24/7. RunPod workers scale to zero, so the first request after idle often fails or times out.

**Therefore:** For a reliable demo, run StoryWorld locally on your machine. The full stack works when all components (main API, Redis, worker) run in a controlled environment.

---

## Step-by-Step: Run StoryWorld Locally

### Prerequisites

- **Python 3.10+**
- **Git**
- **Optional but recommended:** NVIDIA GPU (for real video rendering without RunPod)

### Step 1: Clone the repository

```bash
git clone https://github.com/rengakrishnaa/story_world.git
cd story_world
```

### Step 2: Create environment file

Create a `.env` file in the project root with the following variables. You can sign up for free tiers to get credentials.

| Variable | Required | Where to get it |
|----------|----------|-----------------|
| `REDIS_URL` | Yes | [Upstash](https://upstash.com/) — create a Redis database, copy the `rediss://` URL |
| `GEMINI_API_KEY` | Yes | [Google AI Studio](https://aistudio.google.com/apikey) — create an API key |
| `S3_ENDPOINT` | Yes | [Cloudflare R2](https://developers.cloudflare.com/r2/) — create a bucket, copy endpoint URL |
| `S3_BUCKET` | Yes | Your R2 bucket name |
| `S3_ACCESS_KEY` | Yes | R2 API token (access key ID) |
| `S3_SECRET_KEY` | Yes | R2 API token (secret key) |
| `S3_REGION` | No | Set to `auto` for R2 |
| `DATABASE_URL` | No | Default `sqlite:///./local.db` works |
| `DEFAULT_BACKEND` | No | `veo` (default) or `svd` |
| `USE_DIFFUSION` | No | `true` if using SVD fallback (requires GPU) |

**Example `.env`:**

```
REDIS_URL=rediss://default:xxx@xxx.upstash.io:6379
GEMINI_API_KEY=AIza...
S3_ENDPOINT=https://xxx.r2.cloudflarestorage.com
S3_BUCKET=your-bucket
S3_ACCESS_KEY=xxx
S3_SECRET_KEY=xxx
S3_REGION=auto
DATABASE_URL=sqlite:///./local.db
DEFAULT_BACKEND=veo
```

### Step 3: Install dependencies

```bash
pip install -r requirements-replit.txt
```

**Reason:** We use a minimal requirements file for Replit and local runs. It includes FastAPI, Redis, Gemini, and other core dependencies.

### Step 4: Run the main API

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

**Reason:** This starts the FastAPI server. The UI and API are served from the same process.

### Step 5: Open the UI

Open in your browser:

```
http://localhost:8000
```

- Click **New Run** or **Simulations**
- Enter a simulation goal (e.g. *"A ball rolling down an inclined ramp"*)
- Set **Budget** (e.g. 5) and **Risk Profile** (e.g. medium)
- Click **Run Simulation**

### Step 6: Run the GPU worker (required for video results)

The main API queues jobs to Redis. To actually render video and get results, you need a **worker** that pulls jobs from Redis and runs the backends. Two options:

#### Option A: Local worker (if you have an NVIDIA GPU)

In a **second terminal**:

```bash
pip install -r requirements.gpu.txt
python worker.py
```

**Reason:** The worker polls Redis for jobs, renders video (Veo/SVD), uploads to R2, and pushes results back. It needs a GPU and the full GPU requirements.

#### Option B: Use our RunPod endpoint (may fail)

If you do not have a GPU, jobs can be processed by our RunPod Serverless endpoint. The GitHub Actions bridge (`bridge-cron.yml`) runs every 3 minutes and forwards jobs from Redis to RunPod. This can fail for the reasons listed at the top of this document.

---

## What You Should See

- **Dashboard:** List of simulations
- **Simulation detail:** Goal, status (EXECUTING → COMPLETED or FAILED), confidence score, World State Graph, State Delta (JSON)
- **Ephemeral Debug Artifacts:** Video links (when rendering succeeds; links expire after ~1 hour)

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: pandas` | Run `pip install pandas` |
| `REDIS_URL required` | Add `REDIS_URL` to `.env` |
| `S3 configuration incomplete` | Add all `S3_*` variables to `.env` |
| Simulations stay EXECUTING | Worker is not running or not connected to the same Redis; check Option A or B above |
| Gemini 429 / quota exceeded | Wait and retry, or use a different API key |

---

## Summary

- **Live link:** Often fails because RunPod workers are unreliable and we did not have time to harden the deployment.
- **Local run:** Clone → create `.env` → install → run main API → run worker (with GPU) for full video results.
- **Minimal test:** You can run only the main API and submit simulations; they will stay in EXECUTING until a worker processes them. The UI and API flow are still visible.
