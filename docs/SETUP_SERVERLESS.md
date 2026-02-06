# RunPod Serverless Worker Setup

Step-by-step setup for `Dockerfile.serverless` on RunPod.

---

## 1. Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed
- [Docker Hub](https://hub.docker.com/) account (or RunPod registry)
- [RunPod](https://www.runpod.io/) account

---

## 2. Build the Image

From the project root:

```bash
# Replace YOUR_DOCKERHUB_USER with your Docker Hub username
docker build -f Dockerfile.serverless -t YOUR_DOCKERHUB_USER/storyworld-worker:serverless .
```

---

## 3. Push to Docker Hub

```bash
docker login
docker push YOUR_DOCKERHUB_USER/storyworld-worker:serverless
```

---

## 4. Create RunPod Serverless Endpoint

1. Go to [RunPod Console → Serverless](https://www.runpod.io/console/serverless)
2. Click **New Endpoint**
3. **Select GPU**: T4 (cheapest ~$0.0002/sec) or L4 (faster)
4. **Container Image**: `YOUR_DOCKERHUB_USER/storyworld-worker:serverless`
5. **Container Disk**: 20 GB (or more if models are large)
6. **Handler**: RunPod auto-detects from `runpod.serverless.start({"handler": handler})`
7. Create the endpoint

---

## 5. Set Environment Variables (Once)

In RunPod Console: **Endpoint → Settings → Environment Variables**

Add these (copy values from your `.env` or `runpod-template.env`):

| Variable | Required | Example |
|----------|----------|---------|
| `S3_ENDPOINT` | Yes | `https://xxx.r2.cloudflarestorage.com` |
| `S3_BUCKET` | Yes | `storyworld-artifacts` |
| `S3_ACCESS_KEY` | Yes | Your R2 access key |
| `S3_SECRET_KEY` | Yes | Your R2 secret key |
| `S3_REGION` | Yes | `auto` |
| `DEFAULT_BACKEND` | Yes | `veo` or `svd` |
| `VEO_FALLBACK_BACKEND` | No | `svd` |
| `USE_DIFFUSION` | No | `true` |
| `GEMINI_API_KEY` | If using Veo | Your API key |

**Note:** `REDIS_URL` is **not** needed. The bridge handles Redis; the worker only receives jobs via RunPod HTTP and uploads to R2.

---

## 6. Note Endpoint ID and API Key

- **Endpoint ID**: From the endpoint URL (e.g. `abc123xyz`) or Settings
- **API Key**: RunPod Console → Settings → API Keys

You need these for the bridge (`RUNPOD_ENDPOINT_ID`, `RUNPOD_API_KEY`).

---

## 7. Test Locally (Optional)

```bash
# Without RunPod (uses handler fallback, needs a valid job payload)
python worker_serverless.py '{"input":{"job_id":"test","backend":"stub","input":{"motion":{"engine":"sparse","params":{}},"output":{"path":"test/"}}}}'
```

Full integration test requires the bridge + main server + frontend. See [LOW_COST_DEPLOYMENT.md](LOW_COST_DEPLOYMENT.md).

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `S3 configuration incomplete` | Set all S3_* env vars in RunPod |
| `Unknown backend` | Set `DEFAULT_BACKEND` (veo/svd/animatediff/stub) |
| OOM / timeout | Increase container memory or use a larger GPU |
| Slow cold start | Use a warmer (min workers > 0) or keep endpoint active |
