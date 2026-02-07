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
# We use our Docker Hub username in the image tag
docker build -f Dockerfile.serverless -t <dockerhub-username>/storyworld-worker:serverless .
```

---

## 3. Push to Docker Hub

```bash
docker login
docker push <dockerhub-username>/storyworld-worker:serverless
```

---

## 4. Create RunPod Serverless Endpoint

1. Go to [RunPod Console → Serverless](https://www.runpod.io/console/serverless)
2. Click **New Endpoint**
3. **Select GPU**: T4 (cheapest ~$0.0002/sec) or L4 (faster)
4. **Container Image**: `<dockerhub-username>/storyworld-worker:serverless`
5. **Container Disk**: 20 GB (or more if models are large)
6. **Handler**: RunPod auto-detects from `runpod.serverless.start({"handler": handler})`
7. Create the endpoint

---

## 5. Set Environment Variables (Once)

In RunPod Console: **Endpoint → Settings → Environment Variables**

We add these (values come from `.env` or `runpod-template.env`):

| Variable | Required | Example |
|----------|----------|---------|
| `S3_ENDPOINT` | Yes | R2 endpoint URL |
| `S3_BUCKET` | Yes | `storyworld-artifacts` |
| `S3_ACCESS_KEY` | Yes | R2 access key |
| `S3_SECRET_KEY` | Yes | R2 secret key |
| `S3_REGION` | Yes | `auto` |
| `DEFAULT_BACKEND` | Yes | `veo` or `svd` |
| `VEO_FALLBACK_BACKEND` | No | `svd` |
| `USE_DIFFUSION` | No | `true` |
| `GEMINI_API_KEY` | If using Veo | Gemini API key |

We don't set `REDIS_URL` on the worker. The bridge handles Redis; the worker receives jobs via RunPod HTTP and uploads to R2.

---

## 6. Endpoint ID and API Key

We grab the Endpoint ID from the endpoint URL or Settings, and the API Key from RunPod Console → Settings → API Keys. These go into the bridge as `RUNPOD_ENDPOINT_ID` and `RUNPOD_API_KEY`.

---

## 7. Test Locally

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
