# StoryWorld Distributed Architecture

## Production Setup

### Infrastructure Components

1. **Main Server** (Local/Cloud VM)
   - Runs FastAPI server (`main.py`)
   - Runs episode orchestration (`runtime/run_decision_loop.py`)
   - SQLite database (`local.db`)

2. **Upstash Redis** (Job Queue)
   - URL: `rediss://default:***@climbing-bee-30900.upstash.io:6379`
   - Queues: `storyworld:gpu:jobs`, `storyworld:gpu:results`
   - Enables communication between main server and RunPod workers

3. **RunPod GPU Workers** (Remote GPU Instances)
   - Run `worker.py` for video rendering
   - Use AnimateDiff/Veo/SVD backends
   - Require GPU for diffusion models

4. **Cloudflare R2** (Artifact Storage)
   - S3-compatible object storage
   - Endpoint: `https://ddfb4016d161f6a65f0a2e09e8e62094.r2.cloudflarestorage.com`
   - Bucket: `storyworld-artifacts`
   - Stores generated videos and keyframes

---

## Job Flow

```
┌─────────────────┐
│  Main Server    │
│  (FastAPI)      │
└────────┬────────┘
         │
         │ 1. Submit job
         ▼
┌─────────────────┐
│ Upstash Redis   │
│ GPU_JOB_QUEUE   │
└────────┬────────┘
         │
         │ 2. Poll for jobs
         ▼
┌─────────────────┐
│ RunPod Worker   │
│ (GPU Rendering) │
└────────┬────────┘
         │
         │ 3. Upload artifact
         ▼
┌─────────────────┐
│ Cloudflare R2   │
│ (Video Storage) │
└────────┬────────┘
         │
         │ 4. Push result
         ▼
┌─────────────────┐
│ Upstash Redis   │
│ GPU_RESULT_QUEUE│
└────────┬────────┘
         │
         │ 5. Retrieve result
         ▼
┌─────────────────┐
│  Main Server    │
│  (Mark complete)│
└─────────────────┘
```

---

## Configuration Notes

### Cloudflare R2 vs AWS S3

We use **Cloudflare R2** which is S3-compatible but has some differences:

- **Endpoint**: Custom R2 endpoint (not `s3.amazonaws.com`)
- **Region**: Set to `auto` for R2
- **Authentication**: Uses R2 access keys (not AWS IAM)
- **Benefits**: No egress fees, cheaper than S3

### Environment Variables

Critical settings in `.env`:

```bash
# Redis (Upstash)
REDIS_URL=rediss://default:***@climbing-bee-30900.upstash.io:6379

# Queues
JOB_QUEUE=storyworld:gpu:jobs
RESULT_QUEUE=storyworld:gpu:results

# Storage (Cloudflare R2)
S3_ENDPOINT=https://ddfb4016d161f6a65f0a2e09e8e62094.r2.cloudflarestorage.com
S3_BUCKET=storyworld-artifacts
S3_REGION=auto
```

---

## Deployment Checklist

### Main Server Setup

1. ✅ Configure `.env` with Upstash Redis URL
2. ✅ Configure Cloudflare R2 credentials
3. ✅ Set `DATABASE_URL=sqlite:///./local.db`
4. ✅ Set `LOCAL_MODE=false` for production
5. ⬜ Start FastAPI server: `uvicorn main:app --host 0.0.0.0 --port 8000`
6. ⬜ Start decision loop for episodes (spawned automatically)

### RunPod Worker Setup

1. ⬜ Copy same `.env` to RunPod instance
2. ⬜ Install dependencies: `pip install -r requirements.gpu.txt`
3. ⬜ Verify GPU access: `nvidia-smi`
4. ⬜ Set `USE_DIFFUSION=true` for SDXL support
5. ⬜ Start worker: `python worker.py`
6. ⬜ Worker polls `GPU_JOB_QUEUE` continuously

### Testing

1. ⬜ Create episode via API
2. ⬜ Verify job appears in Upstash Redis queue
3. ⬜ Verify RunPod worker picks up job
4. ⬜ Verify video uploads to R2
5. ⬜ Verify result appears in result queue
6. ⬜ Verify episode completes successfully

---

## Monitoring

### Check Upstash Redis

```bash
# View queue lengths
redis-cli -u $REDIS_URL LLEN storyworld:gpu:jobs
redis-cli -u $REDIS_URL LLEN storyworld:gpu:results
```

### Check R2 Storage

Access Cloudflare dashboard to view:
- Bucket contents
- Storage usage
- Request metrics

### Check Worker Status

On RunPod instance:
```bash
# View worker logs
tail -f worker.log

# Check GPU utilization
watch -n 1 nvidia-smi
```

---

## Cost Optimization

1. **RunPod**: Use spot instances for lower costs
2. **Upstash Redis**: Free tier supports up to 10K commands/day
3. **Cloudflare R2**: No egress fees (major savings vs S3)
4. **GPU Usage**: Workers only run when jobs are queued

---

## Troubleshooting

### Worker Not Picking Up Jobs

- Verify Upstash Redis URL is correct
- Check queue names match exactly
- Ensure worker has network access to Upstash

### Artifacts Not Uploading

- Verify R2 endpoint and credentials
- Check bucket name is correct
- Ensure worker has write permissions

### Jobs Timing Out

- Increase Redis timeout in `worker.py`
- Check GPU memory availability
- Monitor RunPod instance resources
