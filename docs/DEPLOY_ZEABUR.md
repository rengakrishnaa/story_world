# Deploy Main API on Zeabur

**For future deployment.** Current setup runs locally: [LOCAL_SETUP.md](LOCAL_SETUP.md)

When scaling: Zeabur hosts the FastAPI app with Python 3.10.

---

## 1. Connect GitHub

1. Go to [zeabur.com](https://zeabur.com) and sign in (GitHub)
2. **Create Project** → **Deploy from GitHub**
3. Select the `story_world` repo
4. Zeabur will detect the project

---

## 2. Dockerfile (Auto-Used)

We use `zbpack.json` so Zeabur deploys with `Dockerfile.zeabur`:
- Python 3.10
- `requirements-replit.txt` (no CLIP git clone)
- uvicorn

No extra config needed—push and deploy.

---

## 3. Environment Variables

In Zeabur: **Service** → **Variables** (or **Environment**)

Add:

| Key | Value |
|-----|-------|
| `REDIS_URL` | Upstash Redis URL |
| `DATABASE_URL` | `sqlite:///./local.db` |
| `S3_ENDPOINT` | R2 endpoint |
| `S3_BUCKET` | Bucket name |
| `S3_ACCESS_KEY` | R2 access key |
| `S3_SECRET_KEY` | R2 secret key |
| `GEMINI_API_KEY` | Gemini API key |
| `JOB_QUEUE` | `storyworld:gpu:jobs` |
| `RESULT_QUEUE` | `storyworld:gpu:results` |

---

## 4. Deploy

1. Click **Deploy**
2. Wait for the build (Dockerfile uses `requirements-replit.txt` – no CLIP, faster)
3. We note the URL (e.g. `https://storyworld-xxx.zeabur.app`)

---

## 5. Connect Netlify

We update `netlify.toml` with our Zeabur URL (e.g. `https://storyworld-abc123.zeabur.app`):

```toml
to = "https://storyworld-abc123.zeabur.app/episodes"
```

---

## 6. Override Dockerfile (if needed)

If Zeabur ignores the Dockerfile, add env var:  
`ZBPACK_DOCKERFILE_NAME=zeabur`  
so it uses `Dockerfile.zeabur`.
