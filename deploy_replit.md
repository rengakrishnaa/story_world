# Deploy StoryWorld to Replit (100% Free)

## Prerequisites
- GitHub account
- Replit account (free)

## Step 1: Push to GitHub

```bash
# Navigate to your project
cd c:\Users\KRISH\Desktop\LEARN\Vidme\story_world

# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit
git commit -m "Deploy to Replit"

# Create a new repository on GitHub, then add remote
git remote add origin https://github.com/YOUR_USERNAME/story_world.git

# Push
git push -u origin main
```

## Step 2: Import to Replit

1. Go to https://replit.com
2. Click **"Create Repl"**
3. Select **"Import from GitHub"**
4. Paste your GitHub repo URL
5. Click **"Import from GitHub"**

## Step 3: Configure Environment Variables

In Replit, click the **lock icon** (Secrets) and add these variables:

```env
REDIS_URL=rediss://default:AXi0AAIncDIzZDEwNjE3N2U4MGI0YThjYWUzMjljYmE4ODU3NjVhMHAyMzA5MDA@climbing-bee-30900.upstash.io:6379
DATABASE_URL=sqlite:///./local.db
JOB_QUEUE=storyworld:gpu:jobs
RESULT_QUEUE=storyworld:gpu:results
GPU_JOB_QUEUE=storyworld:gpu:jobs
GPU_RESULT_QUEUE=storyworld:gpu:results
LOCAL_MODE=false
USE_LLM_PHYSICS_VETO=true
S3_ENDPOINT=https://ddfb4016d161f6a65f0a2e09e8e62094.r2.cloudflarestorage.com
S3_BUCKET=mei-engine
S3_REGION=auto
S3_ACCESS_KEY=d5e82f19826e68a6eacdc7f0359579a8
S3_SECRET_KEY=acf748adfee94e553c0b0dc6c5dcbd8fd9f8f64e07e7a71d128a60d111e04295
USE_S3=true
GEMINI_API_KEY=AIzaSyDShRNWb9TGs-TTeJsTz8JAEZyFLAJ9Xe0
GEMINI_OBSERVER_MODEL=gemini-2.5-flash-lite
NANOBANANA_API_KEY=921ea38693a66aa2ef7c24b1c2092082
OBSERVER_FALLBACK_ENABLED=true
INTENT_CLASSIFIER_MODEL=gemini-2.0-flash
USE_MOCK_PLANNER=false
GEMINI_USE_REAL=true
PIPELINE_VALIDATE=false
USE_DIFFUSION=true
DEFAULT_BACKEND=veo
VEO_USE_FAST=false
ENV=production
LOG_LEVEL=INFO
REALITY_COMPILER_MODE=true
USE_OBSERVER_IN_PRODUCTION=true
```

## Step 4: Run the Server

1. Click the **"Run"** button in Replit
2. Wait for dependencies to install
3. Server will start automatically
4. Copy your Replit URL: `https://YOUR_REPL.YOUR_USERNAME.repl.co`

## Step 5: Enable Always-On (Free)

1. Go to the **"Deployments"** tab in Replit
2. Click **"Deploy"**
3. Select **"Autoscale"** (free tier)
4. Your server will stay online 24/7

## Step 6: Update Frontend

Edit `static/index.html` and `static/simulation.html`:

```javascript
// Find this line:
const API_URL = 'http://localhost:8000';

// Replace with your Replit URL:
const API_URL = 'https://YOUR_REPL.YOUR_USERNAME.repl.co';
```

## Step 7: Deploy Frontend to Vercel

```bash
# Install Vercel CLI
npm install -g vercel

# Deploy
cd c:\Users\KRISH\Desktop\LEARN\Vidme\story_world
vercel

# Follow prompts:
# - Set up and deploy? Yes
# - Which scope? Your account
# - Link to existing project? No
# - Project name? story-world
# - In which directory is your code? ./
# - Override settings? Yes
# - Build Command? (leave empty)
# - Output Directory? static
# - Development Command? (leave empty)
```

## Step 8: Add CORS to Main Server

Add this to `main.py` (after `app = FastAPI(...)`):

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your Vercel URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

Commit and push to GitHub. Replit will auto-deploy.

## Step 9: Test

1. Open your Vercel URL
2. Create a simulation
3. Check Replit logs to see job queued
4. Wait for RunPod GPU to process (may take 5-30 min if GPUs busy)
5. Result will appear in UI

## Troubleshooting

### Replit keeps sleeping
- Use Deployments → Deploy (free tier)
- This keeps server always-on

### Dependencies fail to install
- Check `requirements-replit.txt` exists
- Replit should auto-detect and install

### Can't connect to Redis
- Check REDIS_URL in Secrets
- Test connection: `redis-cli -u YOUR_REDIS_URL ping`

### CORS errors
- Add CORS middleware (Step 8)
- Make sure Vercel URL is in `allow_origins`

## Done! 🎉

Your StoryWorld is now deployed:
- **Main Server:** Replit (free, always-on)
- **Frontend:** Vercel (free, fast CDN)
- **GPU Worker:** RunPod Serverless (pay-per-use)
- **Storage:** Cloudflare R2 (free tier)

**Total Cost:** $0/month + ~$0.01 per video
