# StoryWorld

**A reality compiler that uses video generation to simulate and validate physical scenarios.**

StoryWorld compiles simulation goals into validated outcomes—not video content, but **state**, **truth**, and **discovered constraints**. Video is the execution medium; the product is epistemic validity.

---

## Current Architecture (No Deployment)

- **Main API** — Run locally: `uvicorn main:app`
- **GPU Worker** — RunPod Serverless (already set up)
- **Bridge** — GitHub Actions cron: Redis ↔ RunPod (every 3 min)

When we upgrade and get users, we will deploy. For now, everything runs locally except the GPU.

---

## Quick Start

```bash
# 1. Copy config
cp env.example .env
# Edit .env: REDIS_URL, GEMINI_API_KEY, S3_*, DEFAULT_BACKEND=veo

# 2. Install
pip install -r requirements-replit.txt

# 3. Start main server (local)
uvicorn main:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000. The bridge (GitHub Actions) and RunPod Serverless handle GPU jobs automatically.

---

## Documentation

| Document | Description |
|----------|-------------|
| [Local Setup](docs/LOCAL_SETUP.md) | How to run StoryWorld (current architecture) |
| [Product Documentation](docs/PRODUCT_DOCUMENTATION.md) | Product vision, what StoryWorld is and is not |
| [User Manual](docs/USER_MANUAL.md) | How to run simulations, interpret results, use the UI |
| [API Reference](docs/API_REFERENCE.md) | Complete API endpoint documentation |
| [Architecture](docs/ARCHITECTURE.md) | System design, components, data flow |
| [Configuration Reference](docs/CONFIGURATION_REFERENCE.md) | Environment variables |
| [RunPod Serverless Setup](docs/SETUP_SERVERLESS.md) | GPU worker on RunPod Serverless |
| [Deployment Guide](docs/DEPLOYMENT_GUIDE.md) | Future deployment (Render, Fly, etc.) |
| [Changelog & Roadmap](docs/CHANGELOG_AND_ROADMAP.md) | Completed features, future plans |
| [Troubleshooting](docs/TROUBLESHOOTING.md) | Common issues and solutions |

---

## What StoryWorld Does

- **Input:** Simulation goal (physics-focused, e.g., stacking boxes, vehicle dynamics)
- **Process:** Plan → Render video (Veo/SVD/AnimateDiff) → Observe with vision AI → Validate physics
- **Output:** Outcome (goal_achieved, goal_impossible, etc.), confidence, constraints_discovered, WorldStateGraph

Video is ephemeral; state and constraints are the product.

---

## License & Contact

See project root for license and contribution guidelines.
