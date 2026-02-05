# StoryWorld

**A reality compiler that uses video generation to simulate and validate physical scenarios.**

StoryWorld compiles simulation goals into validated outcomes—not video content, but **state**, **truth**, and **discovered constraints**. Video is the execution medium; the product is epistemic validity.

---

## Documentation

**[→ Full documentation](docs/README.md)**

| Document | Description |
|----------|-------------|
| [Product Documentation](docs/PRODUCT_DOCUMENTATION.md) | Product vision, what StoryWorld is and is not |
| [User Manual](docs/USER_MANUAL.md) | How to run simulations, interpret results, use the UI |
| [API Reference](docs/API_REFERENCE.md) | Complete API endpoint documentation |
| [Architecture](docs/ARCHITECTURE.md) | System design, components, data flow |
| [Configuration Reference](docs/CONFIGURATION_REFERENCE.md) | Environment variables |
| [Deployment Guide](docs/DEPLOYMENT_GUIDE.md) | Production setup |
| [Changelog & Roadmap](docs/CHANGELOG_AND_ROADMAP.md) | Completed features, future plans |
| [Troubleshooting](docs/TROUBLESHOOTING.md) | Common issues and solutions |

---

## Quick Start

```bash
# 1. Copy config
cp env.example .env
# Edit .env: REDIS_URL, GEMINI_API_KEY, S3_*, DEFAULT_BACKEND=veo

# 2. Install
pip install -r requirements.base.txt

# 3. Start main server
uvicorn main:app --host 0.0.0.0 --port 8000

# 4. (Separate terminal) Start GPU worker (requires GPU, Redis, R2)
pip install -r requirements.gpu.txt
python worker.py
```

Open http://localhost:8000, click **New Run**, enter a goal (e.g., *"A robotic arm stacks three boxes without tipping..."*), and run.

---

## What StoryWorld Does

- **Input:** Simulation goal (physics-focused, e.g., stacking boxes, vehicle dynamics)
- **Process:** Plan → Render video (Veo/SVD/AnimateDiff) → Observe with vision AI → Validate physics
- **Output:** Outcome (goal_achieved, goal_impossible, etc.), confidence, constraints_discovered, WorldStateGraph

Video is ephemeral; state and constraints are the product.

---

## License & Contact

See project root for license and contribution guidelines.
