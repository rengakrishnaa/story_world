# StoryWorld Documentation

Welcome to the StoryWorld documentation. This guide helps you understand, use, and deploy the StoryWorld platform—a **reality compiler** that uses video generation to simulate and validate physical scenarios.

---

## Documentation Index

### Current Setup (Local)

| Document | Description |
|----------|-------------|
| **[Local Setup](LOCAL_SETUP.md)** | How to run StoryWorld — uvicorn + RunPod Serverless |
| **[RunPod Serverless](SETUP_SERVERLESS.md)** | GPU worker setup on RunPod |

### Reference

| Document | Description |
|----------|-------------|
| **[Product Overview](PRODUCT_DOCUMENTATION.md)** | Product vision, value proposition |
| **[User Manual](USER_MANUAL.md)** | Run simulations, interpret results, use the UI |
| **[API Reference](API_REFERENCE.md)** | API endpoint documentation |
| **[Architecture](ARCHITECTURE.md)** | System design, data flow |
| **[Technical Documentation](TECHNICAL_DOCUMENTATION.md)** | System overview, capabilities, data models |
| **[Configuration Reference](CONFIGURATION_REFERENCE.md)** | Environment variables |

### Future Deployment (When Scaling)

| Document | Description |
|----------|-------------|
| **[Deployment Guide](DEPLOYMENT_GUIDE.md)** | Overview of deployment options |
| **[Deploy Render + Netlify](DEPLOY_RENDER_NETLIFY.md)** | Render + Netlify |
| **[Deploy Fly](DEPLOY_FLY.md)** | Fly.io |
| **[Deploy Replit](DEPLOY_REPLIT.md)** | Replit |
| **[Deploy Zeabur](DEPLOY_ZEABUR.md)** | Zeabur |
| **[Deploy Vercel](DEPLOY_VERCEL.md)** | Vercel |
| **[Low-Cost Deployment](LOW_COST_DEPLOYMENT.md)** | Minimal cost setup |

### Other

| Document | Description |
|----------|-------------|
| **[Changelog & Roadmap](CHANGELOG_AND_ROADMAP.md)** | Completed features, future plans |
| **[Troubleshooting](TROUBLESHOOTING.md)** | Common issues and solutions |

---

## Quick Links

- **Run StoryWorld:** [Local Setup](LOCAL_SETUP.md)
- **Run a Simulation:** [User Manual → Running a Simulation](USER_MANUAL.md#running-a-simulation)
- **API Basics:** [API Reference → Primary Endpoints](API_REFERENCE.md#primary-endpoints)
- **Configuration:** [Configuration Reference](CONFIGURATION_REFERENCE.md)

---

## What is StoryWorld?

StoryWorld is a **computational video infrastructure** that compiles simulation goals into validated physical outcomes. It is not a content generator or media platform—it is a **reality compiler**:

- **Input:** A simulation goal (e.g., *"A robotic arm stacks three boxes without tipping"*)
- **Process:** Plan → Render video → Observe with vision model → Validate physics
- **Output:** Outcome (goal_achieved, goal_impossible, etc.), confidence, discovered constraints, world state graph

Video is the **execution medium**, not the product. State, truth, and discovered constraints are the product.

---

## Documentation Conventions

- **Code blocks** show commands, configuration, or API examples
- **Bold** indicates UI elements, endpoints, or important terms
- Links to related documents are provided where relevant
