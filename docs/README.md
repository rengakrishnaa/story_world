# StoryWorld Documentation

Welcome to the StoryWorld documentation. This guide helps you understand, use, and deploy the StoryWorld platform—a **reality compiler** that uses video generation to simulate and validate physical scenarios.

---

## Documentation Index

| Document | Audience | Description |
|----------|----------|-------------|
| **[Product Overview](PRODUCT_DOCUMENTATION.md)** | Everyone | Product vision, what StoryWorld is and is not, value proposition |
| **[User Manual](USER_MANUAL.md)** | End users, operators | How to run simulations, interpret results, use the UI |
| **[API Reference](API_REFERENCE.md)** | Developers, integrators | Complete API endpoint documentation |
| **[Architecture](ARCHITECTURE.md)** | Engineers | System design, components, data flow |
| **[Configuration Reference](CONFIGURATION_REFERENCE.md)** | DevOps, developers | Environment variables, deployment config |
| **[Deployment Guide](DEPLOYMENT_GUIDE.md)** | DevOps | Production setup, main server, GPU worker |
| **[Low-Cost Deployment](LOW_COST_DEPLOYMENT.md)** | DevOps | Netlify + Render + RunPod Serverless, minimal cost setup |
| **[Setup Serverless](SETUP_SERVERLESS.md)** | DevOps | Step-by-step Dockerfile.serverless + RunPod setup |
| **[Deploy Render + Netlify](DEPLOY_RENDER_NETLIFY.md)** | DevOps | Main app (Render) + frontend (Netlify) deployment |
| **[Changelog & Roadmap](CHANGELOG_AND_ROADMAP.md)** | Everyone | Completed features, known issues, future plans |
| **[Troubleshooting](TROUBLESHOOTING.md)** | Support, operators | Common issues and solutions |

---

## Quick Links

- **Get Started:** [User Manual → Quick Start](USER_MANUAL.md#quick-start)
- **Run a Simulation:** [User Manual → Running a Simulation](USER_MANUAL.md#running-a-simulation)
- **API Basics:** [API Reference → Primary Endpoints](API_REFERENCE.md#primary-endpoints)
- **Configuration:** [Configuration Reference](CONFIGURATION_REFERENCE.md)

---

## What is StoryWorld?

StoryWorld is a **computational video infrastructure** that compiles simulation goals into validated physical outcomes. It is not a content generator or media platform—it is a **reality compiler**:

- **Input:** A simulation goal (e.g., *"A robotic arm stacks three boxes without tipping"*)
- **Process:** Plan → Render video → Observe with vision AI → Validate physics
- **Output:** Outcome (goal_achieved, goal_impossible, etc.), confidence, discovered constraints, world state graph

Video is the **execution medium**, not the product. State, truth, and discovered constraints are the product.

---

## Documentation Conventions

- **Code blocks** show commands, configuration, or API examples
- **Bold** indicates UI elements, endpoints, or important terms
- Links to related documents are provided where relevant
