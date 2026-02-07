# StoryWorld Architecture

## Current Architecture (No Deployment)

- **Main API** runs locally (uvicorn)
- **GPU Worker** runs on RunPod Serverless
- **Bridge** (GitHub Actions cron) connects Redis ↔ RunPod

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  LOCAL: uvicorn main:app                                                     │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ FRONTEND (static/) + FASTAPI (main.py)                                 │  │
│  │ /simulate │ /episodes │ /world-state │ ResultConsumer                  │  │
│  └─────┬─────────────┬─────────────┬─────────────────────────────────────┘  │
│        │             │             │                                         │
│        ▼             ▼             ▼                                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────┐                             │
│  │ SQLStore │ │ Redis    │ │ WorldGraphStore  │                             │
│  │ (local   │ │ (Upstash)│ │ (local SQLite)   │                             │
│  │  db)     │ │          │ │                  │                             │
│  └──────────┘ └────┬─────┘ └──────────────────┘                             │
└─────────────────────│────────────────────────────────────────────────────────┘
                      │
         ┌────────────┴────────────┐
         │ Redis Queues            │
         │ storyworld:gpu:jobs     │
         │ storyworld:gpu:results  │
         └────────────┬────────────┘
                      │
    ┌─────────────────┴─────────────────┐
    │ GitHub Actions (every 3 min)       │
    │ bridge_serverless.py               │
    │ Redis blpop → RunPod HTTP → rpush  │
    └─────────────────┬─────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────────────────┐
│ RunPod Serverless (worker_serverless.py)                                     │
│ Receive job → render (veo/svd/animatediff) → upload R2 → callback results    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Components

### Main Server (FastAPI)

- **main.py** — HTTP API, startup, ResultConsumer
- **EpisodeRuntime** — Episode lifecycle, planning, job submission
- **ResultConsumer** — Background loop: consume GPU results, run observer, update DB
- **Persistence:** SQLStore (SQLite), RedisStore, WorldGraphStore

### GPU Worker (RunPod Serverless)

- **worker_serverless.py** — Receives jobs via RunPod HTTP, renders via backends, uploads to R2
- **Bridge** (bridge_serverless.py, GitHub Actions) — Polls Redis, invokes RunPod, pushes results back
- **Backends:** Veo, SVD, AnimateDiff, Stub
- **Fallback:** Veo credit exhausted → SVD (configurable)

### Video Observer

- **Primary:** Gemini (vision)
- **Fallback:** Ollama + LLaVA when Gemini 429
- **Verdicts:** valid, uncertain, impossible, contradicts, blocks

### Epistemic Evaluator

- **Input:** Evidence ledger, intent, verdict, observer availability
- **Output:** EpistemicState (ACCEPTED, EPISTEMICALLY_INCOMPLETE, REJECTED, UNCERTAIN_TERMINATION)
- **Closed-form rule:** Observer optional; solver-only success allowed when `requires_visual_verification=false`

---

## Data Flow

1. **User** → POST /simulate → EpisodeRuntime.create(), plan(), schedule(), submit_pending_beats()
2. **Main API** → rpush job to Redis (storyworld:gpu:jobs)
3. **Bridge** (GitHub Actions, every 3 min) → blpop Redis → invoke RunPod Serverless
4. **RunPod Worker** → render → upload R2 → callback to bridge → bridge rpush result to Redis
5. **ResultConsumer** (local) → blpop result → observer.observe(video_uri) → epistemic_evaluator
6. **ResultConsumer** → mark_beat_success/failure, world_graph_store.record_beat_observation()
7. **UI** → polls /episodes, /world-state, /result

---

## Infrastructure

| Component | Where | Purpose |
|-----------|-------|---------|
| **Main API** | Local | FastAPI, ResultConsumer, SQLite, WorldGraphStore |
| **Upstash Redis** | Cloud | Job queue (storyworld:gpu:jobs), result queue (storyworld:gpu:results) |
| **Cloudflare R2** | Cloud | Video artifact storage (S3-compatible) |
| **GitHub Actions** | Cloud | Bridge: Redis ↔ RunPod |
| **RunPod Serverless** | Cloud | GPU worker (Veo, SVD, AnimateDiff) |

---

## Key Invariants

- **Beats are execution scaffolding, not semantic truth.** Success is defined by observer-validated transitions.
- **Observer is witness, not judge.** Infrastructure failure must NOT block closed-form problems.
- **No state without observation provenance.** Every transition ties to observation_id and video_uri.
- **Video is ephemeral.** Presigned URLs expire; system works if videos are deleted.
