# StoryWorld Architecture

## System Overview

StoryWorld is a **computational video infrastructure** that compiles simulation goals into validated physical outcomes. Video is the execution medium; state, truth, and constraints are the product.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FRONTEND (static/)                              │
│  index.html │ new.html │ simulation.html │ dashboard.js │ style.css          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │ HTTP
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FASTAPI (main.py)                                    │
│  /simulate │ /episodes │ /world-state │ /episodes/{id}/result │ /diagnostics │
└─────┬───────────────────┬───────────────────────┬───────────────────────────┘
      │                   │                       │
      ▼                   ▼                       ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────────┐
│ SQLStore     │  │ RedisStore   │  │ WorldGraphStore  │
│ (episodes,   │  │ (job queue,  │  │ (nodes,          │
│  beats,      │  │  result      │  │  transitions)    │
│  attempts)   │  │  queue)      │  │                  │
└──────────────┘  └──────┬───────┘  └──────────────────┘
                         │
          ┌──────────────┴──────────────┐
          │ Redis Queues                │
          │ storyworld:gpu:jobs         │
          │ storyworld:gpu:results      │
          └──────────────┬──────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────────────────┐
│ GPU WORKER (worker.py) — RunPod                                              │
│  blpop jobs → render (veo/svd/animatediff/stub) → upload R2 → rpush results  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Components

### Main Server (FastAPI)

- **main.py** — HTTP API, startup, ResultConsumer
- **EpisodeRuntime** — Episode lifecycle, planning, job submission
- **ResultConsumer** — Background loop: consume GPU results, run observer, update DB
- **Persistence:** SQLStore (SQLite), RedisStore, WorldGraphStore

### GPU Worker

- **worker.py** — Polls Redis for jobs, renders via backends, uploads to R2, pushes results
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
2. **Redis** → GPU worker blpop → render → upload R2 → rpush result
3. **ResultConsumer** → blpop result → observer.observe(video_uri) → epistemic_evaluator
4. **Epistemic evaluator** → ACCEPTED / EPISTEMICALLY_INCOMPLETE / REJECTED
5. **ResultConsumer** → mark_beat_success/failure, world_graph_store.record_beat_observation()
6. **UI** → polls /episodes, /world-state, /result

---

## Infrastructure

| Component | Purpose |
|-----------|---------|
| **Upstash Redis** | Job queue (storyworld:gpu:jobs), result queue (storyworld:gpu:results) |
| **Cloudflare R2** | Video artifact storage (S3-compatible) |
| **SQLite** | Episodes, beats, attempts, artifacts |
| **RunPod** | GPU worker (Veo, SVD, AnimateDiff) |

---

## Key Invariants

- **Beats are execution scaffolding, not semantic truth.** Success is defined by observer-validated transitions.
- **Observer is witness, not judge.** Infrastructure failure must NOT block closed-form problems.
- **No state without observation provenance.** Every transition ties to observation_id and video_uri.
- **Video is ephemeral.** Presigned URLs expire; system works if videos are deleted.
