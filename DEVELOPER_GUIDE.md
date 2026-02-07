# StoryWorld — Developer Guide

**Purpose:** A technical reference for developers to understand the product, its architecture, file roles, and how components wire together from backend to frontend.

---

## 1. Product Identity

**StoryWorld is a reality compiler, not a content generator.**

| Principle | Meaning |
|-----------|---------|
| Video is the execution medium | Video is used to test hypotheses; it is not the primary output |
| State and truth are the product | Output: outcome, WorldStateGraph, confidence, constraints |
| Failure is first-class | Episodes may end in GOAL_IMPOSSIBLE, DEAD_STATE, GOAL_ABANDONED |
| Observer is authority | Only the VideoObserver may assert facts from video; planners propose |

**Product output:**
- Outcome: `goal_achieved` | `goal_impossible` | `goal_abandoned` | `dead_state` | `uncertain_termination` | `epistemically_incomplete`
- WorldStateGraph (validated state transitions)
- Confidence (0.0–1.0) — see §3.3
- Discovered constraints
- `suggested_alternatives` / `attempts_made` (when exploratory + failure) — what we tried, what might work next

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FRONTEND (static/)                              │
│  index.html │ new.html │ simulation.html │ dashboard.js │ style.css          │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │ HTTP
┌─────────────────────────────────────▼───────────────────────────────────────┐
│                         FASTAPI (main.py)                                    │
│  /simulate │ /episodes │ /world-state │ /episodes/{id}/result │ /diagnostics │
└─────┬───────────────────────┬───────────────────────┬───────────────────────┘
      │                       │                       │
      ▼                       ▼                       ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────────┐
│ SQLStore     │      │ RedisStore   │      │ WorldGraphStore  │
│ (episodes,   │      │ (job queue,  │      │ (nodes,          │
│  beats,      │      │  result      │      │  transitions)    │
│  attempts)   │      │  queue)      │      │                  │
└──────────────┘      └──────┬───────┘      └──────────────────┘
                             │
              ┌──────────────┴──────────────┐
              │ Redis Queues                │
              │ storyworld:gpu:jobs         │
              │ storyworld:gpu:results      │
              └──────────────┬──────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────────────┐
│ GPU WORKER (worker.py) — runs on RunPod                                      │
│  blpop jobs → render (veo/svd/animatediff/stub) → upload R2 → rpush results  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. End-to-End Flow

### 3.1 Simulation Lifecycle

**Important:**
- A completed beat does **not** imply a valid state transition. Only observer-validated transitions update the WorldStateGraph and contribute to goal success.
- **Beats are execution scaffolding, not semantic truth.** They are legacy execution units; success is defined by observer-validated transitions, not beat completion counts.

**Rule:** `physics_veto` may only return GOAL_IMPOSSIBLE when the goal is **self-contradictory in language** (e.g. "stone floats unsupported"), not when it is physically implausible. Physically implausible goals must reach the observer.

```
User submits goal (new.html)
    │
    ▼
POST /simulate?goal=...&budget=5
    │
    ├─ EpisodeRuntime.create() → SQL episodes, beats
    ├─ physics_veto (NLP gate) → if HARD contradiction (logical impossibility): mark GOAL_IMPOSSIBLE (pre-simulation); otherwise tag constraints and continue
    ├─ ExecutionPlanner (ProductionNarrativePlanner) → generates execution beats
    ├─ runtime.schedule() → EXECUTING
    └─ runtime.submit_pending_beats() → Redis push
    │
    ▼
Redis storyworld:gpu:jobs
    │
    ▼
Worker (RunPod) blpop → render → upload R2 → rpush storyworld:gpu:results
    │
    ▼
ResultConsumer (background) blpop results
    │
    ├─ VideoObserver.observe(video_uri) → verdict, confidence
    ├─ runtime.mark_beat_success/failure()
    └─ world_graph_store.record_beat_observation()
    │
    ▼
UI polls /episodes/{id}, /world-state/{id}, /episodes/{id}/result
```

**Non-story example:** "A robot stacks three boxes vertically on a narrow base without external support." — same flow; no narrative logic. Goal is physics/constraints, not story.

### 3.2 Data Flow

| Stage | Source | Sink |
|-------|--------|-----|
| Input | `new.html` form → `dashboard.js` | `POST /simulate` |
| Create | `main.py` | `EpisodeRuntime`, `SQLStore` |
| Plan | ExecutionPlanner (narrative_planner) | `beats` table |
| Submit | `EpisodeRuntime.submit_pending_beats` | `RedisStore.push_gpu_job` |
| Process | `worker.py` | `Redis rpush` result |
| Consume | `ResultConsumer` | `SQLStore`, `WorldGraphStore` |
| Display | `GET /episodes`, `/world-state`, `/result` | `dashboard.js` → UI |

### 3.3 Confidence Definition

Confidence represents the system's certainty in the outcome given observer agreement, temporal consistency, and constraint strength. It **must decrease** when:
- Observers disagree
- Outcomes depend on narrow tolerances
- Failure occurs late in execution

Do not default to constants. Confidence is derived from observer verdicts and metrics, not from planner or beat-completion counts.

---

## 4. File Reference

### 4.1 Root / Entry Points

| File | Role |
|------|------|
| `main.py` | FastAPI app; all HTTP routes; startup (SQL, Redis, ResultConsumer) |
| `worker.py` | GPU worker; polls Redis for jobs; renders via backends; pushes results |
| `config.py` | Central config; env vars for queues, policies, REALITY_COMPILER_MODE |

### 4.2 Frontend (`static/`)

**UI invariant:** The UI must **never** require video playback to understand outcomes. Outcome, confidence, and WorldStateGraph are sufficient.

| File | Role |
|------|------|
| `index.html` | Dashboard; list of simulations |
| `new.html` | New simulation form (goal, budget, risk profile). Risk profile: low/medium/high—exploratory (high) retries with different framings and surfaces suggestions on failure. |
| `simulation.html` | Simulation detail; WorldStateGraph; State Delta JSON |
| `dashboard.js` | `loadDashboard()`, `loadDetail()`, `submitSimulation()`, `renderGraph()`, polling |
| `style.css` | Infrastructure console styling |

### 4.3 Runtime (`runtime/`)

| File | Role |
|------|------|
| `episode_runtime.py` | Episode lifecycle; plan, schedule, submit beats; mark success/failure |
| `episode_state.py` | `EpisodeState` enum (CREATED, PLANNED, EXECUTING, COMPLETED, etc.) |
| `beat_state.py` | `BeatState` enum (PENDING, EXECUTING, ACCEPTED, ABORTED) |
| `snapshot.py` | `EpisodeSnapshot` from runtime for API response |
| `result_consumer.py` | Background loop; pop GPU results; run observer; update SQL + WorldGraph |
| `physics_veto.py` | NLP impossibility gate; HARD veto blocks, SOFT does not |
| `planner_adapter.py` | Adapter for planner interface |
| `run_decision_loop.py` | Standalone decision loop (alternative to ResultConsumer path) |

### 4.4 Policies (`runtime/policies/`)

| File | Role |
|------|------|
| `retry_policy.py` | Decides RETRY vs ABORT; respects observer_verdict impossible/uncertain |
| `quality_policy.py` | Confidence thresholds |
| `cost_policy.py` | Budget limits |

### 4.5 Persistence (`runtime/persistence/`)

| File | Role |
|------|------|
| `sql_store.py` | Episodes, beats (execution scaffolding), attempts, artifacts; `all_beats_completed`, `any_beats_failed` |
| `redis_store.py` | `push_gpu_job`, `pop_gpu_result`; queue names from env |
| `world_graph_store.py` | `record_beat_observation()`; nodes, transitions, branches |

### 4.6 Agents

| File | Role |
|------|------|
| `narrative_planner.py` | ExecutionPlanner (`ProductionNarrativePlanner`); generates execution beats from goal (scaffolding, not semantic truth) |
| `video_observer.py` | `VideoObserverAgent`; analyzes video → verdict, confidence, constraints |
| `policy_engine.py` | Action proposals (secondary path) |

### 4.7 Backends (`agents/backends/`)

| File | Role |
|------|------|
| `veo_backend.py` | Veo API → Gemini image → SDXL motion fallback |
| `svd_backend.py` | SVD + keyframes |
| `animatediff_backend.py` | AnimateDiff |
| `stub_backend.py` | Fake video for testing |

### 4.8 Models

| File | Role |
|------|------|
| `episode_outcome.py` | `EpisodeOutcome`, `EpisodeResult`, `TerminationReason` |
| `observation.py` | `ObservationResult`, `TaskContext`, `CharacterObservation` |
| `state_transition.py` | `TransitionStatus`, `ActionOutcome` |
| `world_state_graph.py` | `WorldStateNode`, `StateTransition` |

---

## 5. API Reference

### Primary Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/simulate` | Create and start simulation (plan → schedule → submit) |
| GET | `/episodes` | List simulations |
| GET | `/episodes/{id}` | Episode snapshot (state, beats, progress) |
| GET | `/world-state/{id}` | WorldStateGraph nodes and transitions |
| GET | `/episodes/{id}/result` | Outcome, confidence, cost, constraints_discovered |
| GET | `/diagnostics/queue` | Redis connection, queue lengths |

### Request/Response

**POST /simulate**
```
Query: world_id, goal, budget?, risk_profile?
Response: { simulation_id, status, initial_state }
```

**GET /episodes/{id}/result**
```
Response: {
  outcome: "goal_achieved" | "goal_impossible" | "goal_abandoned" | "dead_state" | "uncertain_termination" | "epistemically_incomplete" | "in_progress",
  confidence: float,
  total_cost_usd: float,
  constraints_discovered: string[],
  suggested_alternatives: string[],   // populated when exploratory + failure
  attempts_made: [{observability_attempt, render_hint}],
  state_delta: object,
  beats_attempted: int,
  beats_completed: int
}
```

---

## 6. Environment Variables

| Variable | Purpose |
|----------|---------|
| `REDIS_URL` | Redis connection (Upstash) |
| `JOB_QUEUE` | GPU job queue (default: `storyworld:gpu:jobs`) |
| `RESULT_QUEUE` | Result queue (default: `storyworld:gpu:results`) |
| `DATABASE_URL` | SQLite path |
| `REALITY_COMPILER_MODE` | Neutral style; no cinematic specs |
| `USE_OBSERVER_IN_PRODUCTION` | Run observer on each beat |
| `USE_MOCK_PLANNER` | Mock planner vs Gemini |
| `GEMINI_API_KEY` | For planner and observer |
| `S3_*` | Cloudflare R2 / S3 for artifacts |

**RunPod worker must use same `REDIS_URL`, `JOB_QUEUE`, `RESULT_QUEUE`** (no spaces around `=`).

---

## 7. Component Authority & Invariants

**Invariant:** `goal_achieved` is **invalid** if `WorldStateGraph.transitions == 0`. A goal may not be marked `goal_achieved` unless at least one observer-validated state transition exists in the WorldStateGraph. This is a hard correctness rule.

| Component | Authority | Role |
|-----------|-----------|------|
| `physics_veto.py` | NLP sanity filter | Pre-GPU gate; only for logical contradictions; never overrides observer |
| Planner | Hypothesis generator | Proposes beats |
| Video | Computation medium | Tests hypotheses |
| Observer | Ground truth | Verdicts from video |
| WorldStateGraph | Memory of truth | Validated states only |

---

## 8. Deployment

- **Control plane:** `uvicorn main:app --host 0.0.0.0 --port 8000`
- **GPU worker:** `python worker.py` (RunPod)
- **Redis:** Upstash or compatible
- **Storage:** Cloudflare R2 (S3-compatible) for video artifacts

---

## 9. Invariants Summary

| Invariant | Rule |
|-----------|------|
| Goal success | `goal_achieved` invalid if `WorldStateGraph.transitions == 0` |
| Beat semantics | Beats are execution scaffolding; observer-validated transitions define truth |
| Physics veto | May only mark IMPOSSIBLE for logical contradictions in language |
| UI | Must never require video playback to understand outcomes |

---

## 10. Related Documents

- `REALITY_COMPILER_CONTRACT.md` — Product contract
- `VALIDATION_CHECKLIST.md` — Validation checks
- `ARCHITECTURE.md` — Original architecture notes
- `env.example` — Environment template
