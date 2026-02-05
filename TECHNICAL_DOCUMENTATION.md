# StoryWorld Technical Documentation

## System Overview

**StoryWorld** is a **Computational Video Infrastructure** that uses video generation as a state transition function. It simulates complex causal interactions by rendering a stream of video, observing the results, and strictly enforcing physical consistency.

**It is NOT a media engine.** It is a physics engine where the physics are learned by a VLM.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       COMPUTATIONAL VIDEO PIPELINE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Simulation Goal ──► Policy Engine ──► Runtime ──► GPU Renderer            │
│          │                                  │              │                │
│          │                                  ▼              │                │
│          │                        ┌─────────────────┐      │                │
│          │                        │  Decision Loop  │◄─────┘                │
│          │                        │  (Orchestrator) │                       │
│          │                        └────────┬────────┘                       │
│          │                                 │                                │
│          ▼                                 ▼                                │
│    World State ◄─────────────────── Video Observer                          │
│       Graph                           (Verdict)                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Capabilities (Phases 1-6)

| Capability | Description | Phase |
|------------|-------------|-------|
| **World State Persistence** | Versioned, forkable DAG of world states stored in SQLite | Phase 1 |
| **Video Observation** | Machine vision (Gemini) extracts structured state from video | Phase 2 |
| **Adaptive Quality** | Dynamic quality thresholds based on task risk profile | Phase 3 |
| **Elastic Budgeting** | Value-driven resource allocation with diminishing returns logic | Phase 3 |
| **Hierarchical Control** | Macro Intention (fixed) → Story Beat (flexible) → Micro Actions | Phase 4 |
| **Pipeline Orchestration** | Closed-loop execution handling rendering, observing, and updating | Phase 5 |
| **Video-Native Compliance**| First-class failure states, observer vetoes, state-centric APIs | Phase 6 |

---

## Architecture Internal Breakdown

### 1. The World State Graph (Phase 1)
*Treats video as a state transition function.*
- **Model**: `WorldStateGraph` (DAG structure)
- **Persistence**: SQLite (`world_state_nodes`, `state_transitions`)
- **Key Feature**: Every video generation attempts to transition the world from State A → State B. If the video fails (physically or narratively), the transition is rejected.

### 2. The Observer System (Phase 2 & 6)
*The tracking authority of the simulation.*
- **Agent**: `VideoObserverAgent`
- **Mechanism**: Extracts structured JSON (Characters, Environment, Actions) from video frames.
- **Phase 6 Upgrade**: Can issue **Verdicts** (`VALID`, `IMPOSSIBLE`, `CONTRADICTS`).
    - *Example*: If a character flies but has no flight ability, Observer returns `IMPOSSIBLE`.
    - *Result*: Episode terminates or branches to failure state.

### 3. Economic & Quality Control (Phase 3)
*Resource optimization for computation.*
- **Components**: `QualityEvaluator`, `BudgetController`, `ValueEstimator`.
- **Logic**: 
    - High-stakes beats (climax) get higher budget/quality thresholds.
    - Low-stakes beats (transition) accept lower fidelity.
    - Dead-end detection prevents wasting GPU credits on failing branches.

### 4. Hierarchical Director (Phase 4)
*Narrative strategy engine.*
- **Components**: `StoryDirector`, `StoryIntentGraph`.
- **Hierarchy**:
    1. **Macro Intent**: "Hero defeats Villain" (Immutable Goal)
    2. **Beat**: "Sword fight sequence" (Adaptable Plan)
    3. **Action**: "Swing sword" (Reactive Execution)
- **Adaptation**: If a beat fails (e.g., character missing), the Director attempts `SUBSTITUTE` or `ADAPT` strategies before failing the Macro Intent.

### 5. Video-Native Compliance (Phase 6)
*Enforcing simulation integrity.*
- **Failure as First-Class**: Episodes can end in `GOAL_IMPOSSIBLE` or `DEAD_STATE`.
- **Termination Authority**: The Observer can force termination via `forces_termination=True`.
- **State-Centric Outcomes**: Consumers receive a `State Delta`, not just a video file.
- **Video as Debug**: Video files are treated as ephemeral debug artifacts, not the primary product.

---

## API Reference

### Primary Endpoints (Simulation-Centric)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/simulate` | **Primary Interaction**. Initialize a new simulation from a Goal. |
| `POST` | `/simulate/{id}/execute` | Run the decision loop (Plan -> Render -> Observe) |
| `GET`  | `/simulate/{id}/result`  | Returns final world state delta & outcome |
| `GET`  | `/world-state/{id}`      | Inspect the full state change graph |

### Debug & Control Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/episodes/{id}/terminate` | Force-stop simulation (`outcome=goal_abandoned`) |
| `GET`  | `/episodes/{id}/video`     | **Debug Only**. Get ephemeral link to video artifact. |
| `GET`  | `/phase-status`            | Check operational status of all subsystems |

---

## Data Models

### Episode Outcome
Strict outcomes ensure simulation validity:
- `GOAL_ACHIEVED`: Simulation reached target state.
- `GOAL_IMPOSSIBLE`: Laws of physics/world violated (Observer Veto).
- `DEAD_STATE`: No valid actions remain to reach goal.
- `GOAL_ABANDONED`: Budget exhausted or manual termination.

### Observation Verdict
The Observer's judgment on a state transition:
- `VALID`: State transition accepted.
- `DEGRADED`: Accepted with quality warnings.
- `IMPOSSIBLE`: Rejected (Physics violation).
- `CONTRADICTS`: Rejected (Continuity violation).

---

## Deployment & Configuration

### Infrastructure Stack
- **Compute**: Local/Cloud FastAPI Server + Distributed RunPod GPU Workers.
- **Storage**: SQLite (State), Cloudflare R2 (Artifacts), Upstash Redis (Queues).
- **Vision**: Gemini Pro Vision (or local alternative).

### Environment Variables (.env)
```bash
# Core
DATABASE_URL=sqlite:///./local.db
REDIS_URL=rediss://default:***@...

# Feature Flags (Video-Native)
PHASE_6_ENABLED=true
STRICT_FAILURE_MODES=true

# External Services
GEMINI_API_KEY=***
S3_ENDPOINT=https://***.r2.cloudflarestorage.com
```

---

## System Statistics

| Metric | Status |
|--------|--------|
| **Test Coverage** | ~230 Tests covering all 6 phases |
| **Pipeline Latency** | Optimized via parallel GPU dispatch & caching |
| **Failure Handling** | Strict. Impossible actions terminate simulation. |
| **State Tracking** | 100% Observable Facts (No planner hallucinations) |

---

## Future Roadmap

- **Multi-Observer Consensus**: Voting system for higher confidence verdicts.
- **Local Vision Models**: Replace Gemini with local VLM for privacy/cost.
- **Real-time Branching**: User intervention during `IN_PROGRESS` simulation.
