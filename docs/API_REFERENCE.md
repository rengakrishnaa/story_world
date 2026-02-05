# StoryWorld API Reference

Complete API endpoint documentation for the StoryWorld FastAPI server.

**Base URL:** `http://localhost:8000` (or your deployed URL)

---

## Primary Endpoints

### POST /simulate

**Description:** Start a new simulation from a goal. This is the primary interaction.

**Query parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `world_id` | string | Yes | World identifier (e.g., `default`) |
| `goal` | string | Yes | Simulation goal (physics-focused description) |
| `budget` | float | No | Max cost in USD |
| `risk_profile` | string | No | `low`, `medium`, or `high` |
| `requires_visual_verification` | boolean | No | API override for intent classifier |
| `problem_domain` | string | No | API override: `vehicle_dynamics`, `statics`, `structural`, `fluid`, `generic` |

**Example:**
```
POST /simulate?world_id=default&goal=A+robotic+arm+stacks+three+boxes&budget=5&risk_profile=medium
```

**Response (200):**
```json
{
  "simulation_id": "0fc963d8-d2b4-4aa9-b58d-7bded20d1be9",
  "status": "executing",
  "initial_state": "EXECUTING"
}
```

**Response (impossible):** When goal is logically contradictory (e.g., "stone floats unsupported"):
```json
{
  "simulation_id": "...",
  "status": "impossible",
  "initial_state": "IMPOSSIBLE"
}
```

---

### GET /episodes/{episode_id}

**Description:** Get episode status and snapshot.

**Response (200):**
```json
{
  "episode_id": "0fc963d8-d2b4-4aa9-b58d-7bded20d1be9",
  "state": "COMPLETED",
  "beats": [...],
  "progress": {...},
  "epistemic_summary": {...}
}
```

---

### GET /world-state/{episode_id}

**Description:** Get WorldStateGraph (nodes and transitions).

**Response (200):**
```json
{
  "episode_id": "...",
  "nodes": [...],
  "transitions": [...]
}
```

---

### GET /episodes/{episode_id}/result

**Description:** Get final result (outcome, confidence, constraints). Call when episode is terminal.

**Query parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `include_video` | boolean | false | Include video URLs in debug (presigned, expire ~1h) |

**Response (200):**
```json
{
  "episode_id": "0fc963d8-d2b4-4aa9-b58d-7bded20d1be9",
  "outcome": "goal_achieved",
  "state_delta": {
    "state": "COMPLETED",
    "outcome": "goal_achieved",
    "confidence": 0.75,
    "cost": 0.05,
    "constraints_discovered": ["insufficient_observational_evidence"],
    "progress": {"total_beats": 1, "completed": 1, "aborted": 0, "percent": 100},
    "state_nodes": 1,
    "transitions": 1
  },
  "confidence": 0.75,
  "total_cost_usd": 0.05,
  "constraints_discovered": ["insufficient_observational_evidence"],
  "metrics": {"beats_attempted": 1, "beats_completed": 1},
  "debug": {}
}
```

---

## Episode Lifecycle Endpoints

### POST /episodes

**Description:** Create an episode (manual flow; /simulate does this automatically).

**Query parameters:** `world_id`, `intent`

---

### POST /episodes/{episode_id}/plan

**Description:** Plan episode (generate beats). Called automatically by /simulate.

---

### POST /episodes/{episode_id}/execute

**Description:** Start decision loop for episode. /simulate submits beats directly.

---

### POST /episodes/{episode_id}/terminate

**Description:** Force-stop simulation.

**Query parameters:** `reason`, `outcome` (e.g., `goal_abandoned`)

---

## Utility Endpoints

### GET /episodes

**Description:** List recent episodes.

**Query parameters:** `limit`, `world_id`

---

### GET /episodes/{episode_id}/beats

**Description:** Get beat status (rendered, pending, etc.).

---

### GET /diagnostics

**Description:** System diagnostics (Redis, queues, config).

---

## Data Models

### Episode State

- `CREATED` — Initial
- `PLANNED` — Beats generated
- `EXECUTING` — Jobs submitted
- `COMPLETED` — All beats done
- `EPISTEMICALLY_BLOCKED` — Halted due to missing evidence
- `IMPOSSIBLE` — Goal vetoed
- `FAILED` — Error

### Outcome

- `goal_achieved` — Success
- `goal_impossible` — Veto
- `goal_abandoned` — Manual/budget
- `dead_state` — Physics constraint
- `epistemically_incomplete` — Evidence blocked
- `uncertain_termination` — Observer uncertain
