# StoryWorld User Manual

This manual explains how to use StoryWorld to run simulations, interpret results, and operate the system.

---

## Quick Start

### Prerequisites

- Web browser (Chrome, Firefox, Safari, Edge)
- StoryWorld server running (see [Deployment Guide](DEPLOYMENT_GUIDE.md))

### Running a Simulation

1. **Open the dashboard**
   - Navigate to `http://localhost:8000` (or the deployed URL)
   - Click **New Run** or **Simulations**

2. **Enter a simulation goal**
   - Describe a physical scenario. Example:
     > A robotic arm stacks three identical boxes vertically on a flat surface without tipping. Each box weighs 1.5 kg, measures 30 cm per side, and has a friction coefficient of 0.6. Gravity is 9.8 m/s². The arm places each box gently with zero angular velocity.
   - Optionally set **Budget** (e.g., 5) and **Risk Profile** (low, medium, high)

3. **Submit**
   - Click **Run Simulation**
   - You receive a simulation ID (e.g., `0fc963d8-d2b4-4aa9-b58d-7bded20d1be9`)

4. **Monitor progress**
   - The UI polls automatically
   - Status progresses: **EXECUTING** → **COMPLETED** (or **EPISTEMICALLY_BLOCKED**, **FAILED**)

5. **View result**
   - When terminal, the **State Delta (JSON)** and **Confidence Score** appear
   - Optionally expand **Ephemeral Debug Artifacts** for video links (presigned, expire ~1h)

---

## Running a Simulation

### Simulation Goal Best Practices

**Good goals (physics-focused):**
- Specify mass, dimensions, friction, gravity when relevant
- Describe what should happen, not how it should look
- Use physical terms: stack, tip, fall, friction, gravity, load

**Avoid:**
- Creative or narrative language (e.g., "cinematic shot", "epic scene")
- Vague descriptions without physical parameters
- Goals that are logically contradictory (e.g., "stone floats unsupported")

### Budget and Risk Profile

| Parameter | Description | Default |
|-----------|-------------|---------|
| **Budget** | Max cost in USD for the simulation | From policy |
| **Risk profile** | Affects quality thresholds (low/medium/high) | medium |

---

## Understanding Results

### Outcome Values

| Outcome | Meaning |
|---------|---------|
| **goal_achieved** | Simulation reached target state; observer validated |
| **goal_impossible** | Goal violates laws of physics or logic (e.g., veto) |
| **goal_abandoned** | Budget exhausted or manual termination |
| **dead_state** | No valid actions remain; physics constraint triggered |
| **epistemically_incomplete** | Observer could not extract required evidence; perceptual intent blocked |
| **uncertain_termination** | Observer uncertain despite evidence; epistemic halt |

### Confidence Score

- **0.0–1.0** — System certainty in the outcome
- **Decreases** when: observers disagree, outcomes depend on narrow tolerances, failure occurs late
- **Solver-only success:** 0.75 when observer unavailable but intent is closed-form

### State Delta (JSON)

The state delta contains:

| Field | Description |
|-------|-------------|
| `state` | Episode state (COMPLETED, EPISTEMICALLY_BLOCKED, etc.) |
| `outcome` | Final outcome |
| `confidence` | Confidence score |
| `cost` | Compute cost in USD |
| `constraints_discovered` | List of constraints (e.g., `insufficient_observational_evidence`, `observer_unavailable`) |
| `progress` | Beats attempted, completed, aborted |
| `state_nodes` | Number of nodes in WorldStateGraph |
| `transitions` | Number of validated transitions |

### Constraints Discovered

| Constraint | Meaning |
|------------|---------|
| **insufficient_observational_evidence** | Observer could not extract physics from video; solver-sufficient for closed-form |
| **observer_unavailable** | Observer failed (API 429, timeout); solver-only success used |
| **insufficient_physical_evidence** | Perceptual intent; observer required; blocked |
| **observer_exception** | Observer threw an error |

---

## Dashboard and UI

### Dashboard (index.html)

- Lists recent simulations
- Shows episode ID, state, updated time
- Click an episode to view detail

### Simulation Detail (simulation.html)

- **World State Graph (Visual):** Nodes and transitions
- **Confidence Score:** Numeric display
- **Compute Cost:** USD
- **State Delta (JSON):** Full result payload
- **Ephemeral Debug Artifacts:** Video URLs (if available; presigned, expire ~1h)

### UI Invariant

**The UI never requires video playback to understand outcomes.** Outcome, confidence, and WorldStateGraph are sufficient.

---

## API Usage

For programmatic access, see [API Reference](API_REFERENCE.md). Basic flow:

1. `POST /simulate?goal=...&budget=5` — Start simulation
2. `GET /episodes/{id}` — Poll status
3. `GET /world-state/{id}` — World state graph
4. `GET /episodes/{id}/result?include_video=false` — Final result

---

## Troubleshooting

See [Troubleshooting](TROUBLESHOOTING.md) for common issues such as:
- Black/blank video (stub backend; check DEFAULT_BACKEND)
- EPISTEMICALLY_BLOCKED when expecting goal_achieved (closed-form intent; check intent classifier)
- Old jobs with wrong backend (clear Redis queue)
