# StoryWorld — Reality Compiler Product Contract

**Identity:** StoryWorld is a **reality compiler**, not a content generator.  
**Operating principle:** Video is the execution medium. State and truth are the product.

---

## 1. What StoryWorld IS

| Component | Role |
|-----------|------|
| **Simulation runtime** | Video is the state transition function |
| **Decision engine** | Answers: "Is this sequence possible? What constraints make it impossible?" |
| **Verifier** | Discovers constraints through observation |
| **Compiler** | Input: Simulation Goal → Output: Validated World State + Outcome |

**Product output:**
- Final outcome (GOAL_ACHIEVED, GOAL_IMPOSSIBLE, GOAL_ABANDONED, DEAD_STATE)
- WorldStateGraph (validated state transitions)
- Confidence and cost
- Discovered constraints

---

## 2. What StoryWorld IS NOT

| Prohibited | Rule |
|------------|------|
| Video generator | Video is never the primary output |
| Content creation tool | No feature exists only to improve aesthetics |
| Story engine | No story-specific assumptions |
| Media platform | UI is infrastructure console |
| Retry-until-perfect | Goals may fail; no logic forces success |

**Hard rules:**
- If not observable in video → it does not exist in world state
- Planners propose; observers assert
- All state mutations must reference an `observation_id`
- No state without observation provenance

---

## 3. Single Source of Truth

```
Planner → proposes actions
Observer → asserts facts (from video)
WorldStateGraph → DAG of validated states
```

**Rule:** Only observers may assert facts. Planner-invented or off-camera state must be rejected.

---

## 4. Core Responsibilities

| Component | Responsibility |
|-----------|----------------|
| Simulation Goal | Initial objective; may be invalidated or proven impossible |
| Policy Engine | Proposes actions; can stop, abandon, do nothing |
| Video Rendering | Tests hypotheses; disposable after observation |
| Video Observer | Extracts state; issues verdicts (VALID, IMPOSSIBLE, UNCERTAIN) |
| WorldStateGraph | DAG of validated states; every edge tied to observation |
| Outcome | GOAL_ACHIEVED, GOAL_IMPOSSIBLE, DEAD_STATE, GOAL_ABANDONED |

---

## 5. Required Non-Functional Properties

- **Determinism** where possible; explicit uncertainty when not
- **Failure** as first-class outcome
- **Auditability**: state → observation → video
- **Video deletion**: system must work if videos are deleted after observation

---

## 6. Reality Compiler Mode (REALITY_COMPILER_MODE=true)

When enabled via environment variable:

| Setting | Effect |
|---------|--------|
| **Planner** | No cinematic_spec, style_profile, or genre; minimal physics-focused beats |
| **Style** | Always "neutral" (no cinematic/story styling) |
| **Retry** | Observer verdict IMPOSSIBLE/CONTRADICTS → no retry; terminate |
| **Result** | Must include `constraints_discovered`, `observation_id` when available |

---

## 7. API Contract: /episodes/{id}/result

```json
{
  "outcome": "goal_achieved|goal_impossible|goal_abandoned|dead_state|in_progress",
  "state_delta": { ... },
  "confidence": 0.0–1.0,
  "total_cost_usd": 0.0,
  "constraints_discovered": [ "constraint_1", "constraint_2" ],
  "metrics": { "beats_attempted": 3, "beats_completed": 3 },
  "debug": null
}
```

Video URLs are optional. API consumers must not need video to understand results.

---

## 8. Acceptance Test (All Must Be YES)

1. Would the system work if videos were deleted after observation?
2. Can a user understand results without watching video?
3. Can goals fail without retries or "story fixes"?
4. Is video used to discover truth, not to impress humans?

**If any answer is NO → the change violates the product.**

---

## 9. Environment Variables for Reality Compiler

```bash
REALITY_COMPILER_MODE=true
USE_MOCK_PLANNER=false
# Policies
SIM_RETRY_MAX_ATTEMPTS=2
# Queues
RESULT_QUEUE=storyworld:gpu:results
JOB_QUEUE=storyworld:gpu:jobs
```
