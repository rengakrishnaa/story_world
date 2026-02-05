# StoryWorld Product Documentation

## Product Identity

**StoryWorld is a reality compiler, not a content generator.**

| Principle | Meaning |
|-----------|---------|
| **Video is the execution medium** | Video is used to test hypotheses; it is not the primary output |
| **State and truth are the product** | Output: outcome, WorldStateGraph, confidence, constraints |
| **Failure is first-class** | Episodes may end in GOAL_IMPOSSIBLE, DEAD_STATE, GOAL_ABANDONED |
| **Observer is authority** | Only the VideoObserver may assert facts from video; planners propose |
| **Auditability** | Every state transition is tied to an observation and video provenance |

---

## What StoryWorld IS

| Component | Role |
|-----------|------|
| **Simulation runtime** | Video is the state transition function |
| **Decision engine** | Answers: "Is this sequence possible? What constraints make it impossible?" |
| **Verifier** | Discovers constraints through video observation |
| **Compiler** | Input: Simulation Goal → Output: Validated World State + Outcome |
| **Epistemic engine** | Enforces evidence requirements; closed-form physics vs. perceptual dynamics |

**Product output:**
- **Outcome:** `goal_achieved` | `goal_impossible` | `goal_abandoned` | `dead_state` | `epistemically_incomplete`
- **WorldStateGraph:** DAG of validated state transitions with observation provenance
- **Confidence:** 0.0–1.0, derived from observer agreement and evidence quality
- **Cost:** Compute cost in USD
- **Constraints discovered:** List of physical or epistemic constraints inferred from observations

---

## What StoryWorld IS NOT

| Prohibited | Rule |
|------------|------|
| Video generator | Video is never the primary output |
| Content creation tool | No feature exists only to improve aesthetics |
| Story engine | No story-specific assumptions; physics-focused |
| Media platform | UI is infrastructure console |
| Retry-until-perfect | Goals may fail; no logic forces success |
| Creative prompt tool | Goals should describe physical scenarios, not narrative |

**Hard rules:**
- If not observable in video → it does not exist in world state
- Planners propose; observers assert
- All state mutations must reference an `observation_id`
- No state without observation provenance

---

## Core Capabilities

### 1. Simulation Lifecycle

1. **Goal input** — User submits a physics-focused goal (e.g., stacking boxes, vehicle dynamics)
2. **Impossibility gate** — NLP veto blocks only hard logical contradictions (e.g., "stone floats unsupported")
3. **Planning** — Narrative planner generates execution beats with backend selection (Veo, SVD, AnimateDiff, stub)
4. **Rendering** — GPU worker renders video via selected backend; uploads to Cloudflare R2
5. **Observation** — Video Observer (Gemini/Ollama) extracts state and issues verdict
6. **Epistemic evaluation** — Determines if evidence is sufficient; closed-form intents can succeed without full observer evidence
7. **State transition** — Validated transitions recorded in WorldStateGraph
8. **Outcome** — Final outcome, confidence, constraints returned to user

### 2. Epistemic Architecture

- **Closed-form physics** (mass, geometry, friction, gravity specified): observer optional; solver-only success allowed
- **Perceptual dynamics** (vehicle turn, speed, etc.): observer required for epistemic validity
- **Observer infrastructure failure** (429, timeout): must NOT block closed-form problems

### 3. Video Backends

| Backend | Description | Fallback |
|---------|-------------|----------|
| **Veo** | Google Veo 3.1 API; best quality | SVD when credit exhausted |
| **SVD** | Stable Video Diffusion; keyframes + motion | Local SDXL when API fails |
| **AnimateDiff** | Keyframes + motion interpolation | Local SDXL |
| **Stub** | Black placeholder; testing only | — |

**Credit exhaustion:** When Veo returns 429, worker automatically falls back to SVD (free open-source).

### 4. Observer System

- **Primary:** Gemini (vision model)
- **Fallback:** Ollama with LLaVA when Gemini 429
- **Mock:** Returns uncertain when both fail
- **Verdicts:** `valid` | `uncertain` | `impossible` | `contradicts` | `blocks`

### 5. Intent Classification

- **LLM-based** (Gemini → Ollama → rule-based)
- Determines: `requires_visual_verification`, `problem_domain`, `observer_impact`
- Drives epistemic evaluator and physics constraint selection

---

## User Personas

| Persona | Use case |
|---------|----------|
| **Simulation operator** | Run physics simulations; interpret outcomes |
| **Enterprise API consumer** | Integrate simulation into workflows; use `requires_visual_verification` override |
| **DevOps / SRE** | Deploy main server + GPU worker; configure queues, R2, Redis |
| **Developer** | Extend backends, observers; contribute to codebase |

---

## Non-Functional Properties

- **Determinism** where possible; explicit uncertainty when not
- **Failure** as first-class outcome
- **Auditability:** state → observation → video
- **Video deletion:** system must work if videos are deleted after observation
- **Presigned URLs:** video URLs expire (~1h); not primary output
