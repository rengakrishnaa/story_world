# StoryWorld Changelog & Roadmap

## Completed Features

### Video Pipeline

| Feature | Description |
|---------|-------------|
| **Veo as default backend** | Veo 3.1 API is the default for best quality video generation |
| **Credit exhaustion fallback** | When Veo returns 429, worker automatically falls back to SVD (free open-source) |
| **SVD/AnimateDiff fallback** | SVD and AnimateDiff support `_credit_exhausted` to skip API and use local SDXL |
| **Veo internal fallback** | On Veo 429, Veo backend skips Gemini and goes straight to SDXL |
| **Display mapping** | `insufficient_evidence` displayed as `insufficient_observational_evidence` in UI |
| **Backend selection** | `DEFAULT_BACKEND` and `VEO_FALLBACK_BACKEND` configurable via env |

### Epistemic Architecture

| Feature | Description |
|---------|-------------|
| **Closed-form solver-only success** | When intent is closed-form (mass, geometry specified) and observer returns uncertain with missing evidence, accept solver-only success (goal_achieved, 0.75 confidence) |
| **Epistemic evaluator** | Determines ACCEPTED vs EPISTEMICALLY_INCOMPLETE vs REJECTED based on evidence and intent |
| **Observer optional for closed-form** | Infrastructure failure (429, timeout) does NOT block closed-form problems |
| **insufficient_observational_evidence** | New status for observer could not extract evidence; solver-sufficient path |

### Intent Classification

| Feature | Description |
|---------|-------------|
| **LLM-based classifier** | Gemini → Ollama → rule-based fallback |
| **Problem domain** | vehicle_dynamics, statics, structural, fluid, generic |
| **requires_visual_verification** | Drives epistemic evaluator |
| **observer_impact** | blocking vs confidence_only |

### Observer System

| Feature | Description |
|---------|-------------|
| **Gemini primary** | Vision model for video observation |
| **Ollama fallback** | LLaVA when Gemini 429 |
| **Mock fallback** | Uncertain when both fail |
| **Verdicts** | valid, uncertain, impossible, contradicts, blocks |

### Infrastructure

| Feature | Description |
|---------|-------------|
| **Redis job queue** | Upstash Redis for storyworld:gpu:jobs, storyworld:gpu:results |
| **Cloudflare R2** | S3-compatible storage for video artifacts |
| **ResultConsumer** | Background consumer: GPU results → observer → DB |
| **WorldStateGraph** | DAG of validated state transitions with provenance |
| **clear_job_queue.py** | Utility to clear old jobs from Redis |
| **Debug logging** | DEFAULT_BACKEND and beat backend logged at startup/plan time |

### Reality Compiler Mode

| Feature | Description |
|---------|-------------|
| **REALITY_COMPILER_MODE** | Neutral style, minimal physics-focused beats |
| **No cinematic specs** | No cinematic_spec, style_profile in beats |
| **Observer-driven retry** | Re-render for insufficient_evidence (observability cap) |

### Exploratory Mode (Risk Profile High)

| Feature | Description |
|---------|-------------|
| **Exploratory retries** | On uncertain verdict, retry with different camera/framing (side view, overhead, etc.) instead of halting |
| **Observability augmentation** | Each retry uses a different render hint from `get_render_hints()`; stops when cap reached |
| **suggested_alternatives** | When exploratory run fails, result includes actionable suggestions—from observer or derived from framings tried |
| **attempts_made** | List of `{observability_attempt, render_hint}` so you see what we tried |
| **UI panel** | Simulation detail shows "Exploratory: Possible Ways Forward" when these are present |

---

## Known Issues / Limitations

| Issue | Workaround |
|-------|------------|
| **Black video with stub backend** | Set `DEFAULT_BACKEND=veo` or `svd` in `.env`; restart main server; run new simulation |
| **Old jobs with wrong backend** | Run `python clear_job_queue.py` before new simulations |
| **EPISTEMICALLY_BLOCKED when expecting goal_achieved** | Ensure intent is closed-form (mass, geometry specified); epistemic evaluator now accepts solver-only for closed-form when observer has insufficient evidence |
| **SDXL produces zoom-like motion** | SDXL + motion interpolation may show camera motion vs object motion; use Veo for better temporal quality |
| **Video URLs expire** | Presigned URLs expire ~1h; video is debug artifact, not primary output |

---

## Roadmap

### Near-Term (Planned)

| Item | Description |
|------|-------------|
| **Multi-beat temporal coherence** | Improve keyframe selection for SDXL/AnimateDiff to show distinct action phases |
| **Enhanced observability re-render** | Smarter camera/scale choices when observer returns insufficient_evidence (partially done via exploratory mode) |
| **Observer evidence extraction** | Improve observer prompts to extract acceleration_vector, center_of_mass, etc. for structural intents |
| **Metrics dashboard** | Aggregated confidence, cost, success rate over time |
| **Batch simulation API** | Submit multiple goals; return batch results |

### Medium-Term

| Item | Description |
|------|-------------|
| **Multi-backend comparison** | Run same goal on Veo vs SVD vs AnimateDiff; compare outcome and quality |
| **PostgreSQL support** | Production-grade DB for high throughput |
| **Per-episode result queues** | Avoid result mixing when multiple episodes run concurrently |
| **Webhook notifications** | Notify external systems when episode completes |
| **Video stitching** | Compose beat videos into episode video (optional) |

### Long-Term

| Item | Description |
|------|-------------|
| **Real-time streaming** | Stream video as it renders; partial observations |
| **Interactive refinement** | User feedback loop to refine goal or re-run with constraints |
| **Custom physics constraints** | User-defined constraint schemas per domain |
| **Federated observers** | Multiple observer backends; consensus or majority vote |
| **Cost optimization** | Adaptive backend selection based on intent complexity |

---

## Version History

- **Current** — Veo default, credit-exhausted fallback, closed-form solver-only success, epistemic architecture, LLM intent classifier, full documentation
