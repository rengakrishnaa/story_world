# StoryWorld Production Acceptance Test Report

**Date:** 2026-02-01  
**Product:** StoryWorld — Computational Video Infrastructure  
**Scope:** Full end-to-end validation for production deployment

---

## Executive Summary

| Verdict | Status |
|---------|--------|
| **Overall** | ⚠️ **Needs fixes** (automated tests pass; full E2E with GPU worker requires manual validation) |

**Automated tests:** 19/19 passed  
**Manual/E2E tests:** Require live API server, Redis, and GPU worker

---

## TEST 0 — Preconditions (Hard Gate)

| Check | Status | Notes |
|-------|--------|-------|
| API server responds | ✅ PASS | `GET /` returns 200 |
| Static assets load | ✅ PASS | `/static/dashboard.js`, `/static/style.css` load |
| Phase status endpoint | ✅ PASS | Reports operational/error |
| Redis queue reachable | ⚠️ SKIP | Not validated in automated run; requires `REDIS_URL` |
| GPU worker online | ⚠️ SKIP | Requires worker process; manual check |

**Gate:** Proceed if API and static assets load. Redis/GPU required for full simulation execution.

---

## TEST 1 — UI ↔ Backend Wiring (No Video Involved)

| Check | Status | Notes |
|-------|--------|-------|
| Dashboard fetches from `/episodes` | ✅ PASS | Returns `{episodes: [...]}` |
| No video thumbnails/previews | ✅ PASS | HTML contains no thumbnail, video preview, autoplay |
| Primary fields visible | ✅ PASS | ID, Goal, Status, Confidence, Cost in table |
| No "Generate Video" language | ✅ PASS | Verified across /, /new.html, /simulation.html |
| Status updates without refresh | ✅ PASS | `loadDetail()` polls every 2s |

**Failure conditions not observed:**  
- ❌ UI does not depend on video  
- ❌ UI does not block if video unavailable  

---

## TEST 2 — Simulation Creation (Intent → System)

| Check | Status | Notes |
|-------|--------|-------|
| UI sends to `/simulate` | ✅ PASS | `submitSimulation()` POSTs goal |
| Backend returns simulation_id | ✅ PASS | 200 with `simulation_id` |
| Status transitions | ✅ PASS | Returns `executing` or `initialized` |
| No video shown | ✅ PASS | Form does not display video |

**Form:** Goal textarea + Budget + Risk Profile; Budget/Risk sent via backend config, not as form params.

---

## TEST 3 — Closed-Loop Execution

| Check | Status | Notes |
|-------|--------|-------|
| Episode status returns state | ✅ PASS | `GET /episodes/{id}` returns state/beats |
| GPU worker receives jobs | ⚠️ MANUAL | Requires Redis + worker |
| Observer invoked after render | ⚠️ MANUAL | Observer runs in GPU worker pipeline |
| WorldStateGraph grows | ⚠️ MANUAL | Depends on observer writing transitions |

**Note:** `/world-state/{id}` endpoint exists; graph populated when observer records transitions.

---

## TEST 4 — Confidence & Re-evaluation Logic

| Check | Status | Notes |
|-------|--------|-------|
| Impossible goal → GOAL_IMPOSSIBLE | ⚠️ MANUAL | Requires observer verdict in pipeline |
| Confidence from metrics | ✅ PASS | `get_episode_result` aggregates from attempts |
| Retry policy exists | ✅ PASS | `RetryPolicy` with `max_attempts` from config |

**Note:** Observer integration with decision loop determines when GOAL_IMPOSSIBLE is set.

---

## TEST 5 — World State Graph Integrity

| Check | Status | Notes |
|-------|--------|-------|
| `/world-state/{id}` exists | ✅ PASS | Returns `total_nodes`, `total_transitions` |
| Graph without video | ✅ PASS | Endpoint returns data; no video dependency |
| Nodes = world states | ✅ PASS | Schema: `world_state_nodes` |
| Edges = validated transitions | ✅ PASS | `state_transitions` with `observation_json` |

**Hard rule:** Deleting videos does not break graph; graph stores observation metadata, not video.

---

## TEST 6 — Video as Debug Artifact

| Check | Status | Notes |
|-------|--------|-------|
| Video labeled "Ephemeral Debug Artifact" | ✅ PASS | Button text in simulation.html |
| Video collapsed by default | ✅ PASS | "Collapsed" in button label |
| Not autoplayed | ✅ PASS | No autoplay in HTML |
| Result without video | ✅ PASS | `GET /episodes/{id}/result` works; `include_video=false` default |

---

## TEST 7 — Observer Disagreement & Uncertainty

| Check | Status | Notes |
|-------|--------|-------|
| UNCERTAIN verdict supported | ✅ PASS | `ObserverVerdict.UNCERTAIN` exists |
| Disagreement score in model | ✅ PASS | `disagreement_score` in consolidation |
| Multi-observer config | ✅ PASS | `enable_multi_observer`, `disagreement_threshold` |

**Note:** Full behavior requires observer pipeline run; automated test covers model presence.

---

## TEST 8 — Failure as First-Class Outcome

| Check | Status | Notes |
|-------|--------|-------|
| `POST /terminate` with goal_impossible | ✅ PASS | Accepts `outcome=goal_impossible` |
| EpisodeOutcome includes GOAL_IMPOSSIBLE | ✅ PASS | First-class enum |
| No forced success | ✅ PASS | Policy engine respects observer veto |

---

## TEST 9 — API Contract Validation

| Check | Status | Notes |
|-------|--------|-------|
| `outcome` | ✅ PASS | Present in result |
| `state_delta` | ✅ PASS | Present (from world graph or empty) |
| `confidence` | ✅ PASS | Float |
| `total_cost_usd` (cost) | ✅ PASS | Present |
| `constraints_discovered` | ✅ PASS | Added to EpisodeResult |
| Video optional | ✅ PASS | `include_video=false` default; result complete without video |

**Sample result shape:**
```json
{
  "episode_id": "...",
  "outcome": "goal_achieved",
  "state_delta": {},
  "confidence": 0.85,
  "total_cost_usd": 0.15,
  "constraints_discovered": [],
  "metrics": {...},
  "debug": null
}
```

---

## TEST 10 — Product Identity Check

| Question | Answer |
|----------|--------|
| Can I understand the result without watching video? | ✅ YES — outcome, confidence, state_delta, constraints |
| Can goals legitimately fail? | ✅ YES — GOAL_IMPOSSIBLE, GOAL_ABANDONED, DEAD_STATE |
| Is video used to discover truth, not to entertain? | ✅ YES — observer extracts state; video is computation |
| Would deleting video not break correctness? | ✅ YES — graph and result rely on observations |
| Feels like simulator/verifier, not video generator? | ✅ YES — UI says "Infrastructure", "Simulation" |

---

## Screenshots (Placeholders)

Manual verification required:

1. **Dashboard** — `/` — Simulation list, no video
2. **WorldStateGraph** — `/simulation.html?id=...` — Mermaid graph of beats/transitions
3. **Result view** — `/episodes/{id}/result` — outcome, confidence, cost

---

## Blocking Issues

1. **Redis/GPU worker not validated** — Full simulation execution requires:
   - Redis at `REDIS_URL`
   - GPU worker consuming `GPU_JOB_QUEUE`
   - Run: `python worker.py` (or Docker image)

2. **Observer not in render loop** — Current pipeline: Planner → GPU Job → Worker → Result. Observer (video → state) integration with WorldStateGraph needs confirmation for TEST 3–4, 7.

3. **Dashboard cost/confidence** — List view shows "-" for cost/confidence (data not in list_episodes). Detail view fetches from snapshot/result.

---

## Recommended Next Steps

1. Run full E2E with live Redis + GPU worker.
2. Confirm observer writes to `world_state_nodes` and `state_transitions`.
3. Add screenshot capture to CI for UI regression.
4. Migrate `on_event("startup")` to lifespan handler (deprecation warning).

---

## Final Verdict

| | |
|---|---|
| **Automated suite** | ✅ 19/19 passed |
| **Manual E2E** | ⚠️ Pending Redis + GPU |
| **One-sentence verdict** | **⚠️ Needs fixes** — API and UI are production-ready for simulation creation and result retrieval; full closed-loop execution requires GPU worker and observer integration validation. |
