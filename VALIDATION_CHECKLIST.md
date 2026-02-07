# StoryWorld Validation Checklist

**Purpose:** Ensure product matches "reality compiler" contract before deployment.

---

## 1. Input Correctness

| Check | Status |
|-------|--------|
| Input framed as Simulation Goal, not "generate video" | ✅ Form label, placeholder |
| Goal describes what should happen, not how it looks | ✅ Hint: avoid scene, cinematic, shot |
| Budget explicit and enforced | ✅ Form + /simulate?budget= |
| Risk/entropy affects exploration | ✅ Form + policies.risk_profile; exploratory (high) retries with different framings, returns suggested_alternatives on failure |
| Red flags: "scene", "cinematic", "shot", "make it look good" | ✅ Server logs warning; client hint |

---

## 2. Execution Loop Integrity

| Check | Status |
|-------|--------|
| Each proposal triggers render | ✅ Beats → GPU jobs |
| Each render followed by VideoObserver | ✅ ResultConsumer._observe_and_record |
| Observer returns verdict, confidence, causal | ✅ ObservationResult |
| System loops when confidence insufficient | ✅ RetryPolicy |
| Stops on: goal achieved, impossible, dead, budget | ✅ Terminate, RetryPolicy ABORT |

---

## 3. Video Role

| Check | Status |
|-------|--------|
| Video generated to test hypothesis | ✅ Beat → render |
| Video consumed by Observer, not humans | ✅ Observer.observe() |
| Video optional in API | ✅ include_video=false default |
| Labeled "Ephemeral Debug Artifact" | ✅ UI button |
| UI does not highlight video as success | ✅ Collapsed, debug section |

---

## 4. WorldStateGraph

| Check | Status |
|-------|--------|
| Every node = world state | ✅ WorldStateNode |
| Every edge = validated transition | ✅ StateTransition |
| Rejected transitions visible | ✅ Beat ABORTED in graph |
| No state without observation_id | ✅ record_beat_observation provenance |
| Planner cannot inject state | ✅ Only observer writes |

---

## 5. Confidence & Uncertainty

| Check | Status |
|-------|--------|
| Confidence 0.0–1.0 | ✅ EpisodeResult |
| Low confidence → explore or terminate | ✅ RetryPolicy |
| Observer disagreement surfaced | ✅ ObservationResult.disagreement_score |
| UNCERTAIN valid, not error | ✅ Verdict enum |

---

## 6. Failure Semantics

| Check | Status |
|-------|--------|
| Physically impossible → GOAL_IMPOSSIBLE | ✅ terminate_episode |
| No valid actions → DEAD_STATE | ✅ terminate_episode |
| Budget out → GOAL_ABANDONED | ✅ Policies |
| Failure = correct behavior | ✅ First-class outcomes |

---

## 7. UI

| Check | Status |
|-------|--------|
| Dashboard: Outcome, Confidence, Cost, Status | ✅ Detail view |
| WorldStateGraph centerpiece | ✅ Graph + state delta |
| Video collapsed by default | ✅ Debug artifacts |
| No thumbnails, autoplay | ✅ |
| User understands without video | ✅ State, outcome, constraints |

---

## 8. API Contract

| Check | Status |
|-------|--------|
| /episodes/{id}/result without video | ✅ include_video=false |
| outcome, state_delta, confidence, cost, constraints_discovered | ✅ EpisodeResult |
| Video fields optional | ✅ debug only |

---

## 9. Infrastructure

| Check | Status |
|-------|--------|
| GPU worker remote (RunPod) | ✅ |
| Jobs queue if GPU offline | ✅ Redis |
| Resumes when GPU back | ✅ Worker loop |

---

## 10. Non-Story Validation

| Check | Status |
|-------|--------|
| Same code paths for robot/gravity/balance scenarios | ✅ |
| No story-specific logic when REALITY_COMPILER_MODE=true | ✅ |
| Outcome meaningful without narrative | ✅ |

---

## 11. MVP Declaration

> "StoryWorld answers questions about what is possible in the world, not how good a video looks."

✅ Architecture and API support this. Deploy with REALITY_COMPILER_MODE=true.

---

## RunPod Deployment Notes

- Queue env vars: use `JOB_QUEUE=storyworld:gpu:jobs` (no spaces around `=`)
- Main app and worker must use same RESULT_QUEUE / JOB_QUEUE
- REALITY_COMPILER_MODE=true for production
