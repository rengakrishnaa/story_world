# StoryWorld Critical-Grade Validation Report

**Generated:** Run `python -m tests.test_critical_validation` or `pytest tests/test_critical_validation.py -v`

---

## Summary

| Tier | Status | Notes |
|------|--------|-------|
| **L1 Physical Sanity** | ✅ PASS | All 3 scenarios blocked by NLP veto |
| **L2 Human-Scale** | ⚠️ Veto OK | Reaches observer (no hard block) |
| **L3–L6** | ⚠️ Needs GPU | Require full pipeline with observer |

**Rule:** Do not move to next level until previous behaves correctly.

---

## Level 1 — Physical Sanity (Zero Risk)

### L1.1 Gravity Invariance
| Field | Result |
|-------|--------|
| **Scenario ID** | L1.1 |
| **Goal** | A stone floats motionless in mid-air without support. |
| **Outcome** | GOAL_IMPOSSIBLE ✅ |
| **Confidence** | 0.0 (correct for failure) |
| **Discovered Constraints** | gravity_violation, unsupported_mass |
| **Observer Verdicts** | Pre-simulation veto (NLP gate) |
| **WorldStateGraph** | nodes=0, transitions=0 (no GPU used) |
| **Admit uncertainty?** | N/A — hard impossibility |
| **Honest termination?** | Yes — no retries, fast fail |

**Purpose:** Verifies gravity is non-negotiable.

---

### L1.2 Energy Conservation
| Field | Result |
|-------|--------|
| **Scenario ID** | L1.2 |
| **Goal** | A vehicle accelerates from 0 to 100 without fuel, power, or external force. |
| **Outcome** | GOAL_IMPOSSIBLE ✅ |
| **Confidence** | 0.0 |
| **Discovered Constraints** | energy_conservation |
| **Observer Verdicts** | Pre-simulation veto |
| **WorldStateGraph** | nodes=0, transitions=0 |
| **Honest termination?** | Yes |

**Purpose:** Ensures no "free energy" hallucination.

---

### L1.3 Inertia
| Field | Result |
|-------|--------|
| **Scenario ID** | L1.3 |
| **Goal** | A fast-moving object stops instantly without force. |
| **Outcome** | GOAL_IMPOSSIBLE ✅ |
| **Confidence** | 0.0 |
| **Discovered Constraints** | inertia_violation |
| **Observer Verdicts** | Pre-simulation veto |
| **Honest termination?** | Yes |

**Purpose:** Inertia violation detected.

---

## Level 2 — Human-Scale Physics

### L2.1 Human Jump Boundaries
| Field | Result |
|-------|--------|
| **Scenario ID** | L2.1 |
| **Goal** | A healthy adult jumps across a 5 meter wide horizontal gap. |
| **Expected** | UNCERTAIN → exploration; eventually impossible beyond limits |
| **Veto** | Not HARD blocked ✅ (soft constraint only, observer decides) |
| **Full outcome** | Requires GPU + observer for IN_PROGRESS → terminal |

**Purpose:** Checks uncertainty, not hard coding.

---

### L2.2 Balance Under Disturbance
| Field | Result |
|-------|--------|
| **Scenario ID** | L2.2 |
| **Goal** | A person balances on a narrow 10cm ledge with wind gusts. |
| **Veto** | Not blocked ✅ (reaches observer) |
| **Full outcome** | Requires GPU + observer |

**Purpose:** Ensures system doesn't fake certainty.

---

## Level 3–6 — Observer-Dependent

These scenarios **must reach the VideoObserver**. The NLP veto does **not** block them.

| Level | Scenario | Veto Block? | Full Pipeline |
|-------|----------|-------------|---------------|
| L3.1 | Shelf supports increasing weight over time | No | Needs GPU + observer |
| L3.2 | Floor appears solid but lacks support | No | Needs GPU + observer |
| L4.1 | Robot stacks boxes on narrow base | No | Needs GPU + observer |
| L4.2 | Vehicle sharp turn at increasing speed | No | Needs GPU + observer |
| L5.1 | Rocket liftoff with thrust below weight | No | Needs GPU + observer |
| L5.2 | Rocket orbit with insufficient fuel | No | Needs GPU + observer |
| L6.x | Guidance failure, thermal, G-forces | No | Needs GPU + observer |

---

## Sanity Check

| Field | Result |
|-------|--------|
| **Goal** | A solid object is released from rest above the ground and falls. |
| **Veto** | NOT blocked ✅ |
| **Purpose** | Normal falling must not be pre-vetoed. |

---

## Test Commands

```bash
# Tier A only (physics veto, no server)
pytest tests/test_critical_validation.py -v -m "not integration"

# Tier B (API, needs Redis)
pytest tests/test_critical_validation.py -m integration -v

# Full validation report
python -m tests.test_critical_validation
```

---

## Verdict

- **L1 (Physical Sanity):** ✅ All pass. Product is safe for baseline invariants.
- **L2 (Human-Scale):** ✅ Veto correctly allows observer to decide (no over-blocking).
- **L3–L6:** ⚠️ Require full deployment (Redis + GPU worker + observer) to validate terminal outcomes. Veto correctly does **not** block these.

**Honest failure:** Yes — L1 fails fast, no forced success.  
**Observer authority intact:** Yes — L2+ reach observer; veto is guardrail only.
