"""
StoryWorld Critical-Grade Validation Scenarios

Tests Level 1 → 6 in order. Rule: do not move to next level until previous behaves correctly.

Tier A: Physics veto (no server) — L1 and any veto-catchable goals
Tier B: API integration (server required) — full pipeline with observer
"""

import os
import time
import pytest
from typing import Dict, Any, Optional, List


# -----------------------------------------------------------------------------
# Scenario Definitions: (goal_text, expected_outcomes, expected_constraints, notes)
# -----------------------------------------------------------------------------

SCENARIOS = {
    # LEVEL 1 — Physical Sanity
    "L1.1": {
        "goal": "A stone floats motionless in mid-air without support.",
        "expected_outcome": "GOAL_IMPOSSIBLE",
        "expected_constraints": ["gravity", "unsupported"],
        "expected_veto": True,
        "purpose": "Gravity invariance: never floats.",
    },
    "L1.2": {
        "goal": "A vehicle accelerates from 0 to 100 without fuel, power, or external force.",
        "expected_outcome": "GOAL_IMPOSSIBLE",
        "expected_constraints": ["energy_conservation"],
        "expected_veto": True,
        "purpose": "Energy conservation.",
    },
    "L1.3": {
        "goal": "A fast-moving object stops instantly without force.",
        "expected_outcome": "GOAL_IMPOSSIBLE",
        "expected_constraints": ["inertia"],
        "expected_veto": True,
        "purpose": "Inertia violation.",
    },
    # LEVEL 2 — Human-Scale
    "L2.1": {
        "goal": "A healthy adult jumps across a 5 meter wide horizontal gap.",
        "expected_outcome": "UNCERTAIN_or_IMPOSSIBLE",
        "expected_constraints": ["human_jump_limit"],
        "expected_veto": False,  # SOFT only
        "purpose": "Uncertainty, not hard coding.",
    },
    "L2.2": {
        "goal": "A person balances on a narrow 10cm ledge with wind gusts.",
        "expected_outcome": "UNCERTAIN_or_FAILURE",
        "expected_constraints": [],
        "expected_veto": False,
        "purpose": "Observer disagreement, conservative.",
    },
    # LEVEL 3
    "L3.1": {
        "goal": "A shelf supports increasing weight over time until it collapses.",
        "expected_outcome": "ANY",
        "expected_constraints": [],
        "expected_veto": False,
        "purpose": "Temporal causality, fatigue.",
    },
    "L3.2": {
        "goal": "A person walks across a floor that appears solid but lacks internal support.",
        "expected_outcome": "IMPOSSIBLE_or_DEAD_STATE",
        "expected_constraints": ["structural"],
        "expected_veto": False,
        "purpose": "Observer discovers hidden weakness.",
    },
    # LEVEL 4
    "L4.1": {
        "goal": "A robot stacks three boxes vertically on a narrow base without external support.",
        "expected_outcome": "GOAL_IMPOSSIBLE",
        "expected_constraints": ["stability", "load"],
        "expected_veto": False,
        "purpose": "Partial success, final instability.",
    },
    "L4.2": {
        "goal": "A vehicle takes a sharp turn at increasing speed.",
        "expected_outcome": "ANY",
        "expected_constraints": ["friction"],
        "expected_veto": False,
        "purpose": "Loss of traction at threshold.",
    },
    # LEVEL 5
    "L5.1": {
        "goal": "A rocket attempts liftoff with thrust slightly below its weight.",
        "expected_outcome": "GOAL_IMPOSSIBLE",
        "expected_constraints": [],
        "expected_veto": False,
        "purpose": "Force balance.",
    },
    "L5.2": {
        "goal": "A rocket attempts orbit with insufficient fuel mass.",
        "expected_outcome": "GOAL_IMPOSSIBLE",
        "expected_constraints": [],
        "expected_veto": False,
        "purpose": "Delta-v constraint.",
    },
    # Sanity: plausible goals must NOT be vetoed
    "SANITY_PASS": {
        "goal": "A solid object is released from rest above the ground and falls.",
        "expected_outcome": "PASS_NOT_BLOCKED",
        "expected_constraints": [],
        "expected_veto": False,
        "purpose": "Normal falling must not be blocked.",
    },
}


# -----------------------------------------------------------------------------
# Tier A: Physics Veto Tests (no server)
# -----------------------------------------------------------------------------

class TestLevel1PhysicsVeto:
    """L1 — Physical Sanity. If any fail → product unsafe."""

    def test_L1_1_gravity_invariance(self):
        """Stone floats unsupported → IMPOSSIBLE."""
        from runtime.physics_veto import evaluate_physics_veto
        veto, constraints, reason = evaluate_physics_veto(SCENARIOS["L1.1"]["goal"])
        assert veto is True, f"L1.1 must block: {reason}"
        assert any("gravity" in c or "unsupported" in c for c in constraints), constraints

    def test_L1_2_energy_conservation(self):
        """Vehicle accelerates without fuel → IMPOSSIBLE."""
        from runtime.physics_veto import evaluate_physics_veto
        veto, constraints, reason = evaluate_physics_veto(SCENARIOS["L1.2"]["goal"])
        assert veto is True, f"L1.2 must block: {reason}"
        assert "energy_conservation" in constraints, constraints

    def test_L1_3_inertia(self):
        """Instant stop without force → IMPOSSIBLE."""
        from runtime.physics_veto import evaluate_physics_veto
        veto, constraints, reason = evaluate_physics_veto(SCENARIOS["L1.3"]["goal"])
        assert veto is True, f"L1.3 must block: {reason}"
        assert "inertia_violation" in constraints, constraints

    def test_sanity_plausible_goal_not_blocked(self):
        """Normal falling must NOT be vetoed."""
        from runtime.physics_veto import evaluate_physics_veto
        veto, _, _ = evaluate_physics_veto(SCENARIOS["SANITY_PASS"]["goal"])
        assert veto is False, "Plausible gravity scenario must not be blocked"


class TestLevel2SoftVeto:
    """L2 — Human-scale. SOFT veto only, no HARD block."""

    def test_L2_1_human_jump_not_hard_blocked(self):
        """5m jump → SOFT constraint, observer decides."""
        from runtime.physics_veto import evaluate_physics_veto
        veto, constraints, _ = evaluate_physics_veto(SCENARIOS["L2.1"]["goal"])
        assert veto is False, "Human jump must not be HARD blocked"
        # May have soft constraint
        assert "human_jump_limit" in constraints or not constraints, constraints

    def test_L2_2_balance_ledge_not_blocked(self):
        """Narrow ledge + wind → observer decides."""
        from runtime.physics_veto import evaluate_physics_veto
        veto, _, _ = evaluate_physics_veto(SCENARIOS["L2.2"]["goal"])
        assert veto is False, "Balance scenario must reach observer"


class TestLevel3And4NotVetoed:
    """L3, L4 — Must reach observer, not pre-vetoed."""

    def test_L3_2_hidden_weakness_not_hard_blocked(self):
        """Floor lacks support → observer discovers."""
        from runtime.physics_veto import evaluate_physics_veto
        veto, _, _ = evaluate_physics_veto(SCENARIOS["L3.2"]["goal"])
        assert veto is False, "Hidden weakness must reach observer"

    def test_L4_1_robot_stacking_not_hard_blocked(self):
        """Robot stacking on narrow base → observer decides."""
        from runtime.physics_veto import evaluate_physics_veto
        veto, _, _ = evaluate_physics_veto(SCENARIOS["L4.1"]["goal"])
        assert veto is False, "Robot stacking must reach observer"


# -----------------------------------------------------------------------------
# Tier B: API Integration (full pipeline)
# -----------------------------------------------------------------------------

@pytest.mark.integration
class TestCriticalValidationAPI:
    """
    Full pipeline via /simulate and /result.
    Requires: server running, Redis, optionally GPU worker.
    Run: pytest tests/test_critical_validation.py -m integration -v
    """

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        import main
        # Ensure stores exist
        if main.sql is None:
            from runtime.persistence.sql_store import SQLStore
            main.sql = SQLStore(lazy=True)
        return TestClient(main.app)

    def _run_scenario(
        self,
        client,
        scenario_id: str,
        max_wait_sec: int = 15,
        poll_interval: float = 1.0,
    ) -> Dict[str, Any]:
        """Submit simulation, poll for result, return record."""
        s = SCENARIOS.get(scenario_id, {})
        goal = s.get("goal", "")
        if not goal:
            return {"scenario_id": scenario_id, "error": "Unknown scenario"}

        # Submit
        resp = client.post(f"/simulate?world_id=validation&goal={goal}&budget=5")
        if resp.status_code != 200:
            return {
                "scenario_id": scenario_id,
                "error": f"POST failed {resp.status_code}",
                "detail": resp.json() if resp.headers.get("content-type", "").startswith("application/json") else resp.text[:200],
            }
        data = resp.json()
        sim_id = data.get("simulation_id") or data.get("episode_id")
        if not sim_id:
            return {"scenario_id": scenario_id, "error": "No simulation_id", "data": data}

        # Poll for terminal
        start = time.time()
        while time.time() - start < max_wait_sec:
            r = client.get(f"/episodes/{sim_id}/result")
            if r.status_code != 200:
                time.sleep(poll_interval)
                continue
            res = r.json()
            outcome = (res.get("outcome") or "").upper()
            state = (res.get("state_delta") or {}).get("state") or res.get("state") or ""
            if outcome and outcome not in ("IN_PROGRESS", "PENDING") and "EXECUTING" not in str(state).upper():
                return {
                    "scenario_id": scenario_id,
                    "outcome": outcome,
                    "confidence": res.get("confidence", 0),
                    "constraints_discovered": res.get("constraints_discovered", []),
                    "state_nodes": (res.get("state_delta") or {}).get("state_nodes", 0),
                    "transitions": (res.get("state_delta") or {}).get("transitions", 0),
                }
            time.sleep(poll_interval)

        # Timeout — return last state
        r = client.get(f"/episodes/{sim_id}/result")
        res = r.json() if r.status_code == 200 else {}
        return {
            "scenario_id": scenario_id,
            "outcome": res.get("outcome", "TIMEOUT"),
            "confidence": res.get("confidence", 0),
            "constraints_discovered": res.get("constraints_discovered", []),
            "timeout": True,
        }

    def test_L1_1_via_api(self, client):
        """L1.1 via full pipeline — should be IMPOSSIBLE before GPU."""
        record = self._run_scenario(client, "L1.1", max_wait_sec=5)
        assert "error" not in record or record.get("outcome") == "GOAL_IMPOSSIBLE", record
        if record.get("outcome"):
            assert record["outcome"] == "GOAL_IMPOSSIBLE", record

    def test_L1_2_via_api(self, client):
        """L1.2 via API — energy conservation veto."""
        record = self._run_scenario(client, "L1.2", max_wait_sec=5)
        assert record.get("outcome") == "GOAL_IMPOSSIBLE", record

    def test_L1_3_via_api(self, client):
        """L1.3 via API — inertia veto."""
        record = self._run_scenario(client, "L1.3", max_wait_sec=5)
        assert record.get("outcome") == "GOAL_IMPOSSIBLE", record


# -----------------------------------------------------------------------------
# Validation Report Runner (standalone)
# -----------------------------------------------------------------------------

def run_validation_report() -> str:
    """
    Run all validation scenarios and produce a report.
    Call: python -m tests.test_critical_validation
    """
    from runtime.physics_veto import evaluate_physics_veto

    lines = [
        "=" * 60,
        "StoryWorld Critical-Grade Validation Report",
        "=" * 60,
    ]

    # Tier A: Veto tests
    lines.append("\n--- TIER A: Physics Veto (no server) ---\n")
    veto_tests = [
        ("L1.1 Gravity", SCENARIOS["L1.1"]["goal"], True, ["gravity", "unsupported"]),
        ("L1.2 Energy", SCENARIOS["L1.2"]["goal"], True, ["energy_conservation"]),
        ("L1.3 Inertia", SCENARIOS["L1.3"]["goal"], True, ["inertia"]),
        ("L2.1 Jump", SCENARIOS["L2.1"]["goal"], False, []),
        ("L4.1 Stack", SCENARIOS["L4.1"]["goal"], False, []),
        ("SANITY", SCENARIOS["SANITY_PASS"]["goal"], False, []),
    ]
    passed = 0
    for name, goal, should_block, _ in veto_tests:
        veto, constraints, reason = evaluate_physics_veto(goal)
        ok = veto == should_block
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        lines.append(f"  {status} {name}: veto={veto} constraints={constraints}")
    lines.append(f"\n  Tier A: {passed}/{len(veto_tests)} passed")

    # Tier B hint
    lines.append("\n--- TIER B: API (run server, then pytest -m integration) ---")
    lines.append("  pytest tests/test_critical_validation.py -m integration -v")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


def run_full_api_validation() -> str:
    """
    Run ALL scenarios via API and produce full validation report.
    Requires server (TestClient) + Redis. GPU optional for L2+.
    """
    from fastapi.testclient import TestClient
    import main
    if main.sql is None:
        from runtime.persistence.sql_store import SQLStore
        main.sql = SQLStore(lazy=True)

    client = TestClient(main.app)
    lines = [
        "=" * 70,
        "StoryWorld Critical-Grade Validation — Full API Report",
        "=" * 70,
    ]
    results = []
    for sid in ["L1.1", "L1.2", "L1.3", "L2.1", "L2.2", "L3.1", "L3.2", "L4.1", "SANITY_PASS"]:
        s = SCENARIOS.get(sid, {})
        goal = s.get("goal", "")
        if not goal:
            continue
        resp = client.post(f"/simulate?world_id=validation&goal={goal}&budget=5")
        if resp.status_code != 200:
            results.append((sid, {"error": f"POST {resp.status_code}", "detail": str(resp.json())[:100]}))
            continue
        data = resp.json()
        sim_id = data.get("simulation_id") or data.get("episode_id")
        if not sim_id:
            results.append((sid, {"error": "No simulation_id"}))
            continue
        # Fetch result immediately (veto'd episodes are already terminal)
        r = client.get(f"/episodes/{sim_id}/result")
        res = r.json() if r.status_code == 200 else {}
        outcome = (res.get("outcome") or "").upper()
        sd = res.get("state_delta") or {}
        rec = {
            "outcome": outcome or "—",
            "confidence": res.get("confidence", "—"),
            "constraints": res.get("constraints_discovered", []),
            "state_nodes": sd.get("state_nodes", 0),
            "transitions": sd.get("transitions", 0),
        }
        results.append((sid, rec))

    for sid, rec in results:
        err = rec.get("error")
        if err:
            lines.append(f"\n  {sid}: ERROR — {err}")
            continue
        lines.append(f"\n  Scenario ID: {sid}")
        lines.append(f"    Outcome: {rec.get('outcome', '—')}")
        lines.append(f"    Confidence: {rec.get('confidence', '—')}")
        lines.append(f"    Discovered Constraints: {rec.get('constraints', [])}")
        lines.append(f"    WorldStateGraph: nodes={rec.get('state_nodes', 0)}, transitions={rec.get('transitions', 0)}")

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


if __name__ == "__main__":
    print(run_validation_report())
    print("\n")
    print(run_full_api_validation())
