"""
End-to-end validation: submit one scenario, poll until terminal, verify world graph updates.

Usage: Ensure main server is running (uvicorn main:app) and GPU worker is connected.
  python run_e2e_validation.py

Expects API at BASE_URL (default http://localhost:8000).
"""

import os
import sys
import time
import json

try:
    import requests
except ImportError:
    print("Install requests: pip install requests")
    sys.exit(1)

BASE_URL = os.getenv("API_BASE", "http://localhost:8000")
GOAL = "A shelf supports increasing weight over time."
POLL_INTERVAL = 5
MAX_WAIT_SEC = 360  # 6 min: GPU render + observer can take 2+ min per beat


def run():
    print(f"E2E Validation: {GOAL}")
    print(f"API: {BASE_URL}")
    print("-" * 60)

    # 0. Pre-flight: queue diagnostics
    try:
        d = requests.get(f"{BASE_URL}/diagnostics/queue", timeout=5).json()
        if d.get("status") == "ok":
            q = d.get("queues", {})
            jq = q.get("job_queue", {})
            rq = q.get("result_queue", {})
            print(f"Queues: jobs={jq.get('pending_jobs', '?')} pending, results={rq.get('pending_results', '?')} pending")
        else:
            print(f"Queue check: {d.get('message', 'unknown')}")
    except Exception as e:
        print(f"Queue check skipped: {e}")

    # 1. Submit
    resp = requests.post(
        f"{BASE_URL}/simulate",
        params={"world_id": "e2e-validation", "goal": GOAL, "budget": 5},
        timeout=30,
    )
    if resp.status_code != 200:
        print(f"FAIL: POST /simulate returned {resp.status_code}")
        print(resp.text[:500])
        return False

    data = resp.json()
    sim_id = data.get("simulation_id") or data.get("episode_id")
    if not sim_id:
        print("FAIL: No simulation_id in response")
        return False

    print(f"Submitted: {sim_id}")
    print("Polling for result (GPU + observer)...")

    # 2. Poll until terminal (GPU + observer can take 2+ min per beat)
    start = time.time()
    last_outcome, last_elapsed = None, -99
    while time.time() - start < MAX_WAIT_SEC:
        try:
            r = requests.get(f"{BASE_URL}/episodes/{sim_id}/result", timeout=15)
        except Exception as e:
            print(f"Request error: {e}")
            time.sleep(POLL_INTERVAL)
            continue

        if r.status_code != 200:
            time.sleep(POLL_INTERVAL)
            continue

        res = r.json()
        outcome = (res.get("outcome") or "").upper()
        sd = res.get("state_delta") or {}
        progress = sd.get("progress") or res.get("metrics") or {}
        completed = progress.get("completed", progress.get("beats_completed", 0))
        aborted = progress.get("aborted", progress.get("beats_failed", 0))
        total = progress.get("total_beats", progress.get("beats_attempted", "?"))

        elapsed = int(time.time() - start)
        if outcome != last_outcome or elapsed - last_elapsed >= 30:
            print(f"  [{elapsed}s] outcome={outcome} completed={completed}/{total} aborted={aborted}")
            last_outcome, last_elapsed = outcome, elapsed

        if outcome and outcome not in ("IN_PROGRESS", "PENDING"):
            break

        time.sleep(POLL_INTERVAL)

    if time.time() - start >= MAX_WAIT_SEC:
        print("TIMEOUT: Did not reach terminal state within limit")
        return False

    # 3. Extract metrics
    sd = res.get("state_delta") or {}
    state_nodes = sd.get("state_nodes", 0)
    transitions = sd.get("transitions", 0)
    constraints = res.get("constraints_discovered") or []
    confidence = res.get("confidence", 0)
    cost = res.get("cost") or res.get("total_cost_usd", 0)

    # 4. Report
    print("-" * 60)
    print("RESULT:")
    print(json.dumps({
        "episode_id": sim_id,
        "outcome": outcome,
        "confidence": confidence,
        "cost": cost,
        "state_nodes": state_nodes,
        "transitions": transitions,
        "constraints_discovered": constraints,
        "progress": progress,
    }, indent=2))

    # 5. Validate world graph / learning
    learned = state_nodes > 0 or transitions > 0 or len(constraints) > 0
    if learned:
        print("\n[PASS] World graph exercised: state_nodes={} transitions={} constraints={}"
              .format(state_nodes, transitions, len(constraints)))
    else:
        print("\n[FAIL] No learning: state_nodes=0, transitions=0, constraints_discovered=[]")
        print("  GPU result may not have reached ResultConsumer, or observer did not run.")

    return learned


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)
