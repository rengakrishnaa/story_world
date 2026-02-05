import time
from fastapi.testclient import TestClient
import main


def run():
    # Minimal wiring without async startup task
    if main.sql is None:
        from runtime.persistence.sql_store import SQLStore
        main.sql = SQLStore(lazy=True)
    if main.redis_store is None:
        from runtime.persistence.redis_store import RedisStore
        import os
        main.redis_store = RedisStore(url=os.getenv("REDIS_URL"), lazy=True)
    if main.world_graph_store is None:
        from runtime.persistence.world_graph_store import WorldGraphStore
        main.world_graph_store = WorldGraphStore()
    client = TestClient(main.app)
    goal = "A shelf supports increasing weight over time."
    resp = client.post(f"/simulate?world_id=validation&goal={goal}&budget=5")
    data = resp.json()
    sim_id = data.get("simulation_id")
    # Poll briefly for result
    for _ in range(3):
        res = client.get(f"/episodes/{sim_id}/result").json()
        outcome = (res.get("outcome") or "").upper()
        if outcome and outcome not in ("IN_PROGRESS", "PENDING"):
            return res
        time.sleep(1)
    return res


if __name__ == "__main__":
    result = run()
    print(result)
