import json
import importlib
redis = importlib.import_module("redis")


r = redis.Redis(host="localhost", port=6379, db=0)

with open("test_render.json", "r") as f:
    job = json.load(f)

r.lpush("render_queue:demo", json.dumps(job))

print("✅ Test job injected into Redis")
