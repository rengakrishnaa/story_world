import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.beat_observer import BeatObserver

observer = BeatObserver()

# Use a REAL image you already generated
TEST_IMAGE = "keyframes/beat-3_sdxl.png"   # adjust path

# Case 1: happy path
beat_ok = {
    "id": "beat_test_ok",
    "description": "Saitama punches a monster",
    "characters": ["Saitama"]
}

generation_ok = {
    "keyframe_path": TEST_IMAGE,
    "video_path": None,
    "model": "sdxl"
}

obs = observer.observe(beat_ok, generation_ok)
print("OK CASE:", obs)
