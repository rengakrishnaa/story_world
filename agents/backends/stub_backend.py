import tempfile
from pathlib import Path

def render(input_spec: dict) -> dict:
    tmp = Path(tempfile.mkdtemp())
    fake_video = tmp / "video.mp4"
    fake_video.write_bytes(b"FAKE_VIDEO_DATA")
    return {"video": str(fake_video)}
