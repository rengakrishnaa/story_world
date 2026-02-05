"""
Stub video backend for testing without GPU.

Produces a minimal valid MP4 (64x64 black frames) — visually "blank" when played.
Expected when DEFAULT_BACKEND=stub. For real video, use svd/animatediff/veo.
"""
import tempfile
from pathlib import Path

import numpy as np


def _write_minimal_mp4(path: Path, width: int = 64, height: int = 64, num_frames: int = 24) -> None:
    """Write a minimal valid MP4 (black frames) so OpenCV/R2 accept it. Visually blank."""
    try:
        import imageio
    except ImportError:
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, 24.0, (width, height))
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        for _ in range(num_frames):
            writer.write(frame)
        writer.release()
        return

    frames = np.zeros((num_frames, height, width, 3), dtype=np.uint8)
    writer = imageio.get_writer(str(path), format="FFMPEG", fps=24, codec="mpeg4")
    for f in frames:
        writer.append_data(f)
    writer.close()


def render(input_spec: dict) -> dict:
    tmp = Path(tempfile.mkdtemp())
    fake_video = tmp / "video.mp4"
    _write_minimal_mp4(fake_video)
    return {"video": str(fake_video)}
