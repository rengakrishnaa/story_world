from pathlib import Path
from PIL import Image
import numpy as np
from moviepy.editor import ImageSequenceClip
import torch

from agents.motion.sparse_motion_engine import SparseMotionEngine

KEYFRAMES_DIR = Path(r"C:\Users\KRISH\Desktop\LEARN\Vidme\story_world\agents\keyframes")
OUTPUT_DIR = Path(r"C:\Users\KRISH\Desktop\LEARN\Vidme\story_world\agents\videos")
OUTPUT_DIR.mkdir(exist_ok=True)

FPS = 24
DURATION_SEC = 3.0  # keep short for testing

engine = SparseMotionEngine()

keyframes = [
    "beat_1_sdxl.png",
    "beat_2_sdxl.png",
    "beat_3_sdxl.png",
]

for name in keyframes:
    print(f"\n🎬 Testing motion for {name}")

    path = KEYFRAMES_DIR / name
    assert path.exists(), f"Missing {path}"

    start = Image.open(path).convert("RGB")
    end = Image.open(path).convert("RGB")  # MVP: same frame

    frames = engine.render_motion(
        start_frame=start,
        end_frame=end,
        duration_sec=DURATION_SEC,
    )

    # Safety check
    assert isinstance(frames, list)
    assert len(frames) > 0

    # Convert to numpy
    processed = []
    for f in frames:
        if isinstance(f, Image.Image):
            processed.append(np.array(f))
        else:
            processed.append(f)

    out_path = OUTPUT_DIR / f"{path.stem}_motion.mp4"

    clip = ImageSequenceClip(processed, fps=FPS)
    clip.write_videofile(
        str(out_path),
        codec="libx264",
        audio=False,
        verbose=False,
        logger=None
    )

    # GPU cleanup (important for repeated tests)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    print(f"✅ Saved: {out_path}")
