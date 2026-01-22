import json
from pathlib import Path
from PIL import Image
from agents.motion.sparse_motion_engine import SparseMotionEngine

INPUT_JSON = Path("/job/input.json")
OUTPUT_DIR = Path("/job/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    with open(INPUT_JSON) as f:
        job = json.load(f)

    start = Image.open(job["start_frame"])
    end = Image.open(job["end_frame"])

    engine = SparseMotionEngine()

    frames = engine.render_motion(
        start_frame=start,
        end_frame=end,
        duration_sec=job["duration_sec"]
    )

    # Save frames → video
    from moviepy.editor import ImageSequenceClip
    clip = ImageSequenceClip([frame for frame in frames], fps=24)

    output_path = OUTPUT_DIR / "motion.mp4"
    clip.write_videofile(
        str(output_path),
        codec="libx264",
        audio=False,
        verbose=False,
        logger=None
    )

if __name__ == "__main__":
    main()
