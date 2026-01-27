import os
import tempfile
import subprocess
import uuid
from PIL import Image
from agents.motion.sparse_motion_engine import SparseMotionEngine

PIPELINE_VALIDATE = os.getenv("PIPELINE_VALIDATE", "false").lower() == "true"


def render(input_spec):
    """
    Animatediff backend (PRODUCTION READY)

    Produces:
      - MP4 video file
    """

    # -----------------------------
    # Validate required fields
    # -----------------------------
    prompt = input_spec.get("prompt")
    if not prompt:
        raise RuntimeError("Animatediff backend requires prompt")

    duration_sec = input_spec.get("duration_sec")
    if duration_sec is None:
        raise RuntimeError("duration_sec is required")

    engine = SparseMotionEngine()

    # -----------------------------
    # Input resolution
    # -----------------------------
    if PIPELINE_VALIDATE:
        width, height = 512, 512
        start_frame = Image.new("RGB", (width, height), color=(30, 30, 30))
        end_frame = Image.new("RGB", (width, height), color=(120, 120, 120))
    else:
        start_path = input_spec.get("start_frame_path")
        end_path = input_spec.get("end_frame_path")

        if not start_path or not end_path:
            raise RuntimeError(
                "Animatediff backend requires start_frame_path and end_frame_path "
                "unless PIPELINE_VALIDATE=true"
            )

        start_frame = Image.open(start_path).convert("RGB")
        end_frame = Image.open(end_path).convert("RGB")

    # -----------------------------
    # Motion rendering (frames)
    # -----------------------------
    frames = engine.render_motion(
        start_frame=start_frame,
        end_frame=end_frame,
        duration_sec=duration_sec,
    )

    # -----------------------------
    # Persist frames
    # -----------------------------
    out_dir = tempfile.mkdtemp(prefix="animatediff_frames_")

    for i, frame in enumerate(frames):
        Image.fromarray(frame).save(
            os.path.join(out_dir, f"{i:04d}.png")
        )

    # -----------------------------
    # Encode frames → MP4 (CRITICAL)
    # -----------------------------
    video_path = os.path.join(
        out_dir, f"{uuid.uuid4()}.mp4"
    )

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate", "8",
            "-i", os.path.join(out_dir, "%04d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            video_path,
        ],
        check=True,
    )

    # -----------------------------
    # Return FINAL artifact
    # -----------------------------
    return {
        "video": video_path
    }
