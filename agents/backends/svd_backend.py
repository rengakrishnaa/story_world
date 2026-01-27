from agents.motion.motion_guard import assert_sparse_motion
from agents.motion.sparse_motion_engine import SparseMotionEngine


def render(input_spec: dict) -> dict:
    # 🔒 HARD VALIDATION
    motion = assert_sparse_motion(input_spec)

    engine = SparseMotionEngine(
        strength=motion["params"].get("strength", 0.85),
        reuse_poses=motion["params"].get("reuse_poses", True),
        temporal_smoothing=motion["params"].get("temporal_smoothing", True),
    )

    keyframes = engine.generate_keyframes(input_spec)

    video_path = engine.render_video_from_keyframes(
        keyframes=keyframes,
        duration=input_spec.get("duration_sec", 4),
    )

    return {
        "video": video_path,
    }
