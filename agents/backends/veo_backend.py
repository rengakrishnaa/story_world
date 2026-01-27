from agents.motion.motion_guard import assert_sparse_motion
from agents.motion.sparse_motion_engine import SparseMotionEngine


def render(input_spec: dict) -> dict:
    # 🔒 HARD VALIDATION
    motion = assert_sparse_motion(input_spec)

    # Initialize sparse motion engine
    engine = SparseMotionEngine(
        strength=motion["params"].get("strength", 0.85),
        reuse_poses=motion["params"].get("reuse_poses", True),
        temporal_smoothing=motion["params"].get("temporal_smoothing", True),
    )

    # Produce motion-constrained frames / guidance
    motion_plan = engine.build_motion_plan(input_spec)

    # Veo is now a *renderer*, not a decision maker
    video_path = engine.render_with_veo(
        prompt=input_spec["prompt"],
        motion_plan=motion_plan,
        duration=input_spec.get("duration_sec", 4),
        resolution=input_spec.get("resolution", "720p"),
        seed=input_spec.get("seed"),
    )

    return {
        "video": video_path,
    }
