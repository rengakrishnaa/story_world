from agents.pose.pose_extraction import PoseExtractor
from agents.pose.keypose_selector import KeyPoseSelector
from agents.pose.pose_interpolator import PoseInterpolator
from agents.backends.animatediff_backend import AnimateDiffBackend


class CharacterMotionEngine:
    """
    Converts a single keyframe + beat intent into animated frames.
    """

    def __init__(self):
        self.pose_extractor = PoseExtractor()
        self.pose_selector = KeyPoseSelector()
        self.pose_interpolator = PoseInterpolator()
        self._backend = None  # lazy-loaded

    # -------------------------
    # Lazy backend loading
    # -------------------------
    def _get_backend(self):
        if self._backend is None:
            self._backend = AnimateDiffBackend()
        return self._backend

    # -------------------------
    # Main entry point
    # -------------------------
    def render_beat(self, beat: dict, keyframe_path: str):
        """
        Returns: list[PIL.Image] or list[np.ndarray]
        """

        # 1️⃣ Extract base pose from keyframe
        base_pose = self.pose_extractor.extract(keyframe_path)

        # 2️⃣ Select semantic key poses
        key_poses = self.pose_selector.select(
            base_pose=base_pose,
            motion_type=beat.get("motion_type", "idle")
        )

        # 3️⃣ Interpolate poses into per-frame pose sequence
        num_frames = int(beat.get("estimated_duration_sec", 5) * 24)

        pose_frames = self.pose_interpolator.interpolate(
            key_poses=key_poses,
            num_frames=num_frames
        )

        # 4️⃣ AnimateDiff rendering (pose-conditioned)
        backend = self._get_backend()

        frames = backend.render(
            prompt=beat["description"],
            pose_frames=pose_frames
        )

        return frames
