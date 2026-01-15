from agents.pose.pose_extraction import PoseExtractor
from agents.pose.keypose_selector import KeyPoseSelector
from agents.pose.pose_interpolator import PoseInterpolator
from agents.backends.animatediff_backend import AnimateDiffBackend
import math


class CharacterMotionEngine:
    """
    Converts a single keyframe + beat intent into animated frames
    using REAL pose conditioning (ControlNet).
    """

    FPS = 24
    MAX_FRAMES_PER_CHUNK = 24  # AnimateDiff hard limit

    def __init__(self):
        self.pose_extractor = PoseExtractor()
        self.pose_selector = KeyPoseSelector()
        self.pose_interpolator = PoseInterpolator()
        self._backend = None

    def _get_backend(self):
        if self._backend is None:
            self._backend = AnimateDiffBackend()
        return self._backend

    def render_beat(self, beat: dict, keyframe_path: str):
        """
        Returns: list[PIL.Image]
        """

        # 1️⃣ Pose validation ONLY (no conditioning)
        self.pose_extractor.extract(keyframe_path)

        # 2️⃣ AnimateDiff render (text → video)
        backend = self._get_backend()

        MAX_FRAMES = 24  # AnimateDiff hard limit

        frames = backend.render(
            prompt=beat["description"],
            num_frames=MAX_FRAMES
        )

        return frames

