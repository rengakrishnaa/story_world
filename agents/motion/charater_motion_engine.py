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

        # 1️⃣ Extract base pose
        base_pose = self.pose_extractor.extract(keyframe_path)

        # 2️⃣ Select key semantic poses
        key_poses = self.pose_selector.select(
            base_pose=base_pose,
            motion_type=beat.get("motion_type", "idle")
        )

        # 3️⃣ Interpolate poses for full duration
        total_frames = int(beat.get("estimated_duration_sec", 5) * self.FPS)

        pose_frames = self.pose_interpolator.interpolate(
            key_poses=key_poses,
            num_frames=total_frames
        )

        backend = self._get_backend()

        # 4️⃣ Chunking (MANDATORY)
        num_chunks = math.ceil(total_frames / self.MAX_FRAMES_PER_CHUNK)

        all_frames = []

        for i in range(num_chunks):
            start = i * self.MAX_FRAMES_PER_CHUNK
            end = start + self.MAX_FRAMES_PER_CHUNK

            pose_chunk = pose_frames[start:end]

            frames = backend.render(
                prompt=beat["description"],
                pose_frames=pose_chunk  # REAL conditioning
            )

            all_frames.extend(frames)

        return all_frames
