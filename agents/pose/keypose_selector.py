from agents.pose.pose_types import HumanPose
import numpy as np

class KeyPoseSelector:
    def select(self, base_pose: HumanPose, motion_type: str = "idle", **kwargs):
        """
        base_pose: HumanPose extracted from keyframe
        motion_type: semantic hint (idle, punch, walk, etc.)
        """

        base = base_pose.body  # (N, 3) numpy array

        if motion_type == "idle":
            return [base]

        if motion_type == "punch":
            return [
                base * 0.95,   # wind-up
                base * 1.05,   # impact
            ]

        # Fallback: static pose
        return [base]
