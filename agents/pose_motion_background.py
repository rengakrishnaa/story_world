from pathlib import Path

class PoseMotionBackend:
    """
    Placeholder for AnimateDiff / pose-based animation.
    """

    def generate(
        self,
        image_path: str,
        prompt: str,
        duration: float
    ) -> Path:
        """
        Future:
        - ControlNet OpenPose
        - AnimateDiff
        - Stable Video Diffusion
        """

        raise NotImplementedError(
            "Pose-based motion backend not enabled yet"
        )
