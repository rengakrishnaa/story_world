import numpy as np
from PIL import Image

from agents.pose.pose_extraction import PoseExtractor
from agents.motion.optical_flow import FlowWarp


class SparseMotionEngine:
    """
    Sparse motion interpolation engine.

    Guarantees:
    - Always produces frames
    - Never crashes due to CUDA absence
    - Gracefully degrades to CPU interpolation
    """

    FPS = 24

    def __init__(self):
        self.pose = PoseExtractor()

        # FlowWarp may or may not support CUDA
        try:
            self.flow = FlowWarp()
            self.flow_available = True
        except Exception as e:
            print(f"[motion] FlowWarp unavailable, CPU fallback: {e}")
            self.flow = None
            self.flow_available = False

    def render_motion(
        self,
        start_frame: Image.Image,
        end_frame: Image.Image,
        duration_sec: float,
    ):
        # ---------------------------------
        # Optional pose extraction
        # ---------------------------------
        try:
            self.pose.extract(start_frame)
            self.pose.extract(end_frame)
        except Exception as e:
            print(f"[motion] pose skipped: {e}")

        f0 = np.asarray(start_frame)
        f1 = np.asarray(end_frame)

        total = max(int(duration_sec * self.FPS), 1)
        frames = []

        for i in range(total):
            alpha = i / max(total - 1, 1)

            if self.flow_available:
                try:
                    warped = self.flow.warp(f0, f1, alpha)
                except Exception as e:
                    print(f"[motion] flow failed at step {i}, fallback: {e}")
                    warped = self._linear_blend(f0, f1, alpha)
            else:
                warped = self._linear_blend(f0, f1, alpha)

            frames.append(warped)

        return frames

    @staticmethod
    def _linear_blend(f0: np.ndarray, f1: np.ndarray, alpha: float):
        """
        Guaranteed CPU-safe interpolation fallback.
        """
        return ((1.0 - alpha) * f0 + alpha * f1).astype(np.uint8)
