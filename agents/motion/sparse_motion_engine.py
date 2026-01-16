import numpy as np
from PIL import Image
from agents.pose.pose_extraction import PoseExtractor
from agents.motion.optical_flow import FlowWarp
from agents.motion.interpolator import TemporalInterpolator

class SparseMotionEngine:
    FPS = 24

    def __init__(self):
        self.pose_extractor = PoseExtractor()
        self.flow = FlowWarp(device="cuda")
        self.interpolator = TemporalInterpolator(device="cuda")

    def render_motion(self, start_frame: Image.Image, end_frame: Image.Image, duration_sec: float):
        # Pose validation (CPU, cheap)
        self.pose_extractor.extract_from_image(start_frame)
        self.pose_extractor.extract_from_image(end_frame)

        f0 = np.array(start_frame)
        f1 = np.array(end_frame)

        # Sparse warp (cheap)
        warped = self.flow.warp_sequence(f0, f1, steps=4)

        # Temporal interpolation (RIFE)
        total_frames = int(duration_sec * self.FPS)
        frames = self.interpolator.interpolate(warped, total_frames)

        # GPU sync & cleanup (IMPORTANT)
        import torch
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        return frames
