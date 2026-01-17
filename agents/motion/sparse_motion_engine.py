# agents/motion/sparse_motion_engine.py
import numpy as np
from PIL import Image
from agents.pose.pose_extraction import PoseExtractor
from agents.motion.optical_flow import FlowWarp

class SparseMotionEngine:
    FPS = 24

    def __init__(self):
        self.pose = PoseExtractor()
        self.flow = FlowWarp()

    def render_motion(self, start_frame: Image.Image, end_frame: Image.Image, duration_sec: float):
        self.pose.extract_from_image(start_frame)
        self.pose.extract_from_image(end_frame)

        f0 = np.array(start_frame)
        f1 = np.array(end_frame)

        total = int(duration_sec * self.FPS)
        frames = []

        for i in range(total):
            alpha = i / max(total - 1, 1)
            warped = self.flow.warp(f0, f1, alpha)
            frames.append(warped)

        return frames
