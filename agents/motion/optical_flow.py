import torch
import numpy as np
import torch.nn.functional as F

class FlowWarp:
    def __init__(self, device="cuda"):
        self.device = device

    def _to_tensor(self, img):
        t = torch.from_numpy(img).permute(2, 0, 1).float()
        return t.unsqueeze(0).to(self.device) / 255.0

    @torch.no_grad()
    def compute_flow(self, frame0, frame1):
        # VERY cheap dense approximation (no RAFT reloads)
        # This avoids huge latency and GPU memory spikes
        flow = frame1 - frame0
        return flow

    def warp(self, frame, flow, alpha):
        return (frame + alpha * flow).clip(0, 255).astype(np.uint8)

    def warp_sequence(self, frame0, frame1, steps=4):
        flow = self.compute_flow(frame0, frame1)

        frames = []
        for i in range(steps):
            alpha = i / (steps - 1)
            frames.append(self.warp(frame0, flow, alpha))
        return frames
