# agents/motion/flow_warp.py
import cv2
import numpy as np

class FlowWarp:
    def __init__(self):
        if not cv2.cuda.getCudaEnabledDeviceCount():
            raise RuntimeError("CUDA OpenCV not available")

    def warp(self, frame0, frame1, alpha):
        g0 = cv2.cuda_GpuMat()
        g1 = cv2.cuda_GpuMat()
        g0.upload(frame0)
        g1.upload(frame1)

        flow = cv2.cuda_FarnebackOpticalFlow.create().calc(
            cv2.cuda.cvtColor(g0, cv2.COLOR_BGR2GRAY),
            cv2.cuda.cvtColor(g1, cv2.COLOR_BGR2GRAY),
            None
        )

        flow = flow.download()
        h, w = flow.shape[:2]

        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (grid_x + flow[..., 0] * alpha).astype(np.float32)
        map_y = (grid_y + flow[..., 1] * alpha).astype(np.float32)

        return cv2.remap(frame0, map_x, map_y, cv2.INTER_LINEAR)
