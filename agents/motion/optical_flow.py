import cv2
import numpy as np


class FlowWarp:
    """
    Optical flow / frame interpolation helper.

    Guarantees:
    - Never throws due to CUDA absence
    - Automatically falls back to CPU
    - Safe in Docker, CI, and non-GPU machines
    """

    def __init__(self):
        self.cuda_available = False

        try:
            if hasattr(cv2, "cuda"):
                count = cv2.cuda.getCudaEnabledDeviceCount()
                if count > 0:
                    self.cuda_available = True
        except Exception as e:
            print(f"[motion] CUDA check failed, CPU fallback: {e}")
            self.cuda_available = False

        if self.cuda_available:
            print("[motion] FlowWarp using CUDA")
        else:
            print("[motion] FlowWarp running in CPU mode")

    def warp(self, frame0: np.ndarray, frame1: np.ndarray, alpha: float):
        """
        Blend two frames using GPU if available, otherwise CPU.
        """
        if self.cuda_available:
            try:
                return self._warp_cuda(frame0, frame1, alpha)
            except Exception as e:
                print(f"[motion] CUDA warp failed, fallback to CPU: {e}")
                return self._warp_cpu(frame0, frame1, alpha)

        return self._warp_cpu(frame0, frame1, alpha)

    # -----------------------
    # CUDA path
    # -----------------------

    def _warp_cuda(self, frame0, frame1, alpha):
        g0 = cv2.cuda_GpuMat()
        g1 = cv2.cuda_GpuMat()

        g0.upload(frame0)
        g1.upload(frame1)

        blended = cv2.cuda.addWeighted(
            g0, 1.0 - alpha,
            g1, alpha,
            0.0
        )

        return blended.download()

    # -----------------------
    # CPU path (guaranteed)
    # -----------------------

    def _warp_cpu(self, frame0, frame1, alpha):
        return cv2.addWeighted(
            frame0, 1.0 - alpha,
            frame1, alpha,
            0.0
        )
