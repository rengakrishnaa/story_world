from ccvfi import AutoModel, ConfigType
import numpy as np

class TemporalInterpolator:
    def __init__(self, device="cuda"):
        self.device = device
        self.model = AutoModel.from_pretrained(
            "rife",
            config=ConfigType.RIFE_HD
        )

    def interpolate(self, frames, target_len):
        """
        frames: list[np.ndarray] (H, W, 3) uint8
        returns: list[np.ndarray]
        """

        output = frames

        while len(output) < target_len:
            new_frames = []
            for i in range(len(output) - 1):
                f0 = output[i]
                f1 = output[i + 1]

                mid = self.model.infer(f0, f1)
                new_frames.append(f0)
                new_frames.append(mid)

            new_frames.append(output[-1])
            output = new_frames

        return output[:target_len]
