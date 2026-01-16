import torch
from ccvfi import AutoModel, ConfigType

class TemporalInterpolator:
    def __init__(self, device="cuda"):
        self.device = device
        self.model = AutoModel.from_pretrained(
            pretrained_model_name=ConfigType.RIFE_IFNet_v426_heavy
        )
        self.model.to(device).eval()

    @torch.no_grad()
    def interpolate(self, frames, target_len):
        output = frames[:]

        while len(output) < target_len:
            new = []
            for i in range(len(output) - 1):
                mid = self.model.inference_image_list(
                    img_list=[output[i], output[i + 1]]
                )[0]
                new.append(output[i])
                new.append(mid)
            new.append(output[-1])
            output = new

        return output[:target_len]
