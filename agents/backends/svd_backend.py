from diffusers import StableVideoDiffusionPipeline
import torch

class SVDBackend:
    def __init__(self):
        self.pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid",
            torch_dtype=torch.float16
        ).to("cuda")

    def render(
        self,
        image_path,
        num_frames
    ):
        return self.pipe(
            image=image_path,
            num_frames=num_frames
        ).frames
