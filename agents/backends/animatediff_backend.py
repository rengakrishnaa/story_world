import torch
from diffusers import (
    StableDiffusionPipeline,
    AnimateDiffPipeline,
    MotionAdapter,
    DDIMScheduler,
    ControlNetModel
)


class AnimateDiffBackend:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Base SD
        base_pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )

        # Motion Adapter
        motion_adapter = MotionAdapter.from_pretrained(
            "guoyww/animatediff-motion-adapter-v1-5"
        )

        # ControlNet (POSE)
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_openpose",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )

        self.pipe = AnimateDiffPipeline(
            vae=base_pipe.vae,
            text_encoder=base_pipe.text_encoder,
            tokenizer=base_pipe.tokenizer,
            unet=base_pipe.unet,
            controlnet=controlnet,
            scheduler=DDIMScheduler.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                subfolder="scheduler"
            ),
            motion_adapter=motion_adapter,
        ).to(self.device)

        self.pipe.enable_attention_slicing()

        if self.device == "cuda":
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass

    def render(self, prompt: str, pose_frames):
        """
        pose_frames: list[np.ndarray] or pose maps (OpenPose-style)
        """

        num_frames = min(len(pose_frames), 24)

        output = self.pipe(
            prompt=prompt,
            control_image=pose_frames[:num_frames],
            num_frames=num_frames,
            guidance_scale=7.5,
            num_inference_steps=25
        )

        return output.frames[0]
