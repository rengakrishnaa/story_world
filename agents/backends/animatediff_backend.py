import torch
from diffusers import (
    StableDiffusionPipeline,
    MotionAdapter,
    AnimateDiffPipeline,
    DDIMScheduler
)

class AnimateDiffBackend:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 1️⃣ Load base Stable Diffusion (this HAS model_index.json)
        base_pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )

        # 2️⃣ Load motion adapter (NO model_index.json here — correct)
        motion_adapter = MotionAdapter.from_pretrained(
            "guoyww/animatediff-motion-adapter-v1-5"
        )

        # 3️⃣ Compose AnimateDiff pipeline
        self.pipe = AnimateDiffPipeline(
            vae=base_pipe.vae,
            text_encoder=base_pipe.text_encoder,
            tokenizer=base_pipe.tokenizer,
            unet=base_pipe.unet,
            scheduler=DDIMScheduler.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                subfolder="scheduler"
            ),
            motion_adapter=motion_adapter,
        ).to(self.device)

        self.pipe.to(self.device)
        self.pipe.enable_attention_slicing()


        if self.device == "cuda":
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except Exception as e:
                print(f"[WARN] xformers disabled: {e}")


    def render(self, prompt: str, pose_frames=None):
        """
        pose_frames is accepted for future pose-conditioning.
        Currently ignored by AnimateDiff (text-to-video only).
        """

        MAX_ANIMATEDIFF_FRAMES = 24  # hard limit in AnimateDiff

        num_frames = (
            min(len(pose_frames), MAX_ANIMATEDIFF_FRAMES)
            if pose_frames is not None
            else MAX_ANIMATEDIFF_FRAMES
        )


        return self.pipe(
            prompt=prompt,
            num_frames=num_frames,
            guidance_scale=7.5,
            num_inference_steps=25
        ).frames

