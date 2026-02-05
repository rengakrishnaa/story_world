"""
AnimateDiff Backend

Uses keyframe generation (Gemini/SDXL) with motion interpolation.
Produces smooth motion between generated keyframes.
"""

import os
import tempfile
import subprocess
import uuid
import io
from pathlib import Path
from PIL import Image
import numpy as np

from agents.motion.sparse_motion_engine import SparseMotionEngine

PIPELINE_VALIDATE = os.getenv("PIPELINE_VALIDATE", "false").lower() == "true"


def render(input_spec):
    """
    AnimateDiff backend - produces MP4 from keyframe interpolation.
    """
    prompt = input_spec.get("prompt")
    if not prompt:
        raise RuntimeError("AnimateDiff backend requires prompt")

    duration_sec = input_spec.get("duration_sec")
    if duration_sec is None:
        raise RuntimeError("duration_sec is required")

    engine = SparseMotionEngine()

    # Get start/end frames
    start_frame = None
    end_frame = None
    
    if PIPELINE_VALIDATE:
        # Validation mode: use placeholder frames
        width, height = 512, 512
        start_frame = Image.new("RGB", (width, height), color=(30, 30, 30))
        end_frame = Image.new("RGB", (width, height), color=(120, 120, 120))
    else:
        # Try provided paths first
        start_path = input_spec.get("start_frame_path")
        end_path = input_spec.get("end_frame_path")

        if start_path and end_path and os.path.exists(start_path) and os.path.exists(end_path):
            start_frame = Image.open(start_path).convert("RGB")
            end_frame = Image.open(end_path).convert("RGB")
        else:
            # Generate keyframes
            print(f"[animatediff] Generating keyframes...")
            start_frame, end_frame = generate_keyframes(prompt, input_spec)

    # Motion rendering
    frames = engine.render_motion(
        start_frame=start_frame,
        end_frame=end_frame,
        duration_sec=duration_sec,
    )

    # Persist frames
    out_dir = tempfile.mkdtemp(prefix="animatediff_frames_")

    for i, frame in enumerate(frames):
        if isinstance(frame, np.ndarray):
            Image.fromarray(frame).save(os.path.join(out_dir, f"{i:04d}.png"))
        else:
            frame.save(os.path.join(out_dir, f"{i:04d}.png"))

    # Encode to MP4
    video_path = os.path.join(out_dir, f"{uuid.uuid4()}.mp4")

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate", "24",
            "-i", os.path.join(out_dir, "%04d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "medium",
            "-crf", "23",
            video_path,
        ],
        check=True,
        capture_output=True,
    )

    return {"video": video_path}


def generate_keyframes(prompt: str, input_spec: dict = None) -> tuple:
    """
    Generate start and end keyframes.
    Tries Gemini first, falls back to SDXL. Skips Gemini when _credit_exhausted (Veo fallback).
    """
    credit_exhausted = (input_spec or {}).get("_credit_exhausted", False)
    if not credit_exhausted:
        try:
            return generate_keyframes_gemini(prompt)
        except Exception as e:
            print(f"[animatediff] Gemini failed: {e}")
    
    try:
        return generate_keyframes_sdxl(prompt)
    except Exception as e:
        print(f"[animatediff] SDXL failed: {e}")
        raise RuntimeError("All keyframe generation methods failed")


def generate_keyframes_gemini(prompt: str) -> tuple:
    """Generate keyframes using Gemini's image generation."""
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise RuntimeError("google-genai package not installed")
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    
    client = genai.Client(api_key=api_key)
    keyframes = []
    
    prompts = [
        f"cinematic establishing shot, {prompt}, high quality, professional lighting",
        f"cinematic shot with slight camera movement, {prompt}, high quality"
    ]
    
    for i, frame_prompt in enumerate(prompts):
        print(f"[animatediff] Generating keyframe {i+1}/2 with Gemini...")
        
        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=f"Generate a cinematic movie frame: {frame_prompt}",
        )
        
        # Extract image
        image_data = None
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'inline_data') and part.inline_data:
                image_data = part.inline_data.data
                break
        
        if not image_data:
            raise RuntimeError(f"No image in response for keyframe {i+1}")
        
        img = Image.open(io.BytesIO(image_data)).convert("RGB")
        keyframes.append(img)
        print(f"[animatediff] Keyframe {i+1} generated: {img.size}")
    
    return keyframes[0], keyframes[1]


def generate_keyframes_sdxl(prompt: str) -> tuple:
    """Generate keyframes using local SDXL."""
    USE_DIFFUSION = os.getenv("USE_DIFFUSION", "false").lower() == "true"
    
    if not USE_DIFFUSION:
        raise RuntimeError("USE_DIFFUSION not enabled")
    
    import torch
    from diffusers import StableDiffusionXLPipeline
    
    print("[animatediff] Loading SDXL...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=dtype,
        use_safetensors=True,
    ).to(device)
    
    if device == "cuda":
        pipe.enable_attention_slicing()
    
    keyframes = []
    for i in range(2):
        print(f"[animatediff] SDXL generating keyframe {i+1}/2...")
        image = pipe(
            prompt=f"cinematic frame, {prompt}, high quality",
            guidance_scale=7.5,
            num_inference_steps=30,
            generator=torch.Generator(device=device).manual_seed(42 + i * 100),
        ).images[0]
        
        keyframes.append(image.resize((1280, 720), Image.LANCZOS))
    
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return keyframes[0], keyframes[1]
