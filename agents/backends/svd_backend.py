"""
SVD (Stable Video Diffusion) Backend

Uses Gemini or SDXL for keyframe generation, then applies sparse motion interpolation.
This is a fallback backend when Veo is not available.
"""

import os
import tempfile
import io
from pathlib import Path
from typing import Dict, Any, List
from PIL import Image
import numpy as np

from agents.motion.motion_guard import assert_sparse_motion
from agents.motion.sparse_motion_engine import SparseMotionEngine


def render(input_spec: dict) -> dict:
    """
    Render video using keyframe generation + motion interpolation.
    """
    # Validate motion spec
    motion = assert_sparse_motion(input_spec)
    params = motion.get("params", {}) or {}
    
    prompt = input_spec.get("prompt")
    if not prompt:
        raise RuntimeError("SVD backend requires prompt")
    
    duration_sec = input_spec.get("duration_sec", 4)
    
    # Try Gemini for keyframe generation first
    try:
        keyframes = generate_keyframes_gemini(prompt)
    except Exception as e:
        print(f"[svd] Gemini keyframe generation failed: {e}")
        # Fallback to SDXL
        try:
            keyframes = generate_keyframes_sdxl(prompt)
        except Exception as e2:
            print(f"[svd] SDXL keyframe generation also failed: {e2}")
            raise RuntimeError(f"All keyframe generation methods failed")
    
    # Create motion engine
    try:
        engine = SparseMotionEngine(
            strength=params.get("strength", 0.85),
            reuse_poses=params.get("reuse_poses", True),
            temporal_smoothing=params.get("temporal_smoothing", True),
        )
    except TypeError:
        engine = SparseMotionEngine()
    
    # Render video from keyframes
    video_path = engine.render_video_from_keyframes(
        keyframes=keyframes,
        duration=duration_sec,
    )
    
    return {"video": video_path}


def generate_keyframes_gemini(prompt: str) -> List[np.ndarray]:
    """
    Generate keyframes using Gemini's image generation.
    """
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
    
    # Generate keyframes with slight variations
    variations = [
        f"establishing shot, {prompt}, cinematic lighting",
        f"medium shot, {prompt}, dynamic angle",
    ]
    
    for i, var_prompt in enumerate(variations):
        print(f"[svd] Generating keyframe {i+1}/{len(variations)} with Gemini...")
        
        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=f"Generate a cinematic movie frame: {var_prompt}. High quality, detailed.",
        )
        
        # Extract image from response
        image_data = None
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'inline_data') and part.inline_data:
                image_data = part.inline_data.data
                break
        
        if not image_data:
            raise RuntimeError(f"No image in Gemini response for keyframe {i+1}")
        
        img = Image.open(io.BytesIO(image_data)).convert("RGB")
        keyframes.append(np.array(img))
        print(f"[svd] Keyframe {i+1} generated: {img.size}")
    
    return keyframes


def generate_keyframes_sdxl(prompt: str) -> List[np.ndarray]:
    """
    Fallback: Generate keyframes using local SDXL.
    """
    USE_DIFFUSION = os.getenv("USE_DIFFUSION", "false").lower() == "true"
    
    if not USE_DIFFUSION:
        raise RuntimeError("USE_DIFFUSION not enabled for SDXL fallback")
    
    import torch
    from diffusers import StableDiffusionXLPipeline
    
    print("[svd] Loading SDXL for keyframe generation...")
    
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
        print(f"[svd] SDXL generating keyframe {i+1}/2...")
        image = pipe(
            prompt=f"cinematic frame, {prompt}, high quality, detailed",
            guidance_scale=7.5,
            num_inference_steps=30,
            generator=torch.Generator(device=device).manual_seed(42 + i * 100),
        ).images[0]
        
        # Resize to 720p
        image = image.resize((1280, 720), Image.LANCZOS)
        keyframes.append(np.array(image))
    
    # Cleanup
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return keyframes
