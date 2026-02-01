"""
Veo 3.1 Video Generation Backend

Production implementation with robust fallback chain:
1. Veo 3.1 API (best quality)
2. Gemini Image + Motion (API-based fallback)
3. SDXL + Motion (local GPU - no API needed)
"""

import os
import time
import tempfile
import io
from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image
import numpy as np

from agents.motion.motion_guard import assert_sparse_motion


def render(input_spec: dict) -> dict:
    """
    Render video with robust fallback chain.
    
    Order:
    1. Veo 3.1 API
    2. Gemini Image + Motion
    3. SDXL + Motion (local, no API)
    
    Character consistency is maintained by using:
    - enhanced_description: prompt with character details
    - character_conditioning: character-specific prompt fragments
    
    Cinematic styling is applied using:
    - cinematic_prompt: full prompt with camera, lighting, color modifiers
    - cinematic_spec: detailed spec for post-processing
    """
    motion = assert_sparse_motion(input_spec)
    
    # Priority: cinematic_prompt > enhanced_description > prompt
    # cinematic_prompt includes all camera, lighting, color modifiers
    prompt = (
        input_spec.get("cinematic_prompt") or
        input_spec.get("enhanced_description") or
        input_spec.get("prompt")
    )
    if not prompt:
        raise RuntimeError("Veo backend requires prompt")
    
    duration_sec = input_spec.get("duration_sec", 8)
    
    # Include character conditioning in prompt if available
    char_conditioning = input_spec.get("character_conditioning", {})
    if char_conditioning.get("character_prompts"):
        # Add character descriptions to prompt
        char_details = ". ".join(char_conditioning["character_prompts"].values())
        if char_details and char_details not in prompt:
            prompt = f"{prompt}. Characters: {char_details}"
    
    errors = []
    
    # Try Veo 3.1 first
    try:
        print("[veo] Attempting Veo 3.1 API...")
        video_path = generate_with_veo(prompt, duration_sec, input_spec)
        return {"video": video_path}
    except Exception as e:
        errors.append(f"Veo: {e}")
        print(f"[veo] Veo 3.1 failed: {e}")
    
    # Fallback to Gemini image + motion
    try:
        print("[veo] Attempting Gemini image fallback...")
        video_path = generate_with_gemini_image_motion(prompt, duration_sec, input_spec)
        return {"video": video_path}
    except Exception as e:
        errors.append(f"Gemini: {e}")
        print(f"[veo] Gemini image fallback failed: {e}")
    
    # Final fallback to SDXL + motion (local, no API)
    try:
        print("[veo] Attempting SDXL local fallback...")
        video_path = generate_with_sdxl_motion(prompt, duration_sec, input_spec)
        return {"video": video_path}
    except Exception as e:
        errors.append(f"SDXL: {e}")
        print(f"[veo] SDXL fallback failed: {e}")
    
    # All methods failed
    raise RuntimeError(f"All video generation methods failed: {errors}")


def generate_with_veo(prompt: str, duration_sec: float, input_spec: dict) -> str:
    """Generate video using Veo 3.1 API."""
    try:
        from google import genai
    except ImportError:
        raise RuntimeError("google-genai not installed")
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    
    client = genai.Client(api_key=api_key)
    enhanced_prompt = build_cinematic_prompt(prompt, input_spec)
    
    use_fast = os.getenv("VEO_USE_FAST", "false").lower() == "true"
    model = "veo-3.1-fast-generate-preview" if use_fast else "veo-3.1-generate-preview"
    
    print(f"[veo] Starting with {model}")
    
    operation = client.models.generate_videos(
        model=model,
        prompt=enhanced_prompt,
    )
    
    max_wait = 300
    waited = 0
    while not operation.done:
        if waited >= max_wait:
            raise RuntimeError(f"Timeout after {max_wait}s")
        time.sleep(10)
        waited += 10
        operation = client.operations.get(operation)
    
    if not operation.response or not operation.response.generated_videos:
        raise RuntimeError(f"Veo failed: {getattr(operation, 'error', 'Unknown')}")
    
    generated_video = operation.response.generated_videos[0]
    output_dir = Path(tempfile.mkdtemp(prefix="veo_"))
    output_path = output_dir / "veo_output.mp4"
    
    client.files.download(file=generated_video.video)
    generated_video.video.save(str(output_path))
    
    print(f"[veo] ✅ Video saved: {output_path}")
    return str(output_path)


def generate_with_gemini_image_motion(prompt: str, duration_sec: float, input_spec: dict) -> str:
    """Fallback: Gemini image + motion interpolation."""
    try:
        from google import genai
    except ImportError:
        raise RuntimeError("google-genai not installed")
    
    from agents.motion.sparse_motion_engine import SparseMotionEngine
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    
    client = genai.Client(api_key=api_key)
    
    print(f"[gemini-image] Generating keyframe...")
    
    response = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=f"Generate a cinematic movie frame: {prompt}. High quality, detailed.",
    )
    
    # Extract image
    image_data = None
    for part in response.candidates[0].content.parts:
        if hasattr(part, 'inline_data') and part.inline_data:
            image_data = part.inline_data.data
            break
    
    if not image_data:
        raise RuntimeError("No image in Gemini response")
    
    start_frame = Image.open(io.BytesIO(image_data)).convert("RGB")
    end_frame = start_frame.copy()
    
    print(f"[gemini-image] ✅ Keyframe: {start_frame.size}")
    
    # Apply enhanced motion
    try:
        from agents.motion.enhanced_motion_engine import get_motion_engine
        engine = get_motion_engine()
        
        # Detect motion type from prompt
        motion_config = engine.detect_motion_type(prompt)
        print(f"[gemini-image] Motion type: {motion_config.motion_type.value}")
        
        frames = engine.generate_motion_from_image(start_frame, motion_config, duration_sec)
        
        output_dir = Path(tempfile.mkdtemp(prefix="gemini_motion_"))
        output_path = output_dir / "gemini_motion.mp4"
        engine.encode_frames_to_video(frames, output_path, fps=24)
        
    except Exception as e:
        print(f"[gemini-image] Enhanced motion failed, using fallback: {e}")
        from agents.motion.sparse_motion_engine import SparseMotionEngine
        engine = SparseMotionEngine()
        frames = engine.render_motion(start_frame=start_frame, end_frame=start_frame, duration_sec=duration_sec)
        output_dir = Path(tempfile.mkdtemp(prefix="gemini_motion_"))
        output_path = output_dir / "gemini_motion.mp4"
        engine._encode_frames_to_video(frames, output_path, fps=24)
    
    print(f"[gemini-image] ✅ Video saved: {output_path}")
    return str(output_path)


def generate_with_sdxl_motion(prompt: str, duration_sec: float, input_spec: dict) -> str:
    """
    Local fallback: SDXL image generation + motion.
    No API calls - runs entirely on local GPU.
    """
    import torch
    
    # Check if GPU is available
    if not torch.cuda.is_available():
        raise RuntimeError("SDXL requires CUDA GPU")
    
    try:
        from diffusers import StableDiffusionXLPipeline
    except ImportError:
        raise RuntimeError("diffusers not installed. Run: pip install diffusers")
    
    from agents.motion.sparse_motion_engine import SparseMotionEngine
    
    print(f"[sdxl] Loading SDXL pipeline (this may take a moment)...")
    
    device = "cuda"
    dtype = torch.float16
    
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=dtype,
        use_safetensors=True,
        variant="fp16",
    ).to(device)
    
    pipe.enable_attention_slicing()
    
    # Generate keyframe
    print(f"[sdxl] Generating keyframe for: {prompt[:50]}...")
    
    image = pipe(
        prompt=f"cinematic movie frame, {prompt}, high quality, detailed, professional lighting",
        guidance_scale=7.5,
        num_inference_steps=25,
        generator=torch.Generator(device=device).manual_seed(42),
    ).images[0]
    
    start_frame = image.resize((1280, 720), Image.LANCZOS)
    end_frame = start_frame.copy()
    
    print(f"[sdxl] ✅ Keyframe generated: {start_frame.size}")
    
    # Cleanup GPU memory
    del pipe
    torch.cuda.empty_cache()
    
    # Apply enhanced motion
    try:
        from agents.motion.enhanced_motion_engine import get_motion_engine
        engine = get_motion_engine()
        
        # Detect motion type from prompt
        motion_config = engine.detect_motion_type(prompt)
        print(f"[sdxl] Motion type: {motion_config.motion_type.value}")
        
        frames = engine.generate_motion_from_image(start_frame, motion_config, duration_sec)
        
        output_dir = Path(tempfile.mkdtemp(prefix="sdxl_motion_"))
        output_path = output_dir / "sdxl_motion.mp4"
        engine.encode_frames_to_video(frames, output_path, fps=24)
        
    except Exception as e:
        print(f"[sdxl] Enhanced motion failed, using fallback: {e}")
        from agents.motion.sparse_motion_engine import SparseMotionEngine
        engine = SparseMotionEngine()
        frames = engine.render_motion(start_frame=start_frame, end_frame=start_frame, duration_sec=duration_sec)
        output_dir = Path(tempfile.mkdtemp(prefix="sdxl_motion_"))
        output_path = output_dir / "sdxl_motion.mp4"
        engine._encode_frames_to_video(frames, output_path, fps=24)
    
    print(f"[sdxl] ✅ Video saved: {output_path}")
    return str(output_path)


def build_cinematic_prompt(base_prompt: str, input_spec: dict) -> str:
    """
    Enhance prompt with style-specific details.
    
    Uses style_profile if available for better style consistency.
    Falls back to basic style modifier otherwise.
    """
    parts = []
    
    # Check for style profile from StyleDetector
    style_profile = input_spec.get("style_profile", {})
    
    if style_profile:
        # Use StyleDetector's prefix
        if style_profile.get("style_prefix"):
            parts.append(style_profile["style_prefix"])
        
        # Add base prompt
        parts.append(base_prompt)
        
        # Add camera details
        camera = input_spec.get("camera", {})
        if camera:
            shot_type = camera.get("shot_type", "")
            movement = camera.get("movement", "")
            if shot_type:
                parts.append(f"{shot_type.replace('_', ' ')} shot")
            if movement and movement != "static":
                parts.append(f"with {movement.replace('_', ' ')} camera movement")
        
        # Use StyleDetector's suffix
        if style_profile.get("style_suffix"):
            parts.append(style_profile["style_suffix"])
        
        # Add lighting style
        if style_profile.get("lighting_style"):
            parts.append(f"{style_profile['lighting_style']} lighting")
        
    else:
        # Fallback: basic prompt enhancement
        parts.append(base_prompt)
        
        camera = input_spec.get("camera", {})
        if camera:
            shot_type = camera.get("shot_type", "")
            movement = camera.get("movement", "")
            if shot_type:
                parts.append(f"{shot_type.replace('_', ' ')} shot")
            if movement and movement != "static":
                parts.append(f"with {movement.replace('_', ' ')} camera movement")
        
        style = input_spec.get("style", "cinematic")
        parts.append(f"{style} style, high quality, detailed")
    
    return ", ".join(parts)


def get_negative_prompt(input_spec: dict) -> str:
    """Get negative prompt from style profile or default."""
    style_profile = input_spec.get("style_profile", {})
    
    if style_profile and style_profile.get("negative_prompt"):
        return style_profile["negative_prompt"]
    
    # Default negative prompt
    return "blurry, low quality, distorted, watermark, text, logo"

