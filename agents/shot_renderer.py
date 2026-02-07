import os
import json
import time
import base64
import sys

# Force site-packages to take priority over project folders
sys.path.insert(0, "/usr/local/lib/python3.10/site-packages")
import redis
import requests
import boto3
from botocore.exceptions import NoCredentialsError
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont
import io
import logging
import numpy as np
from moviepy.editor import ColorClip, ImageClip, CompositeVideoClip
from dotenv import load_dotenv
import google.genai as genai
import torch
from agents.motion.sparse_motion_engine import SparseMotionEngine
from agents.cinematic_camera import CinematicCamera
from agents.backends.cinematic_backend import CinematicBackend
from agents.motion.charater_motion_engine import CharacterMotionEngine
from moviepy.editor import ImageSequenceClip

load_dotenv()

# Redis (same store we use for job queue)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
r = redis.from_url(REDIS_URL)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SDXL_PIPE = None
USE_DIFFUSION = os.getenv("USE_DIFFUSION", "false").lower() == "true"

if USE_DIFFUSION:
    from diffusers import StableDiffusionXLPipeline

def load_sdxl():
    global SDXL_PIPE

    if not USE_DIFFUSION:
        raise RuntimeError("Diffusion disabled")

    if SDXL_PIPE is None:
        logger.info("Loading SDXL on CPU")
        SDXL_PIPE = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float32,
        ).to("cpu")

        SDXL_PIPE.enable_attention_slicing()
        SDXL_PIPE.enable_vae_slicing()

    return SDXL_PIPE

from pathlib import Path

def generate_fallback_image(prompt: str, output_path: Path):
    pipe = load_sdxl()
    image = pipe(
        prompt=prompt,
        guidance_scale=7.5,
        num_inference_steps=30,
    ).images[0]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)

    return {
        "keyframe_path": str(output_path),
        "keyframe_url": f"http://localhost:8000/static/{output_path.as_posix()}",
    }

@dataclass
class RenderResult:
    beat_id: str
    keyframe_url: str
    video_url: Optional[str] = None
    status: str = "pending"
    duration_sec: float = 0.0
    cost: float = 0.0

class StorageAdapter:
    """Local ↔ S3 abstraction - flip USE_S3 env var."""
    def __init__(self):
        self.use_s3 = os.getenv("USE_S3", "false").lower() == "true"
        self.s3_bucket = os.getenv("S3_BUCKET", "storyworld-keyframes")
        self.region = "ap-south-1"
        
        if self.use_s3:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                region_name=self.region
            )
    
    def store_keyframe(self, image_bytes: bytes, filename: str) -> str:
        """Store PNG keyframe - local or S3."""
        path = Path("keyframes") / filename
        
        # Always save local
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.open(io.BytesIO(image_bytes)).save(path)
        
        if self.use_s3:
            try:
                self.s3_client.upload_fileobj(
                    io.BytesIO(image_bytes),
                    self.s3_bucket,
                    f"keyframes/{filename}",
                    ExtraArgs={'ContentType': 'image/png', 'ACL': 'public-read'}
                )
                return f"https://{self.s3_bucket}.s3.{self.region}.amazonaws.com/keyframes/{filename}"
            except:
                logger.warning("S3 failed, using local")
        
        return f"http://localhost:8000/static/{path.as_posix()}"

    def store_video(self, video_bytes: bytes, filename: str) -> str:
        path = Path("videos") / filename
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            f.write(video_bytes)

        return f"http://localhost:8000/static/{path.as_posix()}"


storage = StorageAdapter()

class ProductionShotRenderer:
    def __init__(self, api_key: str):
        """Production init - YOUR original google.genai."""
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-3-flash-preview"
        self.image_model = "gemini-2.5-flash-image"
        self.motion_engine = SparseMotionEngine()

        
        # Create dirs
        for d in ["keyframes", "thumbnails", "videos"]:
            Path(d).mkdir(exist_ok=True)
    
    def _load_world(self, world_id: str) -> Dict[str, Any]:
        """
        Load world data for rendering.
        
        Args:
            world_id: ID of the world to load
            
        Returns:
            World data dictionary with characters and locations
        """
        from pathlib import Path
        import json
        
        # Try loading cached world
        world_file = Path(f"outputs/{world_id}/world.json")
        if world_file.exists():
            with open(world_file, 'r') as f:
                return json.load(f)
        
        # Fallback to empty world
        return {
            "characters": [],
            "locations": []
        }
    
    def process_redis_queue(self, world_id: str):
        queue_key = f"render_queue:{world_id}"

        job_json = r.brpop(queue_key, timeout=30)
        if not job_json:
            logger.info("No jobs found. Exiting worker.")
            return

        job = json.loads(job_json[1])
        beat = job["beat"]
        beat_id = beat["id"]

        logger.info(f"🎬 Rendering beat {beat_id}")

        try:
            # Load world data
            world = self._load_world(world_id)
            
            keyframe_data = self.render_beat_keyframe(world, beat)

            # Free SDXL memory
            global SDXL_PIPE
            SDXL_PIPE = None
            import gc
            gc.collect()


            video_url = self.render_veo_video(None, beat, keyframe_data)

            result = RenderResult(
                beat_id=beat_id,
                keyframe_url=keyframe_data["keyframe_url"],
                video_url=video_url,
                status="completed",
                duration_sec=beat.get("estimated_duration_sec", 5.0),
                cost=beat.get("estimated_duration_sec", 5.0) * 0.008
            )

            r.hset(f"render_results:{world_id}", beat_id, json.dumps(result.__dict__))
            logger.info(f"✅ {beat_id} completed")

        except Exception as e:
            logger.error(f"❌ {beat_id} failed: {e}")

    
    def render_beat_keyframe(self, world, beat: Dict) -> Dict[str, Optional[str]]:
        """YOUR original NanoBanana → Gemini → placeholder pipeline."""
        try:
            # NanoBanana first
            if os.getenv("NANOBANANA_API_KEY"):
                prompt = self.generate_detailed_shot_prompt(world, beat)
                img = self.generate_nanobanana_image(prompt["nanobananaprompt"])
                return {
                    "keyframe_url": img["keyframe_url"],
                    "keyframe_path": img["keyframe_path"],
                }

        except Exception as e:
            logger.warning(f"NanoBanana failed: {e}")

        try:
            motion_type = beat.get("motion_type", "static")

            if motion_type == "character":
                # 🔒 Pose-safe prompt (STRICT)
                prompt = (
                    "full body human character, standing, facing camera, "
                    "arms visible, legs visible, neutral pose, realistic anatomy, "
                    "well-lit studio shot, plain background, high detail, "
                    f"{beat['description']}"
                )
            else:
                # 🎨 Cinematic freedom
                prompt = f"anime cinematic frame, {beat['description']}"

            path = Path("keyframes") / f"{beat['id']}_sdxl.png"
            return generate_fallback_image(prompt, path)

        except Exception as e:
            raise RuntimeError(
                f"SDXL keyframe generation FAILED on GPU. "
                f"Cannot continue to motion stage. Reason: {e}"
            )

    
    def generate_nanobanana_image(self, prompt: str) -> str:
        """YOUR exact NanoBanana API call."""
        api_key = os.getenv("NANOBANANA_API_KEY")
        if not api_key:
            raise ValueError("NANOBANANA_API_KEY required")
        
        payload = {
            "prompt": prompt,
            "width": 1024,
            "height": 576,
            "steps": 50,
            "seed": 42
        }
        
        resp = requests.post(
            "https://api.nano-banana.pro/v1/generate",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
            timeout=120
        )
        resp.raise_for_status()
        
        result = resp.json()
        image_url = (result.get("image_url") or 
                    result[0].get("data", [{}])[0].get("url") or 
                    result.get("data"))
        
        if image_url:
            resp = requests.get(image_url, timeout=30)
            filename = f"shot_{int(time.time())}.png"
            url = storage.store_keyframe(resp.content, filename)

            return {
                "keyframe_url": url,
                "keyframe_path": str(Path("keyframes") / filename)
            }
        
        raise ValueError("No image URL from NanoBanana")
    
    def create_detailed_placeholder(self, prompt: str, shot_id: str = None) -> str:
        """YOUR original anime-style placeholder."""
        if not shot_id:
            shot_id = f"shot_{int(time.time())}"
        
        width, height = 1024, 576
        
        # Anime gradient (YOUR code)
        gradient = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            gradient[y, :, 0] = int(100 + 55 * (y / height))  # Red
            gradient[y, :, 1] = int(20 + 30 * (y / height))   # Green  
            gradient[y, :, 2] = int(150 - 45 * (y / height))  # Blue
        
        img = Image.fromarray(gradient)
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("arial.ttf", 48)
        except:
            font = ImageFont.load_default()
        
        # YOUR anime text
        draw.text((50, 50), "SHOUNEN ANIME", fill=(255,255,255), font=font)
        draw.text((50, 120), prompt[:60], fill=(255,220,100), font=ImageFont.load_default())
        draw.text((50, 450), f"KEYFRAME {shot_id[:4]}", fill=(150,150,255))
        draw.rectangle([800, 100, 1010, 200], fill=(255,100,100))
        
        filename = f"shot_{shot_id}.png"
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        return {
            "keyframe_url": storage.store_keyframe(img_bytes.getvalue(), filename),
            "keyframe_path": str(Path("keyframes") / filename),
        }

    
    def fallback_gemini_image(self, prompt: str) -> str:
        """YOUR Gemini image fallback."""
        shot_id = f"shot_{int(time.time())}"
        return self.create_detailed_placeholder(prompt, shot_id)
    
    def render_veo_video(self, world, beat: Dict, keyframe_data: Dict) -> Optional[str]:
        if beat.get("motion_type") != "character":
            return None

        keyframe_path = keyframe_data["keyframe_path"]
        if not keyframe_path:
            raise RuntimeError("Missing keyframe")

        start_frame = Image.open(keyframe_path).convert("RGB")
        end_frame = start_frame.copy()

        frames = self.motion_engine.render_motion(
            start_frame=start_frame,
            end_frame=end_frame,
            duration_sec=beat.get("estimated_duration_sec", 5.0)
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        video_path = Path("videos") / f"{beat['id']}.mp4"
        self._write_frames_to_video(frames, video_path)

        return storage.store_video(video_path.read_bytes(), video_path.name)


    def _write_frames_to_video(self, frames, output_path: Path):

        if not isinstance(frames, list) or len(frames) == 0:
            raise RuntimeError("No frames to write video")

        processed = []

        for i, frame in enumerate(frames):
            if isinstance(frame, Image.Image):
                processed.append(np.array(frame))
            elif isinstance(frame, np.ndarray):
                processed.append(frame)
            else:
                raise TypeError(
                    f"Frame {i} has unsupported type: {type(frame)}"
                )

        clip = ImageSequenceClip(processed, fps=24)
        clip.write_videofile(
            str(output_path),
            codec="libx264",
            audio=False,
            verbose=False,
            logger=None
        )

        
    def generate_detailed_shot_prompt(self, world, beat: Dict) -> Dict:
        """YOUR Gemini shot prompt generator."""
        prompt = f"""
        Generate NanoBanana Pro prompt for anime beat: {beat.get('description')}
        Return ONLY JSON: {{"shottype": "wide/medium/close", "camera": "low angle", "nanobananaprompt": "exact prompt"}}
        """
        
        response = self.client.models.generate_content(
            model=self.model,
            contents=[{"role": "user", "parts": [{"text": prompt}]}]
        )
        
        return self.safe_parse_shot_prompt(response.text, beat)
    
    @staticmethod
    def safe_parse_shot_prompt(raw_text: str, beat: Dict) -> Dict:
        """YOUR robust JSON parser."""
        cleaned = raw_text.strip()
        for marker in ['``````']:
            if cleaned.startswith(marker):
                cleaned = cleaned[len(marker):].strip()
            if cleaned.endswith(marker):
                cleaned = cleaned[:-len(marker)].strip()
        
        try:
            return json.loads(cleaned)
        except:
            return {
                "shottype": "wide",
                "camera": "low angle dramatic",
                "nanobananaprompt": f"Shounen anime: {beat.get('description', '')}"
            }
        
    def _validate_pose(self, keyframe_path: str):
        from agents.pose.pose_extraction import PoseExtractor
        PoseExtractor().extract(keyframe_path)


# Production worker
def run_worker(world_id: str = "demo"):
    """Redis worker CLI."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY required")
    
    renderer = ProductionShotRenderer(api_key)
    renderer.process_redis_queue(world_id)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-id", default="demo")
    args = parser.parse_args()
    run_worker(args.world_id)
