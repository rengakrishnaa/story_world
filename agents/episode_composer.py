"""
Episode Composer

Collects beat videos from Redis/R2 storage and stitches them into a single episode.
Handles:
- Retrieving beat render results from Redis
- Downloading beat videos from R2 storage
- Concatenating videos with ffmpeg
- Applying transitions (optional)
- Uploading final episode back to R2
"""

import json
import os
import tempfile
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
import importlib

redis = importlib.import_module("redis")
import boto3
from botocore.client import Config

from models.composed_shot import ComposedShot
from models.episode_plan import EpisodePlan

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
r = redis.from_url(REDIS_URL)


class EpisodeComposer:
    """
    Composes individual beat videos into a complete episode.
    
    Pipeline:
    1. Get beat results from Redis
    2. Download videos from R2
    3. Stitch with ffmpeg
    4. Upload final episode
    """
    
    def __init__(self, world_id: str):
        self.world_id = world_id
        self.temp_dir = None
        
        # S3/R2 configuration
        self.s3_endpoint = os.getenv("S3_ENDPOINT")
        self.s3_bucket = os.getenv("S3_BUCKET")
        self.s3_region = os.getenv("S3_REGION", "auto")
        self.s3_access_key = os.getenv("S3_ACCESS_KEY")
        self.s3_secret_key = os.getenv("S3_SECRET_KEY")
        
        self._s3_client = None
    
    @property
    def s3(self):
        """Lazy S3 client initialization."""
        if self._s3_client is None:
            if not all([self.s3_endpoint, self.s3_bucket, self.s3_access_key, self.s3_secret_key]):
                raise RuntimeError("S3/R2 configuration incomplete. Check env vars.")
            
            self._s3_client = boto3.client(
                "s3",
                endpoint_url=self.s3_endpoint,
                aws_access_key_id=self.s3_access_key,
                aws_secret_access_key=self.s3_secret_key,
                region_name=self.s3_region,
                config=Config(signature_version="s3v4"),
            )
        return self._s3_client
    
    def _ensure_temp_dir(self):
        """Create temp directory if needed."""
        if self.temp_dir is None:
            self.temp_dir = Path(tempfile.mkdtemp(prefix=f"episode_{self.world_id}_"))
        return self.temp_dir
    
    def compose(self, plan: EpisodePlan) -> List[ComposedShot]:
        """
        Collect shots from Redis matching the episode plan.
        Returns list of ComposedShot objects for stitching.
        """
        shots = []

        for beat_id in plan.beats:
            raw = r.hget(f"render_results:{self.world_id}", beat_id)
            if not raw:
                if not plan.allow_gaps:
                    raise RuntimeError(f"Missing beat {beat_id}")
                continue

            result = json.loads(raw)

            if result["status"] != "completed":
                continue

            if result.get("confidence", 0) < plan.required_confidence:
                continue

            video_url = result.get("video_url")
            if not video_url:
                logger.warning(f"Beat {beat_id} has no video_url in result: {result}")
                continue

            shots.append(
                ComposedShot(
                    beat_id=beat_id,
                    # Don't use Path() for URLs - it corrupts slashes on Windows
                    video_path=video_url,  # Store as string, not Path
                    duration=result.get("duration_sec", 5.0),
                    confidence=result.get("confidence", 0.5)
                )
            )

        if not shots:
            raise RuntimeError("No valid shots to compose")

        return shots
    
    def download_video(self, video_url: str, beat_id: str) -> Optional[Path]:
        """Download a video from R2 or copy from local path to temp directory."""
        temp_dir = self._ensure_temp_dir()
        
        # Sanitize beat_id for Windows filename (remove colons and other invalid chars)
        safe_beat_id = beat_id.replace(":", "_").replace("/", "_").replace("\\", "_")
        local_path = temp_dir / f"{safe_beat_id}.mp4"
        
        print(f"\n[DEBUG] download_video called:")
        print(f"[DEBUG]   beat_id: {beat_id}")
        print(f"[DEBUG]   video_url: {video_url}")
        print(f"[DEBUG]   local_path: {local_path}")
        print(f"[DEBUG]   s3_bucket: {self.s3_bucket}")
        
        # Handle None or empty URLs
        if not video_url or video_url == "None":
            print(f"[DEBUG]   ERROR: No video URL")
            logger.warning(f"No video URL for beat {beat_id}")
            return None
        
        try:
            # Check if it's a local file path first
            source_path = Path(video_url)
            if source_path.exists():
                print(f"[DEBUG]   Using local file: {source_path}")
                logger.info(f"Using local file for {beat_id}: {source_path}")
                import shutil
                shutil.copy(str(source_path), str(local_path))
                if local_path.exists() and local_path.stat().st_size > 0:
                    logger.info(f"✅ Copied {beat_id}: {local_path.stat().st_size} bytes")
                    return local_path
                return None
            
            # Check if it's an HTTP URL (R2 or other)
            if video_url.startswith("http"):
                print(f"[DEBUG]   HTTP URL detected")
                # R2 URL format: https://account.r2.cloudflarestorage.com/bucket/key
                # Need to extract the key after the bucket name
                
                key = None
                
                # Check if it's an R2 cloudflare storage URL
                if "r2.cloudflarestorage.com" in video_url:
                    # Extract key: everything after bucket name in the path
                    # URL: https://xxx.r2.cloudflarestorage.com/storyworld-artifacts/episodes/...
                    if self.s3_bucket and self.s3_bucket in video_url:
                        key = video_url.split(f"{self.s3_bucket}/")[-1]
                    else:
                        # Try to extract from path (everything after 3rd /)
                        parts = video_url.split("/")
                        if len(parts) > 4:
                            # parts[0]=https:, [1]="", [2]=host, [3]=bucket, [4+]=key
                            key = "/".join(parts[4:])
                
                # Standard S3 endpoint URL
                elif self.s3_bucket and self.s3_bucket in video_url:
                    key = video_url.split(f"{self.s3_bucket}/")[-1]
                
                # If we extracted a key, download via S3
                if key:
                    logger.info(f"Downloading {beat_id} from S3 key: {key}")
                    self.s3.download_file(
                        self.s3_bucket,
                        key,
                        str(local_path),
                    )
                    if local_path.exists() and local_path.stat().st_size > 0:
                        logger.info(f"✅ Downloaded {beat_id}: {local_path.stat().st_size} bytes")
                        return local_path
                    logger.error(f"Download failed for {beat_id}: file empty or missing")
                    return None
                    
                # Fallback: try direct HTTP download
                import requests
                logger.info(f"Attempting direct HTTP download for {beat_id}")
                response = requests.get(video_url, stream=True, timeout=60)
                if response.status_code == 200:
                    with open(local_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    if local_path.exists() and local_path.stat().st_size > 0:
                        logger.info(f"✅ Downloaded {beat_id} via HTTP: {local_path.stat().st_size} bytes")
                        return local_path
                logger.error(f"HTTP download failed for {beat_id}: status {response.status_code}")
                return None
            else:
                # Assume it's an S3 key
                key = video_url
            
            logger.info(f"Downloading {beat_id} from S3: {key}")
            
            self.s3.download_file(
                self.s3_bucket,
                key,
                str(local_path),
            )
            
            if local_path.exists() and local_path.stat().st_size > 0:
                logger.info(f"✅ Downloaded {beat_id}: {local_path.stat().st_size} bytes")
                return local_path
            else:
                logger.error(f"Download failed for {beat_id}: file empty or missing")
                return None
                
        except Exception as e:
            logger.error(f"Failed to download {beat_id} from {video_url}: {e}")
            return None

    
    def stitch_videos(
        self,
        shots: List[ComposedShot],
        output_name: Optional[str] = None,
        transition: str = "none",
        transition_duration: float = 0.5,
    ) -> Path:
        """
        Download and stitch all shot videos into a single episode.
        
        Args:
            shots: List of ComposedShot objects
            output_name: Output filename (defaults to episode_{world_id}.mp4)
            transition: Transition type (none, crossfade)
            transition_duration: Duration of transition in seconds
            
        Returns:
            Path to the stitched video
        """
        if not shots:
            raise ValueError("No shots provided for stitching")
        
        temp_dir = self._ensure_temp_dir()
        
        # Download all videos
        video_paths = []
        for shot in shots:
            video_url = str(shot.video_path)
            local_path = self.download_video(video_url, shot.beat_id)
            if local_path:
                video_paths.append(local_path)
            else:
                logger.warning(f"Skipping beat {shot.beat_id} - download failed")
        
        if not video_paths:
            raise RuntimeError("No videos could be downloaded")
        
        # Create output path
        if output_name is None:
            output_name = f"episode_{self.world_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        output_path = temp_dir / output_name
        
        # Create concat file
        concat_file = temp_dir / "concat_list.txt"
        with open(concat_file, "w") as f:
            for path in video_paths:
                escaped_path = str(path).replace("\\", "/")
                f.write(f"file '{escaped_path}'\n")
        
        logger.info(f"Stitching {len(video_paths)} videos with transition={transition}...")
        
        # Get ffmpeg path from env or fall back to PATH
        ffmpeg_path = os.getenv("FFMPEG_PATH", "ffmpeg")
        logger.info(f"Using ffmpeg: {ffmpeg_path}")
        
        # Build ffmpeg command
        if transition == "none":
            cmd = [
                ffmpeg_path,
                "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_file),
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "23",
                "-c:a", "aac",
                "-b:a", "192k",
                "-movflags", "+faststart",
                str(output_path),
            ]
        elif transition == "crossfade":
            cmd = self._build_crossfade_cmd(video_paths, output_path, transition_duration)
        else:
            raise ValueError(f"Unknown transition type: {transition}")
        
        # Run ffmpeg
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
            
            if output_path.exists():
                size_mb = output_path.stat().st_size / (1024 * 1024)
                logger.info(f"✅ Episode stitched: {output_path} ({size_mb:.2f} MB)")
                return output_path
            else:
                raise RuntimeError("ffmpeg completed but output file missing")
                
        except subprocess.CalledProcessError as e:
            logger.error(f"ffmpeg error: {e.stderr}")
            raise RuntimeError(f"Video stitching failed: {e.stderr}")
    
    def _build_crossfade_cmd(
        self,
        video_paths: List[Path],
        output_path: Path,
        duration: float
    ) -> List[str]:
        """Build ffmpeg command for crossfade transitions."""
        if len(video_paths) <= 1:
            return [
                "ffmpeg", "-y", "-i", str(video_paths[0]),
                "-c:v", "libx264", "-preset", "medium", "-crf", "23",
                str(output_path)
            ]
        
        inputs = []
        for path in video_paths:
            inputs.extend(["-i", str(path)])
        
        # Build xfade filter chain
        n = len(video_paths)
        filter_parts = []
        
        for i in range(n - 1):
            if i == 0:
                filter_parts.append(f"[0:v][1:v]xfade=transition=fade:duration={duration}:offset=3[v{i}]")
            else:
                filter_parts.append(f"[v{i-1}][{i+1}:v]xfade=transition=fade:duration={duration}:offset=3[v{i}]")
        
        filter_str = ";".join(filter_parts)
        
        cmd = ["ffmpeg", "-y"] + inputs
        cmd.extend([
            "-filter_complex", filter_str,
            "-map", f"[v{n-2}]",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-movflags", "+faststart",
            str(output_path)
        ])
        
        return cmd
    
    def upload_episode(self, local_path: Path) -> str:
        """Upload final episode to R2 and return the URL."""
        key = f"episodes/{self.world_id}/final/{local_path.name}"
        
        try:
            logger.info(f"Uploading final episode to {key}...")
            
            self.s3.upload_file(
                str(local_path),
                self.s3_bucket,
                key,
                ExtraArgs={"ContentType": "video/mp4"},
            )
            
            # Generate URL
            url = f"{self.s3_endpoint}/{self.s3_bucket}/{key}"
            
            # Store result in Redis
            r.hset(
                f"episode_results:{self.world_id}",
                "final",
                json.dumps({
                    "status": "completed",
                    "url": url,
                    "key": key,
                    "size_bytes": local_path.stat().st_size,
                    "created_at": datetime.now().isoformat(),
                })
            )
            
            logger.info(f"✅ Episode uploaded: {url}")
            return url
            
        except Exception as e:
            logger.error(f"Failed to upload episode: {e}")
            raise
    
    def compose_and_stitch(
        self,
        plan: EpisodePlan,
        transition: str = "none",
        upload: bool = True,
    ) -> Dict[str, Any]:
        """
        Full composition pipeline:
        1. Compose shots from plan
        2. Download and stitch videos
        3. Upload final episode
        
        Returns:
            Dict with episode_url, shot_count, duration, etc.
        """
        start_time = datetime.now()
        
        # Step 1: Compose shots
        try:
            shots = self.compose(plan)
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to compose shots: {e}",
                "world_id": self.world_id,
            }
        
        # Step 2: Stitch
        try:
            final_path = self.stitch_videos(shots, transition=transition)
        except Exception as e:
            return {
                "status": "error",
                "message": f"Stitching failed: {e}",
                "world_id": self.world_id,
                "shot_count": len(shots),
            }
        
        # Step 3: Upload
        episode_url = None
        if upload:
            try:
                episode_url = self.upload_episode(final_path)
            except Exception as e:
                return {
                    "status": "partial",
                    "message": f"Stitching succeeded but upload failed: {e}",
                    "world_id": self.world_id,
                    "local_path": str(final_path),
                }
        
        duration = (datetime.now() - start_time).total_seconds()
        total_video_duration = sum(shot.duration for shot in shots)
        
        return {
            "status": "success",
            "world_id": self.world_id,
            "episode_url": episode_url,
            "local_path": str(final_path),
            "shot_count": len(shots),
            "total_duration_sec": total_video_duration,
            "file_size_mb": final_path.stat().st_size / (1024 * 1024),
            "composition_time_sec": duration,
        }
    
    def cleanup(self):
        """Remove temporary files."""
        import shutil
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            logger.info(f"Cleaned up temp dir: {self.temp_dir}")


# Convenience function
def compose_episode(
    world_id: str,
    beat_ids: List[str],
    transition: str = "none",
    upload: bool = True,
) -> Dict[str, Any]:
    """
    Compose an episode from a list of beat IDs.
    
    Usage:
        result = compose_episode("world-123", ["beat_1", "beat_2", "beat_3"])
        print(result["episode_url"])
    """
    plan = EpisodePlan(
        beats=beat_ids,
        allow_gaps=True,
        required_confidence=0.3,
    )
    
    composer = EpisodeComposer(world_id)
    try:
        return composer.compose_and_stitch(plan, transition=transition, upload=upload)
    finally:
        composer.cleanup()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compose episode from beat videos")
    parser.add_argument("world_id", help="World ID")
    parser.add_argument("--beats", nargs="+", help="Beat IDs to include")
    parser.add_argument("--transition", default="none", choices=["none", "crossfade"])
    parser.add_argument("--no-upload", action="store_true", help="Skip uploading to R2")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    if args.beats:
        result = compose_episode(
            args.world_id,
            args.beats,
            transition=args.transition,
            upload=not args.no_upload,
        )
    else:
        # Get all beats from Redis
        beat_ids = r.lrange(f"episode_beats:{args.world_id}", 0, -1)
        beat_ids = [b.decode() if isinstance(b, bytes) else b for b in beat_ids]
        
        if not beat_ids:
            print(f"No beats found for world {args.world_id}")
        else:
            result = compose_episode(
                args.world_id,
                beat_ids,
                transition=args.transition,
                upload=not args.no_upload,
            )
            print(json.dumps(result, indent=2, default=str))
