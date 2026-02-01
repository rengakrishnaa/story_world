"""
Test Suite for Episode Composer

Tests:
1. Individual component tests (unit tests)
2. Full pipeline integration test

Run:
    python tests/test_episode_composer.py
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()


def test_imports():
    """Test 1: Verify all imports work"""
    print("\n" + "="*50)
    print("TEST 1: Import Verification")
    print("="*50)
    
    try:
        from agents.episode_composer import EpisodeComposer, compose_episode
        print("✅ EpisodeComposer imported")
    except Exception as e:
        print(f"❌ EpisodeComposer import failed: {e}")
        return False
    
    try:
        from models.episode_plan import EpisodePlan
        print("✅ EpisodePlan imported")
    except Exception as e:
        print(f"❌ EpisodePlan import failed: {e}")
        return False
    
    try:
        from models.composed_shot import ComposedShot
        print("✅ ComposedShot imported")
    except Exception as e:
        print(f"❌ ComposedShot import failed: {e}")
        return False
    
    return True


def test_env_vars():
    """Test 2: Check required environment variables"""
    print("\n" + "="*50)
    print("TEST 2: Environment Variables")
    print("="*50)
    
    required = ["S3_ENDPOINT", "S3_BUCKET", "S3_ACCESS_KEY", "S3_SECRET_KEY"]
    all_set = True
    
    for var in required:
        value = os.getenv(var)
        if value:
            print(f"✅ {var} is set")
        else:
            print(f"⚠️ {var} not set (R2 upload will fail)")
            all_set = False
    
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        print(f"✅ REDIS_URL is set")
    else:
        print(f"⚠️ REDIS_URL not set (using localhost)")
    
    return True  # Don't fail on missing env vars


def test_ffmpeg_available():
    """Test 3: Check ffmpeg is available"""
    print("\n" + "="*50)
    print("TEST 3: FFmpeg Availability")
    print("="*50)
    
    import subprocess
    
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            version_line = result.stdout.split("\n")[0]
            print(f"✅ FFmpeg available: {version_line}")
            return True
        else:
            print(f"❌ FFmpeg error: {result.stderr}")
            return False
    except FileNotFoundError:
        print("❌ FFmpeg not found. Install from: https://ffmpeg.org/download.html")
        return False


def test_video_stitching_mock():
    """Test 4: Test video stitching with mock data"""
    print("\n" + "="*50)
    print("TEST 4: Video Stitching (Mock Data)")
    print("="*50)
    
    import subprocess
    from PIL import Image
    
    # Create temp directory
    temp_dir = Path(tempfile.mkdtemp(prefix="test_stitch_"))
    
    try:
        # Create mock video frames
        print("Creating mock videos...")
        
        for i in range(3):
            frame_dir = temp_dir / f"frames_{i}"
            frame_dir.mkdir()
            
            # Create solid color frames
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
            for j in range(24):  # 1 second at 24fps
                img = Image.new("RGB", (640, 480), colors[i])
                img.save(frame_dir / f"{j:04d}.png")
            
            # Encode to video
            video_path = temp_dir / f"beat_{i}.mp4"
            subprocess.run([
                "ffmpeg", "-y",
                "-framerate", "24",
                "-i", str(frame_dir / "%04d.png"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                str(video_path),
            ], capture_output=True, check=True)
            
            print(f"  Created {video_path.name}")
        
        # Create concat file
        concat_file = temp_dir / "concat.txt"
        with open(concat_file, "w") as f:
            for i in range(3):
                video_path = temp_dir / f"beat_{i}.mp4"
                f.write(f"file '{str(video_path).replace(chr(92), '/')}'\n")
        
        # Stitch videos
        output_path = temp_dir / "stitched.mp4"
        print("Stitching videos...")
        
        result = subprocess.run([
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            str(output_path),
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ FFmpeg failed: {result.stderr}")
            return False
        
        if output_path.exists():
            size_kb = output_path.stat().st_size / 1024
            print(f"✅ Stitched video created: {size_kb:.2f} KB")
            
            # Verify duration with ffprobe
            probe = subprocess.run([
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "json",
                str(output_path),
            ], capture_output=True, text=True)
            
            if probe.returncode == 0:
                duration = json.loads(probe.stdout)["format"]["duration"]
                print(f"✅ Video duration: {float(duration):.2f} seconds")
            
            return True
        else:
            print("❌ Output file not created")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_composer_class():
    """Test 5: Test EpisodeComposer class initialization"""
    print("\n" + "="*50)
    print("TEST 5: EpisodeComposer Class")
    print("="*50)
    
    try:
        from agents.episode_composer import EpisodeComposer
        
        composer = EpisodeComposer(world_id="test-world-123")
        
        print(f"✅ EpisodeComposer created for world: {composer.world_id}")
        print(f"   S3 endpoint: {composer.s3_endpoint or 'not set'}")
        print(f"   S3 bucket: {composer.s3_bucket or 'not set'}")
        
        # Test temp dir creation
        temp_dir = composer._ensure_temp_dir()
        print(f"✅ Temp dir created: {temp_dir}")
        
        composer.cleanup()
        print("✅ Cleanup successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_api_endpoint():
    """Test 6: Test API endpoint import"""
    print("\n" + "="*50)
    print("TEST 6: API Endpoint")
    print("="*50)
    
    try:
        from main import app
        
        # Check if compose endpoint exists
        routes = [route.path for route in app.routes]
        
        if "/episodes/{episode_id}/compose" in routes:
            print("✅ /episodes/{episode_id}/compose endpoint registered")
        else:
            print("⚠️ Compose endpoint not found in routes")
            print(f"   Available routes: {routes}")
        
        if "/episodes/{episode_id}/video" in routes:
            print("✅ /episodes/{episode_id}/video endpoint registered")
        else:
            print("⚠️ Video endpoint not found in routes")
        
        return True
        
    except Exception as e:
        print(f"❌ API test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "#"*60)
    print("# EPISODE COMPOSER TEST SUITE")
    print("#"*60)
    
    results = {}
    
    results["imports"] = test_imports()
    results["env_vars"] = test_env_vars()
    results["ffmpeg"] = test_ffmpeg_available()
    results["stitching_mock"] = test_video_stitching_mock()
    results["composer_class"] = test_composer_class()
    results["api_endpoint"] = test_api_endpoint()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️ Some tests failed. Check output above.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
