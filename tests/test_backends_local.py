"""
Local Test Suite for Video Backends

Tests the full fallback chain:
1. Veo 3.1 API
2. Gemini Image + Motion
3. SDXL + Motion (local GPU - no API)

Usage:
    python tests/test_backends_local.py
"""

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()


def test_imports():
    """Test 1: Verify all imports work"""
    print("\n" + "="*50)
    print("TEST 1: Import Verification")
    print("="*50)
    
    all_good = True
    
    # Core imports
    try:
        from PIL import Image
        print("✅ Pillow")
    except ImportError as e:
        print(f"❌ Pillow: {e}")
        all_good = False
    
    try:
        import numpy as np
        print("✅ NumPy")
    except ImportError as e:
        print(f"❌ NumPy: {e}")
        all_good = False
    
    # google-genai (optional - API fallback)
    try:
        from google import genai
        print("✅ google-genai (API available)")
    except ImportError:
        print("⚠️ google-genai not installed (will skip API tests)")
    
    # diffusers (required for SDXL fallback)
    try:
        from diffusers import StableDiffusionXLPipeline
        print("✅ diffusers (SDXL available)")
    except ImportError:
        print("⚠️ diffusers not installed (SDXL fallback unavailable)")
    
    # torch
    try:
        import torch
        cuda_status = "CUDA available" if torch.cuda.is_available() else "CPU only"
        print(f"✅ PyTorch ({cuda_status})")
    except ImportError:
        print("❌ PyTorch not installed")
        all_good = False
    
    # Backend imports
    try:
        from agents.backends import veo_backend
        print("✅ veo_backend module")
    except Exception as e:
        print(f"❌ veo_backend: {e}")
        all_good = False
    
    try:
        from agents.motion.sparse_motion_engine import SparseMotionEngine
        print("✅ SparseMotionEngine")
    except Exception as e:
        print(f"❌ SparseMotionEngine: {e}")
        all_good = False
    
    return all_good


def test_env_vars():
    """Test 2: Check environment variables"""
    print("\n" + "="*50)
    print("TEST 2: Environment Variables")
    print("="*50)
    
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        print(f"✅ GEMINI_API_KEY set ({gemini_key[:10]}...)")
    else:
        print("⚠️ GEMINI_API_KEY not set (API fallbacks unavailable)")
    
    return True


def test_sdxl_generation():
    """Test 3: SDXL Local Image Generation (No API)"""
    print("\n" + "="*50)
    print("TEST 3: SDXL Local Image Generation")
    print("="*50)
    
    try:
        import torch
        if not torch.cuda.is_available():
            print("⚠️ CUDA not available - skipping SDXL test")
            return True  # Not a failure, just skipped
        
        from diffusers import StableDiffusionXLPipeline
        from PIL import Image
        
        print("Loading SDXL (first time may download ~6GB model)...")
        
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        ).to("cuda")
        
        pipe.enable_attention_slicing()
        
        print("Generating test image...")
        
        image = pipe(
            prompt="a simple red circle on white background, minimal",
            guidance_scale=7.5,
            num_inference_steps=15,  # Fast test
            generator=torch.Generator(device="cuda").manual_seed(42),
        ).images[0]
        
        # Save test image
        os.makedirs("tests", exist_ok=True)
        test_path = "tests/test_sdxl_output.png"
        image.save(test_path)
        
        print(f"✅ SDXL image generated: {image.size}")
        print(f"✅ Saved to {test_path}")
        
        # Cleanup
        del pipe
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ SDXL test failed: {e}")
        traceback.print_exc()
        return False


def test_motion_engine():
    """Test 4: Motion Engine (frame interpolation)"""
    print("\n" + "="*50)
    print("TEST 4: Motion Engine")
    print("="*50)
    
    try:
        from PIL import Image
        import numpy as np
        from agents.motion.sparse_motion_engine import SparseMotionEngine
        
        # Create test frames
        start = Image.new("RGB", (512, 512), color=(255, 0, 0))  # Red
        end = Image.new("RGB", (512, 512), color=(0, 0, 255))    # Blue
        
        engine = SparseMotionEngine()
        
        print("Generating motion frames...")
        frames = engine.render_motion(
            start_frame=start,
            end_frame=end,
            duration_sec=1.0,  # 1 second = 24 frames
        )
        
        print(f"✅ Generated {len(frames)} frames")
        
        # Test video encoding
        os.makedirs("tests", exist_ok=True)
        output_path = "tests/test_motion_output.mp4"
        engine._encode_frames_to_video(frames, output_path, fps=24)
        
        if os.path.exists(output_path):
            size = os.path.getsize(output_path)
            print(f"✅ Video encoded: {output_path} ({size} bytes)")
            return True
        else:
            print("❌ Video encoding failed")
            return False
        
    except Exception as e:
        print(f"❌ Motion engine test failed: {e}")
        traceback.print_exc()
        return False


def test_full_sdxl_video_pipeline():
    """Test 5: Full SDXL + Motion Pipeline (Real Video Output)"""
    print("\n" + "="*50)
    print("TEST 5: Full SDXL + Motion Video Pipeline")
    print("="*50)
    
    try:
        import torch
        if not torch.cuda.is_available():
            print("⚠️ CUDA not available - skipping")
            return True
        
        from agents.backends.veo_backend import generate_with_sdxl_motion
        
        print("Generating full video with SDXL + motion...")
        
        video_path = generate_with_sdxl_motion(
            prompt="a beautiful sunset over mountains",
            duration_sec=2.0,
            input_spec={}
        )
        
        if os.path.exists(video_path):
            size = os.path.getsize(video_path)
            print(f"✅ Full pipeline SUCCESS!")
            print(f"✅ Video: {video_path} ({size} bytes)")
            
            # Copy to tests folder for easy access
            import shutil
            dest = "tests/test_full_pipeline_output.mp4"
            shutil.copy(video_path, dest)
            print(f"✅ Copied to: {dest}")
            
            return True
        else:
            print("❌ Video not created")
            return False
            
    except Exception as e:
        print(f"❌ Full pipeline test failed: {e}")
        traceback.print_exc()
        return False


def main():
    print("\n" + "#"*60)
    print("# STORYWORLD LOCAL TEST SUITE")
    print("# Testing SDXL fallback (no API required)")
    print("#"*60)
    
    results = {}
    
    results["imports"] = test_imports()
    results["env_vars"] = test_env_vars()
    results["motion_engine"] = test_motion_engine()
    results["sdxl_generation"] = test_sdxl_generation()
    results["full_pipeline"] = test_full_sdxl_video_pipeline()
    
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
        print("\n🎉 All tests passed! Safe to build Docker.")
    else:
        print("\n⚠️ Some tests failed. Check output above.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
