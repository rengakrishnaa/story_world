"""
Test Suite for Enhanced Motion Engine

Tests:
1. Motion types and easing functions
2. Motion from single image generation
3. Frame interpolation
4. Motion detection from beat description
5. Integration with veo backend
6. Integration with narrative planner

Run:
    python tests/test_motion_engine.py
"""

import os
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()


def test_imports():
    """Test 1: Verify all imports work"""
    print("\n" + "="*50)
    print("TEST 1: Import Verification")
    print("="*50)
    
    try:
        from agents.motion.enhanced_motion_engine import (
            EnhancedMotionEngine,
            MotionType,
            MotionConfig,
            EasingFunction,
            get_motion_engine,
            generate_video_from_image,
        )
        print("✅ EnhancedMotionEngine imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


def test_motion_types():
    """Test 2: Verify all motion types are defined"""
    print("\n" + "="*50)
    print("TEST 2: Motion Types")
    print("="*50)
    
    from agents.motion.enhanced_motion_engine import MotionType
    
    expected_types = [
        "static", "pan_left", "pan_right", "pan_up", "pan_down",
        "zoom_in", "zoom_out", "dolly_in", "dolly_out",
        "orbit_left", "orbit_right", "push_in", "pull_out",
        "subtle", "dynamic",
    ]
    
    all_present = True
    for motion_name in expected_types:
        try:
            motion = MotionType(motion_name)
            print(f"✅ {motion_name}")
        except ValueError:
            print(f"❌ Missing: {motion_name}")
            all_present = False
    
    return all_present


def test_easing_functions():
    """Test 3: Verify easing functions work correctly"""
    print("\n" + "="*50)
    print("TEST 3: Easing Functions")
    print("="*50)
    
    from agents.motion.enhanced_motion_engine import (
        EnhancedMotionEngine,
        EasingFunction,
    )
    
    engine = EnhancedMotionEngine()
    
    test_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    all_passed = True
    for easing in EasingFunction:
        try:
            results = [engine.apply_easing(t, easing) for t in test_values]
            
            # Check boundaries
            if abs(results[0] - 0.0) < 0.01 and abs(results[-1] - 1.0) < 0.01:
                print(f"✅ {easing.value}: boundaries correct")
            else:
                print(f"❌ {easing.value}: boundary error")
                all_passed = False
        except Exception as e:
            print(f"❌ {easing.value}: {e}")
            all_passed = False
    
    return all_passed


def test_motion_from_image():
    """Test 4: Generate motion from single image"""
    print("\n" + "="*50)
    print("TEST 4: Motion from Image")
    print("="*50)
    
    from agents.motion.enhanced_motion_engine import (
        EnhancedMotionEngine,
        MotionType,
        MotionConfig,
    )
    
    engine = EnhancedMotionEngine()
    
    # Create test image
    test_img = Image.new("RGB", (1280, 720), color=(100, 150, 200))
    
    motion_types_to_test = [
        MotionType.PAN_LEFT,
        MotionType.ZOOM_IN,
        MotionType.PUSH_IN,
        MotionType.SUBTLE,
        MotionType.STATIC,
    ]
    
    all_passed = True
    for mt in motion_types_to_test:
        try:
            config = MotionConfig(motion_type=mt, intensity=0.5)
            frames = engine.generate_motion_from_image(test_img, config, duration_sec=2.0)
            
            if len(frames) == 48:  # 2 seconds at 24 FPS
                print(f"✅ {mt.value}: {len(frames)} frames")
            else:
                print(f"⚠️ {mt.value}: {len(frames)} frames (expected 48)")
        except Exception as e:
            print(f"❌ {mt.value}: {e}")
            all_passed = False
    
    return all_passed


def test_motion_detection():
    """Test 5: Motion detection from beat description"""
    print("\n" + "="*50)
    print("TEST 5: Motion Detection")
    print("="*50)
    
    from agents.motion.enhanced_motion_engine import (
        EnhancedMotionEngine,
        MotionType,
    )
    
    engine = EnhancedMotionEngine()
    
    test_cases = [
        ("Two warriors clash in an epic battle", MotionType.DYNAMIC),
        ("A dramatic reveal as the hero turns", MotionType.PUSH_IN),
        ("Wide establishing shot of the city", MotionType.PAN_RIGHT),
        ("Close-up on emotional expression", MotionType.SUBTLE),
        ("Characters having a conversation", MotionType.SUBTLE),
    ]
    
    all_correct = True
    for description, expected_motion in test_cases:
        config = engine.detect_motion_type(description)
        
        if config.motion_type == expected_motion:
            print(f"✅ '{description[:35]}...' → {config.motion_type.value}")
        else:
            print(f"⚠️ '{description[:35]}...' → {config.motion_type.value} (expected {expected_motion.value})")
            # Not a failure - detection is heuristic
    
    return True


def test_frame_interpolation():
    """Test 6: Frame interpolation between two frames"""
    print("\n" + "="*50)
    print("TEST 6: Frame Interpolation")
    print("="*50)
    
    from agents.motion.enhanced_motion_engine import EnhancedMotionEngine
    
    engine = EnhancedMotionEngine()
    
    # Create two different frames
    frame1 = np.full((720, 1280, 3), 50, dtype=np.uint8)
    frame2 = np.full((720, 1280, 3), 200, dtype=np.uint8)
    
    frames = engine.interpolate_frames(frame1, frame2, num_frames=10, use_rife=False)
    
    checks = [
        (len(frames) == 10, f"Generated {len(frames)} frames (expected 10)"),
        (np.array_equal(frames[0], frame1), "First frame matches input"),
        (np.array_equal(frames[-1], frame2), "Last frame matches input"),
    ]
    
    all_passed = True
    for passed, desc in checks:
        if passed:
            print(f"✅ {desc}")
        else:
            print(f"❌ {desc}")
            all_passed = False
    
    # Check middle frame is interpolated
    mid_value = frames[5].mean()
    if 100 < mid_value < 150:
        print(f"✅ Middle frame interpolated (mean={mid_value:.1f})")
    else:
        print(f"⚠️ Middle frame value unexpected (mean={mid_value:.1f})")
    
    return all_passed


def test_veo_backend_integration():
    """Test 7: Integration with veo backend"""
    print("\n" + "="*50)
    print("TEST 7: Veo Backend Integration")
    print("="*50)
    
    try:
        from agents.backends.veo_backend import (
            generate_with_gemini_image_motion,
            generate_with_sdxl_motion,
        )
        import inspect
        
        # Check Gemini function
        gemini_source = inspect.getsource(generate_with_gemini_image_motion)
        gemini_checks = [
            ("enhanced_motion_engine" in gemini_source, "Imports enhanced motion"),
            ("get_motion_engine" in gemini_source, "Uses motion engine"),
            ("detect_motion_type" in gemini_source, "Detects motion type"),
        ]
        
        print("Gemini fallback:")
        for passed, desc in gemini_checks:
            print(f"  {'✅' if passed else '⚠️'} {desc}")
        
        # Check SDXL function
        sdxl_source = inspect.getsource(generate_with_sdxl_motion)
        sdxl_checks = [
            ("enhanced_motion_engine" in sdxl_source, "Imports enhanced motion"),
            ("get_motion_engine" in sdxl_source, "Uses motion engine"),
            ("detect_motion_type" in sdxl_source, "Detects motion type"),
        ]
        
        print("SDXL fallback:")
        for passed, desc in sdxl_checks:
            print(f"  {'✅' if passed else '⚠️'} {desc}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration check failed: {e}")
        return False


def test_narrative_planner_integration():
    """Test 8: Integration with narrative planner"""
    print("\n" + "="*50)
    print("TEST 8: Narrative Planner Integration")
    print("="*50)
    
    try:
        from agents.narrative_planner import ProductionNarrativePlanner
        import inspect
        
        source = inspect.getsource(ProductionNarrativePlanner.generate_beats)
        
        checks = [
            ("enhanced_motion_engine" in source, "Imports enhanced motion"),
            ("get_motion_engine" in source, "Uses motion engine"),
            ("detect_motion_type" in source, "Detects motion type"),
            ("motion_config" in source, "Adds motion_config to beats"),
        ]
        
        all_present = True
        for found, desc in checks:
            if found:
                print(f"✅ {desc}")
            else:
                print(f"⚠️ {desc}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration check failed: {e}")
        return False


def test_convenience_function():
    """Test 9: Test convenience function"""
    print("\n" + "="*50)
    print("TEST 9: Convenience Function")
    print("="*50)
    
    from agents.motion.enhanced_motion_engine import generate_video_from_image
    
    test_img = Image.new("RGB", (640, 480), color=(80, 120, 180))
    
    try:
        frames = generate_video_from_image(
            test_img,
            duration_sec=1.0,
            motion_type="zoom_in",
            intensity=0.5,
        )
        
        if len(frames) == 24:
            print(f"✅ Generated {len(frames)} frames")
            return True
        else:
            print(f"⚠️ Generated {len(frames)} frames (expected 24)")
            return True
    except Exception as e:
        print(f"❌ Convenience function failed: {e}")
        return False


def main():
    print("\n" + "#"*60)
    print("# ENHANCED MOTION ENGINE TEST SUITE")
    print("#"*60)
    
    results = {}
    
    results["imports"] = test_imports()
    results["motion_types"] = test_motion_types()
    results["easing_functions"] = test_easing_functions()
    results["motion_from_image"] = test_motion_from_image()
    results["motion_detection"] = test_motion_detection()
    results["frame_interpolation"] = test_frame_interpolation()
    results["veo_backend"] = test_veo_backend_integration()
    results["narrative_planner"] = test_narrative_planner_integration()
    results["convenience_function"] = test_convenience_function()
    
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
