"""
Test Suite for Cinematic System

Comprehensive tests for the production-level cinematic system.

Run:
    python tests/test_cinematic_system.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_camera_system():
    """Test 1: Camera System"""
    print("\n" + "="*50)
    print("TEST 1: Camera System")
    print("="*50)
    
    try:
        from agents.cinematic.camera_system import (
            CameraSystem,
            CameraMovement,
            CameraLens,
            CameraRig,
        )
        
        system = CameraSystem()
        
        # Test movement count
        movement_count = len(CameraMovement)
        print(f"✅ {movement_count} camera movements defined")
        
        # Test camera selection
        spec = system.select_camera_for_beat("Epic battle at sunset", "action")
        print(f"✅ Camera selection: {spec.movement.value}, {spec.lens.focal_length}mm")
        
        # Test lens creation
        lens = CameraLens.from_mm(85)
        print(f"✅ Lens creation: {lens.focal_length}mm ({lens.lens_type.value})")
        
        # Test rig count
        rig_count = len(CameraRig)
        print(f"✅ {rig_count} camera rigs defined")
        
        return True
        
    except Exception as e:
        print(f"❌ Camera system test failed: {e}")
        return False


def test_shot_composer():
    """Test 2: Shot Composer"""
    print("\n" + "="*50)
    print("TEST 2: Shot Composer")
    print("="*50)
    
    try:
        from agents.cinematic.shot_composer import (
            ShotComposer,
            ShotType,
            CompositionType,
        )
        
        composer = ShotComposer()
        
        # Test shot type count
        shot_count = len(ShotType)
        print(f"✅ {shot_count} shot types defined")
        
        # Test composition count
        comp_count = len(CompositionType)
        print(f"✅ {comp_count} composition types defined")
        
        # Test shot composition
        spec = composer.compose_shot("Close-up on tearful expression", "drama")
        print(f"✅ Shot selection: {spec.shot_type.value} ({spec.composition.value})")
        
        return True
        
    except Exception as e:
        print(f"❌ Shot composer test failed: {e}")
        return False


def test_lighting_engine():
    """Test 3: Lighting Engine"""
    print("\n" + "="*50)
    print("TEST 3: Lighting Engine")
    print("="*50)
    
    try:
        from agents.cinematic.lighting_engine import (
            LightingEngine,
            LightingSetup,
            TimeOfDay,
            GENRE_LIGHTING,
        )
        
        engine = LightingEngine()
        
        # Test lighting setup count
        setup_count = len(LightingSetup)
        print(f"✅ {setup_count} lighting setups defined")
        
        # Test time of day count
        time_count = len(TimeOfDay)
        print(f"✅ {time_count} time-of-day options defined")
        
        # Test genre presets
        genre_count = len(GENRE_LIGHTING)
        print(f"✅ {genre_count} genre lighting presets")
        
        # Test lighting creation
        spec = engine.create_lighting("Dark mysterious alley at night", "noir")
        print(f"✅ Lighting: {spec.setup.value}, {spec.time_of_day.value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Lighting engine test failed: {e}")
        return False


def test_color_grading():
    """Test 4: Color Grading"""
    print("\n" + "="*50)
    print("TEST 4: Color Grading")
    print("="*50)
    
    try:
        from agents.cinematic.color_grading import (
            ColorGradingEngine,
            ColorLUT,
            PostProcessEffect,
            GENRE_COLOR_PRESETS,
        )
        
        engine = ColorGradingEngine()
        
        # Test LUT count
        lut_count = len(ColorLUT)
        print(f"✅ {lut_count} LUTs defined")
        
        # Test post effect count
        effect_count = len(PostProcessEffect)
        print(f"✅ {effect_count} post-processing effects defined")
        
        # Test genre presets
        preset_count = len(GENRE_COLOR_PRESETS)
        print(f"✅ {preset_count} color grading presets")
        
        # Test grading creation
        grade, post = engine.create_grade("Romantic sunset scene", "romance", "golden_pm")
        print(f"✅ Color grade: {grade.lut.value}, sat={grade.saturation:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Color grading test failed: {e}")
        return False


def test_genre_profiles():
    """Test 5: Genre Profiles"""
    print("\n" + "="*50)
    print("TEST 5: Genre Profiles")
    print("="*50)
    
    try:
        from agents.cinematic.genre_profiles import (
            GENRE_PROFILES,
            get_genre_profile,
            list_genres,
        )
        
        # Test genre count
        genre_count = len(GENRE_PROFILES)
        print(f"✅ {genre_count} genre profiles defined")
        
        # Test genre list
        genres = list_genres()
        print(f"✅ Genres: {', '.join(genres[:5])}...")
        
        # Test profile retrieval
        for genre in ["anime", "pixar", "noir"]:
            profile = get_genre_profile(genre)
            print(f"✅ {genre}: {profile.name}, LUT={profile.color.get('lut', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Genre profiles test failed: {e}")
        return False


def test_cinematic_director():
    """Test 6: Cinematic Director"""
    print("\n" + "="*50)
    print("TEST 6: Cinematic Director (Main Orchestrator)")
    print("="*50)
    
    try:
        from agents.cinematic.cinematic_director import (
            CinematicDirector,
            direct_beat,
            get_cinematic_prompt,
        )
        
        director = CinematicDirector()
        
        # Test directing a beat
        spec = director.direct_beat("Hero realizes the truth", "anime")
        
        print(f"✅ Genre: {spec.genre}")
        print(f"✅ Camera: {spec.camera.movement.value if spec.camera else 'N/A'}")
        print(f"✅ Shot: {spec.shot.shot_type.value if spec.shot else 'N/A'}")
        print(f"✅ Lighting: {spec.lighting.setup.value if spec.lighting else 'N/A'}")
        print(f"✅ LUT: {spec.color_grade.lut.value if spec.color_grade else 'N/A'}")
        
        # Test summary
        summary = spec.get_summary()
        print(f"✅ Summary: {summary}")
        
        # Test full prompt
        full_prompt = spec.get_full_prompt("Hero realizes the truth")
        print(f"✅ Full prompt: {full_prompt[:80]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Cinematic director test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Test 7: Full Integration Test"""
    print("\n" + "="*50)
    print("TEST 7: Full Integration")
    print("="*50)
    
    try:
        from agents.cinematic import (
            direct_beat,
            get_cinematic_prompt,
            list_genres,
        )
        
        # Test all genres
        genres = list_genres()
        
        test_desc = "Epic dramatic scene with tension"
        
        all_passed = True
        for genre in genres:
            try:
                spec = direct_beat(test_desc, genre)
                if spec.camera and spec.shot and spec.lighting and spec.color_grade:
                    print(f"✅ {genre}: complete")
                else:
                    print(f"⚠️ {genre}: partial spec")
            except Exception as e:
                print(f"❌ {genre}: {e}")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


def main():
    print("\n" + "#"*60)
    print("# CINEMATIC SYSTEM TEST SUITE")
    print("#"*60)
    
    results = {}
    
    results["camera_system"] = test_camera_system()
    results["shot_composer"] = test_shot_composer()
    results["lighting_engine"] = test_lighting_engine()
    results["color_grading"] = test_color_grading()
    results["genre_profiles"] = test_genre_profiles()
    results["cinematic_director"] = test_cinematic_director()
    results["integration"] = test_integration()
    
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
        print("\n⚠️ Some tests failed.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
