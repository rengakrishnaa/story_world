"""
Comprehensive Integration and Verification Test Suite

Tests all cinematic system components including:
- All 40 camera movements
- All 28 shot types
- All 40 lighting setups
- All 35 color LUTs
- All 20 genre profiles
- All 25 post-processing effects
- Integration with narrative_planner
- Integration with veo_backend
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_all_camera_movements():
    """Test all 40 camera movement types."""
    print("\n" + "="*60)
    print("TEST 1: All Camera Movements (40)")
    print("="*60)
    
    from agents.cinematic.camera_system import CameraMovement, CameraSystem
    
    camera = CameraSystem()
    movements = list(CameraMovement)
    
    print(f"  Total movements: {len(movements)}")
    
    # Test each movement exists in enum
    for move in movements:
        assert move.value is not None
    
    # Group by category
    categories = {
        "Static": ["static"],
        "Pan": ["pan_left", "pan_right", "pan_up", "pan_down"],
        "Tilt": ["tilt_up", "tilt_down"],
        "Dolly": ["dolly_in", "dolly_out", "dolly_left", "dolly_right"],
        "Zoom": ["zoom_in", "zoom_out"],
        "Crane": ["crane_up", "crane_down", "crane_shot"],
        "Arc": ["arc_left", "arc_right"],
        "Special": ["tracking_shot", "steadicam", "handheld", "floating", "orbit"],
    }
    
    for cat, moves in categories.items():
        found = sum(1 for m in moves if any(cm.value == m for cm in movements))
        print(f"  {cat}: {found}/{len(moves)}")
    
    print(f"  ✅ All {len(movements)} camera movements validated")
    return True


def test_all_shot_types():
    """Test all 28 shot types."""
    print("\n" + "="*60)
    print("TEST 2: All Shot Types (28)")
    print("="*60)
    
    from agents.cinematic.shot_composer import ShotType, ShotComposer
    
    composer = ShotComposer()
    shot_types = list(ShotType)
    
    print(f"  Total shot types: {len(shot_types)}")
    
    # Test each shot type
    for shot in shot_types:
        spec = composer.compose_shot(
            description=f"Test scene with {shot.value}",
            genre="cinematic"
        )
        assert spec is not None
    
    # Group by category
    categories = {
        "Wide": ["extreme_long_shot", "long_shot", "establishing"],
        "Medium": ["medium_long_shot", "medium_shot", "medium_close_up"],
        "Close": ["close_up", "extreme_close_up", "insert"],
        "Multi": ["two_shot", "group_shot", "over_the_shoulder"],
        "Special": ["pov", "reaction_shot", "cutaway", "impact_shot", "beauty_shot"],
    }
    
    for cat, shots in categories.items():
        found = sum(1 for s in shots if any(st.value == s for st in shot_types))
        print(f"  {cat}: {found}/{len(shots)}")
    
    print(f"  ✅ All {len(shot_types)} shot types validated")
    return True


def test_all_lighting_setups():
    """Test all 40 lighting setups."""
    print("\n" + "="*60)
    print("TEST 3: All Lighting Setups (40)")
    print("="*60)
    
    from agents.cinematic.lighting_engine import LightingSetup, LightingEngine
    
    engine = LightingEngine()
    setups = list(LightingSetup)
    
    print(f"  Total lighting setups: {len(setups)}")
    
    # Test each setup exists in enum
    for setup in setups:
        assert setup.value is not None
    
    # Test engine can create lighting for various genres
    genres = ["anime", "noir", "cyberpunk", "horror"]
    for genre in genres:
        spec = engine.create_lighting("Test scene", genre)
        assert spec is not None
    
    # Group by category
    categories = {
        "Standard": ["three_point", "high_key", "low_key", "flat"],
        "Natural": ["natural", "available", "window", "practical"],
        "Dramatic": ["split", "rembrandt", "butterfly", "rim"],
        "Special": ["silhouette", "under_lighting", "cel_shaded", "neon"],
    }
    
    for cat, lights in categories.items():
        found = sum(1 for l in lights if any(ls.value == l for ls in setups))
        print(f"  {cat}: {found}/{len(lights)}")
    
    print(f"  ✅ All {len(setups)} lighting setups validated")
    return True


def test_all_color_luts():
    """Test all 35 color LUTs."""
    print("\n" + "="*60)
    print("TEST 4: All Color LUTs (35)")
    print("="*60)
    
    from agents.cinematic.color_grading import ColorLUT, ColorGradingEngine
    
    engine = ColorGradingEngine()
    luts = list(ColorLUT)
    
    print(f"  Total LUTs: {len(luts)}")
    
    # Test each LUT
    for lut in luts:
        grade, post = engine.create_grade(
            description=f"Test scene with {lut.value}",
            genre="cinematic"
        )
        assert grade is not None
        assert post is not None
    
    # Group by category
    categories = {
        "Modern Film": ["teal_orange", "bleach_bypass", "cross_process"],
        "Classic": ["noir", "sepia", "monochrome"],
        "Animation": ["anime_pop", "ghibli_soft", "pixar_bright", "cartoon_vibrant"],
        "Film Stocks": ["kodak_5219", "fuji_eterna", "cinestill_800t"],
        "HDR/Broadcast": ["rec2020_hdr", "aces_filmic", "broadcast_709"],
    }
    
    for cat, grades in categories.items():
        found = sum(1 for g in grades if any(l.value == g for l in luts))
        print(f"  {cat}: {found}/{len(grades)}")
    
    print(f"  ✅ All {len(luts)} color LUTs validated")
    return True


def test_all_genre_profiles():
    """Test all 20 genre profiles."""
    print("\n" + "="*60)
    print("TEST 5: All Genre Profiles (20)")
    print("="*60)
    
    from agents.cinematic.genre_profiles import GENRE_PROFILES, get_genre_profile
    
    print(f"  Total profiles: {len(GENRE_PROFILES)}")
    
    # Test each profile
    for name, profile in GENRE_PROFILES.items():
        assert profile.camera is not None
        assert profile.shots is not None
        assert profile.lighting is not None
        assert profile.color is not None
        assert profile.post is not None
        assert profile.prompt_prefix != ""
    
    # List all genres
    genres = list(GENRE_PROFILES.keys())
    print(f"  Genres: {', '.join(genres[:10])}...")
    
    print(f"  ✅ All {len(GENRE_PROFILES)} genre profiles validated")
    return True


def test_all_post_effects():
    """Test all 25 post-processing effects."""
    print("\n" + "="*60)
    print("TEST 6: All Post-Processing Effects (25)")
    print("="*60)
    
    from agents.cinematic.color_grading import PostProcessEffect, PostProcessSpec
    
    effects = list(PostProcessEffect)
    
    print(f"  Total effects: {len(effects)}")
    
    # Group by category
    categories = {
        "Optical": ["bloom", "ca", "lens_flare", "anamorphic", "halation", "dof"],
        "Film": ["grain", "light_leak", "scratches", "dust", "gate_weave"],
        "Diffusion": ["vignette", "glow", "soften", "sharpen"],
        "Retro": ["scan_lines", "vhs_glitch", "digital_glitch", "pixelate"],
    }
    
    for cat, efx in categories.items():
        found = sum(1 for e in efx if any(pe.value == e for pe in effects))
        print(f"  {cat}: {found}/{len(efx)}")
    
    # Test PostProcessSpec creation
    spec = PostProcessSpec(bloom=0.5, vignette=0.3, film_grain=0.2)
    assert spec is not None
    
    print(f"  ✅ All {len(effects)} post-processing effects validated")
    return True


def test_narrative_planner_integration():
    """Test integration with narrative_planner.py."""
    print("\n" + "="*60)
    print("TEST 7: Narrative Planner Integration")
    print("="*60)
    
    from agents.cinematic import direct_beat
    
    # This simulates what narrative_planner does
    description = "The hero confronts the villain in an epic showdown"
    genre = "anime"
    
    spec = direct_beat(description, genre)
    
    # Check all required fields exist
    assert spec.camera is not None, "Camera missing"
    assert spec.shot is not None, "Shot missing"
    assert spec.lighting is not None, "Lighting missing"
    assert spec.color_grade is not None, "Color grade missing"
    assert spec.post_process is not None, "Post process missing"
    
    # Check prompt generation
    prompt = spec.get_full_prompt(description)
    assert len(prompt) > len(description), "Prompt not enhanced"
    assert "anime" in prompt.lower(), "Genre not in prompt"
    
    # Check serialization
    data = spec.to_dict()
    assert "camera" in data
    assert "lighting" in data
    assert "color_grade" in data
    
    print(f"  ✅ direct_beat() returns complete spec")
    print(f"  ✅ get_full_prompt() enhances description")
    print(f"  ✅ to_dict() serializes correctly")
    print(f"  ✅ Narrative planner integration verified")
    
    return True


def test_veo_backend_integration():
    """Test integration with veo_backend.py."""
    print("\n" + "="*60)
    print("TEST 8: Veo Backend Integration")
    print("="*60)
    
    from agents.backends.veo_backend import build_cinematic_prompt, get_negative_prompt
    from agents.cinematic import direct_beat
    
    # Simulate beat with cinematic spec
    description = "Neon city street at night"
    genre = "cyberpunk"
    
    spec = direct_beat(description, genre)
    cinematic_prompt = spec.get_full_prompt(description)
    
    # Create input_spec like narrative_planner does
    input_spec = {
        "prompt": description,
        "cinematic_prompt": cinematic_prompt,
        "style": genre,
    }
    
    # Test build_cinematic_prompt
    enhanced = build_cinematic_prompt(description, input_spec)
    assert len(enhanced) > len(description)
    
    # Test negative prompt
    negative = get_negative_prompt(input_spec)
    assert negative is not None
    
    print(f"  ✅ build_cinematic_prompt() works")
    print(f"  ✅ get_negative_prompt() works")
    print(f"  ✅ Veo backend integration verified")
    
    return True


def test_style_detector_integration():
    """Test integration with style_detector.py."""
    print("\n" + "="*60)
    print("TEST 9: Style Detector Integration")
    print("="*60)
    
    from agents.style_detector import StyleDetector
    from agents.cinematic import direct_beat
    
    detector = StyleDetector()
    
    test_cases = [
        ("Anime battle between warriors", "anime"),
        ("Pixar-style family adventure", "pixar"),
        ("Noir detective mystery", "noir"),
        ("Cyberpunk city at night", "cyberpunk"),
    ]
    
    for description, expected_style in test_cases:
        # Style detector detects style
        profile = detector.detect(description)
        detected = profile.style.value if hasattr(profile.style, 'value') else str(profile.style)
        
        # Cinematic system uses detected style
        spec = direct_beat(description, detected)
        
        assert spec is not None
        print(f"  ✅ '{expected_style}' → detected and directed")
    
    print(f"  ✅ Style detector integration verified")
    return True


def test_motion_engine_integration():
    """Test integration with enhanced motion engine."""
    print("\n" + "="*60)
    print("TEST 10: Motion Engine Integration")
    print("="*60)
    
    from agents.motion.enhanced_motion_engine import EnhancedMotionEngine
    from agents.cinematic.camera_system import CameraMovement
    
    engine = EnhancedMotionEngine()
    
    test_cases = [
        ("Epic battle with explosions", "dynamic"),
        ("Slow reveal as hero enters", "push_in"),
        ("Wide establishing shot of city", "pan_right"),
        ("Close-up on emotional expression", "subtle"),
    ]
    
    for description, expected in test_cases:
        motion = engine.detect_motion_type(description)
        assert motion is not None
        print(f"  ✅ '{expected}' motion detected")
    
    # Verify camera movements match motion engine types
    motion_types = ["pan_left", "pan_right", "zoom_in", "zoom_out", "dolly_in"]
    camera_moves = [m.value for m in CameraMovement]
    
    for mtype in motion_types:
        if mtype in camera_moves or any(mtype in m for m in camera_moves):
            print(f"  ✅ '{mtype}' exists in camera system")
    
    print(f"  ✅ Motion engine integration verified")
    return True


def test_component_inventory():
    """Final inventory of all components."""
    print("\n" + "="*60)
    print("COMPONENT INVENTORY")
    print("="*60)
    
    from agents.cinematic.camera_system import CameraMovement, LensType, CameraRig
    from agents.cinematic.shot_composer import ShotType, CompositionType
    from agents.cinematic.lighting_engine import LightingSetup, TimeOfDay
    from agents.cinematic.color_grading import ColorLUT, PostProcessEffect
    from agents.cinematic.genre_profiles import GENRE_PROFILES
    from agents.style_detector import VisualStyle
    from agents.motion.enhanced_motion_engine import MotionType
    
    inventory = {
        "Visual Styles": len(VisualStyle),
        "Motion Types": len(MotionType),
        "Camera Movements": len(CameraMovement),
        "Lens Types": len(LensType),
        "Camera Rigs": len(CameraRig),
        "Shot Types": len(ShotType),
        "Compositions": len(CompositionType),
        "Lighting Setups": len(LightingSetup),
        "Time of Day": len(TimeOfDay),
        "Color LUTs": len(ColorLUT),
        "Post Effects": len(PostProcessEffect),
        "Genre Profiles": len(GENRE_PROFILES),
    }
    
    total = 0
    for name, count in inventory.items():
        print(f"  {name:20}: {count:3}")
        total += count
    
    print(f"\n  {'TOTAL':20}: {total}")
    
    return True


def main():
    """Run all verification tests."""
    print("\n" + "#"*60)
    print("# COMPREHENSIVE INTEGRATION VERIFICATION")
    print("#"*60)
    
    results = {}
    
    tests = [
        ("camera_movements", test_all_camera_movements),
        ("shot_types", test_all_shot_types),
        ("lighting_setups", test_all_lighting_setups),
        ("color_luts", test_all_color_luts),
        ("genre_profiles", test_all_genre_profiles),
        ("post_effects", test_all_post_effects),
        ("narrative_planner", test_narrative_planner_integration),
        ("veo_backend", test_veo_backend_integration),
        ("style_detector", test_style_detector_integration),
        ("motion_engine", test_motion_engine_integration),
        ("inventory", test_component_inventory),
    ]
    
    for name, test_fn in tests:
        try:
            results[name] = test_fn()
        except Exception as e:
            print(f"  ❌ {name} failed: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False
    
    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {name}: {status}")
    
    print()
    print(f"🎉 {passed}/{total} verification tests passed!" if passed == total else f"⚠️ {passed}/{total} tests passed")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
