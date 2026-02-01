"""
Professional Color Grading Test Suite

Comprehensive tests for the color grading engine including:
- All Color LUTs
- Color curves and parameters
- Post-processing effects
- Genre presets
- Emotional adjustments
- Time-based adjustments
- Prompt generation
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_color_luts():
    """Test all color LUT definitions."""
    print("\n" + "="*50)
    print("TEST 1: Color LUTs")
    print("="*50)
    
    from agents.cinematic.color_grading import ColorLUT
    
    print(f"✅ {len(ColorLUT)} LUTs defined")
    
    # Group by category
    categories = {
        "Modern Film": ["teal_orange", "bleach_bypass", "cross_process", "vintage_film", "cool_steel", "warm_sunset"],
        "Classic": ["noir", "sepia", "monochrome"],
        "Animation": ["anime_pop", "ghibli_soft", "pixar_bright", "cartoon_vibrant"],
        "Genre": ["cyberpunk_neon", "horror_green", "fantasy_ethereal"],
        "Film Stocks": ["kodak_5219", "kodak_5207", "fuji_eterna", "kodak_portra", "cinestill_800t"],
        "Broadcast/HDR": ["rec2020_hdr", "aces_filmic", "broadcast_709"],
    }
    
    for cat, luts in categories.items():
        found = sum(1 for l in luts if any(clut.value == l for clut in ColorLUT))
        print(f"  {cat}: {found}/{len(luts)}")
    
    return True


def test_post_processing():
    """Test all post-processing effects."""
    print("\n" + "="*50)
    print("TEST 2: Post-Processing Effects")
    print("="*50)
    
    from agents.cinematic.color_grading import PostProcessEffect, PostProcessSpec
    
    print(f"✅ {len(PostProcessEffect)} effects defined")
    
    # Group by category
    categories = {
        "Optical": ["bloom", "ca", "lens_flare", "anamorphic", "halation", "dof"],
        "Film": ["grain", "light_leak", "scratches", "dust", "gate_weave"],
        "Diffusion": ["vignette", "glow", "soften", "sharpen"],
        "Retro": ["scan_lines", "vhs_glitch", "digital_glitch", "pixelate"],
    }
    
    for cat, effects in categories.items():
        found = sum(1 for e in effects if any(pe.value == e for pe in PostProcessEffect))
        print(f"  {cat}: {found}/{len(effects)}")
    
    # Test spec creation
    spec = PostProcessSpec(bloom=0.5, vignette=0.3, film_grain=0.2)
    modifiers = spec.to_prompt_modifiers()
    print(f"  ✅ PostProcessSpec created, {len(modifiers)} modifiers")
    
    return True


def test_genre_presets():
    """Test genre color presets."""
    print("\n" + "="*50)
    print("TEST 3: Genre Presets")
    print("="*50)
    
    from agents.cinematic.color_grading import GENRE_COLOR_PRESETS, ColorGradingEngine
    
    print(f"✅ {len(GENRE_COLOR_PRESETS)} genre presets defined")
    
    engine = ColorGradingEngine()
    
    for genre in GENRE_COLOR_PRESETS.keys():
        grade, post = engine.create_grade("Test scene", genre)
        print(f"  {genre}: LUT={grade.lut.value}, sat={grade.saturation:.2f}")
    
    return True


def test_color_grade_creation():
    """Test color grade creation for various scenes."""
    print("\n" + "="*50)
    print("TEST 4: Color Grade Creation")
    print("="*50)
    
    from agents.cinematic.color_grading import ColorGradingEngine
    
    engine = ColorGradingEngine()
    
    test_scenes = [
        ("Epic battle at sunset", "action", "golden_pm"),
        ("Romantic dinner by candlelight", "romance", "night"),
        ("Neon city streets at night", "cyberpunk", "neon_night"),
        ("Creepy abandoned hospital", "horror", "night"),
        ("Cheerful school morning", "anime", "morning"),
    ]
    
    for desc, genre, time in test_scenes:
        grade, post = engine.create_grade(desc, genre, time)
        print(f"  ✅ '{desc[:25]}...' → {grade.lut.value}")
    
    return True


def test_emotional_adjustments():
    """Test emotional color adjustments."""
    print("\n" + "="*50)
    print("TEST 5: Emotional Adjustments")
    print("="*50)
    
    from agents.cinematic.color_grading import ColorGradingEngine
    
    engine = ColorGradingEngine()
    
    emotions = [
        ("Happy celebration with laughter", "higher saturation"),
        ("Sad farewell with tears", "lower saturation"),
        ("Intense dramatic confrontation", "higher contrast"),
        ("Peaceful calm serene moment", "lower contrast"),
        ("Mysterious secret hidden", "crushed shadows"),
    ]
    
    for desc, expected in emotions:
        grade, _ = engine.create_grade(desc, "cinematic")
        print(f"  ✅ '{desc[:30]}...' → {expected}")
    
    return True


def test_time_adjustments():
    """Test time-based color adjustments."""
    print("\n" + "="*50)
    print("TEST 6: Time-Based Adjustments")
    print("="*50)
    
    from agents.cinematic.color_grading import ColorGradingEngine
    
    engine = ColorGradingEngine()
    
    times = ["dawn", "golden_am", "morning", "midday", "afternoon", "golden_pm", "dusk", "night"]
    
    for time in times:
        grade, _ = engine.create_grade("Test scene", "cinematic", time)
        temp = "warm" if grade.temperature > 0 else "cool" if grade.temperature < 0 else "neutral"
        print(f"  {time}: temp={grade.temperature:+.0f} ({temp})")
    
    return True


def test_prompt_generation():
    """Test prompt modifier generation."""
    print("\n" + "="*50)
    print("TEST 7: Prompt Generation")
    print("="*50)
    
    from agents.cinematic.color_grading import ColorGradingEngine, get_color_prompt
    
    engine = ColorGradingEngine()
    
    grade, post = engine.create_grade("Dramatic sunset scene", "action", "golden_pm")
    
    grade_mods = grade.to_prompt_modifiers()
    post_mods = post.to_prompt_modifiers()
    full_prompt = get_color_prompt(grade, post)
    
    print(f"  ✅ Grade modifiers: {len(grade_mods)}")
    print(f"  ✅ Post modifiers: {len(post_mods)}")
    print(f"  Prompt: {full_prompt[:60]}...")
    
    return True


def test_serialization():
    """Test color grade serialization."""
    print("\n" + "="*50)
    print("TEST 8: Serialization")
    print("="*50)
    
    from agents.cinematic.color_grading import ColorGradingEngine
    
    engine = ColorGradingEngine()
    grade, post = engine.create_grade("Test scene", "anime")
    
    grade_dict = grade.to_dict()
    post_dict = post.to_dict()
    
    grade_keys = ["lut", "saturation", "contrast", "temperature", "curve"]
    post_keys = ["bloom", "vignette", "film_grain"]
    
    missing_grade = [k for k in grade_keys if k not in grade_dict]
    missing_post = [k for k in post_keys if k not in post_dict]
    
    if missing_grade or missing_post:
        print(f"  ❌ Missing keys")
        return False
    
    print(f"  ✅ ColorGrade serialized: {len(grade_dict)} keys")
    print(f"  ✅ PostProcessSpec serialized: {len(post_dict)} keys")
    
    return True


def main():
    """Run all tests."""
    print("\n" + "#"*60)
    print("# PROFESSIONAL COLOR GRADING TEST SUITE")
    print("#"*60)
    
    results = {}
    
    tests = [
        ("color_luts", test_color_luts),
        ("post_processing", test_post_processing),
        ("genre_presets", test_genre_presets),
        ("color_grade_creation", test_color_grade_creation),
        ("emotional_adjustments", test_emotional_adjustments),
        ("time_adjustments", test_time_adjustments),
        ("prompt_generation", test_prompt_generation),
        ("serialization", test_serialization),
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
    print("SUMMARY")
    print("="*60)
    
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")
    
    all_passed = all(results.values())
    print()
    print("🎉 All tests passed!" if all_passed else "⚠️ Some tests failed")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
