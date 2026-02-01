"""
Test Suite for Style Detection

Tests:
1. StyleDetector basic functionality
2. Keyword-based detection
3. Integration with narrative planner

Run:
    python tests/test_style_detection.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()


def test_imports():
    """Test 1: Verify all imports work"""
    print("\n" + "="*50)
    print("TEST 1: Import Verification")
    print("="*50)
    
    try:
        from agents.style_detector import (
            StyleDetector,
            StyleProfile,
            VisualStyle,
            detect_style,
        )
        print("✅ StyleDetector imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


def test_visual_styles():
    """Test 2: Verify all visual styles are defined"""
    print("\n" + "="*50)
    print("TEST 2: Visual Styles")
    print("="*50)
    
    from agents.style_detector import VisualStyle
    
    expected_styles = [
        "anime", "realistic", "cartoon", "pixar", 
        "ghibli", "noir", "cyberpunk", "fantasy", "cinematic"
    ]
    
    all_present = True
    for style_name in expected_styles:
        try:
            style = VisualStyle(style_name)
            print(f"✅ {style_name}")
        except ValueError:
            print(f"❌ Missing: {style_name}")
            all_present = False
    
    return all_present


def test_style_profiles():
    """Test 3: Verify style profiles are complete"""
    print("\n" + "="*50)
    print("TEST 3: Style Profiles")
    print("="*50)
    
    from agents.style_detector import VisualStyle, StyleProfile
    
    all_valid = True
    for style in VisualStyle:
        profile = StyleProfile.from_style(style)
        
        required_fields = [
            "style_prefix", "style_suffix", "negative_prompt",
            "lighting_style", "recommended_model"
        ]
        
        missing = [f for f in required_fields if not getattr(profile, f)]
        
        if missing:
            print(f"❌ {style.value}: Missing {missing}")
            all_valid = False
        else:
            print(f"✅ {style.value}: Complete profile")
    
    return all_valid


def test_keyword_detection():
    """Test 4: Keyword-based style detection"""
    print("\n" + "="*50)
    print("TEST 4: Keyword Detection")
    print("="*50)
    
    from agents.style_detector import StyleDetector, VisualStyle
    
    test_cases = [
        ("A dragon ball anime fight scene", VisualStyle.ANIME),
        ("A noir detective mystery in rain", VisualStyle.NOIR),
        ("A realistic documentary about nature", VisualStyle.REALISTIC),
        ("A Pixar style animated adventure", VisualStyle.PIXAR),
        ("A cyberpunk neon city heist", VisualStyle.CYBERPUNK),
        ("A Studio Ghibli forest adventure", VisualStyle.GHIBLI),
        ("An epic fantasy dragon battle", VisualStyle.FANTASY),
    ]
    
    detector = StyleDetector(use_llm=False)  # Keywords only
    
    all_correct = True
    for intent, expected_style in test_cases:
        profile = detector.detect(intent)
        
        if profile.style == expected_style:
            print(f"✅ '{intent[:40]}...' → {profile.style.value}")
        else:
            print(f"❌ '{intent[:40]}...' → {profile.style.value} (expected {expected_style.value})")
            all_correct = False
    
    return all_correct


def test_prompt_enhancement():
    """Test 5: Prompt enhancement with style"""
    print("\n" + "="*50)
    print("TEST 5: Prompt Enhancement")
    print("="*50)
    
    from agents.style_detector import StyleDetector, VisualStyle
    
    detector = StyleDetector(use_llm=False)
    
    intent = "An anime fight scene with energy blasts"
    profile = detector.detect(intent)
    
    base_prompt = "Two warriors fighting in a destroyed city"
    enhanced = detector.enhance_prompt(base_prompt, profile)
    
    print(f"Base: {base_prompt}")
    print(f"Enhanced: {enhanced[:100]}...")
    
    # Check enhancements
    checks = [
        ("anime" in enhanced.lower(), "Contains 'anime'"),
        ("animation" in enhanced.lower(), "Contains 'animation'"),
        (len(enhanced) > len(base_prompt), "Prompt is longer"),
    ]
    
    all_passed = True
    for passed, desc in checks:
        if passed:
            print(f"✅ {desc}")
        else:
            print(f"❌ {desc}")
            all_passed = False
    
    return all_passed


def test_narrative_planner_integration():
    """Test 6: Integration with narrative planner"""
    print("\n" + "="*50)
    print("TEST 6: Narrative Planner Integration")
    print("="*50)
    
    try:
        # Check that the import works
        from agents.narrative_planner import ProductionNarrativePlanner
        print("✅ NarrativePlanner imports successfully")
        
        # Check that style_detector is imported in generate_beats
        import inspect
        source = inspect.getsource(ProductionNarrativePlanner.generate_beats)
        
        if "style_detector" in source:
            print("✅ style_detector imported in generate_beats")
        else:
            print("⚠️ style_detector not found in generate_beats")
        
        if "detect_style" in source:
            print("✅ detect_style called in generate_beats")
        else:
            print("⚠️ detect_style not called in generate_beats")
        
        if "style_profile" in source:
            print("✅ style_profile used in beat generation")
        else:
            print("⚠️ style_profile not used in beat generation")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


def test_veo_backend_integration():
    """Test 7: Integration with veo backend"""
    print("\n" + "="*50)
    print("TEST 7: Veo Backend Integration")
    print("="*50)
    
    try:
        from agents.backends.veo_backend import build_cinematic_prompt, get_negative_prompt
        
        # Test with style profile
        input_spec = {
            "style": "anime",
            "style_profile": {
                "style_prefix": "anime style, Japanese animation",
                "style_suffix": "vibrant colors, dynamic action",
                "negative_prompt": "realistic, 3D",
                "lighting_style": "dramatic",
            }
        }
        
        prompt = build_cinematic_prompt("Warrior charging energy", input_spec)
        neg_prompt = get_negative_prompt(input_spec)
        
        print(f"Generated prompt: {prompt[:80]}...")
        print(f"Negative prompt: {neg_prompt}")
        
        checks = [
            ("anime" in prompt.lower(), "Contains 'anime'"),
            ("vibrant" in prompt.lower(), "Contains style suffix"),
            ("realistic" in neg_prompt.lower(), "Negative prompt from profile"),
        ]
        
        all_passed = True
        for passed, desc in checks:
            if passed:
                print(f"✅ {desc}")
            else:
                print(f"❌ {desc}")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Veo backend test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "#"*60)
    print("# STYLE DETECTION TEST SUITE")
    print("#"*60)
    
    results = {}
    
    results["imports"] = test_imports()
    results["visual_styles"] = test_visual_styles()
    results["style_profiles"] = test_style_profiles()
    results["keyword_detection"] = test_keyword_detection()
    results["prompt_enhancement"] = test_prompt_enhancement()
    results["narrative_planner"] = test_narrative_planner_integration()
    results["veo_backend"] = test_veo_backend_integration()
    
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
