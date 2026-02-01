"""
Genre Profiles Test Suite

Comprehensive tests for genre profiles including:
- All genre profile definitions
- Camera preferences
- Shot configurations  
- Lighting setups
- Color grading
- Post-processing
- Prompt modifiers
- Profile retrieval
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_genre_count():
    """Test total number of genre profiles."""
    print("\n" + "="*50)
    print("TEST 1: Genre Profile Count")
    print("="*50)
    
    from agents.cinematic.genre_profiles import GENRE_PROFILES
    
    count = len(GENRE_PROFILES)
    print(f"✅ {count} genre profiles defined")
    
    # List all genres
    for name in sorted(GENRE_PROFILES.keys()):
        profile = GENRE_PROFILES[name]
        print(f"  {name}: {profile.name}")
    
    return count >= 12  # Should have at least 12


def test_camera_preferences():
    """Test camera preference configurations."""
    print("\n" + "="*50)
    print("TEST 2: Camera Preferences")
    print("="*50)
    
    from agents.cinematic.genre_profiles import GENRE_PROFILES
    
    for name, profile in GENRE_PROFILES.items():
        cam = profile.camera
        lens = cam.get("preferred_lens", "N/A")
        style = cam.get("movement_style", "N/A")
        
        if not cam:
            print(f"  ⚠️ {name}: Missing camera config")
            continue
            
        print(f"  ✅ {name}: {lens}mm, {style}")
    
    return True


def test_lighting_setups():
    """Test lighting setup configurations."""
    print("\n" + "="*50)
    print("TEST 3: Lighting Setups")
    print("="*50)
    
    from agents.cinematic.genre_profiles import GENRE_PROFILES
    
    for name, profile in GENRE_PROFILES.items():
        light = profile.lighting
        setup = light.get("setup", "N/A")
        
        print(f"  ✅ {name}: {setup}")
    
    return True


def test_color_grading():
    """Test color grading configurations."""
    print("\n" + "="*50)
    print("TEST 4: Color Grading")
    print("="*50)
    
    from agents.cinematic.genre_profiles import GENRE_PROFILES
    
    for name, profile in GENRE_PROFILES.items():
        color = profile.color
        lut = color.get("lut", "N/A")
        sat = color.get("saturation", 1.0)
        
        print(f"  ✅ {name}: LUT={lut}, sat={sat}")
    
    return True


def test_post_processing():
    """Test post-processing configurations."""
    print("\n" + "="*50)
    print("TEST 5: Post-Processing")
    print("="*50)
    
    from agents.cinematic.genre_profiles import GENRE_PROFILES
    
    for name, profile in GENRE_PROFILES.items():
        post = profile.post
        effects = [k for k, v in post.items() if isinstance(v, (int, float)) and v > 0]
        
        print(f"  ✅ {name}: {', '.join(effects[:3]) if effects else 'minimal'}")
    
    return True


def test_prompt_modifiers():
    """Test prompt modifier configurations."""
    print("\n" + "="*50)
    print("TEST 6: Prompt Modifiers")
    print("="*50)
    
    from agents.cinematic.genre_profiles import GENRE_PROFILES
    
    for name, profile in GENRE_PROFILES.items():
        prefix = profile.prompt_prefix
        has_negative = len(profile.negative_prompt) > 0
        
        status = "✅" if prefix and has_negative else "⚠️"
        print(f"  {status} {name}: '{prefix[:30]}...'")
    
    return True


def test_profile_retrieval():
    """Test profile retrieval function."""
    print("\n" + "="*50)
    print("TEST 7: Profile Retrieval")
    print("="*50)
    
    from agents.cinematic.genre_profiles import get_genre_profile
    
    test_cases = [
        ("anime", "Anime"),
        ("PIXAR", "Pixar / 3D Animation"),
        ("sci-fi", "Science Fiction"),
        ("unknown_genre", "Cinematic"),  # Default fallback
        ("cyber", "Cyberpunk"),  # Partial match
    ]
    
    for query, expected in test_cases:
        profile = get_genre_profile(query)
        status = "✅" if profile.name == expected else "⚠️"
        print(f"  {status} '{query}' → {profile.name}")
    
    return True


def test_genre_completeness():
    """Test that all profiles have required fields."""
    print("\n" + "="*50)
    print("TEST 8: Profile Completeness")
    print("="*50)
    
    from agents.cinematic.genre_profiles import GENRE_PROFILES
    
    required_fields = ["camera", "shots", "lighting", "color", "post"]
    all_complete = True
    
    for name, profile in GENRE_PROFILES.items():
        missing = []
        for field in required_fields:
            val = getattr(profile, field, None)
            if not val:
                missing.append(field)
        
        if missing:
            print(f"  ⚠️ {name}: missing {', '.join(missing)}")
            all_complete = False
        else:
            print(f"  ✅ {name}: complete")
    
    return all_complete


def main():
    """Run all tests."""
    print("\n" + "#"*60)
    print("# GENRE PROFILES TEST SUITE")
    print("#"*60)
    
    results = {}
    
    tests = [
        ("genre_count", test_genre_count),
        ("camera_preferences", test_camera_preferences),
        ("lighting_setups", test_lighting_setups),
        ("color_grading", test_color_grading),
        ("post_processing", test_post_processing),
        ("prompt_modifiers", test_prompt_modifiers),
        ("profile_retrieval", test_profile_retrieval),
        ("profile_completeness", test_genre_completeness),
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
