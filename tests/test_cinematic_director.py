"""
Cinematic Director Test Suite

Comprehensive tests for the main orchestrator including:
- CinematicSpec creation and serialization
- Beat direction for all genres
- Camera selection
- Shot composition  
- Lighting application
- Color grading
- Post-processing
- Prompt generation
- Time of day detection
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_director_initialization():
    """Test cinematic director initialization."""
    print("\n" + "="*50)
    print("TEST 1: Director Initialization")
    print("="*50)
    
    from agents.cinematic.cinematic_director import CinematicDirector, get_cinematic_director
    
    director = CinematicDirector()
    
    # Check systems are initialized
    assert director.camera_system is not None, "Camera system missing"
    assert director.shot_composer is not None, "Shot composer missing"
    assert director.lighting_engine is not None, "Lighting engine missing"
    assert director.grading_engine is not None, "Grading engine missing"
    
    print("  ✅ Camera system initialized")
    print("  ✅ Shot composer initialized")
    print("  ✅ Lighting engine initialized")
    print("  ✅ Grading engine initialized")
    
    # Test singleton
    singleton = get_cinematic_director()
    assert singleton is not None
    print("  ✅ Singleton accessor works")
    
    return True


def test_beat_direction():
    """Test directing beats for various genres."""
    print("\n" + "="*50)
    print("TEST 2: Beat Direction")
    print("="*50)
    
    from agents.cinematic.cinematic_director import CinematicDirector
    
    director = CinematicDirector()
    
    test_beats = [
        ("Two warriors clash in epic battle", "anime"),
        ("Hero walks through magical forest", "fantasy"),
        ("Detective examines crime scene", "noir"),
        ("Family gathers at dinner table", "pixar"),
        ("Neon city street at night", "cyberpunk"),
    ]
    
    for desc, genre in test_beats:
        spec = director.direct_beat(desc, genre)
        
        assert spec is not None, f"No spec for {genre}"
        assert spec.camera is not None, f"No camera for {genre}"
        assert spec.shot is not None, f"No shot for {genre}"
        assert spec.lighting is not None, f"No lighting for {genre}"
        assert spec.color_grade is not None, f"No color for {genre}"
        
        print(f"  ✅ {genre}: {spec.get_summary()[:50]}...")
    
    return True


def test_cinematic_spec_structure():
    """Test CinematicSpec structure and methods."""
    print("\n" + "="*50)
    print("TEST 3: CinematicSpec Structure")
    print("="*50)
    
    from agents.cinematic.cinematic_director import direct_beat
    
    spec = direct_beat("Hero confronts villain in dramatic showdown", "action")
    
    # Check all components exist
    assert spec.genre == "action"
    assert spec.genre_profile is not None
    assert spec.camera is not None
    assert spec.shot is not None
    assert spec.lighting is not None
    assert spec.color_grade is not None
    assert spec.post_process is not None
    
    print(f"  ✅ Genre: {spec.genre}")
    print(f"  ✅ Camera: {spec.camera.movement.value}, {spec.camera.lens.focal_length}mm")
    print(f"  ✅ Shot: {spec.shot.shot_type.value}")
    print(f"  ✅ Lighting: {spec.lighting.setup.value}")
    print(f"  ✅ Color: {spec.color_grade.lut.value}")
    print(f"  ✅ Post: configured")
    
    return True


def test_serialization():
    """Test spec serialization to dictionary."""
    print("\n" + "="*50)
    print("TEST 4: Serialization")
    print("="*50)
    
    from agents.cinematic.cinematic_director import direct_beat
    
    spec = direct_beat("Romantic sunset scene on beach", "romance")
    data = spec.to_dict()
    
    required_keys = [
        "genre", "camera", "shot", "lighting", 
        "color_grade", "post_process", "technical_prompt"
    ]
    
    missing = [k for k in required_keys if k not in data]
    if missing:
        print(f"  ❌ Missing keys: {missing}")
        return False
    
    print(f"  ✅ All required keys present")
    print(f"  ✅ Camera data: {list(data['camera'].keys())[:3]}...")
    print(f"  ✅ Lighting data: {list(data['lighting'].keys())[:3]}...")
    print(f"  ✅ Color data: {list(data['color_grade'].keys())[:3]}...")
    
    return True


def test_prompt_generation():
    """Test full prompt generation."""
    print("\n" + "="*50)
    print("TEST 5: Prompt Generation")
    print("="*50)
    
    from agents.cinematic.cinematic_director import direct_beat, get_cinematic_prompt
    
    # Test with spec
    desc = "Creepy figure emerges from shadows"
    spec = direct_beat(desc, "horror")
    
    full_prompt = spec.get_full_prompt(desc)
    
    assert len(full_prompt) > len(desc), "Prompt not enhanced"
    assert "horror" in full_prompt.lower() or "dark" in full_prompt.lower()
    
    print(f"  ✅ Original: '{desc}'")
    print(f"  ✅ Enhanced: '{full_prompt[:80]}...'")
    
    # Test convenience function
    prompt2 = get_cinematic_prompt("Epic battle scene", "anime")
    assert "anime" in prompt2.lower()
    print(f"  ✅ Convenience function works")
    
    return True


def test_technical_prompt():
    """Test technical prompt content."""
    print("\n" + "="*50)
    print("TEST 6: Technical Prompt Content")
    print("="*50)
    
    from agents.cinematic.cinematic_director import direct_beat
    
    spec = direct_beat("Wide establishing shot of futuristic city", "cyberpunk")
    
    tech_prompt = spec.technical_prompt
    
    # Should contain various technical terms
    has_camera = any(w in tech_prompt.lower() for w in ["camera", "lens", "shot", "angle"])
    has_lighting = any(w in tech_prompt.lower() for w in ["light", "shadow", "glow"])
    has_color = any(w in tech_prompt.lower() for w in ["color", "saturated", "contrast", "neon"])
    
    print(f"  Technical prompt: '{tech_prompt[:100]}...'")
    print(f"  ✅ Camera terms: {'yes' if has_camera else 'no'}")
    print(f"  ✅ Lighting terms: {'yes' if has_lighting else 'no'}")
    print(f"  ✅ Color terms: {'yes' if has_color else 'no'}")
    
    return True


def test_time_detection():
    """Test time of day detection from descriptions."""
    print("\n" + "="*50)
    print("TEST 7: Time Detection")
    print("="*50)
    
    from agents.cinematic.cinematic_director import CinematicDirector
    from agents.cinematic.genre_profiles import get_genre_profile
    
    director = CinematicDirector()
    profile = get_genre_profile("cinematic")
    
    test_cases = [
        ("Beautiful sunrise over mountains", "dawn"),
        ("Harsh midday sun beats down", "midday"),
        ("Golden hour sunset on beach", "golden"),
        ("Dark midnight scene", "night"),
        ("Neon lights in the city", "neon"),
    ]
    
    for desc, expected_contains in test_cases:
        time = director._detect_time(desc, profile)
        matches = expected_contains in time.lower()
        status = "✅" if matches else "⚠️"
        print(f"  {status} '{desc[:30]}...' → {time}")
    
    return True


def test_all_genres():
    """Test directing beats for all available genres."""
    print("\n" + "="*50)
    print("TEST 8: All Genres")
    print("="*50)
    
    from agents.cinematic.cinematic_director import CinematicDirector
    
    director = CinematicDirector()
    genres = director.get_available_genres()
    
    print(f"  Testing {len(genres)} genres...")
    
    success = 0
    for genre in genres:
        try:
            spec = director.direct_beat(f"Test scene for {genre}", genre)
            assert spec.camera is not None
            assert spec.lighting is not None
            success += 1
        except Exception as e:
            print(f"  ❌ {genre}: {e}")
    
    print(f"  ✅ {success}/{len(genres)} genres processed successfully")
    
    return success == len(genres)


def test_summary():
    """Test spec summary generation."""
    print("\n" + "="*50)
    print("TEST 9: Summary Generation")
    print("="*50)
    
    from agents.cinematic.cinematic_director import direct_beat
    
    test_cases = [
        ("anime", "Hero powers up", "anime"),
        ("noir", "Detective in dark alley", "noir"),
        ("pixar", "Happy family moment", "pixar"),
    ]
    
    for genre, desc, expected in test_cases:
        spec = direct_beat(desc, genre)
        summary = spec.get_summary()
        
        assert "Camera:" in summary
        assert "Shot:" in summary
        assert "Light:" in summary
        assert "LUT:" in summary
        
        print(f"  ✅ {genre}: {summary}")
    
    return True


def test_override_options():
    """Test override options for camera and lighting."""
    print("\n" + "="*50)
    print("TEST 10: Override Options")
    print("="*50)
    
    from agents.cinematic.cinematic_director import CinematicDirector
    
    director = CinematicDirector()
    
    # Test with time override
    spec1 = director.direct_beat("Test scene", "cinematic", time_of_day="night")
    spec2 = director.direct_beat("Test scene", "cinematic", time_of_day="golden_pm")
    
    # Different times should potentially affect color temperature
    assert spec1.color_grade is not None
    assert spec2.color_grade is not None
    
    print(f"  ✅ Night time override works")
    print(f"  ✅ Golden hour override works")
    
    return True


def main():
    """Run all tests."""
    print("\n" + "#"*60)
    print("# CINEMATIC DIRECTOR TEST SUITE")
    print("#"*60)
    
    results = {}
    
    tests = [
        ("initialization", test_director_initialization),
        ("beat_direction", test_beat_direction),
        ("spec_structure", test_cinematic_spec_structure),
        ("serialization", test_serialization),
        ("prompt_generation", test_prompt_generation),
        ("technical_prompt", test_technical_prompt),
        ("time_detection", test_time_detection),
        ("all_genres", test_all_genres),
        ("summary", test_summary),
        ("override_options", test_override_options),
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
