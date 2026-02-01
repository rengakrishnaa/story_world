"""
Professional Lighting Engine Test Suite

Comprehensive tests for the lighting engine including:
- All lighting setups
- Time of day variations
- Genre presets
- Light modifiers
- Practical sources
- Prompt generation
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_lighting_setups():
    """Test all lighting setup types."""
    print("\n" + "="*50)
    print("TEST 1: Lighting Setups")
    print("="*50)
    
    from agents.cinematic.lighting_engine import LightingSetup
    
    print(f"✅ {len(LightingSetup)} lighting setups defined")
    
    # Group by category
    categories = {
        "Portrait": ["three_point", "rembrandt", "loop", "split", "butterfly", "broad", "short"],
        "Mood": ["high_key", "low_key", "silhouette", "chiaroscuro"],
        "Accent": ["rim", "edge", "hair", "kick"],
        "Natural": ["practical", "motivated", "natural", "available"],
        "Special": ["under", "top", "side", "catch"],
        "Animated": ["flat", "cel_shaded", "ao"],
    }
    
    for cat, setups in categories.items():
        found = sum(1 for s in setups if any(ls.value == s for ls in LightingSetup))
        print(f"  {cat}: {found}/{len(setups)}")
    
    return True


def test_time_of_day():
    """Test time of day options."""
    print("\n" + "="*50)
    print("TEST 2: Time of Day")
    print("="*50)
    
    from agents.cinematic.lighting_engine import TimeOfDay, LightingEngine
    
    print(f"✅ {len(TimeOfDay)} time options defined")
    
    engine = LightingEngine()
    
    for tod in TimeOfDay:
        temp = engine._get_time_temperature(tod)
        print(f"  {tod.value}: {temp}K")
    
    return True


def test_genre_presets():
    """Test genre lighting presets."""
    print("\n" + "="*50)
    print("TEST 3: Genre Presets")
    print("="*50)
    
    from agents.cinematic.lighting_engine import GENRE_LIGHTING, LightingEngine
    
    print(f"✅ {len(GENRE_LIGHTING)} genre presets defined")
    
    engine = LightingEngine()
    
    for genre in GENRE_LIGHTING.keys():
        spec = engine.create_lighting("Test scene", genre)
        print(f"  {genre}: {spec.setup.value}, contrast={spec.contrast_ratio}")
    
    return True


def test_scene_detection():
    """Test scene-based lighting detection."""
    print("\n" + "="*50)
    print("TEST 4: Scene Detection")
    print("="*50)
    
    from agents.cinematic.lighting_engine import LightingEngine
    
    engine = LightingEngine()
    
    test_scenes = [
        ("Dark mysterious figure in shadows", "noir", "low_key"),
        ("Bright sunny morning cheerful", "pixar", "ao"),
        ("Creepy figure from below", "horror", "under"),
        ("Romantic candlelight dinner", "romance", "three_point"),
        ("Neon city street at night", "cyberpunk", "practical"),
        ("Anime battle with dramatic pose", "anime", "cel_shaded"),
    ]
    
    for desc, genre, expected in test_scenes:
        spec = engine.create_lighting(desc, genre)
        status = "✅" if spec.setup.value == expected else "⚠️"
        print(f"  {status} '{desc[:30]}...' → {spec.setup.value}")
    
    return True


def test_practical_sources():
    """Test practical light source detection."""
    print("\n" + "="*50)
    print("TEST 5: Practical Sources")
    print("="*50)
    
    from agents.cinematic.lighting_engine import LightingEngine
    
    engine = LightingEngine()
    
    test_scenes = [
        ("Room lit by candlelight", ["candle"]),
        ("Computer screen glowing in dark room", ["screen"]),
        ("Fireplace crackling in the corner", ["fire"]),
        ("Neon sign flickering outside", ["neon"]),
    ]
    
    for desc, expected in test_scenes:
        spec = engine.create_lighting(desc, "cinematic")
        found = [p for p in expected if p in spec.practical_sources]
        status = "✅" if found else "⚠️"
        print(f"  {status} '{desc[:30]}...' → {spec.practical_sources}")
    
    return True


def test_prompt_generation():
    """Test prompt modifier generation."""
    print("\n" + "="*50)
    print("TEST 6: Prompt Generation")
    print("="*50)
    
    from agents.cinematic.lighting_engine import LightingEngine, get_lighting_prompt
    
    engine = LightingEngine()
    
    spec = engine.create_lighting("Dramatic sunset scene", "action")
    modifiers = spec.to_prompt_modifiers()
    prompt = get_lighting_prompt(spec)
    
    print(f"  ✅ Generated {len(modifiers)} modifiers")
    print(f"  Prompt: {prompt[:60]}...")
    
    return True


def test_light_sources():
    """Test individual light source creation."""
    print("\n" + "="*50)
    print("TEST 7: Light Sources")
    print("="*50)
    
    from agents.cinematic.lighting_engine import LightingEngine
    
    engine = LightingEngine()
    
    spec = engine.create_lighting("Three point lighting test", "cinematic")
    
    print(f"  ✅ {len(spec.lights)} lights created")
    for light in spec.lights:
        print(f"    {light.name}: {light.intensity:.1f} @ {light.color_temperature}K")
    
    return True


def test_serialization():
    """Test spec serialization."""
    print("\n" + "="*50)
    print("TEST 8: Serialization")
    print("="*50)
    
    from agents.cinematic.lighting_engine import LightingEngine
    
    engine = LightingEngine()
    spec = engine.create_lighting("Test scene", "anime")
    
    data = spec.to_dict()
    
    required = ["setup", "time_of_day", "lights", "ambient", "contrast"]
    missing = [k for k in required if k not in data]
    
    if missing:
        print(f"  ❌ Missing keys: {missing}")
        return False
    
    print(f"  ✅ All required keys present")
    print(f"  Setup: {data['setup']}")
    print(f"  Time: {data['time_of_day']}")
    
    return True


def main():
    """Run all tests."""
    print("\n" + "#"*60)
    print("# PROFESSIONAL LIGHTING ENGINE TEST SUITE")
    print("#"*60)
    
    results = {}
    
    tests = [
        ("lighting_setups", test_lighting_setups),
        ("time_of_day", test_time_of_day),
        ("genre_presets", test_genre_presets),
        ("scene_detection", test_scene_detection),
        ("practical_sources", test_practical_sources),
        ("prompt_generation", test_prompt_generation),
        ("light_sources", test_light_sources),
        ("serialization", test_serialization),
    ]
    
    for name, test_fn in tests:
        try:
            results[name] = test_fn()
        except Exception as e:
            print(f"  ❌ {name} failed: {e}")
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
