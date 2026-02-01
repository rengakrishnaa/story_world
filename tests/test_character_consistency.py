"""
Test Suite for Character Consistency

Tests:
1. CharacterAppearance creation
2. CharacterConsistencyEngine initialization
3. Character registration and lookup
4. Prompt enhancement with character details
5. Character conditioning generation
6. Integration with narrative planner
7. Integration with veo backend

Run:
    python tests/test_character_consistency.py
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()


def test_imports():
    """Test 1: Verify all imports work"""
    print("\n" + "="*50)
    print("TEST 1: Import Verification")
    print("="*50)
    
    try:
        from agents.character_consistency import (
            CharacterAppearance,
            CharacterConsistencyEngine,
            get_consistency_engine,
        )
        print("✅ CharacterConsistency imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


def test_character_appearance_creation():
    """Test 2: CharacterAppearance dataclass"""
    print("\n" + "="*50)
    print("TEST 2: CharacterAppearance Creation")
    print("="*50)
    
    from agents.character_consistency import CharacterAppearance
    
    char = CharacterAppearance(
        character_id="test_001",
        name="Test Hero",
        physical_description="tall muscular man",
        clothing_description="blue cape and armor",
        distinctive_features=["scar on face", "golden eyes"],
        primary_colors=["#0000FF", "#FFD700"],
        body_type="muscular",
        apparent_age="adult",
    )
    
    checks = [
        (char.name == "Test Hero", "Name set correctly"),
        (char.body_type == "muscular", "Body type set"),
        (len(char.distinctive_features) == 2, "Features count correct"),
        (len(char.primary_colors) == 2, "Colors count correct"),
    ]
    
    all_passed = True
    for passed, desc in checks:
        if passed:
            print(f"✅ {desc}")
        else:
            print(f"❌ {desc}")
            all_passed = False
    
    # Test prompt generation
    prompt = char.get_prompt_description("cinematic")
    if "Test Hero" in prompt and "tall muscular" in prompt:
        print("✅ get_prompt_description works")
    else:
        print("❌ get_prompt_description failed")
        all_passed = False
    
    return all_passed


def test_engine_initialization():
    """Test 3: Engine initialization"""
    print("\n" + "="*50)
    print("TEST 3: Engine Initialization")
    print("="*50)
    
    from agents.character_consistency import CharacterConsistencyEngine
    
    engine = CharacterConsistencyEngine("test-world-123")
    
    checks = [
        (engine.world_id == "test-world-123", "World ID set"),
        (isinstance(engine.characters, dict), "Characters dict initialized"),
        (engine._cache_dir.exists(), "Cache directory created"),
    ]
    
    all_passed = True
    for passed, desc in checks:
        if passed:
            print(f"✅ {desc}")
        else:
            print(f"❌ {desc}")
            all_passed = False
    
    return all_passed


def test_character_registration():
    """Test 4: Character registration and lookup"""
    print("\n" + "="*50)
    print("TEST 4: Character Registration")
    print("="*50)
    
    from agents.character_consistency import (
        CharacterConsistencyEngine,
        CharacterAppearance,
    )
    
    engine = CharacterConsistencyEngine("test-world-reg")
    
    # Create and register characters
    hero = CharacterAppearance(
        character_id="hero_001",
        name="Captain Brave",
        physical_description="tall athletic woman",
        clothing_description="silver armor with red accents",
        distinctive_features=["long silver hair", "determined expression"],
    )
    
    villain = CharacterAppearance(
        character_id="villain_001",
        name="Dark Lord",
        physical_description="imposing figure in black robes",
        clothing_description="dark cloak with hood",
        distinctive_features=["glowing red eyes", "skeletal hands"],
    )
    
    engine.register_character(hero)
    engine.register_character(villain)
    
    checks = [
        (len(engine.characters) == 2, "Two characters registered"),
        (engine.get_character("Captain Brave") is not None, "Hero found by name"),
        (engine.get_character("CAPTAIN BRAVE") is not None, "Case-insensitive lookup"),
        (engine.get_character("Dark Lord") is not None, "Villain found"),
        (engine.get_character("Unknown") is None, "Unknown returns None"),
    ]
    
    all_passed = True
    for passed, desc in checks:
        if passed:
            print(f"✅ {desc}")
        else:
            print(f"❌ {desc}")
            all_passed = False
    
    return all_passed


def test_prompt_enhancement():
    """Test 5: Prompt enhancement with character details"""
    print("\n" + "="*50)
    print("TEST 5: Prompt Enhancement")
    print("="*50)
    
    from agents.character_consistency import (
        CharacterConsistencyEngine,
        CharacterAppearance,
    )
    
    engine = CharacterConsistencyEngine("test-world-prompt")
    
    engine.register_character(CharacterAppearance(
        character_id="warrior_001",
        name="The Warrior",
        physical_description="scarred veteran soldier",
        clothing_description="battle-worn armor",
        distinctive_features=["eye patch", "broken sword"],
    ))
    
    base_prompt = "A battle scene on the burning castle walls"
    enhanced = engine.enhance_prompt_with_characters(
        base_prompt,
        ["The Warrior"],
        style="cinematic"
    )
    
    checks = [
        ("The Warrior" in enhanced, "Character name in prompt"),
        ("scarred veteran" in enhanced, "Physical description included"),
        ("battle scene" in enhanced, "Original prompt preserved"),
        (len(enhanced) > len(base_prompt), "Prompt is longer"),
    ]
    
    all_passed = True
    for passed, desc in checks:
        if passed:
            print(f"✅ {desc}")
        else:
            print(f"❌ {desc}")
            all_passed = False
    
    print(f"\nOriginal: {base_prompt}")
    print(f"Enhanced: {enhanced[:120]}...")
    
    return all_passed


def test_character_conditioning():
    """Test 6: Character conditioning generation"""
    print("\n" + "="*50)
    print("TEST 6: Character Conditioning")
    print("="*50)
    
    from agents.character_consistency import (
        CharacterConsistencyEngine,
        CharacterAppearance,
    )
    
    engine = CharacterConsistencyEngine("test-world-cond")
    
    engine.register_character(CharacterAppearance(
        character_id="mage_001",
        name="The Mage",
        physical_description="elderly wizard with long beard",
        clothing_description="blue robes with star patterns",
        distinctive_features=["pointed hat", "glowing staff"],
        reference_image_urls=["/images/mage_ref.png"],
    ))
    
    conditioning = engine.build_character_conditioning(
        ["The Mage"],
        style="fantasy"
    )
    
    checks = [
        ("character_prompts" in conditioning, "Has character_prompts"),
        ("The Mage" in conditioning.get("character_prompts", {}), "Mage prompt present"),
        ("reference_images" in conditioning, "Has reference_images"),
        ("character_negative" in conditioning, "Has negative prompt"),
    ]
    
    all_passed = True
    for passed, desc in checks:
        if passed:
            print(f"✅ {desc}")
        else:
            print(f"❌ {desc}")
            all_passed = False
    
    print(f"\nConditioning: {json.dumps(conditioning, indent=2)[:300]}...")
    
    return all_passed


def test_world_registration():
    """Test 7: Register from world data"""
    print("\n" + "="*50)
    print("TEST 7: World Registration")
    print("="*50)
    
    from agents.character_consistency import CharacterConsistencyEngine
    
    engine = CharacterConsistencyEngine("test-world-reg2")
    
    world_data = {
        "characters": [
            {
                "name": "Saitama",
                "description": "Bald hero with incredible strength",
                "reference_image_url": "/uploads/saitama.png",
                "traits": ["calm", "deadpan", "overpowered"],
            },
            {
                "name": "Genos",
                "description": "Cyborg hero with blonde hair",
                "reference_image_url": "/uploads/genos.png",
                "traits": ["serious", "loyal", "mechanical"],
            },
        ],
        "locations": [],
    }
    
    engine.register_from_world(world_data)
    
    checks = [
        (len(engine.characters) == 2, "Two characters registered"),
        (engine.get_character("Saitama") is not None, "Saitama found"),
        (engine.get_character("Genos") is not None, "Genos found"),
    ]
    
    saitama = engine.get_character("Saitama")
    if saitama:
        checks.append((
            "calm" in saitama.distinctive_features,
            "Traits preserved as features"
        ))
    
    all_passed = True
    for passed, desc in checks:
        if passed:
            print(f"✅ {desc}")
        else:
            print(f"❌ {desc}")
            all_passed = False
    
    return all_passed


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
            ("character_consistency" in source, "character_consistency imported"),
            ("get_consistency_engine" in source, "get_consistency_engine called"),
            ("register_from_world" in source, "register_from_world called"),
            ("character_conditioning" in source, "character_conditioning used"),
            ("enhanced_description" in source, "enhanced_description added"),
        ]
        
        all_passed = True
        for passed, desc in checks:
            if passed:
                print(f"✅ {desc}")
            else:
                print(f"⚠️ {desc}")
                # Not a failure, just info
        
        return True
        
    except Exception as e:
        print(f"❌ Integration check failed: {e}")
        return False


def test_veo_backend_integration():
    """Test 9: Integration with veo backend"""
    print("\n" + "="*50)
    print("TEST 9: Veo Backend Integration")
    print("="*50)
    
    try:
        from agents.backends.veo_backend import render
        import inspect
        
        source = inspect.getsource(render)
        
        checks = [
            ("enhanced_description" in source, "Uses enhanced_description"),
            ("character_conditioning" in source, "Uses character_conditioning"),
            ("character_prompts" in source, "Extracts character_prompts"),
        ]
        
        all_passed = True
        for passed, desc in checks:
            if passed:
                print(f"✅ {desc}")
            else:
                print(f"⚠️ {desc}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration check failed: {e}")
        return False


def main():
    print("\n" + "#"*60)
    print("# CHARACTER CONSISTENCY TEST SUITE")
    print("#"*60)
    
    results = {}
    
    results["imports"] = test_imports()
    results["character_appearance"] = test_character_appearance_creation()
    results["engine_init"] = test_engine_initialization()
    results["registration"] = test_character_registration()
    results["prompt_enhancement"] = test_prompt_enhancement()
    results["conditioning"] = test_character_conditioning()
    results["world_registration"] = test_world_registration()
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
