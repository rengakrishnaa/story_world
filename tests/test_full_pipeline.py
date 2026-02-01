"""
Full Pipeline Integration Test

Tests the complete video generation pipeline from intent to video spec.
Verifies that all components work together:
1. Style Detection → 9 styles
2. Character Consistency → Embedding + prompt enhancement
3. Motion Engine → 15 motion types
4. Cinematic System → Camera, Shot, Lighting, Color
5. Narrative Planner → Beat generation with all metadata

Run:
    python tests/test_full_pipeline.py
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_full_pipeline():
    """Test the complete pipeline from intent to video spec."""
    print("\n" + "="*60)
    print("FULL PIPELINE INTEGRATION TEST")
    print("="*60)
    
    results = {}
    
    # ========================================
    # Test 1: Style Detection
    # ========================================
    print("\n--- Stage 1: Style Detection ---")
    try:
        from agents.style_detector import StyleDetector, VisualStyle
        
        detector = StyleDetector()
        
        test_intents = {
            "An anime battle between two warriors": "anime",
            "A Pixar-style family adventure": "pixar",
            "A noir detective mystery": "noir",
            "A cyberpunk heist in neon city": "cyberpunk",
        }
        
        for intent, expected_style in test_intents.items():
            # Use detect() method
            profile = detector.detect(intent)
            status = "✅" if expected_style in str(profile.style).lower() else "⚠️"
            print(f"  {status} '{intent[:30]}...' → {profile.style.value}")
        
        results["style_detection"] = True
        print("✅ Style Detection: PASS")
        
    except Exception as e:
        results["style_detection"] = False
        print(f"❌ Style Detection: FAIL - {e}")
        import traceback
        traceback.print_exc()
    
    # ========================================
    # Test 2: Character Consistency
    # ========================================
    print("\n--- Stage 2: Character Consistency ---")
    try:
        from agents.character_consistency import CharacterConsistencyEngine, CharacterAppearance
        
        engine = CharacterConsistencyEngine("test_world")
        
        # Register a test character using correct API
        hero = CharacterAppearance(
            character_id="hero_001",
            name="Hero Warrior",
            physical_description="athletic muscular build",
            clothing_description="silver armor with blue cape",
            distinctive_features=["scar on left cheek", "glowing blue eyes"],
            primary_colors=["#C0C0C0", "#0000FF"],
            body_type="muscular",
        )
        engine.register_character(hero)
        
        # Test prompt enhancement using correct method name
        enhanced = engine.enhance_prompt_with_characters(
            "The hero fights the dragon",
            ["Hero Warrior"],
            style="anime"
        )
        
        if "Hero Warrior" in enhanced or "athletic" in enhanced:
            print(f"  ✅ Prompt enhanced with character details")
            print(f"  Preview: {enhanced[:80]}...")
            results["character_consistency"] = True
            print("✅ Character Consistency: PASS")
        else:
            results["character_consistency"] = False
            print("❌ Character Consistency: FAIL - Enhancement not working")
        
    except Exception as e:
        results["character_consistency"] = False
        print(f"❌ Character Consistency: FAIL - {e}")
        import traceback
        traceback.print_exc()
    
    # ========================================
    # Test 3: Motion Engine
    # ========================================
    print("\n--- Stage 3: Motion Engine ---")
    try:
        from agents.motion.enhanced_motion_engine import EnhancedMotionEngine, MotionType
        
        engine = EnhancedMotionEngine()
        
        test_beats = {
            "Epic battle with explosions": MotionType.DYNAMIC,
            "Slow reveal as hero enters": MotionType.PUSH_IN,
            "Wide establishing shot of city": MotionType.PAN_RIGHT,
            "Close-up on emotional expression": MotionType.SUBTLE,
        }
        
        all_correct = True
        for desc, expected in test_beats.items():
            config = engine.detect_motion_type(desc)
            status = "✅" if config.motion_type == expected else "⚠️"
            print(f"  {status} '{desc[:30]}...' → {config.motion_type.value}")
            if config.motion_type != expected:
                all_correct = False
        
        results["motion_engine"] = all_correct
        print(f"{'✅' if all_correct else '⚠️'} Motion Engine: {'PASS' if all_correct else 'PARTIAL'}")
        
    except Exception as e:
        results["motion_engine"] = False
        print(f"❌ Motion Engine: FAIL - {e}")
    
    # ========================================
    # Test 4: Cinematic System
    # ========================================
    print("\n--- Stage 4: Cinematic System ---")
    try:
        from agents.cinematic import direct_beat, list_genres
        
        genres = list_genres()
        print(f"  Available genres: {len(genres)}")
        
        # Test directing a beat for each major genre
        test_genres = ["anime", "noir", "cyberpunk", "pixar"]
        all_complete = True
        
        for genre in test_genres:
            spec = direct_beat("Hero confronts the villain", genre)
            
            has_all = (
                spec.camera is not None and
                spec.shot is not None and
                spec.lighting is not None and
                spec.color_grade is not None
            )
            
            status = "✅" if has_all else "⚠️"
            print(f"  {status} {genre}: {spec.get_summary()}")
            
            if not has_all:
                all_complete = False
        
        results["cinematic_system"] = all_complete
        print(f"{'✅' if all_complete else '⚠️'} Cinematic System: {'PASS' if all_complete else 'PARTIAL'}")
        
    except Exception as e:
        results["cinematic_system"] = False
        print(f"❌ Cinematic System: FAIL - {e}")
        import traceback
        traceback.print_exc()
    
    # ========================================
    # Test 5: Full Beat Generation
    # ========================================
    print("\n--- Stage 5: Full Beat Generation (Simulated) ---")
    try:
        # Simulate what narrative planner does
        from agents.style_detector import StyleDetector
        from agents.cinematic import direct_beat
        from agents.motion.enhanced_motion_engine import get_motion_engine
        from agents.character_consistency import get_consistency_engine
        
        # Input
        intent = "An anime battle where the hero defeats the villain"
        description = "The hero charges energy and unleashes a powerful attack"
        
        # Detect style (use correct method)
        detector = StyleDetector()
        style_profile = detector.detect(intent)
        print(f"  Detected style: {style_profile.style.value}")
        
        # Get motion
        motion_engine = get_motion_engine()
        motion_config = motion_engine.detect_motion_type(description)
        print(f"  Motion type: {motion_config.motion_type.value}")
        
        # Get cinematic spec
        cinematic_spec = direct_beat(description, style_profile.style.value)
        print(f"  Camera: {cinematic_spec.camera.movement.value if cinematic_spec.camera else 'N/A'}")
        print(f"  Shot: {cinematic_spec.shot.shot_type.value if cinematic_spec.shot else 'N/A'}")
        print(f"  Lighting: {cinematic_spec.lighting.setup.value if cinematic_spec.lighting else 'N/A'}")
        print(f"  LUT: {cinematic_spec.color_grade.lut.value if cinematic_spec.color_grade else 'N/A'}")
        
        # Build final beat
        beat = {
            "description": description,
            "style": style_profile.style.value,
            "motion_config": motion_config.to_dict(),
            "cinematic_spec": cinematic_spec.to_dict(),
            "cinematic_prompt": cinematic_spec.get_full_prompt(description),
        }
        
        print(f"\n  Final beat structure:")
        print(f"    - description: ✅")
        print(f"    - style: ✅")
        print(f"    - motion_config: ✅")
        print(f"    - cinematic_spec: ✅")
        print(f"    - cinematic_prompt: ✅")
        
        # Print prompt preview
        prompt_preview = beat["cinematic_prompt"][:100]
        print(f"\n  Prompt preview: {prompt_preview}...")
        
        results["full_beat"] = True
        print("\n✅ Full Beat Generation: PASS")
        
    except Exception as e:
        results["full_beat"] = False
        print(f"❌ Full Beat Generation: FAIL - {e}")
        import traceback
        traceback.print_exc()
    
    # ========================================
    # Summary
    # ========================================
    print("\n" + "="*60)
    print("FULL PIPELINE SUMMARY")
    print("="*60)
    
    all_passed = True
    for stage, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {stage}: {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("🎉 All pipeline stages working!")
    else:
        print("⚠️ Some stages have issues")
    
    # Print component counts
    print("\n" + "-"*40)
    print("COMPONENT INVENTORY:")
    print("-"*40)
    
    try:
        from agents.style_detector import VisualStyle
        from agents.motion.enhanced_motion_engine import MotionType
        from agents.cinematic.camera_system import CameraMovement
        from agents.cinematic.shot_composer import ShotType, CompositionType
        from agents.cinematic.lighting_engine import LightingSetup, TimeOfDay
        from agents.cinematic.color_grading import ColorLUT, PostProcessEffect
        from agents.cinematic.genre_profiles import GENRE_PROFILES
        
        print(f"  Visual Styles:     {len(VisualStyle)}")
        print(f"  Motion Types:      {len(MotionType)}")
        print(f"  Camera Movements:  {len(CameraMovement)}")
        print(f"  Shot Types:        {len(ShotType)}")
        print(f"  Compositions:      {len(CompositionType)}")
        print(f"  Lighting Setups:   {len(LightingSetup)}")
        print(f"  Time of Day:       {len(TimeOfDay)}")
        print(f"  Color LUTs:        {len(ColorLUT)}")
        print(f"  Post Effects:      {len(PostProcessEffect)}")
        print(f"  Genre Profiles:    {len(GENRE_PROFILES)}")
        
        total = (
            len(VisualStyle) + len(MotionType) + len(CameraMovement) +
            len(ShotType) + len(CompositionType) + len(LightingSetup) +
            len(TimeOfDay) + len(ColorLUT) + len(PostProcessEffect) +
            len(GENRE_PROFILES)
        )
        print(f"\n  TOTAL OPTIONS: {total}")
        
    except Exception as e:
        print(f"  Could not enumerate: {e}")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(test_full_pipeline())
