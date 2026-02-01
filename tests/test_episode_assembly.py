"""
Episode Assembly Test Suite

Comprehensive tests for:
- Beat/Scene/Act/Episode structures
- Transition types
- Episode creation and organization
- Timeline calculation
- Assembly pipeline
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_transition_types():
    """Test transition type definitions."""
    print("\n" + "="*60)
    print("TEST 1: Transition Types")
    print("="*60)
    
    from agents.assembly.episode_assembler import TransitionType
    
    transitions = list(TransitionType)
    print(f"  Total transitions: {len(transitions)}")
    
    expected = ["cut", "crossfade", "fade_black", "wipe_left", "iris_in"]
    for exp in expected:
        found = any(t.value == exp for t in transitions)
        status = "✅" if found else "❌"
        print(f"  {status} {exp}")
    
    assert len(transitions) >= 10, "Expected at least 10 transition types"
    print(f"  ✅ {len(transitions)} transition types validated")
    return True


def test_scene_types():
    """Test scene type definitions."""
    print("\n" + "="*60)
    print("TEST 2: Scene Types")
    print("="*60)
    
    from agents.assembly.episode_assembler import SceneType
    
    types = list(SceneType)
    print(f"  Total scene types: {len(types)}")
    
    expected = ["opening", "dialogue", "action", "climax", "closing"]
    for exp in expected:
        found = any(t.value == exp for t in types)
        status = "✅" if found else "❌"
        print(f"  {status} {exp}")
    
    assert len(types) >= 8, "Expected at least 8 scene types"
    print(f"  ✅ {len(types)} scene types validated")
    return True


def test_act_structure():
    """Test act structure definitions."""
    print("\n" + "="*60)
    print("TEST 3: Act Structure")
    print("="*60)
    
    from agents.assembly.episode_assembler import ActStructure
    
    acts = list(ActStructure)
    print(f"  Total act types: {len(acts)}")
    
    expected = ["act_1", "act_2a", "act_2b", "act_3"]
    for exp in expected:
        found = any(a.value == exp for a in acts)
        status = "✅" if found else "❌"
        print(f"  {status} {exp}")
    
    assert len(acts) == 4, "Expected 4 act types"
    print(f"  ✅ {len(acts)} act structure types validated")
    return True


def test_beat_creation():
    """Test beat creation and serialization."""
    print("\n" + "="*60)
    print("TEST 4: Beat Creation")
    print("="*60)
    
    from agents.assembly.episode_assembler import Beat, SceneType, TransitionType
    
    beat = Beat(
        beat_id="beat_001",
        description="Hero enters the scene",
        duration_seconds=8.0,
        scene_type=SceneType.ESTABLISHING,
        transition_out=TransitionType.CROSSFADE,
        characters=["hero", "villain"],
        location="forest"
    )
    
    assert beat.beat_id == "beat_001"
    assert beat.duration_seconds == 8.0
    assert beat.scene_type == SceneType.ESTABLISHING
    
    data = beat.to_dict()
    assert "beat_id" in data
    assert "scene_type" in data
    assert data["scene_type"] == "establishing"
    
    print(f"  ✅ Created beat: {beat.beat_id}")
    print(f"  ✅ Duration: {beat.duration_seconds}s")
    print(f"  ✅ Serialized to dict with {len(data)} fields")
    return True


def test_scene_creation():
    """Test scene creation with beats."""
    print("\n" + "="*60)
    print("TEST 5: Scene Creation")
    print("="*60)
    
    from agents.assembly.episode_assembler import Scene, Beat, SceneType
    
    beats = [
        Beat(beat_id="b1", description="Dialog 1", duration_seconds=5.0),
        Beat(beat_id="b2", description="Dialog 2", duration_seconds=7.0),
        Beat(beat_id="b3", description="Dialog 3", duration_seconds=6.0),
    ]
    
    scene = Scene(
        scene_id="scene_1",
        title="The Conversation",
        beats=beats,
        scene_type=SceneType.DIALOGUE,
        location="coffee_shop"
    )
    
    assert scene.beat_count == 3
    assert scene.duration == 18.0  # 5+7+6
    assert scene.location == "coffee_shop"
    
    data = scene.to_dict()
    assert "beats" in data
    assert len(data["beats"]) == 3
    
    print(f"  ✅ Created scene: {scene.title}")
    print(f"  ✅ Beats: {scene.beat_count}")
    print(f"  ✅ Duration: {scene.duration}s")
    return True


def test_episode_creation():
    """Test episode creation with acts and scenes."""
    print("\n" + "="*60)
    print("TEST 6: Episode Creation")
    print("="*60)
    
    from agents.assembly.episode_assembler import (
        Episode, Act, Scene, Beat, ActStructure
    )
    
    # Create beats
    beats1 = [Beat(beat_id=f"a1_b{i}", description=f"Beat {i}", duration_seconds=5.0) for i in range(3)]
    beats2 = [Beat(beat_id=f"a2_b{i}", description=f"Beat {i}", duration_seconds=6.0) for i in range(4)]
    
    # Create scenes
    scene1 = Scene(scene_id="s1", title="Opening", beats=beats1)
    scene2 = Scene(scene_id="s2", title="Development", beats=beats2)
    
    # Create acts
    act1 = Act(act_type=ActStructure.ACT_1, scenes=[scene1])
    act2 = Act(act_type=ActStructure.ACT_2A, scenes=[scene2])
    
    # Create episode
    episode = Episode(
        episode_id="ep_001",
        title="Test Episode",
        acts=[act1, act2],
        world_id="test_world"
    )
    
    assert episode.beat_count == 7  # 3 + 4
    assert episode.scene_count == 2
    assert episode.duration == 39.0  # 15 + 24
    
    # Test flat accessors
    all_beats = episode.get_all_beats()
    all_scenes = episode.get_all_scenes()
    
    assert len(all_beats) == 7
    assert len(all_scenes) == 2
    
    print(f"  ✅ Created episode: {episode.title}")
    print(f"  ✅ Acts: {len(episode.acts)}")
    print(f"  ✅ Scenes: {episode.scene_count}")
    print(f"  ✅ Beats: {episode.beat_count}")
    print(f"  ✅ Duration: {episode.duration}s ({episode.duration_minutes:.1f} min)")
    return True


def test_assembler_initialization():
    """Test assembler initialization."""
    print("\n" + "="*60)
    print("TEST 7: Assembler Initialization")
    print("="*60)
    
    from agents.assembly.episode_assembler import EpisodeAssembler
    
    assembler = EpisodeAssembler()
    
    assert os.path.exists(assembler.output_dir)
    assert os.path.exists(assembler.temp_dir)
    
    print(f"  ✅ Output dir: {assembler.output_dir}")
    print(f"  ✅ Temp dir: {assembler.temp_dir}")
    print(f"  ✅ FFmpeg available: {assembler.ffmpeg_available}")
    return True


def test_episode_from_beats():
    """Test creating episode from beat dictionaries."""
    print("\n" + "="*60)
    print("TEST 8: Episode from Beats")
    print("="*60)
    
    from agents.assembly.episode_assembler import EpisodeAssembler
    
    assembler = EpisodeAssembler()
    
    test_beats = [
        {"beat_id": "beat_1", "description": "Hero wakes up", "duration": 8.0, "location": "bedroom"},
        {"beat_id": "beat_2", "description": "Hero gets ready", "duration": 6.0, "location": "bedroom"},
        {"beat_id": "beat_3", "description": "Hero leaves house", "duration": 5.0, "location": "street"},
        {"beat_id": "beat_4", "description": "Hero meets friend", "duration": 10.0, "location": "street"},
        {"beat_id": "beat_5", "description": "They walk to cafe", "duration": 7.0, "location": "street"},
        {"beat_id": "beat_6", "description": "They order coffee", "duration": 8.0, "location": "cafe"},
        {"beat_id": "beat_7", "description": "Deep conversation", "duration": 15.0, "location": "cafe"},
        {"beat_id": "beat_8", "description": "Hero leaves", "duration": 5.0, "location": "cafe"},
    ]
    
    episode = assembler.create_episode_from_beats(
        test_beats,
        title="Morning Adventure",
        world_id="test_world",
        auto_organize=True
    )
    
    assert episode.beat_count == 8
    assert episode.scene_count >= 2  # Should auto-organize by location
    
    # Check timeline
    all_beats = episode.get_all_beats()
    assert all_beats[0].start_time == 0.0
    assert all_beats[1].start_time > 0  # Should have start time
    
    print(f"  ✅ Created episode: {episode.title}")
    print(f"  ✅ Scenes: {episode.scene_count} (auto-organized)")
    print(f"  ✅ Beats: {episode.beat_count}")
    print(f"  ✅ Duration: {episode.duration:.1f}s")
    return True


def test_episode_stats():
    """Test episode statistics."""
    print("\n" + "="*60)
    print("TEST 9: Episode Stats")
    print("="*60)
    
    from agents.assembly.episode_assembler import EpisodeAssembler
    
    assembler = EpisodeAssembler()
    
    test_beats = [
        {"beat_id": f"b{i}", "description": f"Beat {i}", "duration": 8.0}
        for i in range(10)
    ]
    
    episode = assembler.create_episode_from_beats(test_beats, title="Stats Test")
    stats = assembler.get_episode_stats(episode)
    
    assert "duration_seconds" in stats
    assert "beat_count" in stats
    assert "scene_count" in stats
    assert stats["beat_count"] == 10
    
    print(f"  ✅ Stats retrieved:")
    print(f"    Duration: {stats['duration_seconds']}s")
    print(f"    Beats: {stats['beat_count']}")
    print(f"    Scenes: {stats['scene_count']}")
    print(f"    Acts: {stats['act_count']}")
    return True


def test_component_inventory():
    """Test assembly component inventory."""
    print("\n" + "="*60)
    print("ASSEMBLY COMPONENT INVENTORY")
    print("="*60)
    
    from agents.assembly.episode_assembler import (
        TransitionType, SceneType, ActStructure
    )
    
    inventory = {
        "Transition Types": len(TransitionType),
        "Scene Types": len(SceneType),
        "Act Types": len(ActStructure),
    }
    
    total = 0
    for name, count in inventory.items():
        print(f"  {name:20}: {count:3}")
        total += count
    
    print(f"\n  {'TOTAL':20}: {total}")
    
    return True


def main():
    """Run all assembly tests."""
    print("\n" + "#"*60)
    print("# EPISODE ASSEMBLY TEST SUITE")
    print("#"*60)
    
    results = {}
    
    tests = [
        ("transition_types", test_transition_types),
        ("scene_types", test_scene_types),
        ("act_structure", test_act_structure),
        ("beat_creation", test_beat_creation),
        ("scene_creation", test_scene_creation),
        ("episode_creation", test_episode_creation),
        ("assembler_init", test_assembler_initialization),
        ("episode_from_beats", test_episode_from_beats),
        ("episode_stats", test_episode_stats),
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
    print("SUMMARY")
    print("="*60)
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {name}: {status}")
    
    print()
    print(f"🎉 {passed}/{total} assembly tests passed!" if passed == total else f"⚠️ {passed}/{total} tests passed")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
