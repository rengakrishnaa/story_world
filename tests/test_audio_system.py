"""
Audio System Test Suite

Comprehensive tests for:
- Voice synthesis engine
- Lip-sync generation
- Sound mixing
- Audio director
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_voice_synthesis_providers():
    """Test voice provider definitions."""
    print("\n" + "="*60)
    print("TEST 1: Voice Providers")
    print("="*60)
    
    from agents.audio.voice_synthesis import VoiceProvider
    
    providers = list(VoiceProvider)
    print(f"  Total providers: {len(providers)}")
    
    expected = ["elevenlabs", "google_tts", "azure_tts", "openai_tts", "coqui", "bark", "pyttsx3"]
    for exp in expected:
        found = any(p.value == exp for p in providers)
        status = "✅" if found else "❌"
        print(f"  {status} {exp}")
    
    assert len(providers) >= 7, "Expected at least 7 providers"
    print(f"  ✅ {len(providers)} voice providers defined")
    return True


def test_voice_emotions():
    """Test voice emotion options."""
    print("\n" + "="*60)
    print("TEST 2: Voice Emotions")
    print("="*60)
    
    from agents.audio.voice_synthesis import VoiceEmotion
    
    emotions = list(VoiceEmotion)
    print(f"  Total emotions: {len(emotions)}")
    
    expected = ["neutral", "happy", "sad", "angry", "fearful", "surprised"]
    for exp in expected:
        found = any(e.value == exp for e in emotions)
        status = "✅" if found else "❌"
        print(f"  {status} {exp}")
    
    assert len(emotions) >= 10, "Expected at least 10 emotions"
    print(f"  ✅ {len(emotions)} voice emotions defined")
    return True


def test_voice_profile_creation():
    """Test voice profile creation."""
    print("\n" + "="*60)
    print("TEST 3: Voice Profile Creation")
    print("="*60)
    
    from agents.audio.voice_synthesis import (
        VoiceProfile, VoiceGender, VoiceAge, VoiceStyle, VoiceEmotion
    )
    
    profile = VoiceProfile(
        character_id="test_hero",
        character_name="Test Hero",
        gender=VoiceGender.MALE,
        age=VoiceAge.YOUNG_ADULT,
        default_style=VoiceStyle.DRAMATIC,
        description="A brave test hero"
    )
    
    assert profile.character_id == "test_hero"
    assert profile.gender == VoiceGender.MALE
    
    # Test serialization
    data = profile.to_dict()
    assert "character_id" in data
    assert "gender" in data
    assert data["gender"] == "male"
    
    print(f"  ✅ Created profile: {profile.character_name}")
    print(f"  ✅ Serialized to dict with {len(data)} fields")
    return True


def test_voice_engine_initialization():
    """Test voice synthesis engine initialization."""
    print("\n" + "="*60)
    print("TEST 4: Voice Engine Initialization")
    print("="*60)
    
    from agents.audio.voice_synthesis import VoiceSynthesisEngine
    
    engine = VoiceSynthesisEngine()
    
    # Check default voices
    voices = engine.list_voices()
    assert "narrator" in voices, "Missing narrator voice"
    assert len(voices) >= 3, "Expected at least 3 default voices"
    
    print(f"  ✅ Engine initialized")
    print(f"  ✅ Default voices: {voices}")
    print(f"  ✅ Available providers: {[p.value for p in engine.available_providers]}")
    return True


def test_lip_sync_visemes():
    """Test viseme definitions."""
    print("\n" + "="*60)
    print("TEST 5: Lip-Sync Visemes")
    print("="*60)
    
    from agents.audio.lip_sync import Viseme, PHONEME_TO_VISEME
    
    visemes = list(Viseme)
    print(f"  Total visemes: {len(visemes)}")
    
    expected = ["sil", "a", "e", "i", "o", "u", "bmp", "fv", "th"]
    for exp in expected:
        found = any(v.value == exp for v in visemes)
        status = "✅" if found else "❌"
        print(f"  {status} {exp}")
    
    # Check phoneme mapping
    assert len(PHONEME_TO_VISEME) > 30, "Expected 30+ phoneme mappings"
    print(f"  ✅ {len(PHONEME_TO_VISEME)} phoneme-to-viseme mappings")
    
    return True


def test_lip_sync_styles():
    """Test lip-sync animation styles."""
    print("\n" + "="*60)
    print("TEST 6: Lip-Sync Styles")
    print("="*60)
    
    from agents.audio.lip_sync import LipSyncStyle
    
    styles = list(LipSyncStyle)
    print(f"  Total styles: {len(styles)}")
    
    expected = ["realistic", "anime", "cartoon", "minimal", "puppet"]
    for exp in expected:
        found = any(s.value == exp for s in styles)
        status = "✅" if found else "❌"
        print(f"  {status} {exp}")
    
    assert len(styles) >= 5, "Expected at least 5 styles"
    print(f"  ✅ {len(styles)} lip-sync styles defined")
    return True


def test_lip_sync_generation():
    """Test lip-sync data generation."""
    print("\n" + "="*60)
    print("TEST 7: Lip-Sync Generation")
    print("="*60)
    
    from agents.audio.lip_sync import LipSyncEngine, LipSyncStyle
    
    engine = LipSyncEngine()
    
    # Test word timing estimation
    text = "Hello, this is a test of the lip-sync engine."
    words = engine._estimate_word_timing(text, 3.0)
    
    assert len(words) > 0, "Expected word timing"
    print(f"  ✅ Estimated timing for {len(words)} words")
    
    # Test viseme generation
    visemes = engine._words_to_visemes(words, 3.0, LipSyncStyle.REALISTIC)
    
    assert len(visemes) > 0, "Expected visemes"
    print(f"  ✅ Generated {len(visemes)} visemes")
    
    # Test phoneme extraction
    phonemes = engine._text_to_phonemes("hello")
    assert len(phonemes) > 0, "Expected phonemes"
    print(f"  ✅ Extracted {len(phonemes)} phonemes from 'hello'")
    
    return True


def test_sound_mixer_track_types():
    """Test sound mixer track types."""
    print("\n" + "="*60)
    print("TEST 8: Sound Mixer Track Types")
    print("="*60)
    
    from agents.audio.sound_mixer import TrackType
    
    types = list(TrackType)
    print(f"  Total track types: {len(types)}")
    
    expected = ["dialogue", "narration", "music", "ambient", "sfx", "foley"]
    for exp in expected:
        found = any(t.value == exp for t in types)
        status = "✅" if found else "❌"
        print(f"  {status} {exp}")
    
    assert len(types) >= 6, "Expected at least 6 track types"
    print(f"  ✅ {len(types)} track types defined")
    return True


def test_sound_mixer_presets():
    """Test mixing presets."""
    print("\n" + "="*60)
    print("TEST 9: Mix Presets")
    print("="*60)
    
    from agents.audio.sound_mixer import MixPreset, MIX_PRESET_LEVELS, TrackType
    
    presets = list(MixPreset)
    print(f"  Total presets: {len(presets)}")
    
    expected = ["dialogue_focus", "action", "dramatic", "suspense", "battle"]
    for exp in expected:
        found = any(p.value == exp for p in presets)
        status = "✅" if found else "❌"
        print(f"  {status} {exp}")
    
    # Check preset levels
    for preset in presets:
        assert preset in MIX_PRESET_LEVELS, f"Missing levels for {preset.value}"
        levels = MIX_PRESET_LEVELS[preset]
        assert TrackType.DIALOGUE in levels, f"Missing dialogue level in {preset.value}"
    
    print(f"  ✅ {len(presets)} mix presets with complete levels")
    return True


def test_audio_track_creation():
    """Test audio track creation."""
    print("\n" + "="*60)
    print("TEST 10: Audio Track Creation")
    print("="*60)
    
    from agents.audio.sound_mixer import AudioTrack, TrackType
    
    track = AudioTrack(
        path="test.wav",
        track_type=TrackType.DIALOGUE,
        volume=0.8,
        fade_in=0.5,
        fade_out=0.5,
        pan=-0.3
    )
    
    assert track.path == "test.wav"
    assert track.volume == 0.8
    assert track.pan == -0.3
    
    # Test serialization
    data = track.to_dict()
    assert "path" in data
    assert "type" in data
    assert data["type"] == "dialogue"
    
    print(f"  ✅ Created track: {track.path}")
    print(f"  ✅ Volume: {track.volume}, Pan: {track.pan}")
    return True


def test_audio_director():
    """Test audio director initialization."""
    print("\n" + "="*60)
    print("TEST 11: Audio Director")
    print("="*60)
    
    from agents.audio import AudioDirector, get_audio_director
    
    director = get_audio_director()
    
    assert director.voice_engine is not None
    assert director.lipsync_engine is not None
    assert director.sound_mixer is not None
    
    print(f"  ✅ Voice engine initialized")
    print(f"  ✅ Lip-sync engine initialized")
    print(f"  ✅ Sound mixer initialized")
    
    # Get stats
    stats = director.get_stats()
    assert "voice_profiles" in stats
    assert "viseme_count" in stats
    
    print(f"  ✅ Stats: {stats}")
    return True


def test_component_inventory():
    """Test complete audio component inventory."""
    print("\n" + "="*60)
    print("AUDIO COMPONENT INVENTORY")
    print("="*60)
    
    from agents.audio.voice_synthesis import VoiceProvider, VoiceEmotion, VoiceStyle
    from agents.audio.lip_sync import Viseme, LipSyncStyle
    from agents.audio.sound_mixer import TrackType, MixPreset, AudioFormat
    
    inventory = {
        "Voice Providers": len(VoiceProvider),
        "Voice Emotions": len(VoiceEmotion),
        "Voice Styles": len(VoiceStyle),
        "Visemes": len(Viseme),
        "Lip-Sync Styles": len(LipSyncStyle),
        "Track Types": len(TrackType),
        "Mix Presets": len(MixPreset),
        "Audio Formats": len(AudioFormat),
    }
    
    total = 0
    for name, count in inventory.items():
        print(f"  {name:20}: {count:3}")
        total += count
    
    print(f"\n  {'TOTAL':20}: {total}")
    
    return True


def main():
    """Run all audio tests."""
    print("\n" + "#"*60)
    print("# AUDIO SYSTEM TEST SUITE")
    print("#"*60)
    
    results = {}
    
    tests = [
        ("voice_providers", test_voice_synthesis_providers),
        ("voice_emotions", test_voice_emotions),
        ("voice_profile", test_voice_profile_creation),
        ("voice_engine", test_voice_engine_initialization),
        ("lipsync_visemes", test_lip_sync_visemes),
        ("lipsync_styles", test_lip_sync_styles),
        ("lipsync_generation", test_lip_sync_generation),
        ("mixer_tracks", test_sound_mixer_track_types),
        ("mixer_presets", test_sound_mixer_presets),
        ("audio_track", test_audio_track_creation),
        ("audio_director", test_audio_director),
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
    print(f"🎉 {passed}/{total} audio tests passed!" if passed == total else f"⚠️ {passed}/{total} tests passed")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
