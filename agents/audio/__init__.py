"""
Audio System Package

Complete audio production pipeline for StoryWorld:
- voice_synthesis: Text-to-speech with multiple providers
- lip_sync: Phoneme/viseme extraction for animation
- sound_mixer: Multi-track audio mixing and processing

Usage:
    from agents.audio import (
        synthesize_speech,
        generate_lipsync,
        mix_audio,
        get_voice_engine,
        get_lipsync_engine,
        get_sound_mixer,
    )
    
    # Voice synthesis
    result = synthesize_speech("Hello world", character_id="narrator")
    
    # Lip-sync generation
    lipsync = generate_lipsync(result.audio_path, text="Hello world")
    
    # Audio mixing
    mixed = mix_audio([
        {"path": "dialogue.wav", "type": "dialogue", "volume": 1.0},
        {"path": "music.mp3", "type": "music", "volume": 0.3},
    ], output_path="final_mix.aac")
"""

# Voice synthesis
from agents.audio.voice_synthesis import (
    # Engine
    VoiceSynthesisEngine,
    get_voice_engine,
    synthesize_speech,
    
    # Types
    VoiceProvider,
    VoiceGender,
    VoiceAge,
    VoiceEmotion,
    VoiceStyle,
    VoiceProfile,
    SpeechRequest,
    SpeechResult,
)

# Lip-sync
from agents.audio.lip_sync import (
    # Engine
    LipSyncEngine,
    get_lipsync_engine,
    generate_lipsync,
    
    # Types
    Viseme,
    LipSyncStyle,
    VisemeTimestamp,
    LipSyncData,
)

# Sound mixer
from agents.audio.sound_mixer import (
    # Engine
    SoundMixer,
    get_sound_mixer,
    mix_audio,
    
    # Types
    TrackType,
    MixPreset,
    AudioFormat,
    AudioTrack,
    MixSpec,
    MixResult,
)


# ============================================
# AUDIO DIRECTOR
# ============================================

class AudioDirector:
    """
    Main orchestrator for all audio operations.
    
    Coordinates voice synthesis, lip-sync, and mixing.
    """
    
    def __init__(self):
        self.voice_engine = get_voice_engine()
        self.lipsync_engine = get_lipsync_engine()
        self.sound_mixer = get_sound_mixer()
    
    def create_dialogue(
        self,
        text: str,
        character_id: str = "narrator",
        emotion: str = None,
        generate_lipsync: bool = True
    ) -> dict:
        """
        Create dialogue with speech and optional lip-sync.
        
        Args:
            text: Dialogue text
            character_id: Character voice to use
            emotion: Optional emotion
            generate_lipsync: Whether to generate lip-sync data
            
        Returns:
            Dict with speech_result, lipsync_data
        """
        # Synthesize speech
        speech = synthesize_speech(text, character_id, emotion)
        
        result = {
            "speech": speech,
            "audio_path": speech.audio_path,
            "duration": speech.duration_seconds,
        }
        
        # Generate lip-sync
        if generate_lipsync and speech.success:
            lipsync = self.lipsync_engine.generate_lipsync(
                speech.audio_path,
                text=text
            )
            result["lipsync"] = lipsync
        
        return result
    
    def mix_scene_audio(
        self,
        dialogue_path: str = None,
        music_path: str = None,
        sfx_paths: list = None,
        ambient_path: str = None,
        output_path: str = None,
        scene_type: str = "dialogue_focus"
    ) -> MixResult:
        """
        Mix audio for a complete scene.
        
        Args:
            dialogue_path: Path to dialogue audio
            music_path: Path to background music
            sfx_paths: List of sound effect paths with timing
            ambient_path: Path to ambient audio
            output_path: Output file path
            scene_type: Scene type for mix preset
            
        Returns:
            MixResult with final audio
        """
        tracks = []
        
        if dialogue_path:
            tracks.append(AudioTrack(
                path=dialogue_path,
                track_type=TrackType.DIALOGUE,
                volume=1.0
            ))
        
        if music_path:
            tracks.append(AudioTrack(
                path=music_path,
                track_type=TrackType.MUSIC,
                volume=0.5,
                fade_in=2.0,
                fade_out=2.0
            ))
        
        if ambient_path:
            tracks.append(AudioTrack(
                path=ambient_path,
                track_type=TrackType.AMBIENT,
                volume=0.3
            ))
        
        if sfx_paths:
            for sfx in sfx_paths:
                if isinstance(sfx, dict):
                    tracks.append(AudioTrack(
                        path=sfx["path"],
                        track_type=TrackType.SFX,
                        start_time=sfx.get("start", 0),
                        volume=sfx.get("volume", 0.8)
                    ))
                else:
                    tracks.append(AudioTrack(
                        path=sfx,
                        track_type=TrackType.SFX,
                        volume=0.8
                    ))
        
        preset = MixPreset[scene_type.upper()] if scene_type else MixPreset.DIALOGUE_FOCUS
        
        spec = MixSpec(
            tracks=tracks,
            preset=preset,
            output_path=output_path
        )
        
        return self.sound_mixer.mix(spec)
    
    def get_stats(self) -> dict:
        """Get audio system statistics."""
        return {
            "voice_profiles": self.voice_engine.get_voice_count(),
            "available_providers": [p.value for p in self.voice_engine.available_providers],
            "viseme_count": len(Viseme),
            "lipsync_styles": len(LipSyncStyle),
            "track_types": len(TrackType),
            "mix_presets": len(MixPreset),
            "audio_formats": len(AudioFormat),
        }


# Singleton
_audio_director = None


def get_audio_director() -> AudioDirector:
    """Get or create the audio director."""
    global _audio_director
    if _audio_director is None:
        _audio_director = AudioDirector()
    return _audio_director


# ============================================
# EXPORTS
# ============================================

__all__ = [
    # Main orchestrator
    "AudioDirector",
    "get_audio_director",
    
    # Voice synthesis
    "VoiceSynthesisEngine",
    "get_voice_engine",
    "synthesize_speech",
    "VoiceProvider",
    "VoiceGender",
    "VoiceAge",
    "VoiceEmotion",
    "VoiceStyle",
    "VoiceProfile",
    "SpeechRequest",
    "SpeechResult",
    
    # Lip-sync
    "LipSyncEngine",
    "get_lipsync_engine",
    "generate_lipsync",
    "Viseme",
    "LipSyncStyle",
    "VisemeTimestamp",
    "LipSyncData",
    
    # Sound mixer
    "SoundMixer",
    "get_sound_mixer",
    "mix_audio",
    "TrackType",
    "MixPreset",
    "AudioFormat",
    "AudioTrack",
    "MixSpec",
    "MixResult",
]
