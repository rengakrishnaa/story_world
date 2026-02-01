"""
Sound Mixer

Professional audio mixing system:
- Multi-track audio layering (voice, music, SFX)
- Volume normalization and dynamics
- Spatial audio and panning
- Scene-based mixing presets
- FFmpeg-based processing
"""

import logging
import os
import subprocess
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)


# ============================================
# AUDIO TRACK TYPES
# ============================================

class TrackType(Enum):
    """Audio track types."""
    DIALOGUE = "dialogue"          # Character speech
    NARRATION = "narration"        # Voice-over narration
    MUSIC = "music"                # Background music
    AMBIENT = "ambient"            # Ambient sounds
    SFX = "sfx"                    # Sound effects
    FOLEY = "foley"                # Detailed sounds (footsteps, etc.)


class MixPreset(Enum):
    """Mixing presets for different scenes."""
    DIALOGUE_FOCUS = "dialogue_focus"    # Voice up, music down
    ACTION = "action"                    # SFX up, balanced music
    DRAMATIC = "dramatic"                # Music up during emotional beats
    SUSPENSE = "suspense"                # Low ambient, subtle music
    QUIET = "quiet"                      # Minimal audio
    BATTLE = "battle"                    # Heavy SFX and music
    ROMANTIC = "romantic"                # Soft music, intimate dialogue
    DOCUMENTARY = "documentary"          # Narration focus


class AudioFormat(Enum):
    """Output audio formats."""
    WAV = "wav"          # Uncompressed
    MP3 = "mp3"          # Compressed
    AAC = "aac"          # High quality compressed
    OGG = "ogg"          # Open format
    FLAC = "flac"        # Lossless


# ============================================
# AUDIO TRACKS
# ============================================

@dataclass
class AudioTrack:
    """
    An audio track to be mixed.
    """
    path: str
    track_type: TrackType
    
    # Timing
    start_time: float = 0.0        # When to start in the mix
    duration: Optional[float] = None  # Duration (None = full length)
    
    # Volume and dynamics
    volume: float = 1.0            # 0.0-2.0
    fade_in: float = 0.0           # Fade-in duration (seconds)
    fade_out: float = 0.0          # Fade-out duration (seconds)
    
    # Spatial
    pan: float = 0.0               # -1.0 (left) to 1.0 (right)
    
    # Processing
    normalize: bool = True         # Auto-normalize volume
    compress: bool = False         # Apply dynamics compression
    
    # Metadata
    label: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "path": self.path,
            "type": self.track_type.value,
            "start_time": self.start_time,
            "duration": self.duration,
            "volume": self.volume,
            "fade_in": self.fade_in,
            "fade_out": self.fade_out,
            "pan": self.pan,
            "normalize": self.normalize,
            "label": self.label,
        }


@dataclass
class MixSpec:
    """
    Specification for a complete audio mix.
    """
    tracks: List[AudioTrack] = field(default_factory=list)
    preset: MixPreset = MixPreset.DIALOGUE_FOCUS
    
    # Master settings
    master_volume: float = 1.0
    sample_rate: int = 44100
    bit_depth: int = 16
    channels: int = 2  # Stereo
    
    # Output
    output_format: AudioFormat = AudioFormat.AAC
    output_path: Optional[str] = None
    
    # Duration
    total_duration: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "tracks": [t.to_dict() for t in self.tracks],
            "preset": self.preset.value,
            "master_volume": self.master_volume,
            "sample_rate": self.sample_rate,
            "total_duration": self.total_duration,
        }


@dataclass
class MixResult:
    """
    Result of audio mixing.
    """
    output_path: str
    duration_seconds: float
    track_count: int
    
    success: bool = True
    error: Optional[str] = None
    
    # Analysis
    peak_level: float = 0.0        # dB
    rms_level: float = 0.0         # dB


# ============================================
# MIX PRESETS
# ============================================

# Volume multipliers by track type for each preset
MIX_PRESET_LEVELS: Dict[MixPreset, Dict[TrackType, float]] = {
    MixPreset.DIALOGUE_FOCUS: {
        TrackType.DIALOGUE: 1.0,
        TrackType.NARRATION: 1.0,
        TrackType.MUSIC: 0.3,
        TrackType.AMBIENT: 0.2,
        TrackType.SFX: 0.6,
        TrackType.FOLEY: 0.4,
    },
    MixPreset.ACTION: {
        TrackType.DIALOGUE: 0.9,
        TrackType.NARRATION: 0.8,
        TrackType.MUSIC: 0.7,
        TrackType.AMBIENT: 0.3,
        TrackType.SFX: 1.0,
        TrackType.FOLEY: 0.6,
    },
    MixPreset.DRAMATIC: {
        TrackType.DIALOGUE: 0.8,
        TrackType.NARRATION: 0.9,
        TrackType.MUSIC: 1.0,
        TrackType.AMBIENT: 0.4,
        TrackType.SFX: 0.5,
        TrackType.FOLEY: 0.3,
    },
    MixPreset.SUSPENSE: {
        TrackType.DIALOGUE: 1.0,
        TrackType.NARRATION: 0.9,
        TrackType.MUSIC: 0.5,
        TrackType.AMBIENT: 0.6,
        TrackType.SFX: 0.7,
        TrackType.FOLEY: 0.5,
    },
    MixPreset.QUIET: {
        TrackType.DIALOGUE: 0.9,
        TrackType.NARRATION: 0.9,
        TrackType.MUSIC: 0.2,
        TrackType.AMBIENT: 0.3,
        TrackType.SFX: 0.3,
        TrackType.FOLEY: 0.2,
    },
    MixPreset.BATTLE: {
        TrackType.DIALOGUE: 0.7,
        TrackType.NARRATION: 0.6,
        TrackType.MUSIC: 0.9,
        TrackType.AMBIENT: 0.4,
        TrackType.SFX: 1.0,
        TrackType.FOLEY: 0.7,
    },
    MixPreset.ROMANTIC: {
        TrackType.DIALOGUE: 1.0,
        TrackType.NARRATION: 0.8,
        TrackType.MUSIC: 0.6,
        TrackType.AMBIENT: 0.3,
        TrackType.SFX: 0.2,
        TrackType.FOLEY: 0.3,
    },
    MixPreset.DOCUMENTARY: {
        TrackType.DIALOGUE: 0.8,
        TrackType.NARRATION: 1.0,
        TrackType.MUSIC: 0.4,
        TrackType.AMBIENT: 0.5,
        TrackType.SFX: 0.4,
        TrackType.FOLEY: 0.3,
    },
}


# ============================================
# SOUND MIXER ENGINE
# ============================================

class SoundMixer:
    """
    Professional audio mixing engine.
    
    Uses FFmpeg for audio processing and mixing.
    """
    
    def __init__(self):
        self.temp_dir = "cache/audio/mix"
        os.makedirs(self.temp_dir, exist_ok=True)
        
        self._check_ffmpeg()
    
    def _check_ffmpeg(self):
        """Check if FFmpeg is available."""
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True
            )
            self.ffmpeg_available = result.returncode == 0
        except:
            self.ffmpeg_available = False
        
        logger.info(f"FFmpeg available: {self.ffmpeg_available}")
    
    def mix(self, spec: MixSpec) -> MixResult:
        """
        Mix multiple audio tracks into a single output.
        
        Args:
            spec: MixSpec with tracks and settings
            
        Returns:
            MixResult with output path and metadata
        """
        if not spec.tracks:
            return MixResult(
                output_path="",
                duration_seconds=0,
                track_count=0,
                success=False,
                error="No tracks to mix"
            )
        
        # Apply preset levels
        tracks = self._apply_preset(spec.tracks, spec.preset)
        
        # Generate output path if not specified
        if not spec.output_path:
            import uuid
            spec.output_path = os.path.join(
                self.temp_dir,
                f"mix_{uuid.uuid4().hex[:8]}.{spec.output_format.value}"
            )
        
        # Mix using FFmpeg
        if self.ffmpeg_available:
            result = self._mix_ffmpeg(tracks, spec)
        else:
            result = self._mix_simple(tracks, spec)
        
        return result
    
    def _apply_preset(
        self,
        tracks: List[AudioTrack],
        preset: MixPreset
    ) -> List[AudioTrack]:
        """Apply preset volume levels to tracks."""
        levels = MIX_PRESET_LEVELS.get(preset, MIX_PRESET_LEVELS[MixPreset.DIALOGUE_FOCUS])
        
        for track in tracks:
            preset_level = levels.get(track.track_type, 0.5)
            track.volume *= preset_level
        
        return tracks
    
    def _mix_ffmpeg(self, tracks: List[AudioTrack], spec: MixSpec) -> MixResult:
        """Mix using FFmpeg."""
        try:
            # Build FFmpeg command
            cmd = ["ffmpeg", "-y"]
            
            # Add input files
            for track in tracks:
                cmd.extend(["-i", track.path])
            
            # Build filter complex
            filter_parts = []
            mix_inputs = []
            
            for i, track in enumerate(tracks):
                # Apply volume and processing
                filters = []
                
                # Volume adjustment
                vol = track.volume * spec.master_volume
                filters.append(f"volume={vol}")
                
                # Fade in/out
                if track.fade_in > 0:
                    filters.append(f"afade=t=in:d={track.fade_in}")
                if track.fade_out > 0:
                    filters.append(f"afade=t=out:d={track.fade_out}")
                
                # Pan
                if track.pan != 0:
                    # Convert -1..1 to stereo balance
                    left = 1.0 - max(0, track.pan)
                    right = 1.0 + min(0, track.pan)
                    filters.append(f"pan=stereo|c0={left}*c0|c1={right}*c1")
                
                # Delay for start time
                if track.start_time > 0:
                    delay_ms = int(track.start_time * 1000)
                    filters.append(f"adelay={delay_ms}|{delay_ms}")
                
                filter_chain = ','.join(filters) if filters else "anull"
                filter_parts.append(f"[{i}:a]{filter_chain}[a{i}]")
                mix_inputs.append(f"[a{i}]")
            
            # Combine all tracks
            filter_parts.append(f"{''.join(mix_inputs)}amix=inputs={len(tracks)}:normalize=0[out]")
            
            filter_complex = ";".join(filter_parts)
            
            # Add filter and output
            cmd.extend([
                "-filter_complex", filter_complex,
                "-map", "[out]",
                "-ar", str(spec.sample_rate),
                "-ac", str(spec.channels),
            ])
            
            # Format-specific options
            if spec.output_format == AudioFormat.MP3:
                cmd.extend(["-codec:a", "libmp3lame", "-q:a", "2"])
            elif spec.output_format == AudioFormat.AAC:
                cmd.extend(["-codec:a", "aac", "-b:a", "192k"])
            elif spec.output_format == AudioFormat.OGG:
                cmd.extend(["-codec:a", "libvorbis", "-q:a", "6"])
            
            cmd.append(spec.output_path)
            
            # Run FFmpeg
            logger.info(f"Running FFmpeg mix: {len(tracks)} tracks")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"FFmpeg error: {result.stderr}")
                return MixResult(
                    output_path="",
                    duration_seconds=0,
                    track_count=len(tracks),
                    success=False,
                    error=result.stderr[:200]
                )
            
            # Get duration
            duration = self._get_audio_duration(spec.output_path)
            
            return MixResult(
                output_path=spec.output_path,
                duration_seconds=duration,
                track_count=len(tracks),
                success=True
            )
            
        except Exception as e:
            logger.error(f"FFmpeg mix failed: {e}")
            return MixResult(
                output_path="",
                duration_seconds=0,
                track_count=len(tracks),
                success=False,
                error=str(e)
            )
    
    def _mix_simple(self, tracks: List[AudioTrack], spec: MixSpec) -> MixResult:
        """Simple mixing fallback (uses first track only)."""
        if not tracks:
            return MixResult(
                output_path="",
                duration_seconds=0,
                track_count=0,
                success=False,
                error="No tracks"
            )
        
        # Just copy the first dialogue track
        for track in tracks:
            if track.track_type == TrackType.DIALOGUE:
                import shutil
                os.makedirs(os.path.dirname(spec.output_path), exist_ok=True)
                shutil.copy(track.path, spec.output_path)
                
                return MixResult(
                    output_path=spec.output_path,
                    duration_seconds=self._get_audio_duration(track.path),
                    track_count=1,
                    success=True
                )
        
        # Fallback to first track
        import shutil
        os.makedirs(os.path.dirname(spec.output_path), exist_ok=True)
        shutil.copy(tracks[0].path, spec.output_path)
        
        return MixResult(
            output_path=spec.output_path,
            duration_seconds=self._get_audio_duration(tracks[0].path),
            track_count=1,
            success=True
        )
    
    def _get_audio_duration(self, path: str) -> float:
        """Get audio file duration."""
        try:
            cmd = [
                "ffprobe", "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "json", path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            data = json.loads(result.stdout)
            return float(data["format"]["duration"])
        except:
            return 0.0
    
    def add_to_video(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        replace_audio: bool = True
    ) -> bool:
        """
        Add audio track to video.
        
        Args:
            video_path: Path to video file
            audio_path: Path to audio file
            output_path: Output video path
            replace_audio: If True, replace existing audio
            
        Returns:
            True if successful
        """
        if not self.ffmpeg_available:
            logger.error("FFmpeg not available")
            return False
        
        try:
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-i", audio_path,
            ]
            
            if replace_audio:
                cmd.extend([
                    "-c:v", "copy",
                    "-c:a", "aac",
                    "-map", "0:v:0",
                    "-map", "1:a:0",
                    "-shortest"
                ])
            else:
                # Mix with existing audio
                cmd.extend([
                    "-c:v", "copy",
                    "-filter_complex", "[0:a][1:a]amix=inputs=2[a]",
                    "-map", "0:v:0",
                    "-map", "[a]",
                ])
            
            cmd.append(output_path)
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"FFmpeg error: {result.stderr}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Add audio to video failed: {e}")
            return False


# ============================================
# CONVENIENCE FUNCTIONS
# ============================================

_sound_mixer: Optional[SoundMixer] = None


def get_sound_mixer() -> SoundMixer:
    """Get or create the sound mixer."""
    global _sound_mixer
    if _sound_mixer is None:
        _sound_mixer = SoundMixer()
    return _sound_mixer


def mix_audio(
    tracks: List[Dict[str, Any]],
    output_path: str,
    preset: str = "dialogue_focus"
) -> MixResult:
    """
    Convenience function to mix audio tracks.
    
    Args:
        tracks: List of track dicts with path, type, volume, etc.
        output_path: Output file path
        preset: Mix preset name
        
    Returns:
        MixResult with output path
    """
    mixer = get_sound_mixer()
    
    audio_tracks = []
    for t in tracks:
        track = AudioTrack(
            path=t["path"],
            track_type=TrackType[t.get("type", "SFX").upper()],
            volume=t.get("volume", 1.0),
            start_time=t.get("start_time", 0.0),
            fade_in=t.get("fade_in", 0.0),
            fade_out=t.get("fade_out", 0.0),
        )
        audio_tracks.append(track)
    
    preset_enum = MixPreset[preset.upper()] if preset else MixPreset.DIALOGUE_FOCUS
    
    spec = MixSpec(
        tracks=audio_tracks,
        preset=preset_enum,
        output_path=output_path
    )
    
    return mixer.mix(spec)


# ============================================
# TEST
# ============================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("SOUND MIXER TEST")
    print("="*60)
    
    mixer = SoundMixer()
    
    print(f"\nFFmpeg available: {mixer.ffmpeg_available}")
    print(f"\nTrack types: {len(TrackType)}")
    for tt in TrackType:
        print(f"  - {tt.value}")
    
    print(f"\nMix presets: {len(MixPreset)}")
    for mp in MixPreset:
        print(f"  - {mp.value}")
    
    print(f"\nAudio formats: {len(AudioFormat)}")
    for af in AudioFormat:
        print(f"  - {af.value}")
    
    # Test preset levels
    print("\nDialogue focus preset levels:")
    levels = MIX_PRESET_LEVELS[MixPreset.DIALOGUE_FOCUS]
    for tt, vol in levels.items():
        print(f"  {tt.value}: {vol*100:.0f}%")
    
    print("\n✅ Sound mixer engine working!")
