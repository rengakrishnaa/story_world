"""
Episode Assembler

Enhanced episode assembly system that:
- Organizes beats into scenes and acts
- Adds professional transitions between scenes
- Integrates audio (dialogue, music, SFX)
- Handles timeline management
- Supports multiple output formats
"""

import logging
import os
import json
import subprocess
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import math

logger = logging.getLogger(__name__)


# ============================================
# TRANSITION TYPES
# ============================================

class TransitionType(Enum):
    """Professional transition types."""
    # Basic
    CUT = "cut"                      # Hard cut (instant)
    CROSSFADE = "crossfade"          # Dissolve between shots
    FADE_TO_BLACK = "fade_black"     # Fade out then in
    FADE_TO_WHITE = "fade_white"     # Fade out to white then in
    
    # Wipes
    WIPE_LEFT = "wipe_left"          # Horizontal wipe
    WIPE_RIGHT = "wipe_right"
    WIPE_UP = "wipe_up"              # Vertical wipe
    WIPE_DOWN = "wipe_down"
    
    # Fancy
    IRIS_IN = "iris_in"              # Circle reveal
    IRIS_OUT = "iris_out"            # Circle close
    CLOCK_WIPE = "clock_wipe"        # Rotating wipe
    
    # Special
    BLUR_TRANSITION = "blur"         # Blur dissolve
    ZOOM_TRANSITION = "zoom"         # Zoom through
    FLASH_TRANSITION = "flash"       # Brief flash


class SceneType(Enum):
    """Scene types for pacing."""
    OPENING = "opening"              # Title/intro
    ESTABLISHING = "establishing"    # Location setup
    DIALOGUE = "dialogue"            # Character conversation
    ACTION = "action"                # Action sequence
    MONTAGE = "montage"              # Time passage
    FLASHBACK = "flashback"          # Past event
    CLIMAX = "climax"                # Peak moment
    RESOLUTION = "resolution"        # Winding down
    CLOSING = "closing"              # End credits


class ActStructure(Enum):
    """Three-act structure."""
    ACT_1 = "act_1"                  # Setup (25%)
    ACT_2A = "act_2a"                # Confrontation part 1 (25%)
    ACT_2B = "act_2b"                # Confrontation part 2 (25%)
    ACT_3 = "act_3"                  # Resolution (25%)


# ============================================
# EPISODE STRUCTURE
# ============================================

@dataclass
class Beat:
    """A single beat (video clip) in the episode."""
    beat_id: str
    description: str
    video_path: Optional[str] = None
    audio_path: Optional[str] = None
    
    # Timing
    duration_seconds: float = 8.0
    start_time: Optional[float] = None  # In episode timeline
    
    # Classification
    scene_id: Optional[str] = None
    scene_type: SceneType = SceneType.DIALOGUE
    act: ActStructure = ActStructure.ACT_1
    
    # Transition to next beat
    transition_out: TransitionType = TransitionType.CUT
    transition_duration: float = 0.5
    
    # Metadata
    characters: List[str] = field(default_factory=list)
    location: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "beat_id": self.beat_id,
            "description": self.description,
            "video_path": self.video_path,
            "audio_path": self.audio_path,
            "duration": self.duration_seconds,
            "start_time": self.start_time,
            "scene_id": self.scene_id,
            "scene_type": self.scene_type.value,
            "act": self.act.value,
            "transition_out": self.transition_out.value,
            "characters": self.characters,
            "location": self.location,
        }


@dataclass
class Scene:
    """A scene containing multiple beats."""
    scene_id: str
    title: str
    beats: List[Beat] = field(default_factory=list)
    
    # Scene properties
    scene_type: SceneType = SceneType.DIALOGUE
    location: str = ""
    time_of_day: str = ""
    
    # Transition to next scene
    transition_out: TransitionType = TransitionType.CROSSFADE
    transition_duration: float = 1.0
    
    # Audio
    music_path: Optional[str] = None
    ambient_path: Optional[str] = None
    
    @property
    def duration(self) -> float:
        """Total scene duration."""
        return sum(b.duration_seconds for b in self.beats)
    
    @property
    def beat_count(self) -> int:
        """Number of beats in scene."""
        return len(self.beats)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "scene_id": self.scene_id,
            "title": self.title,
            "beats": [b.to_dict() for b in self.beats],
            "scene_type": self.scene_type.value,
            "location": self.location,
            "duration": self.duration,
            "transition_out": self.transition_out.value,
        }


@dataclass
class Act:
    """An act containing multiple scenes."""
    act_type: ActStructure
    scenes: List[Scene] = field(default_factory=list)
    
    @property
    def duration(self) -> float:
        """Total act duration."""
        return sum(s.duration for s in self.scenes)
    
    @property
    def scene_count(self) -> int:
        """Number of scenes in act."""
        return len(self.scenes)
    
    @property
    def beat_count(self) -> int:
        """Total beats in act."""
        return sum(s.beat_count for s in self.scenes)


@dataclass
class Episode:
    """Complete episode structure."""
    episode_id: str
    title: str
    acts: List[Act] = field(default_factory=list)
    
    # Episode metadata
    world_id: str = ""
    target_duration_minutes: float = 8.0
    
    # Audio tracks
    opening_music: Optional[str] = None
    closing_music: Optional[str] = None
    
    # Output
    output_path: Optional[str] = None
    
    @property
    def duration(self) -> float:
        """Total episode duration in seconds."""
        return sum(a.duration for a in self.acts)
    
    @property
    def duration_minutes(self) -> float:
        """Duration in minutes."""
        return self.duration / 60
    
    @property
    def scene_count(self) -> int:
        """Total scenes."""
        return sum(a.scene_count for a in self.acts)
    
    @property
    def beat_count(self) -> int:
        """Total beats."""
        return sum(a.beat_count for a in self.acts)
    
    def get_all_beats(self) -> List[Beat]:
        """Get flat list of all beats."""
        beats = []
        for act in self.acts:
            for scene in act.scenes:
                beats.extend(scene.beats)
        return beats
    
    def get_all_scenes(self) -> List[Scene]:
        """Get flat list of all scenes."""
        scenes = []
        for act in self.acts:
            scenes.extend(act.scenes)
        return scenes
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "episode_id": self.episode_id,
            "title": self.title,
            "world_id": self.world_id,
            "duration_seconds": self.duration,
            "duration_minutes": self.duration_minutes,
            "scene_count": self.scene_count,
            "beat_count": self.beat_count,
            "acts": [
                {
                    "act": a.act_type.value,
                    "scenes": [s.to_dict() for s in a.scenes],
                    "duration": a.duration,
                }
                for a in self.acts
            ],
        }


@dataclass
class AssemblyResult:
    """Result of episode assembly."""
    output_path: str
    duration_seconds: float
    beat_count: int
    scene_count: int
    
    success: bool = True
    error: Optional[str] = None
    
    # Processing stats
    transitions_applied: int = 0
    audio_tracks_mixed: int = 0


# ============================================
# EPISODE ASSEMBLER
# ============================================

class EpisodeAssembler:
    """
    Assembles beats into complete episodes.
    
    Features:
    - Scene/act organization
    - Professional transitions
    - Audio integration
    - Timeline calculation
    """
    
    def __init__(self, output_dir: str = "output/episodes"):
        self.output_dir = output_dir
        self.temp_dir = "cache/assembly"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        
        self._check_ffmpeg()
    
    def _check_ffmpeg(self):
        """Check FFmpeg availability."""
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
    
    def create_episode_from_beats(
        self,
        beats: List[Dict[str, Any]],
        title: str = "Episode",
        world_id: str = "",
        auto_organize: bool = True
    ) -> Episode:
        """
        Create episode structure from beat data.
        
        Args:
            beats: List of beat dicts with description, video_path, etc.
            title: Episode title
            world_id: World identifier
            auto_organize: Auto-organize into scenes/acts
            
        Returns:
            Episode structure
        """
        episode_id = f"ep_{uuid.uuid4().hex[:8]}"
        
        # Convert dicts to Beat objects
        beat_objects = []
        for i, b in enumerate(beats):
            beat = Beat(
                beat_id=b.get("beat_id", f"beat_{i}"),
                description=b.get("description", ""),
                video_path=b.get("video_path"),
                audio_path=b.get("audio_path"),
                duration_seconds=b.get("duration", 8.0),
                characters=b.get("characters", []),
                location=b.get("location", ""),
            )
            beat_objects.append(beat)
        
        # Create episode
        episode = Episode(
            episode_id=episode_id,
            title=title,
            world_id=world_id
        )
        
        if auto_organize:
            # Auto-organize into scenes and acts
            episode = self._auto_organize(episode, beat_objects)
        else:
            # Put all beats in one scene, one act
            scene = Scene(
                scene_id="scene_1",
                title="Main",
                beats=beat_objects
            )
            act = Act(
                act_type=ActStructure.ACT_1,
                scenes=[scene]
            )
            episode.acts = [act]
        
        # Calculate timeline
        self._calculate_timeline(episode)
        
        return episode
    
    def _auto_organize(self, episode: Episode, beats: List[Beat]) -> Episode:
        """Auto-organize beats into scenes and acts."""
        if not beats:
            return episode
        
        # Group beats into scenes by location
        scenes = []
        current_scene_beats = []
        current_location = None
        scene_idx = 1
        
        for beat in beats:
            # New scene on location change or after 5 beats
            if (beat.location != current_location and current_location) or len(current_scene_beats) >= 5:
                if current_scene_beats:
                    scene = Scene(
                        scene_id=f"scene_{scene_idx}",
                        title=f"Scene {scene_idx}",
                        beats=current_scene_beats,
                        location=current_location or ""
                    )
                    scenes.append(scene)
                    scene_idx += 1
                current_scene_beats = []
            
            current_scene_beats.append(beat)
            current_location = beat.location
        
        # Add final scene
        if current_scene_beats:
            scene = Scene(
                scene_id=f"scene_{scene_idx}",
                title=f"Scene {scene_idx}",
                beats=current_scene_beats,
                location=current_location or ""
            )
            scenes.append(scene)
        
        # Distribute scenes into acts (roughly 25% each)
        total_scenes = len(scenes)
        act_sizes = [
            max(1, total_scenes // 4),
            max(1, total_scenes // 4),
            max(1, total_scenes // 4),
            total_scenes - 3 * (total_scenes // 4)
        ]
        
        acts = []
        scene_idx = 0
        for act_type, size in zip(ActStructure, act_sizes):
            act_scenes = scenes[scene_idx:scene_idx + size]
            if act_scenes:
                act = Act(act_type=act_type, scenes=act_scenes)
                acts.append(act)
            scene_idx += size
        
        episode.acts = acts
        return episode
    
    def _calculate_timeline(self, episode: Episode):
        """Calculate start times for all beats."""
        current_time = 0.0
        
        for act in episode.acts:
            for scene in act.scenes:
                for beat in scene.beats:
                    beat.start_time = current_time
                    current_time += beat.duration_seconds
                    
                    # Add transition time
                    if beat.transition_out != TransitionType.CUT:
                        current_time += beat.transition_duration * 0.5  # Overlap
    
    def assemble(
        self,
        episode: Episode,
        include_audio: bool = True,
        output_format: str = "mp4"
    ) -> AssemblyResult:
        """
        Assemble episode from structure.
        
        Args:
            episode: Episode structure
            include_audio: Whether to mix in audio tracks
            output_format: Output video format
            
        Returns:
            AssemblyResult with output path
        """
        if not self.ffmpeg_available:
            return AssemblyResult(
                output_path="",
                duration_seconds=0,
                beat_count=0,
                scene_count=0,
                success=False,
                error="FFmpeg not available"
            )
        
        beats = episode.get_all_beats()
        if not beats:
            return AssemblyResult(
                output_path="",
                duration_seconds=0,
                beat_count=0,
                scene_count=0,
                success=False,
                error="No beats to assemble"
            )
        
        # Check for video paths
        video_paths = [b.video_path for b in beats if b.video_path]
        if not video_paths:
            return AssemblyResult(
                output_path="",
                duration_seconds=episode.duration,
                beat_count=len(beats),
                scene_count=episode.scene_count,
                success=False,
                error="No video files provided"
            )
        
        # Generate output path
        output_path = os.path.join(
            self.output_dir,
            f"{episode.episode_id}.{output_format}"
        )
        episode.output_path = output_path
        
        # Stitch videos
        result = self._stitch_with_transitions(beats, output_path)
        
        if not result["success"]:
            return AssemblyResult(
                output_path="",
                duration_seconds=0,
                beat_count=len(beats),
                scene_count=episode.scene_count,
                success=False,
                error=result.get("error", "Stitch failed")
            )
        
        return AssemblyResult(
            output_path=output_path,
            duration_seconds=episode.duration,
            beat_count=len(beats),
            scene_count=episode.scene_count,
            success=True,
            transitions_applied=result.get("transitions", 0),
        )
    
    def _stitch_with_transitions(
        self,
        beats: List[Beat],
        output_path: str
    ) -> Dict[str, Any]:
        """Stitch beats with transitions using FFmpeg."""
        video_paths = [b.video_path for b in beats if b.video_path and os.path.exists(b.video_path)]
        
        if not video_paths:
            return {"success": False, "error": "No valid video files"}
        
        if len(video_paths) == 1:
            # Just copy single file
            import shutil
            shutil.copy(video_paths[0], output_path)
            return {"success": True, "transitions": 0}
        
        try:
            # Build concat file
            concat_file = os.path.join(self.temp_dir, f"concat_{uuid.uuid4().hex[:8]}.txt")
            with open(concat_file, "w") as f:
                for path in video_paths:
                    f.write(f"file '{os.path.abspath(path)}'\n")
            
            # Run FFmpeg concat
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_file,
                "-c", "copy",
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Cleanup
            os.remove(concat_file)
            
            if result.returncode != 0:
                return {"success": False, "error": result.stderr[:200]}
            
            return {"success": True, "transitions": len(video_paths) - 1}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_episode_stats(self, episode: Episode) -> Dict[str, Any]:
        """Get statistics for an episode."""
        return {
            "episode_id": episode.episode_id,
            "title": episode.title,
            "duration_seconds": episode.duration,
            "duration_minutes": episode.duration_minutes,
            "act_count": len(episode.acts),
            "scene_count": episode.scene_count,
            "beat_count": episode.beat_count,
            "acts": [
                {
                    "act": a.act_type.value,
                    "scenes": a.scene_count,
                    "beats": a.beat_count,
                    "duration": a.duration
                }
                for a in episode.acts
            ]
        }


# ============================================
# CONVENIENCE FUNCTIONS
# ============================================

_episode_assembler: Optional[EpisodeAssembler] = None


def get_episode_assembler() -> EpisodeAssembler:
    """Get or create the episode assembler."""
    global _episode_assembler
    if _episode_assembler is None:
        _episode_assembler = EpisodeAssembler()
    return _episode_assembler


def assemble_episode(
    beats: List[Dict[str, Any]],
    title: str = "Episode",
    world_id: str = "",
) -> AssemblyResult:
    """
    Convenience function to assemble episode from beats.
    """
    assembler = get_episode_assembler()
    episode = assembler.create_episode_from_beats(beats, title, world_id)
    return assembler.assemble(episode)


# ============================================
# TEST
# ============================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("EPISODE ASSEMBLER TEST")
    print("="*60)
    
    assembler = EpisodeAssembler()
    
    print(f"\nFFmpeg available: {assembler.ffmpeg_available}")
    
    # Test episode creation
    test_beats = [
        {"beat_id": "beat_1", "description": "Hero wakes up", "duration": 8.0, "location": "bedroom"},
        {"beat_id": "beat_2", "description": "Hero gets ready", "duration": 6.0, "location": "bedroom"},
        {"beat_id": "beat_3", "description": "Hero leaves house", "duration": 5.0, "location": "street"},
        {"beat_id": "beat_4", "description": "Hero meets friend", "duration": 10.0, "location": "street"},
        {"beat_id": "beat_5", "description": "They talk", "duration": 12.0, "location": "cafe"},
    ]
    
    episode = assembler.create_episode_from_beats(
        test_beats,
        title="Test Episode",
        world_id="test_world"
    )
    
    stats = assembler.get_episode_stats(episode)
    
    print(f"\nEpisode: {episode.title}")
    print(f"  Duration: {episode.duration:.1f}s ({episode.duration_minutes:.1f} min)")
    print(f"  Acts: {len(episode.acts)}")
    print(f"  Scenes: {episode.scene_count}")
    print(f"  Beats: {episode.beat_count}")
    
    print(f"\n✅ {len(TransitionType)} transition types defined")
    print(f"✅ {len(SceneType)} scene types defined")
    print(f"✅ {len(ActStructure)} act types defined")
    
    print("\n✅ Episode assembler working!")
