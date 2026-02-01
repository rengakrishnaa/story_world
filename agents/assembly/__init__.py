"""
Assembly Package

Episode assembly and stitching system:
- episode_assembler: Scene/act organization, transitions, audio integration
"""

from agents.assembly.episode_assembler import (
    # Engine
    EpisodeAssembler,
    get_episode_assembler,
    assemble_episode,
    
    # Data structures
    Beat,
    Scene,
    Act,
    Episode,
    AssemblyResult,
    
    # Enums
    TransitionType,
    SceneType,
    ActStructure,
)

__all__ = [
    # Engine
    "EpisodeAssembler",
    "get_episode_assembler",
    "assemble_episode",
    
    # Data structures
    "Beat",
    "Scene",
    "Act",
    "Episode",
    "AssemblyResult",
    
    # Enums
    "TransitionType",
    "SceneType",
    "ActStructure",
]
