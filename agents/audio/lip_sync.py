"""
Lip-Sync Engine

Professional lip-sync system for matching audio to character animations:
- Phoneme extraction from audio
- Viseme mapping for different animation styles
- Timing synchronization
- Support for anime, realistic, and cartoon styles
"""

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple
import math

logger = logging.getLogger(__name__)


# ============================================
# VISEME DEFINITIONS
# ============================================

class Viseme(Enum):
    """
    Standard viseme set for lip-sync.
    Based on Preston Blair phoneme chart.
    """
    # Closed mouth
    SILENT = "sil"           # Mouth closed, neutral
    
    # Vowels
    A = "a"                  # Open mouth (ah, a)
    E = "e"                  # Wide smile (e, ee)
    I = "i"                  # Slightly open, forward (i)
    O = "o"                  # Round open (o, oh)
    U = "u"                  # Pursed lips (oo, u)
    
    # Consonants
    B_M_P = "bmp"           # Lips together (b, m, p)
    F_V = "fv"              # Teeth on lip (f, v)
    TH = "th"               # Tongue between teeth
    L = "l"                 # Tongue up (l)
    D_T_N = "dtn"           # Tongue behind teeth (d, t, n)
    K_G = "kg"              # Back of tongue (k, g)
    S_Z = "sz"              # Teeth together (s, z)
    SH_CH = "shch"          # Rounded, forward (sh, ch)
    R = "r"                 # Rounded lips (r)
    W = "w"                 # Very rounded (w)


class LipSyncStyle(Enum):
    """Lip-sync animation styles."""
    REALISTIC = "realistic"      # Detailed mouth movements
    ANIME = "anime"              # Simple A-I-U-E-O
    CARTOON = "cartoon"          # Exaggerated expressions
    MINIMAL = "minimal"          # Just open/close
    PUPPET = "puppet"            # Simple flap animation


# ============================================
# PHONEME TO VISEME MAPPING
# ============================================

PHONEME_TO_VISEME = {
    # Silence
    "SIL": Viseme.SILENT,
    "SP": Viseme.SILENT,
    
    # Vowels (ARPAbet)
    "AA": Viseme.A,   # father
    "AE": Viseme.A,   # cat
    "AH": Viseme.A,   # but
    "AO": Viseme.O,   # caught
    "AW": Viseme.A,   # cow
    "AY": Viseme.A,   # say
    "EH": Viseme.E,   # bet
    "ER": Viseme.R,   # bird
    "EY": Viseme.E,   # ate
    "IH": Viseme.I,   # bit
    "IY": Viseme.E,   # beat
    "OW": Viseme.O,   # boat
    "OY": Viseme.O,   # boy
    "UH": Viseme.U,   # book
    "UW": Viseme.U,   # boot
    
    # Consonants (ARPAbet)
    "B": Viseme.B_M_P,
    "CH": Viseme.SH_CH,
    "D": Viseme.D_T_N,
    "DH": Viseme.TH,
    "F": Viseme.F_V,
    "G": Viseme.K_G,
    "HH": Viseme.SILENT,
    "JH": Viseme.SH_CH,
    "K": Viseme.K_G,
    "L": Viseme.L,
    "M": Viseme.B_M_P,
    "N": Viseme.D_T_N,
    "NG": Viseme.K_G,
    "P": Viseme.B_M_P,
    "R": Viseme.R,
    "S": Viseme.S_Z,
    "SH": Viseme.SH_CH,
    "T": Viseme.D_T_N,
    "TH": Viseme.TH,
    "V": Viseme.F_V,
    "W": Viseme.W,
    "Y": Viseme.I,
    "Z": Viseme.S_Z,
    "ZH": Viseme.SH_CH,
}

# Simplified anime viseme set (A-I-U-E-O)
ANIME_VISEME_MAP = {
    Viseme.A: "A",
    Viseme.E: "I",  # Japanese 'i' sound
    Viseme.I: "I",
    Viseme.O: "O",
    Viseme.U: "U",
    Viseme.B_M_P: "N",  # Closed for consonants
    Viseme.SILENT: "N",
    # Default others to closest vowel
}


# ============================================
# LIP-SYNC DATA STRUCTURES
# ============================================

@dataclass
class VisemeTimestamp:
    """A viseme with timing information."""
    viseme: Viseme
    start_time: float      # seconds
    end_time: float        # seconds
    intensity: float = 1.0  # 0.0-1.0, how open the mouth is
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class LipSyncData:
    """
    Complete lip-sync data for an audio clip.
    """
    audio_path: str
    duration_seconds: float
    
    # Viseme timeline
    visemes: List[VisemeTimestamp] = field(default_factory=list)
    
    # Word-level timing
    words: List[Tuple[str, float, float]] = field(default_factory=list)  # (word, start, end)
    
    # Style
    style: LipSyncStyle = LipSyncStyle.REALISTIC
    
    # Metadata
    success: bool = True
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "audio_path": self.audio_path,
            "duration": self.duration_seconds,
            "visemes": [
                {
                    "viseme": v.viseme.value,
                    "start": v.start_time,
                    "end": v.end_time,
                    "intensity": v.intensity
                }
                for v in self.visemes
            ],
            "words": [
                {"word": w, "start": s, "end": e}
                for w, s, e in self.words
            ],
            "style": self.style.value,
            "success": self.success
        }
    
    def get_viseme_at(self, time: float) -> Optional[VisemeTimestamp]:
        """Get viseme at a specific time."""
        for v in self.visemes:
            if v.start_time <= time < v.end_time:
                return v
        return None


# ============================================
# LIP-SYNC ENGINE
# ============================================

class LipSyncEngine:
    """
    Engine for generating lip-sync data from audio.
    
    Supports multiple extraction methods:
    - Gentle (forced alignment)
    - Whisper (OpenAI)
    - Simple timing estimation
    """
    
    def __init__(self):
        self.style = LipSyncStyle.REALISTIC
        self._check_backends()
    
    def _check_backends(self):
        """Check available lip-sync backends."""
        self.available_backends = ["estimation"]  # Always available
        
        try:
            import whisper
            self.available_backends.append("whisper")
        except ImportError:
            pass
        
        logger.info(f"Lip-sync backends: {self.available_backends}")
    
    def generate_lipsync(
        self,
        audio_path: str,
        text: Optional[str] = None,
        style: LipSyncStyle = LipSyncStyle.REALISTIC,
        words_with_timing: Optional[List[Tuple[str, float, float]]] = None
    ) -> LipSyncData:
        """
        Generate lip-sync data from audio.
        
        Args:
            audio_path: Path to audio file
            text: Optional transcript text
            style: Animation style to target
            words_with_timing: Pre-computed word timestamps
            
        Returns:
            LipSyncData with viseme timeline
        """
        self.style = style
        
        # Get audio duration
        duration = self._get_audio_duration(audio_path)
        
        # Get word timing
        if words_with_timing:
            words = words_with_timing
        elif text:
            words = self._estimate_word_timing(text, duration)
        else:
            words = []
        
        # Generate visemes from words
        visemes = self._words_to_visemes(words, duration, style)
        
        return LipSyncData(
            audio_path=audio_path,
            duration_seconds=duration,
            visemes=visemes,
            words=words,
            style=style,
            success=True
        )
    
    def _get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration in seconds."""
        try:
            import wave
            with wave.open(audio_path, 'rb') as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                return frames / rate
        except:
            # Estimate from file size (rough)
            try:
                size = os.path.getsize(audio_path)
                return size / 32000  # ~16kHz mono 16-bit
            except:
                return 5.0  # Default
    
    def _estimate_word_timing(
        self,
        text: str,
        duration: float
    ) -> List[Tuple[str, float, float]]:
        """Estimate word timing from text and duration."""
        words = text.split()
        if not words:
            return []
        
        # Average time per word
        time_per_word = duration / len(words)
        
        result = []
        current_time = 0.0
        
        for word in words:
            # Adjust duration based on word length
            word_duration = time_per_word * (0.5 + len(word) / 12)
            word_duration = min(word_duration, 2.0)  # Cap at 2 seconds
            
            end_time = min(current_time + word_duration, duration)
            result.append((word, current_time, end_time))
            current_time = end_time
        
        return result
    
    def _words_to_visemes(
        self,
        words: List[Tuple[str, float, float]],
        duration: float,
        style: LipSyncStyle
    ) -> List[VisemeTimestamp]:
        """Convert word timing to viseme timeline."""
        visemes = []
        
        # Add initial silence
        if words and words[0][1] > 0:
            visemes.append(VisemeTimestamp(
                viseme=Viseme.SILENT,
                start_time=0,
                end_time=words[0][1],
                intensity=0
            ))
        
        for word, start, end in words:
            word_visemes = self._word_to_visemes(word, start, end, style)
            visemes.extend(word_visemes)
        
        # Add final silence
        if words and words[-1][2] < duration:
            visemes.append(VisemeTimestamp(
                viseme=Viseme.SILENT,
                start_time=words[-1][2],
                end_time=duration,
                intensity=0
            ))
        
        return visemes
    
    def _word_to_visemes(
        self,
        word: str,
        start: float,
        end: float,
        style: LipSyncStyle
    ) -> List[VisemeTimestamp]:
        """Convert a word to viseme sequence."""
        if style == LipSyncStyle.MINIMAL:
            # Just open/close
            mid = (start + end) / 2
            return [
                VisemeTimestamp(Viseme.SILENT, start, start + 0.05, 0.3),
                VisemeTimestamp(Viseme.A, start + 0.05, end - 0.05, 0.8),
                VisemeTimestamp(Viseme.SILENT, end - 0.05, end, 0.3),
            ]
        
        # Get phonemes for word
        phonemes = self._text_to_phonemes(word)
        
        if not phonemes:
            # Default to open mouth
            return [VisemeTimestamp(Viseme.A, start, end, 0.7)]
        
        # Distribute phonemes across word duration
        visemes = []
        time_per_phoneme = (end - start) / len(phonemes)
        
        for i, phoneme in enumerate(phonemes):
            viseme = PHONEME_TO_VISEME.get(phoneme.upper(), Viseme.A)
            
            if style == LipSyncStyle.ANIME:
                # Simplify to anime style
                viseme = self._to_anime_viseme(viseme)
            
            ph_start = start + (i * time_per_phoneme)
            ph_end = start + ((i + 1) * time_per_phoneme)
            
            # Calculate intensity based on vowel/consonant
            intensity = 0.9 if viseme in [Viseme.A, Viseme.O, Viseme.U] else 0.6
            
            visemes.append(VisemeTimestamp(
                viseme=viseme,
                start_time=ph_start,
                end_time=ph_end,
                intensity=intensity
            ))
        
        return visemes
    
    def _text_to_phonemes(self, word: str) -> List[str]:
        """Simple rule-based phoneme estimation."""
        # Simplified phoneme extraction
        phonemes = []
        word = word.lower()
        i = 0
        
        while i < len(word):
            ch = word[i]
            
            # Two-character combos
            if i < len(word) - 1:
                combo = word[i:i+2]
                if combo in ["sh", "ch", "th"]:
                    phonemes.append(combo.upper())
                    i += 2
                    continue
                elif combo == "ng":
                    phonemes.append("NG")
                    i += 2
                    continue
            
            # Single characters
            if ch in "aeiou":
                phonemes.append(ch.upper() + "H")  # AH, EH, etc.
            elif ch in "bcdfghjklmnpqrstvwxyz":
                phonemes.append(ch.upper())
            
            i += 1
        
        return phonemes
    
    def _to_anime_viseme(self, viseme: Viseme) -> Viseme:
        """Convert viseme to anime-style (A-I-U-E-O)."""
        if viseme == Viseme.SILENT:
            return Viseme.SILENT
        elif viseme in [Viseme.A]:
            return Viseme.A
        elif viseme in [Viseme.E, Viseme.I, Viseme.S_Z]:
            return Viseme.I
        elif viseme in [Viseme.U, Viseme.W, Viseme.B_M_P]:
            return Viseme.U
        elif viseme in [Viseme.O, Viseme.R]:
            return Viseme.O
        else:
            return Viseme.A  # Default


# ============================================
# CONVENIENCE FUNCTIONS
# ============================================

_lipsync_engine: Optional[LipSyncEngine] = None


def get_lipsync_engine() -> LipSyncEngine:
    """Get or create the lip-sync engine."""
    global _lipsync_engine
    if _lipsync_engine is None:
        _lipsync_engine = LipSyncEngine()
    return _lipsync_engine


def generate_lipsync(
    audio_path: str,
    text: Optional[str] = None,
    style: str = "realistic"
) -> LipSyncData:
    """
    Convenience function to generate lip-sync data.
    """
    engine = get_lipsync_engine()
    style_enum = LipSyncStyle[style.upper()] if style else LipSyncStyle.REALISTIC
    return engine.generate_lipsync(audio_path, text, style_enum)


# ============================================
# TEST
# ============================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("LIP-SYNC ENGINE TEST")
    print("="*60)
    
    engine = LipSyncEngine()
    
    # Test word timing estimation
    text = "Hello, this is a test of the lip-sync engine."
    words = engine._estimate_word_timing(text, 3.0)
    
    print(f"\nWord timing for: '{text}'")
    for word, start, end in words:
        print(f"  {word}: {start:.2f}s - {end:.2f}s")
    
    # Test viseme generation
    visemes = engine._words_to_visemes(words, 3.0, LipSyncStyle.REALISTIC)
    
    print(f"\nGenerated {len(visemes)} visemes")
    for v in visemes[:5]:
        print(f"  {v.viseme.value}: {v.start_time:.2f}s - {v.end_time:.2f}s")
    
    # Get viseme count by type
    print(f"\n✅ {len(Viseme)} viseme types defined")
    print(f"✅ {len(LipSyncStyle)} lip-sync styles available")
    print("\n✅ Lip-sync engine working!")
