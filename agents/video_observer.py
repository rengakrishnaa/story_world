"""
Video Observer Agent

Watches rendered video and extracts structured observations.
Bootstrap with Gemini Vision, then internalize to local model.

Architecture:
    Video → Observer → ObservationResult → WorldStateGraph
"""

from __future__ import annotations

import os
import json
import uuid
import time
import base64
import logging
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field

from models.observation import (
    ObservationResult,
    CharacterObservation,
    EnvironmentObservation,
    ActionObservation,
    QualityMetrics,
    ContinuityError,
    ContinuityErrorType,
    ActionOutcome,
    EmotionState,
    TaskContext,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ObserverConfig:
    """Configuration for VideoObserverAgent."""
    # Model selection
    use_gemini: bool = True
    gemini_model: str = "gemini-2.0-flash"
    
    # Local model (for internalization)
    use_local: bool = False
    local_model_path: Optional[str] = None
    local_confidence_threshold: float = 0.7
    
    # Frame extraction
    frames_to_analyze: int = 5  # Key frames per video
    frame_extraction_method: str = "uniform"  # uniform, keyframe, scene_change
    
    # Retry configuration
    max_retries: int = 3
    retry_delay_sec: float = 1.0
    
    # Training data
    record_for_training: bool = True
    training_data_dir: str = "training_data/observations"
    
    # Timeouts
    observation_timeout_sec: float = 30.0


# ============================================================================
# Frame Extraction
# ============================================================================

class FrameExtractor:
    """Extract key frames from video for analysis."""
    
    def __init__(self, method: str = "uniform"):
        self.method = method
    
    def extract_frames(
        self,
        video_path: str,
        num_frames: int = 5,
    ) -> List[Tuple[int, bytes]]:
        """
        Extract key frames from video.
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract
            
        Returns:
            List of (frame_index, frame_bytes) tuples
        """
        try:
            import cv2
        except ImportError:
            logger.warning("OpenCV not available, using stub frames")
            return self._stub_frames(num_frames)
        
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return self._stub_frames(num_frames)
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            return self._stub_frames(num_frames)
        
        # Calculate frame indices based on method
        if self.method == "uniform":
            indices = self._uniform_indices(total_frames, num_frames)
        else:
            indices = self._uniform_indices(total_frames, num_frames)
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Encode as JPEG
                _, buffer = cv2.imencode('.jpg', frame)
                frames.append((idx, buffer.tobytes()))
        
        cap.release()
        return frames
    
    def _uniform_indices(self, total: int, count: int) -> List[int]:
        """Get uniformly distributed frame indices."""
        if count >= total:
            return list(range(total))
        step = total / count
        return [int(i * step) for i in range(count)]
    
    def _stub_frames(self, count: int) -> List[Tuple[int, bytes]]:
        """Return stub frames when video processing unavailable."""
        # Return empty list - observer will use text-based analysis
        return []


# ============================================================================
# Gemini Observer
# ============================================================================

class GeminiObserver:
    """
    Observe video using Google Gemini Vision API.
    This is the bootstrap observer before internalization.
    """
    
    OBSERVATION_PROMPT = """You are a video analysis AI for anime/game storytelling.
Analyze this video and extract structured observations.

Return ONLY valid JSON in this exact format:
{
    "characters": {
        "<character_id>": {
            "visible": true/false,
            "position": {"x": 0.5, "y": 0.5},
            "pose": "standing|sitting|running|fighting|etc",
            "emotion": "neutral|happy|angry|sad|fearful|surprised|determined|defeated|excited|confused",
            "motion_intensity": 0.0-1.0,
            "appearance_consistent": true/false
        }
    },
    "environment": {
        "location_description": "brief description",
        "time_of_day": "dawn|morning|noon|afternoon|dusk|night",
        "lighting": "bright|dim|dramatic|natural",
        "mood": "tense|peaceful|chaotic|etc"
    },
    "action": {
        "action_description": "what happened in the video",
        "outcome": "success|partial|failed|interrupted|unknown",
        "action_type": "attack|dialogue|movement|etc",
        "participants": ["character_ids"],
        "narrative_beat_achieved": true/false,
        "narrative_implications": ["implication1", "implication2"]
    },
    "quality": {
        "visual_clarity": 0.0-1.0,
        "motion_smoothness": 0.0-1.0,
        "temporal_coherence": 0.0-1.0,
        "style_consistency": 0.0-1.0,
        "action_clarity": 0.0-1.0,
        "character_recognizability": 0.0-1.0,
        "artifacts_detected": 0
    },
    "continuity_errors": [
        {
            "error_type": "character_missing|location_mismatch|etc",
            "description": "what's wrong",
            "severity": 0.0-1.0,
            "affected_entities": ["entity_ids"]
        }
    ],
    "confidence": 0.0-1.0
}

CONTEXT:
- Expected characters: {expected_characters}
- Expected action: {expected_action}
- Expected location: {expected_location}
- Previous state: {previous_state}

Analyze the video frames provided and return ONLY the JSON, no explanation."""

    def __init__(self, config: ObserverConfig):
        self.config = config
        self.client = None
        self._init_client()
    
    def _init_client(self) -> None:
        """Initialize Gemini client."""
        try:
            from google import genai
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                self.client = genai.Client(api_key=api_key)
                logger.info(f"[gemini_observer] initialized with {self.config.gemini_model}")
            else:
                logger.warning("[gemini_observer] GEMINI_API_KEY not set")
        except ImportError:
            logger.warning("[gemini_observer] google-genai not installed")
    
    def observe(
        self,
        frames: List[Tuple[int, bytes]],
        context: TaskContext,
    ) -> ObservationResult:
        """
        Analyze video frames and return structured observation.
        
        Args:
            frames: List of (frame_index, frame_bytes) tuples
            context: Task context with expectations
            
        Returns:
            ObservationResult with structured observations
        """
        start_time = time.time()
        observation_id = str(uuid.uuid4())
        
        if not self.client:
            logger.warning("[gemini_observer] no client, returning mock observation")
            return self._mock_observation(observation_id, context)
        
        # Build prompt with context
        prompt = self.OBSERVATION_PROMPT.format(
            expected_characters=", ".join(context.expected_characters) or "any",
            expected_action=context.expected_action or "any action",
            expected_location=context.expected_location or "any location",
            previous_state=json.dumps(context.previous_world_state) if context.previous_world_state else "none",
        )
        
        # Prepare content with frames
        contents = [prompt]
        for idx, frame_bytes in frames:
            contents.append({
                "mime_type": "image/jpeg",
                "data": base64.standard_b64encode(frame_bytes).decode("utf-8"),
            })
        
        # Call Gemini
        for attempt in range(self.config.max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.config.gemini_model,
                    contents=contents,
                )
                
                raw_text = response.text
                observation = self._parse_response(
                    raw_text,
                    observation_id,
                    context,
                    time.time() - start_time,
                )
                observation.raw_response = raw_text
                
                logger.info(
                    f"[gemini_observer] observation complete: "
                    f"quality={observation.get_quality_score():.2f}, "
                    f"confidence={observation.confidence:.2f}"
                )
                return observation
                
            except Exception as e:
                logger.warning(f"[gemini_observer] attempt {attempt+1} failed: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay_sec)
        
        # Fallback to mock
        logger.error("[gemini_observer] all attempts failed, returning mock")
        return self._mock_observation(observation_id, context)
    
    def _parse_response(
        self,
        raw_text: str,
        observation_id: str,
        context: TaskContext,
        latency_sec: float,
    ) -> ObservationResult:
        """Parse Gemini response into ObservationResult."""
        # Clean JSON from markdown
        text = raw_text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(f"[gemini_observer] JSON parse error: {e}")
            return self._mock_observation(observation_id, context)
        
        # Parse characters
        characters = {}
        for char_id, char_data in data.get("characters", {}).items():
            emotion = None
            if char_data.get("emotion"):
                try:
                    emotion = EmotionState(char_data["emotion"])
                except ValueError:
                    emotion = EmotionState.NEUTRAL
            
            characters[char_id] = CharacterObservation(
                character_id=char_id,
                visible=char_data.get("visible", True),
                position=char_data.get("position"),
                pose=char_data.get("pose"),
                emotion=emotion,
                motion_intensity=char_data.get("motion_intensity", 0.0),
                appearance_consistent=char_data.get("appearance_consistent", True),
            )
        
        # Parse environment
        env_data = data.get("environment", {})
        environment = EnvironmentObservation(
            location_description=env_data.get("location_description"),
            time_of_day=env_data.get("time_of_day"),
            lighting=env_data.get("lighting"),
            mood=env_data.get("mood"),
        )
        
        # Parse action
        action_data = data.get("action", {})
        action = None
        if action_data:
            outcome = ActionOutcome.UNKNOWN
            if action_data.get("outcome"):
                try:
                    outcome = ActionOutcome(action_data["outcome"])
                except ValueError:
                    pass
            
            action = ActionObservation(
                action_description=action_data.get("action_description", ""),
                outcome=outcome,
                action_type=action_data.get("action_type"),
                participants=action_data.get("participants", []),
                narrative_beat_achieved=action_data.get("narrative_beat_achieved", False),
                narrative_implications=action_data.get("narrative_implications", []),
            )
        
        # Parse quality
        quality_data = data.get("quality", {})
        quality = QualityMetrics(
            visual_clarity=quality_data.get("visual_clarity", 0.5),
            motion_smoothness=quality_data.get("motion_smoothness", 0.5),
            temporal_coherence=quality_data.get("temporal_coherence", 0.5),
            style_consistency=quality_data.get("style_consistency", 0.5),
            action_clarity=quality_data.get("action_clarity", 0.5),
            character_recognizability=quality_data.get("character_recognizability", 0.5),
            artifacts_detected=quality_data.get("artifacts_detected", 0),
        )
        quality.compute_overall()
        
        # Parse continuity errors
        continuity_errors = []
        for err in data.get("continuity_errors", []):
            try:
                error_type = ContinuityErrorType(err.get("error_type", "temporal_artifact"))
            except ValueError:
                error_type = ContinuityErrorType.TEMPORAL_ARTIFACT
            
            continuity_errors.append(ContinuityError(
                error_type=error_type,
                description=err.get("description", ""),
                severity=err.get("severity", 0.5),
                affected_entities=err.get("affected_entities", []),
            ))
        
        return ObservationResult(
            observation_id=observation_id,
            video_uri=context.beat_id or "unknown",
            beat_id=context.beat_id,
            created_at=datetime.utcnow(),
            observation_latency_ms=int(latency_sec * 1000),
            characters=characters,
            environment=environment,
            action=action,
            quality=quality,
            continuity_errors=continuity_errors,
            confidence=data.get("confidence", 0.7),
            observer_type="gemini",
            model_version=self.config.gemini_model,
        )
    
    def _mock_observation(
        self,
        observation_id: str,
        context: TaskContext,
    ) -> ObservationResult:
        """Return mock observation when Gemini unavailable."""
        characters = {}
        for char_id in context.expected_characters:
            characters[char_id] = CharacterObservation(
                character_id=char_id,
                visible=True,
                emotion=EmotionState.NEUTRAL,
            )
        
        return ObservationResult(
            observation_id=observation_id,
            video_uri=context.beat_id or "mock",
            beat_id=context.beat_id,
            characters=characters,
            environment=EnvironmentObservation(
                location_description=context.expected_location,
            ),
            action=ActionObservation(
                action_description=context.expected_action or "mock action",
                outcome=ActionOutcome.SUCCESS,
                narrative_beat_achieved=True,
            ),
            quality=QualityMetrics(
                overall_quality=0.75,
                visual_clarity=0.75,
                motion_smoothness=0.75,
                temporal_coherence=0.75,
                style_consistency=0.75,
                action_clarity=0.75,
                character_recognizability=0.75,
            ),
            confidence=0.5,
            observer_type="mock",
            model_version="mock",
        )


# ============================================================================
# Main Observer Agent
# ============================================================================

class VideoObserverAgent:
    """
    Main video observer agent.
    
    Strategy:
    1. Bootstrap with Gemini Vision
    2. Record observations for training
    3. After internalization, use local model with Gemini fallback
    
    Usage:
        observer = VideoObserverAgent()
        context = TaskContext(expected_characters=["saitama"])
        observation = observer.observe(video_url, context)
    """
    
    def __init__(self, config: Optional[ObserverConfig] = None):
        self.config = config or ObserverConfig()
        
        # Initialize components
        self.frame_extractor = FrameExtractor(self.config.frame_extraction_method)
        self.gemini_observer = GeminiObserver(self.config)
        self.local_model = None
        
        # Training data buffer
        self.training_buffer: List[Tuple[str, ObservationResult]] = []
        self._init_training_dir()
        
        # Load local model if available
        if self.config.use_local and self.config.local_model_path:
            self._load_local_model()
        
        logger.info(
            f"[video_observer] initialized: "
            f"gemini={self.config.use_gemini}, local={self.config.use_local}"
        )
    
    def _init_training_dir(self) -> None:
        """Initialize training data directory."""
        if self.config.record_for_training:
            Path(self.config.training_data_dir).mkdir(parents=True, exist_ok=True)
    
    def _load_local_model(self) -> None:
        """Load internalized local model."""
        # Placeholder for local model loading
        # Will be implemented in observer_internalization.py
        pass
    
    def observe(
        self,
        video_path: str,
        context: TaskContext,
    ) -> ObservationResult:
        """
        Observe video and return structured observation.
        
        Args:
            video_path: Path or URL to video
            context: Task context with expectations
            
        Returns:
            ObservationResult with structured observations
        """
        logger.info(f"[video_observer] observing: {video_path[:50]}...")
        start_time = time.time()
        
        # Download if URL
        local_path = self._ensure_local(video_path)
        
        # Extract frames
        frames = self.frame_extractor.extract_frames(
            local_path,
            self.config.frames_to_analyze,
        )
        
        # Try local model first if available and confident enough
        if self._should_use_local(context):
            observation = self._observe_local(frames, context)
            if observation and observation.confidence >= self.config.local_confidence_threshold:
                logger.info(f"[video_observer] using local model (confidence: {observation.confidence:.2f})")
                return observation
        
        # Use Gemini
        if self.config.use_gemini:
            observation = self.gemini_observer.observe(frames, context)
            
            # Record for training
            if self.config.record_for_training:
                self._record_for_training(video_path, observation)
            
            return observation
        
        # Fallback to mock
        return self.gemini_observer._mock_observation(str(uuid.uuid4()), context)
    
    def _should_use_local(self, context: TaskContext) -> bool:
        """Determine if local model should be tried."""
        if not self.config.use_local or not self.local_model:
            return False
        
        # Don't use local for high-stakes observations
        if context.is_branch_point:
            return False
        
        return True
    
    def _observe_local(
        self,
        frames: List[Tuple[int, bytes]],
        context: TaskContext,
    ) -> Optional[ObservationResult]:
        """Observe using local internalized model."""
        # Placeholder - will be implemented with actual local model
        return None
    
    def _ensure_local(self, video_path: str) -> str:
        """Ensure video is available locally."""
        if video_path.startswith(("http://", "https://")):
            # Download video
            return self._download_video(video_path)
        return video_path
    
    def _download_video(self, url: str) -> str:
        """Download video from URL to temp file."""
        try:
            import requests
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                f.write(response.content)
                return f.name
        except Exception as e:
            logger.error(f"[video_observer] download failed: {e}")
            return url  # Return URL, frame extraction will fail gracefully
    
    def _record_for_training(
        self,
        video_path: str,
        observation: ObservationResult,
    ) -> None:
        """Record observation for training internalized model."""
        if observation.observer_type != "gemini":
            return  # Only record Gemini observations
        
        self.training_buffer.append((video_path, observation))
        
        # Save to disk periodically
        if len(self.training_buffer) >= 10:
            self._flush_training_buffer()
    
    def _flush_training_buffer(self) -> None:
        """Save training buffer to disk."""
        if not self.training_buffer:
            return
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_file = Path(self.config.training_data_dir) / f"batch_{timestamp}.jsonl"
        
        with open(output_file, "w") as f:
            for video_path, observation in self.training_buffer:
                record = {
                    "video_path": video_path,
                    "observation": observation.to_dict(),
                }
                f.write(json.dumps(record) + "\n")
        
        logger.info(f"[video_observer] saved {len(self.training_buffer)} training samples to {output_file}")
        self.training_buffer.clear()
    
    def get_training_sample_count(self) -> int:
        """Get total number of training samples collected."""
        count = len(self.training_buffer)
        
        training_dir = Path(self.config.training_data_dir)
        if training_dir.exists():
            for f in training_dir.glob("*.jsonl"):
                with open(f) as file:
                    count += sum(1 for _ in file)
        
        return count


# ============================================================================
# Convenience functions
# ============================================================================

def create_observer(
    use_gemini: bool = True,
    use_local: bool = False,
    local_model_path: Optional[str] = None,
) -> VideoObserverAgent:
    """
    Factory function to create observer with common configurations.
    
    Args:
        use_gemini: Whether to use Gemini API
        use_local: Whether to use local internalized model
        local_model_path: Path to local model weights
        
    Returns:
        Configured VideoObserverAgent
    """
    config = ObserverConfig(
        use_gemini=use_gemini,
        use_local=use_local,
        local_model_path=local_model_path,
    )
    return VideoObserverAgent(config)


def quick_observe(video_path: str, expected_characters: List[str] = None) -> ObservationResult:
    """
    Quick observation with minimal configuration.
    
    Args:
        video_path: Path or URL to video
        expected_characters: List of expected character IDs
        
    Returns:
        ObservationResult
    """
    observer = VideoObserverAgent()
    context = TaskContext(
        expected_characters=expected_characters or [],
    )
    return observer.observe(video_path, context)
