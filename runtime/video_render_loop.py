"""
Video Render Loop - Complete Pipeline Orchestration

Integrates all Phase 1-4 components into a closed loop:
    1. Story Director → decides next beat
    2. Action Reactor → selects moment-to-moment actions
    3. GPU Worker → renders video
    4. Video Observer → extracts observations
    5. Quality Evaluator → assesses quality
    6. Budget Controller → manages retry/proceed decisions
    7. World State Graph → updates state
    8. Story Director → decides next beat (loop)

This is the production-grade integration layer that closes
the video→observation→decision loop.
"""

from __future__ import annotations

import os
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from datetime import datetime
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class RenderLoopConfig:
    """Configuration for the video render loop."""
    # Polling
    poll_interval_sec: float = 0.5
    max_wait_sec: float = 300.0
    
    # Quality
    quality_task_type: str = "storytelling"
    
    # Budget
    max_retries_per_beat: int = 3
    max_episode_budget_usd: float = 10.0
    
    # Observer
    use_mock_observer: bool = True  # For testing without Gemini API
    
    # Rendering
    mock_render: bool = False  # For testing without GPU
    mock_render_delay_sec: float = 0.1


class LoopState(Enum):
    """State of the render loop."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETE = "complete"
    ERROR = "error"


# ============================================================================
# Loop Status
# ============================================================================

@dataclass
class LoopStatus:
    """Current status of the render loop."""
    state: LoopState = LoopState.IDLE
    
    # Progress
    beats_completed: int = 0
    beats_failed: int = 0
    beats_pending: int = 0
    
    # Current
    current_beat_id: Optional[str] = None
    current_attempt: int = 0
    
    # Timing
    started_at: Optional[datetime] = None
    last_update: Optional[datetime] = None
    
    # Budget
    budget_spent_usd: float = 0.0
    budget_remaining_usd: float = 0.0
    
    # Errors
    last_error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "beats_completed": self.beats_completed,
            "beats_failed": self.beats_failed,
            "beats_pending": self.beats_pending,
            "current_beat_id": self.current_beat_id,
            "budget_spent_usd": self.budget_spent_usd,
            "elapsed_sec": (
                (datetime.utcnow() - self.started_at).total_seconds()
                if self.started_at else 0
            ),
        }


# ============================================================================
# Video Render Loop
# ============================================================================

class VideoRenderLoop:
    """
    Production-grade video render loop that integrates all components.
    
    This is the main orchestrator that:
    1. Gets the next beat from StoryDirector
    2. Selects actions via ActionReactor
    3. Submits render jobs
    4. Observes rendered videos
    5. Evaluates quality
    6. Manages budget and retries
    7. Updates world state
    8. Repeats until complete
    
    Usage:
        loop = VideoRenderLoop(
            episode_id="ep-001",
            intent_graph=intent_graph,
            redis_client=redis,
        )
        
        result = loop.run()
        print(f"Completed: {result['beats_completed']} beats")
    """
    
    def __init__(
        self,
        episode_id: str,
        intent_graph: Optional[Any] = None,  # StoryIntentGraph
        redis_client: Optional[Any] = None,
        config: Optional[RenderLoopConfig] = None,
    ):
        self.episode_id = episode_id
        self.config = config or RenderLoopConfig()
        self.redis = redis_client
        
        # Status
        self.status = LoopStatus()
        
        # Components (lazy initialization)
        # Components (lazy initialization)
        self._intent_graph = intent_graph
        self._policy_engine = None # Was _story_director
        self._action_reactor = None
        self._world_graph = None
        self._observer = None
        self._quality_evaluator = None
        self._budget_controller = None
        self._value_estimator = None
        
        # Event callbacks
        self._on_beat_complete: Optional[Callable] = None
        self._on_beat_failed: Optional[Callable] = None
        self._on_state_update: Optional[Callable] = None
        
        logger.info(f"[video_render_loop] initialized for {episode_id}")
    
    # =========================================================================
    # Lazy Component Initialization
    # =========================================================================
    
    @property
    def policy_engine(self):
        if self._policy_engine is None:
            from agents.policy_engine import PolicyEngine
            from models.goal_graph import GoalGraph
            
            if self._intent_graph is None:
                self._intent_graph = GoalGraph(episode_id=self.episode_id)
            
            self._policy_engine = PolicyEngine(self._intent_graph)
            
            # Hydrate from SQL if empty
            if not self._policy_engine.goal_graph.proposals:
                self._hydrate_policy_from_sql()
                
        return self._policy_engine

    @property
    def story_director(self):
        """Legacy alias."""
        return self.policy_engine
    
    @property
    def action_reactor(self):
        if self._action_reactor is None:
            from runtime.action_reactor import ActionReactor
            self._action_reactor = ActionReactor()
        return self._action_reactor
    
    @property
    def world_graph(self):
        if self._world_graph is None:
            from models.world_state_graph import WorldStateGraph, WorldState
            self._world_graph = WorldStateGraph(episode_id=self.episode_id)
            self._world_graph.initialize(WorldState())
        return self._world_graph
    
    @property
    def observer(self):
        if self._observer is None:
            from agents.video_observer import VideoObserverAgent, ObserverConfig
            # Use mock (no Gemini) by default for testing
            config = ObserverConfig(
                use_gemini=not self.config.use_mock_observer,
                fallback_enabled=os.getenv("OBSERVER_FALLBACK_ENABLED", "").lower() in ("true", "1", "yes"),
                ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                ollama_model=os.getenv("OLLAMA_MODEL", "llava"),
            )
            self._observer = VideoObserverAgent(config=config)
        return self._observer
    
    @property
    def quality_evaluator(self):
        if self._quality_evaluator is None:
            from runtime.quality_evaluator import QualityEvaluator
            self._quality_evaluator = QualityEvaluator()
        return self._quality_evaluator
    
    @property
    def budget_controller(self):
        if self._budget_controller is None:
            from runtime.budget_controller import BudgetController, BudgetConfig
            config = BudgetConfig(
                max_episode_budget_usd=self.config.max_episode_budget_usd,
                max_beat_retries=self.config.max_retries_per_beat,
            )
            self._budget_controller = BudgetController(
                episode_id=self.episode_id,
                config=config,
            )
        return self._budget_controller
    
    @property
    def value_estimator(self):
        if self._value_estimator is None:
            from runtime.value_estimator import ValueEstimator
            self._value_estimator = ValueEstimator()
        return self._value_estimator
    
    # =========================================================================
    # Main Run Loop
    # =========================================================================
    
    def run(self) -> Dict[str, Any]:
        """
        Run the complete video render loop.
        
        Returns:
            Dict with completion status and metrics
        """
        self.status.state = LoopState.RUNNING
        self.status.started_at = datetime.utcnow()
        
        logger.info(f"[video_render_loop] starting episode {self.episode_id}")
        
        try:
            while not self._is_complete():
                self._process_one_cycle()
                self.status.last_update = datetime.utcnow()
            
            self.status.state = LoopState.COMPLETE
            logger.info(
                f"[video_render_loop] completed: "
                f"{self.status.beats_completed} beats, "
                f"${self.status.budget_spent_usd:.2f} spent"
            )
            
        except Exception as e:
            self.status.state = LoopState.ERROR
            self.status.last_error = str(e)
            logger.error(f"[video_render_loop] error: {e}")
            raise
        
        return self.get_result()
    
    def _is_complete(self) -> bool:
        """Check if loop should complete."""
        # Check story director
        if self.policy_engine.is_complete():
            return True
        
        # Check budget exhaustion
        if self.budget_controller.state.remaining_budget_usd <= 0:
            logger.warning("[video_render_loop] budget exhausted")
            return True
        
        return False
    
    def _process_one_cycle(self) -> None:
        """Process one cycle of the render loop."""
        # 1. Get next decision from policy engine
        from agents.policy_engine import DecisionType
        
        world_state = self._get_current_world_state()
        decision = self.policy_engine.next_decision(world_state)
        
        if decision.decision_type == DecisionType.COMPLETE:
            return
            
        if decision.decision_type == DecisionType.WAIT:
            logger.info("[video_render_loop] policy decision: WAIT (rational inaction)")
            return
        
        # Map: proposal_id -> beat_id (for legacy compatibility)
        beat_id = decision.proposal_id or decision.decision_id
        
        if not beat_id:
            logger.warning("[video_render_loop] decision has no proposal_id")
            return
        
        self.status.current_beat_id = beat_id
        self.status.current_attempt = 0
        
        # 2. Request budget
        from runtime.budget_controller import BudgetDecision
        
        estimate = self.value_estimator.estimate_from_beat(
            beat_id=beat_id,
            information_gain_potential=1.0,  # Default
            constraint_complexity=1.0,       # Default
            is_branch_point=getattr(decision, 'is_branch_point', False),
            downstream_beats=getattr(decision, 'downstream_beats', 0),
        )
        
        budget_result = self.budget_controller.request_budget(
            beat_id=beat_id,
            expected_value=estimate.expected_value,
        )
        
        if budget_result.decision == BudgetDecision.ABORT:
            logger.warning(f"[video_render_loop] budget rejected for {beat_id}")
            self._mark_beat_failed(beat_id, "Budget rejected")
            return
        
        # 3. Render loop with retries
        success = self._render_with_retries(beat_id, decision)
        
        if success:
            self.status.beats_completed += 1
            if self._on_beat_complete:
                self._on_beat_complete(beat_id)
        else:
            self.status.beats_failed += 1
            if self._on_beat_failed:
                self._on_beat_failed(beat_id)
    
    def _render_with_retries(
        self,
        beat_id: str,
        decision: Any,
    ) -> bool:
        """
        Render a beat with quality-based retries.
        
        Returns:
            True if beat completed successfully
        """
        from runtime.budget_controller import BudgetDecision
        from runtime.quality_evaluator import (
            QualityScores, EvaluationContext, TaskType
        )
        
        max_attempts = self.config.max_retries_per_beat
        
        for attempt in range(max_attempts):
            self.status.current_attempt = attempt + 1
            
            # Render video
            video_uri = self._render_video(beat_id, decision)
            if not video_uri:
                logger.warning(f"[video_render_loop] render failed for {beat_id}")
                continue
            
            # Observe video
            observation = self._observe_video(video_uri, beat_id)
            
            # Evaluate quality
            quality_result = self._evaluate_quality(observation, beat_id)
            
            # Record attempt
            cost = 0.05  # Default render cost
            self.budget_controller.record_attempt(
                beat_id=beat_id,
                success=quality_result.is_acceptable,
                cost_usd=cost,
                quality_score=quality_result.overall_score,
            )
            self.status.budget_spent_usd += cost
            
            if quality_result.is_acceptable:
                # Update world state
                self._update_world_state(video_uri, observation, beat_id, quality_result)
                
                # Record outcome with director
                self.policy_engine.record_outcome(
                    beat_id=beat_id,
                    observation=observation,
                    quality_result={"is_acceptable": True},
                    video_uri=video_uri,
                    proposal_id=beat_id
                )
                
                return True
            
            # Check if we should retry
            retry_decision = self.budget_controller.should_retry(
                beat_id=beat_id,
                quality_score=quality_result.overall_score,
            )
            
            if retry_decision.decision != BudgetDecision.RETRY:
                break
            
            logger.info(
                f"[video_render_loop] retrying {beat_id} "
                f"(attempt {attempt + 2}, score={quality_result.overall_score:.2f})"
            )
        
        # Failed after all attempts
        self._mark_beat_failed(beat_id, "Quality threshold not met")
        return False
    
    # =========================================================================
    # Rendering
    # =========================================================================
    
    def _render_video(
        self,
        beat_id: str,
        decision: Any,
    ) -> Optional[str]:
        """
        Render video for a beat.
        
        Returns:
            Video URI or None if failed
        """
        if self.config.mock_render:
            # Mock rendering for testing
            time.sleep(self.config.mock_render_delay_sec)
            return f"mock://videos/{self.episode_id}/{beat_id}.mp4"
        
        # Real rendering via Redis queue
        if not self.redis:
            logger.warning("[video_render_loop] no redis client, using mock")
            return f"mock://videos/{self.episode_id}/{beat_id}.mp4"
        
        # Submit job
        job_id = str(uuid.uuid4())
        gpu_job = self._build_gpu_job(beat_id, decision, job_id)
        
        result_queue = f"storyworld:gpu:results:{self.episode_id}"
        gpu_job["meta"] = {"result_queue": result_queue}
        
        self.redis.rpush("storyworld:gpu:jobs", json.dumps(gpu_job))
        
        # Wait for result
        start_time = time.time()
        while time.time() - start_time < self.config.max_wait_sec:
            result = self.redis.blpop(result_queue, timeout=1)
            if result:
                _, payload = result
                result_data = json.loads(payload)
                
                if result_data.get("job_id") == job_id:
                    if result_data.get("status") == "success":
                        return result_data.get("artifacts", {}).get("video")
                    return None
        
        logger.warning(f"[video_render_loop] timeout waiting for {beat_id}")
        return None
    
    def _build_gpu_job(
        self,
        beat_id: str,
        decision: Any,
        job_id: str,
    ) -> Dict[str, Any]:
        """Build GPU job payload."""
        # Get render hints from director
        hints = self.policy_engine.get_render_hints()
        
        return {
            "job_id": job_id,
            "beat_id": beat_id,
            "episode_id": self.episode_id,
            "actions": [a.to_dict() for a in getattr(decision, 'actions', [])],
            "characters": hints.get("characters", []),
            "location": hints.get("location"),
            "style_hints": hints.get("style_hints", {}),
        }
    
    # =========================================================================
    # Observation
    # =========================================================================
    
    def _observe_video(
        self,
        video_uri: str,
        beat_id: str,
    ) -> Dict[str, Any]:
        """
        Observe video and extract structured observations.
        
        Returns:
            Observation dict
        """
        try:
            observation_result = self.observer.observe(
                video_uri=video_uri,
                beat_id=beat_id,
            )
            return observation_result.to_dict()
        except Exception as e:
            logger.warning(f"[video_render_loop] observation failed: {e}")
            return {}
    
    # =========================================================================
    # Quality Evaluation
    # =========================================================================
    
    def _evaluate_quality(
        self,
        observation: Dict[str, Any],
        beat_id: str,
    ) -> Any:
        """Evaluate quality of rendered video."""
        from runtime.quality_evaluator import (
            QualityScores, EvaluationContext, TaskType
        )
        from models.observation import QualityMetrics
        
        # Extract quality metrics from observation
        quality_data = observation.get("quality", {})
        
        scores = QualityScores(
            visual_clarity=quality_data.get("visual_clarity", 0.5),
            motion_smoothness=quality_data.get("motion_smoothness", 0.5),
            temporal_coherence=quality_data.get("temporal_coherence", 0.5),
            style_consistency=quality_data.get("style_consistency", 0.5),
            action_clarity=quality_data.get("action_clarity", 0.5),
            character_recognizability=quality_data.get("character_recognizability", 0.5),
            narrative_coherence=quality_data.get("narrative_coherence", 0.5),
            continuity=quality_data.get("continuity", 0.5),
        )
        
        context = EvaluationContext(
            task_type=TaskType(self.config.quality_task_type),
            beat_id=beat_id,
        )
        
        return self.quality_evaluator.evaluate(scores, context)
    
    # =========================================================================
    # World State
    # =========================================================================
    
    def _get_current_world_state(self) -> Dict[str, Any]:
        """Get current world state as dict."""
        return self.world_graph.current.world_state.to_dict()
    
    def _update_world_state(
        self,
        video_uri: str,
        observation: Dict[str, Any],
        beat_id: str,
        quality_result: Any,
    ) -> None:
        """Update world state graph with observation."""
        self.world_graph.transition(
            video_uri=video_uri,
            observation=observation,
            beat_id=beat_id,
            quality_score=quality_result.overall_score,
        )
        
        if self._on_state_update:
            self._on_state_update(self.world_graph.current)
    
    def _mark_beat_failed(self, beat_id: str, error: str) -> None:
        """Mark a beat as failed."""
        self.policy_engine.record_outcome(
            beat_id=beat_id,
            quality_result={"is_acceptable": False},
            proposal_id=beat_id
        )
        logger.warning(f"[video_render_loop] beat {beat_id} failed: {error}")
    
    # =========================================================================
    # Results & Callbacks
    # =========================================================================
    
    def get_result(self) -> Dict[str, Any]:
        """Get final result of the loop."""
        return {
            "episode_id": self.episode_id,
            "state": self.status.state.value,
            "beats_completed": self.status.beats_completed,
            "beats_failed": self.status.beats_failed,
            "budget_spent_usd": self.status.budget_spent_usd,
            "world_graph_depth": self.world_graph.current.depth,
            "budget_spent_usd": self.status.budget_spent_usd,
            "world_graph_depth": self.world_graph.current.depth,
            "director_progress": self.policy_engine.get_progress(),
        }
    
    
    def _hydrate_policy_from_sql(self) -> None:
        """Hydrate policy engine goal graph from SQL beats."""
        if not self._policy_engine:
            return

        try:
            beats = self.sql.get_beats(self.episode_id)
        except Exception as e:
            logger.warning(f"[video_render_loop] failed to load beats: {e}")
            return

        from models.goal_graph import create_action_proposal, ProposalStatus, ActionProposal
        
        count = 0
        for beat in beats:
            # Skip if already exists
            if beat["beat_id"] in self._policy_engine.goal_graph.proposals:
                continue
                
            spec = beat.get("spec", {})
            if isinstance(spec, str):
                import json
                try:
                    spec = json.loads(spec)
                except:
                    spec = {}
            
            # Map state
            status = ProposalStatus.PENDING
            if beat.get("state") == "ACCEPTED":
                status = ProposalStatus.COMPLETED
            elif beat.get("state") == "ABORTED":
                status = ProposalStatus.FAILED
            
            # Create proposal
            proposal = ActionProposal(
                proposal_id=beat["beat_id"],
                description=spec.get("description", "Unknown Action"),
                objectives=spec.get("objectives", []),
                participants=spec.get("characters", []),
                context_location=spec.get("location"),
                status=status,
                is_optional=spec.get("is_optional", False),
            )
            
            self._policy_engine.goal_graph.add_proposal(proposal)
            count += 1
            
        if count > 0:
            logger.info(f"[video_render_loop] hydrated {count} proposals from SQL")

    def on_beat_complete(self, callback: Callable[[str], None]) -> None:
        """Register callback for beat completion."""
        self._on_beat_complete = callback
    
    def on_beat_failed(self, callback: Callable[[str], None]) -> None:
        """Register callback for beat failure."""
        self._on_beat_failed = callback
    
    def on_state_update(self, callback: Callable[[Any], None]) -> None:
        """Register callback for state updates."""
        self._on_state_update = callback


# ============================================================================
# Enhanced Decision Loop
# ============================================================================

class EnhancedDecisionLoop:
    """
    Enhanced decision loop that integrates with the video render loop.
    
    This extends the existing RuntimeDecisionLoop with:
    - Video observation after render
    - Quality evaluation
    - Budget control
    - World state updates
    """
    
    def __init__(
        self,
        runtime,
        gpu_job_queue: str,
        gpu_result_queue: str,
        redis_client,
        poll_interval: float = 0.5,
    ):
        self.runtime = runtime
        self.redis = redis_client
        self.gpu_job_queue = gpu_job_queue
        self.gpu_result_queue = gpu_result_queue
        self.poll_interval = poll_interval
        
        self.active_jobs: Dict[str, str] = {}
        
        # Phase 1-4 components
        self._world_graph = None
        self._observer = None
        self._quality_evaluator = None
        self._budget_controller = None
        self._story_director = None
    
    @property
    def world_graph(self):
        if self._world_graph is None:
            from models.world_state_graph import WorldStateGraph, WorldState
            self._world_graph = WorldStateGraph(episode_id=self.runtime.episode_id)
            self._world_graph.initialize(WorldState())
        return self._world_graph
    
    @property
    def observer(self):
        if self._observer is None:
            from agents.video_observer import VideoObserverAgent, ObserverConfig
            config = ObserverConfig(use_gemini=False)
            self._observer = VideoObserverAgent(config=config)
        return self._observer
    
    @property
    def quality_evaluator(self):
        if self._quality_evaluator is None:
            from runtime.quality_evaluator import QualityEvaluator
            self._quality_evaluator = QualityEvaluator()
        return self._quality_evaluator
    
    @property
    def budget_controller(self):
        if self._budget_controller is None:
            from runtime.budget_controller import BudgetController
            self._budget_controller = BudgetController(
                episode_id=self.runtime.episode_id,
            )
        return self._budget_controller
    
    def run(self):
        """Run the enhanced decision loop."""
        print(f"[enhanced] decision loop started for {self.runtime.episode_id}")
        
        self._submit_ready_beats()
        
        while not self.runtime.is_terminal():
            self._consume_gpu_results()
            self._submit_ready_beats()
            time.sleep(self.poll_interval)
        
        print(f"[enhanced] episode {self.runtime.episode_id} completed")
    
    def _submit_ready_beats(self):
        """Submit ready beats with budget check."""
        from runtime.budget_controller import BudgetDecision
        
        ready_beats = self.runtime.get_executable_beats()
        
        for beat in ready_beats:
            beat_id = beat.get("beat_id") or beat.get("id")
            if not beat_id:
                continue
            
            if beat_id in self.active_jobs.values():
                continue
            
            # Check budget
            budget_result = self.budget_controller.request_budget(beat_id=beat_id)
            if budget_result.decision == BudgetDecision.ABORT:
                print(f"[enhanced] budget rejected for {beat_id}")
                continue
            
            job_id = str(uuid.uuid4())
            gpu_job = self.runtime.build_gpu_job(beat_id=beat_id, job_id=job_id)
            
            meta = gpu_job.get("meta") or {}
            meta["result_queue"] = self.gpu_result_queue
            gpu_job["meta"] = meta
            
            self.redis.rpush(self.gpu_job_queue, json.dumps(gpu_job))
            self.active_jobs[job_id] = beat_id
            
            print(f"[enhanced] submitted beat {beat_id} as job {job_id}")
    
    def _consume_gpu_results(self):
        """Consume GPU results with observation and quality check."""
        result = self.redis.blpop(self.gpu_result_queue, timeout=1)
        if not result:
            return
        
        _, payload = result
        result_data = json.loads(payload)
        
        job_id = result_data.get("job_id")
        if job_id not in self.active_jobs:
            return
        
        beat_id = self.active_jobs.pop(job_id)
        self._handle_result_enhanced(beat_id, result_data)
    
    def _handle_result_enhanced(self, beat_id: str, result: dict):
        """Handle result with video observation and quality evaluation."""
        from runtime.quality_evaluator import QualityScores, EvaluationContext, TaskType
        from runtime.budget_controller import BudgetDecision
        
        status = result.get("status")
        artifacts = result.get("artifacts", {})
        video_uri = artifacts.get("video")
        
        if status != "success" or not video_uri:
            self._handle_failure(beat_id, result.get("error", "Render failed"))
            return
        
        # 1. Observe video
        try:
            observation = self.observer.observe(video_uri, beat_id)
            obs_dict = observation.to_dict()
        except Exception as e:
            obs_dict = {}
        
        # 2. Evaluate quality
        quality_data = obs_dict.get("quality", {})
        scores = QualityScores(
            visual_clarity=quality_data.get("visual_clarity", 0.7),
            motion_smoothness=quality_data.get("motion_smoothness", 0.7),
            action_clarity=quality_data.get("action_clarity", 0.7),
            character_recognizability=quality_data.get("character_recognizability", 0.7),
        )
        
        context = EvaluationContext(task_type=TaskType.STORYTELLING, beat_id=beat_id)
        quality_result = self.quality_evaluator.evaluate(scores, context)
        
        # 3. Record attempt
        self.budget_controller.record_attempt(
            beat_id=beat_id,
            success=quality_result.is_acceptable,
            cost_usd=0.05,
            quality_score=quality_result.overall_score,
        )
        
        if quality_result.is_acceptable:
            # 4. Update world state
            self.world_graph.transition(
                video_uri=video_uri,
                observation=obs_dict,
                beat_id=beat_id,
                quality_score=quality_result.overall_score,
            )
            
            # 5. Mark success
            self.runtime.mark_beat_success(
                beat_id=beat_id,
                artifacts=artifacts,
                metrics=result.get("runtime", {}),
            )
            print(f"[enhanced] beat {beat_id} succeeded (quality={quality_result.overall_score:.2f})")
        else:
            # Check if should retry
            retry_result = self.budget_controller.should_retry(
                beat_id=beat_id,
                quality_score=quality_result.overall_score,
            )
            
            if retry_result.decision == BudgetDecision.RETRY:
                # Re-submit for retry
                print(f"[enhanced] retrying {beat_id} (quality={quality_result.overall_score:.2f})")
                self._resubmit_beat(beat_id)
            else:
                self._handle_failure(beat_id, "Quality threshold not met")
    
    def _handle_failure(self, beat_id: str, error: str):
        """Handle beat failure."""
        self.runtime.mark_beat_failure(beat_id=beat_id, error=error, metrics={})
        print(f"[enhanced] beat {beat_id} failed: {error}")
    
    def _resubmit_beat(self, beat_id: str):
        """Resubmit a beat for retry."""
        job_id = str(uuid.uuid4())
        gpu_job = self.runtime.build_gpu_job(beat_id=beat_id, job_id=job_id)
        
        meta = gpu_job.get("meta") or {}
        meta["result_queue"] = self.gpu_result_queue
        gpu_job["meta"] = meta
        
        self.redis.rpush(self.gpu_job_queue, json.dumps(gpu_job))
        self.active_jobs[job_id] = beat_id


# ============================================================================
# Convenience Functions
# ============================================================================

def create_render_loop(
    episode_id: str,
    intent_graph: Optional[Any] = None,
    redis_client: Optional[Any] = None,
    mock: bool = True,
) -> VideoRenderLoop:
    """
    Factory to create VideoRenderLoop.
    
    Args:
        episode_id: Episode ID
        intent_graph: Optional StoryIntentGraph
        redis_client: Optional Redis client
        mock: Use mock rendering/observation
        
    Returns:
        Configured VideoRenderLoop
    """
    config = RenderLoopConfig(
        mock_render=mock,
        use_mock_observer=mock,
    )
    
    return VideoRenderLoop(
        episode_id=episode_id,
        intent_graph=intent_graph,
        redis_client=redis_client,
        config=config,
    )


def run_episode(
    episode_id: str,
    beats: List[Dict[str, Any]],
    mock: bool = True,
) -> Dict[str, Any]:
    """
    Run a complete episode.
    
    Args:
        episode_id: Episode ID
        beats: List of beat configurations
        mock: Use mock rendering
        
    Returns:
        Episode result
    """
    from models.goal_graph import GoalGraph, ActionProposal, SimulationGoal
    from models.simulation_goal import GoalType
    
    # Build intent graph
    intent_graph = StoryIntentGraph(episode_id=episode_id)
    
    intent_graph.add_macro_intent(MacroIntent(
        intent_id="episode-goal",
        goal_type=NarrativeGoalType.PLOT_MILESTONE,
        description="Complete episode",
    ))
    
    for i, beat_config in enumerate(beats):
        intent_graph.add_story_beat(StoryBeat(
            beat_id=beat_config.get("beat_id", f"beat-{i+1}"),
            description=beat_config.get("description", f"Scene {i+1}"),
            characters=beat_config.get("characters", []),
            suggested_position=i + 1,
            contributes_to=["episode-goal"],
        ))
    
    # Create and run loop
    loop = create_render_loop(
        episode_id=episode_id,
        intent_graph=intent_graph,
        mock=mock,
    )
    
    return loop.run()
