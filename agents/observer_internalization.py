"""
Observer Internalization

Pipeline for training local vision model to replace Gemini API calls.
Strategy: Distill Gemini observations into smaller, faster local model.

Training Flow:
    1. Collect (video_frames, gemini_observation) pairs
    2. Train local vision encoder + observation head
    3. Evaluate against held-out Gemini observations
    4. Deploy with confidence threshold fallback
"""

from __future__ import annotations

import os
import json
import logging
import time
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
    TaskContext,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class InternalizationConfig:
    """Configuration for internalization training."""
    # Data
    training_data_dir: str = "training_data/observations"
    validation_split: float = 0.1
    
    # Model architecture
    vision_encoder: str = "resnet18"  # resnet18, resnet50, vit_small
    hidden_dim: int = 512
    num_character_slots: int = 8
    
    # Training
    batch_size: int = 16
    learning_rate: float = 1e-4
    epochs: int = 50
    early_stopping_patience: int = 5
    
    # Output
    model_output_dir: str = "models/local_observer"
    checkpoint_every: int = 5
    
    # Evaluation
    min_quality_match: float = 0.8
    min_confidence: float = 0.7


# ============================================================================
# Data Loading
# ============================================================================

@dataclass
class TrainingSample:
    """A single training sample."""
    video_path: str
    frames: List[bytes]  # Encoded frame bytes
    observation: ObservationResult
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingSample":
        return cls(
            video_path=data["video_path"],
            frames=data.get("frames", []),
            observation=ObservationResult.from_dict(data["observation"]),
        )


class TrainingDataLoader:
    """Load and preprocess training data from Gemini observations."""
    
    def __init__(self, config: InternalizationConfig):
        self.config = config
        self.samples: List[TrainingSample] = []
    
    def load_all(self) -> int:
        """Load all training samples from disk."""
        data_dir = Path(self.config.training_data_dir)
        if not data_dir.exists():
            logger.warning(f"[data_loader] training dir not found: {data_dir}")
            return 0
        
        total = 0
        for jsonl_file in data_dir.glob("*.jsonl"):
            with open(jsonl_file) as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        sample = TrainingSample.from_dict(data)
                        self.samples.append(sample)
                        total += 1
                    except Exception as e:
                        logger.warning(f"[data_loader] failed to parse: {e}")
        
        logger.info(f"[data_loader] loaded {total} training samples")
        return total
    
    def split_train_val(self) -> Tuple[List[TrainingSample], List[TrainingSample]]:
        """Split data into training and validation sets."""
        import random
        random.shuffle(self.samples)
        
        val_size = int(len(self.samples) * self.config.validation_split)
        val_samples = self.samples[:val_size]
        train_samples = self.samples[val_size:]
        
        logger.info(f"[data_loader] train: {len(train_samples)}, val: {len(val_samples)}")
        return train_samples, val_samples
    
    def stats(self) -> Dict[str, Any]:
        """Get statistics about training data."""
        if not self.samples:
            return {"total": 0}
        
        observers = {}
        quality_scores = []
        confidences = []
        
        for sample in self.samples:
            obs = sample.observation
            observers[obs.observer_type] = observers.get(obs.observer_type, 0) + 1
            quality_scores.append(obs.get_quality_score())
            confidences.append(obs.confidence)
        
        return {
            "total": len(self.samples),
            "by_observer": observers,
            "avg_quality": sum(quality_scores) / len(quality_scores),
            "avg_confidence": sum(confidences) / len(confidences),
        }


# ============================================================================
# Local Model Architecture (Stub)
# ============================================================================

class LocalObserverModel:
    """
    Local vision model for observation extraction.
    
    Architecture:
        Frame → VisionEncoder → FrameFeatures
        FrameFeatures → TemporalPool → VideoFeatures
        VideoFeatures → CharacterHead → CharacterObservations
        VideoFeatures → EnvironmentHead → EnvironmentObservation
        VideoFeatures → ActionHead → ActionObservation
        VideoFeatures → QualityHead → QualityMetrics
    
    This is a stub - actual implementation would use PyTorch.
    """
    
    def __init__(self, config: InternalizationConfig):
        self.config = config
        self.is_trained = False
        self._model = None
        
        # Check PyTorch availability
        try:
            import torch
            self.has_torch = True
        except ImportError:
            self.has_torch = False
            logger.warning("[local_model] PyTorch not available")
    
    def build(self) -> None:
        """Build the model architecture."""
        if not self.has_torch:
            logger.warning("[local_model] cannot build without PyTorch")
            return
        
        import torch
        import torch.nn as nn
        
        # Simplified architecture for demonstration
        # Real implementation would use proper vision encoder
        
        class ObserverNetwork(nn.Module):
            def __init__(self, hidden_dim: int, num_chars: int):
                super().__init__()
                # Simple CNN for frame encoding (placeholder)
                self.encoder = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((8, 8)),
                    nn.Flatten(),
                    nn.Linear(64 * 8 * 8, hidden_dim),
                )
                
                # Output heads
                self.char_head = nn.Linear(hidden_dim, num_chars * 10)  # 10 features per char
                self.env_head = nn.Linear(hidden_dim, 16)
                self.action_head = nn.Linear(hidden_dim, 32)
                self.quality_head = nn.Linear(hidden_dim, 8)
                self.confidence_head = nn.Linear(hidden_dim, 1)
            
            def forward(self, x):
                features = self.encoder(x)
                return {
                    "characters": self.char_head(features),
                    "environment": self.env_head(features),
                    "action": self.action_head(features),
                    "quality": self.quality_head(features),
                    "confidence": torch.sigmoid(self.confidence_head(features)),
                }
        
        self._model = ObserverNetwork(
            self.config.hidden_dim,
            self.config.num_character_slots,
        )
        logger.info("[local_model] model built")
    
    def train_step(
        self,
        batch: List[TrainingSample],
        optimizer,
    ) -> float:
        """Single training step."""
        if not self.has_torch or not self._model:
            return 0.0
        
        import torch
        
        # Placeholder training step
        # Real implementation would:
        # 1. Extract frames from videos
        # 2. Forward through model
        # 3. Compare to Gemini observations
        # 4. Compute loss and backprop
        
        loss = 0.0
        return loss
    
    def predict(
        self,
        frames: List[bytes],
        context: TaskContext,
    ) -> Optional[ObservationResult]:
        """Run inference on frames."""
        if not self.has_torch or not self._model or not self.is_trained:
            return None
        
        # Placeholder inference
        # Real implementation would decode frames, run model, parse outputs
        return None
    
    def save(self, path: str) -> None:
        """Save model weights."""
        if not self.has_torch or not self._model:
            return
        
        import torch
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), path)
        logger.info(f"[local_model] saved to {path}")
    
    def load(self, path: str) -> bool:
        """Load model weights."""
        if not self.has_torch:
            return False
        
        import torch
        
        if not Path(path).exists():
            logger.warning(f"[local_model] checkpoint not found: {path}")
            return False
        
        self.build()
        self._model.load_state_dict(torch.load(path))
        self.is_trained = True
        logger.info(f"[local_model] loaded from {path}")
        return True


# ============================================================================
# Training Pipeline
# ============================================================================

class InternalizationTrainer:
    """
    Training pipeline for observer internalization.
    
    Usage:
        trainer = InternalizationTrainer()
        trainer.load_data()
        metrics = trainer.train()
        trainer.export_model()
    """
    
    def __init__(self, config: Optional[InternalizationConfig] = None):
        self.config = config or InternalizationConfig()
        self.data_loader = TrainingDataLoader(self.config)
        self.model = LocalObserverModel(self.config)
        self.training_metrics: List[Dict[str, float]] = []
    
    def load_data(self) -> Dict[str, Any]:
        """Load training data and return statistics."""
        count = self.data_loader.load_all()
        stats = self.data_loader.stats()
        return {"loaded": count, **stats}
    
    def train(self) -> Dict[str, Any]:
        """
        Train the local observer model.
        
        Returns:
            Training metrics
        """
        if not self.model.has_torch:
            return {"error": "PyTorch not available"}
        
        if len(self.data_loader.samples) < 10:
            return {"error": "Insufficient training data (need >= 10 samples)"}
        
        import torch
        
        # Build model
        self.model.build()
        
        # Split data
        train_samples, val_samples = self.data_loader.split_train_val()
        
        # Setup optimizer
        optimizer = torch.optim.Adam(
            self.model._model.parameters(),
            lr=self.config.learning_rate,
        )
        
        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0
        
        for epoch in range(self.config.epochs):
            # Training
            train_loss = 0.0
            for i in range(0, len(train_samples), self.config.batch_size):
                batch = train_samples[i:i + self.config.batch_size]
                loss = self.model.train_step(batch, optimizer)
                train_loss += loss
            
            # Validation
            val_loss = self._validate(val_samples)
            
            metrics = {
                "epoch": epoch + 1,
                "train_loss": train_loss / max(1, len(train_samples)),
                "val_loss": val_loss,
            }
            self.training_metrics.append(metrics)
            logger.info(f"[trainer] epoch {epoch + 1}: train_loss={metrics['train_loss']:.4f}, val_loss={val_loss:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.model.save(
                    Path(self.config.model_output_dir) / "best_model.pt"
                )
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"[trainer] early stopping at epoch {epoch + 1}")
                    break
            
            # Checkpoint
            if (epoch + 1) % self.config.checkpoint_every == 0:
                self.model.save(
                    Path(self.config.model_output_dir) / f"checkpoint_epoch_{epoch + 1}.pt"
                )
        
        self.model.is_trained = True
        return {
            "epochs_trained": len(self.training_metrics),
            "best_val_loss": best_val_loss,
            "training_samples": len(train_samples),
            "validation_samples": len(val_samples),
        }
    
    def _validate(self, samples: List[TrainingSample]) -> float:
        """Validate model on samples."""
        # Placeholder validation
        return 0.0
    
    def evaluate_against_gemini(self) -> Dict[str, Any]:
        """
        Evaluate local model against Gemini observations.
        
        Returns:
            Evaluation metrics including match rates
        """
        if not self.model.is_trained:
            return {"error": "Model not trained"}
        
        _, val_samples = self.data_loader.split_train_val()
        
        quality_matches = 0
        confidence_matches = 0
        total = 0
        
        for sample in val_samples:
            # Run local model
            local_obs = self.model.predict(sample.frames, TaskContext())
            if not local_obs:
                continue
            
            # Compare to Gemini observation
            gemini_obs = sample.observation
            
            quality_diff = abs(local_obs.get_quality_score() - gemini_obs.get_quality_score())
            if quality_diff < (1 - self.config.min_quality_match):
                quality_matches += 1
            
            if local_obs.confidence >= self.config.min_confidence:
                confidence_matches += 1
            
            total += 1
        
        return {
            "total_evaluated": total,
            "quality_match_rate": quality_matches / max(1, total),
            "high_confidence_rate": confidence_matches / max(1, total),
        }
    
    def export_model(self, output_path: Optional[str] = None) -> str:
        """
        Export trained model for deployment.
        
        Returns:
            Path to exported model
        """
        path = output_path or str(Path(self.config.model_output_dir) / "exported_model.pt")
        self.model.save(path)
        
        # Also save config
        config_path = Path(path).parent / "config.json"
        with open(config_path, "w") as f:
            json.dump({
                "vision_encoder": self.config.vision_encoder,
                "hidden_dim": self.config.hidden_dim,
                "num_character_slots": self.config.num_character_slots,
                "min_confidence": self.config.min_confidence,
            }, f, indent=2)
        
        logger.info(f"[trainer] exported model to {path}")
        return path


# ============================================================================
# Integration with VideoObserverAgent
# ============================================================================

def integrate_local_observer(observer_agent, model_path: str) -> bool:
    """
    Integrate trained local model into VideoObserverAgent.
    
    Args:
        observer_agent: VideoObserverAgent instance
        model_path: Path to trained model
        
    Returns:
        True if integration successful
    """
    model = LocalObserverModel(InternalizationConfig())
    if model.load(model_path):
        observer_agent.local_model = model
        observer_agent.config.use_local = True
        observer_agent.config.local_model_path = model_path
        logger.info("[integrate] local model integrated successfully")
        return True
    return False


# ============================================================================
# Convenience Functions
# ============================================================================

def check_training_readiness() -> Dict[str, Any]:
    """
    Check if system is ready for internalization training.
    
    Returns:
        Readiness status and any issues
    """
    issues = []
    
    # Check PyTorch
    try:
        import torch
        torch_ok = True
    except ImportError:
        torch_ok = False
        issues.append("PyTorch not installed")
    
    # Check training data
    config = InternalizationConfig()
    data_dir = Path(config.training_data_dir)
    if not data_dir.exists():
        issues.append(f"Training data directory not found: {data_dir}")
        sample_count = 0
    else:
        sample_count = sum(1 for f in data_dir.glob("*.jsonl") for _ in open(f))
        if sample_count < 100:
            issues.append(f"Need more training data: {sample_count}/100 minimum")
    
    return {
        "ready": len(issues) == 0,
        "pytorch_available": torch_ok,
        "training_samples": sample_count,
        "issues": issues,
    }


def train_observer(
    epochs: int = 50,
    min_samples: int = 100,
) -> Dict[str, Any]:
    """
    Convenience function to train local observer.
    
    Args:
        epochs: Number of training epochs
        min_samples: Minimum samples required
        
    Returns:
        Training results
    """
    readiness = check_training_readiness()
    if not readiness["ready"]:
        return {"error": "Not ready", **readiness}
    
    config = InternalizationConfig(epochs=epochs)
    trainer = InternalizationTrainer(config)
    
    data_stats = trainer.load_data()
    if data_stats["loaded"] < min_samples:
        return {"error": f"Insufficient data: {data_stats['loaded']}/{min_samples}"}
    
    train_results = trainer.train()
    eval_results = trainer.evaluate_against_gemini()
    model_path = trainer.export_model()
    
    return {
        "data_stats": data_stats,
        "train_results": train_results,
        "eval_results": eval_results,
        "model_path": model_path,
    }
