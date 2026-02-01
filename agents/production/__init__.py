"""
Production Package

Production-ready video generation with:
- Multi-backend fallback chains
- Quality gates and validation
- Health monitoring
"""

from agents.production.production_controller import (
    # Controller
    ProductionController,
    get_production_controller,
    generate_video_production,
    
    # Request/Result
    GenerationRequest,
    GenerationResult,
    
    # Quality
    QualityMetrics,
    QualityGate,
    QualityLevel,
    QUALITY_GATES,
    
    # Backend
    Backend,
    BackendStatus,
    BackendConfig,
    DEFAULT_BACKEND_CONFIGS,
)

__all__ = [
    # Controller
    "ProductionController",
    "get_production_controller",
    "generate_video_production",
    
    # Request/Result
    "GenerationRequest",
    "GenerationResult",
    
    # Quality
    "QualityMetrics",
    "QualityGate",
    "QualityLevel",
    "QUALITY_GATES",
    
    # Backend
    "Backend",
    "BackendStatus",
    "BackendConfig",
    "DEFAULT_BACKEND_CONFIGS",
]
