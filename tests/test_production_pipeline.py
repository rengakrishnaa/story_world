"""
Production Pipeline Test Suite

Comprehensive tests for:
- Backend definitions and configuration
- Quality gates and validation
- Fallback chain generation
- Production controller
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_backend_definitions():
    """Test backend definitions."""
    print("\n" + "="*60)
    print("TEST 1: Backend Definitions")
    print("="*60)
    
    from agents.production.production_controller import Backend
    
    backends = list(Backend)
    print(f"  Total backends: {len(backends)}")
    
    expected = ["veo", "gemini", "animatediff", "svd", "sdxl", "stub"]
    for exp in expected:
        found = any(b.value == exp for b in backends)
        status = "✅" if found else "❌"
        print(f"  {status} {exp}")
    
    assert len(backends) >= 6, "Expected at least 6 backends"
    print(f"  ✅ {len(backends)} backends validated")
    return True


def test_backend_status():
    """Test backend status definitions."""
    print("\n" + "="*60)
    print("TEST 2: Backend Status")
    print("="*60)
    
    from agents.production.production_controller import BackendStatus
    
    statuses = list(BackendStatus)
    print(f"  Total statuses: {len(statuses)}")
    
    expected = ["healthy", "degraded", "unhealthy", "unavailable"]
    for exp in expected:
        found = any(s.value == exp for s in statuses)
        status = "✅" if found else "❌"
        print(f"  {status} {exp}")
    
    assert len(statuses) >= 4, "Expected at least 4 statuses"
    print(f"  ✅ {len(statuses)} backend statuses validated")
    return True


def test_quality_levels():
    """Test quality level definitions."""
    print("\n" + "="*60)
    print("TEST 3: Quality Levels")
    print("="*60)
    
    from agents.production.production_controller import QualityLevel
    
    levels = list(QualityLevel)
    print(f"  Total levels: {len(levels)}")
    
    expected = ["preview", "draft", "standard", "high", "production"]
    for exp in expected:
        found = any(l.value == exp for l in levels)
        status = "✅" if found else "❌"
        print(f"  {status} {exp}")
    
    assert len(levels) >= 5, "Expected at least 5 quality levels"
    print(f"  ✅ {len(levels)} quality levels validated")
    return True


def test_quality_gate_creation():
    """Test quality gate creation and validation."""
    print("\n" + "="*60)
    print("TEST 4: Quality Gate Creation")
    print("="*60)
    
    from agents.production.production_controller import QualityGate, QualityMetrics
    
    gate = QualityGate(
        name="test_gate",
        description="Test quality gate",
        min_resolution=(1280, 720),
        min_fps=24.0,
        min_duration=2.0,
    )
    
    # Test passing metrics
    good_metrics = QualityMetrics(
        resolution=(1920, 1080),
        fps=30.0,
        duration_seconds=5.0,
        file_size_mb=10.0,
    )
    
    passed, reason = gate.validate(good_metrics)
    assert passed, f"Should pass: {reason}"
    print(f"  ✅ Good metrics pass: {reason}")
    
    # Test failing metrics
    bad_metrics = QualityMetrics(
        resolution=(640, 360),  # Too low
        fps=30.0,
        duration_seconds=5.0,
        file_size_mb=10.0,
    )
    
    passed, reason = gate.validate(bad_metrics)
    assert not passed, "Should fail on resolution"
    print(f"  ✅ Bad metrics fail: {reason}")
    
    return True


def test_quality_gates_config():
    """Test default quality gate configurations."""
    print("\n" + "="*60)
    print("TEST 5: Quality Gates Config")
    print("="*60)
    
    from agents.production.production_controller import QUALITY_GATES, QualityLevel
    
    print(f"  Total gates: {len(QUALITY_GATES)}")
    
    for level in QualityLevel:
        assert level in QUALITY_GATES, f"Missing gate for {level.value}"
        gate = QUALITY_GATES[level]
        print(f"  ✅ {level.value}: {gate.min_resolution[0]}x{gate.min_resolution[1]}, {gate.min_fps}fps")
    
    print(f"  ✅ All {len(QUALITY_GATES)} quality gates configured")
    return True


def test_backend_configs():
    """Test default backend configurations."""
    print("\n" + "="*60)
    print("TEST 6: Backend Configs")
    print("="*60)
    
    from agents.production.production_controller import DEFAULT_BACKEND_CONFIGS, Backend
    
    print(f"  Total configs: {len(DEFAULT_BACKEND_CONFIGS)}")
    
    for backend in Backend:
        assert backend in DEFAULT_BACKEND_CONFIGS, f"Missing config for {backend.value}"
        config = DEFAULT_BACKEND_CONFIGS[backend]
        print(f"  ✅ {backend.value}: priority={config.priority}, timeout={config.timeout_seconds}s")
    
    print(f"  ✅ All {len(DEFAULT_BACKEND_CONFIGS)} backend configs present")
    return True


def test_generation_request():
    """Test generation request creation."""
    print("\n" + "="*60)
    print("TEST 7: Generation Request")
    print("="*60)
    
    from agents.production.production_controller import GenerationRequest, QualityLevel
    
    request = GenerationRequest(
        request_id="test_001",
        prompt="A hero battles a dragon",
        genre="fantasy",
        duration_seconds=10.0,
        quality_level=QualityLevel.HIGH
    )
    
    assert request.request_id == "test_001"
    assert request.genre == "fantasy"
    assert request.quality_level == QualityLevel.HIGH
    
    print(f"  ✅ Created request: {request.request_id}")
    print(f"  ✅ Prompt: {request.prompt[:30]}...")
    print(f"  ✅ Quality: {request.quality_level.value}")
    return True


def test_controller_initialization():
    """Test production controller initialization."""
    print("\n" + "="*60)
    print("TEST 8: Controller Initialization")
    print("="*60)
    
    from agents.production.production_controller import ProductionController
    
    controller = ProductionController()
    
    assert controller.backend_configs is not None
    assert controller.quality_gates is not None
    assert controller.backend_health is not None
    
    health = controller.get_health_status()
    
    print(f"  ✅ Backend configs: {len(controller.backend_configs)}")
    print(f"  ✅ Quality gates: {len(controller.quality_gates)}")
    print(f"  ✅ Health status: {health['healthy_count']}/{health['total_count']}")
    return True


def test_fallback_chain():
    """Test fallback chain generation."""
    print("\n" + "="*60)
    print("TEST 9: Fallback Chain")
    print("="*60)
    
    from agents.production.production_controller import (
        ProductionController, Backend, QualityLevel
    )
    
    controller = ProductionController()
    
    # Default chain
    chain = controller.get_fallback_chain()
    assert len(chain) > 0, "Chain should not be empty"
    assert Backend.STUB in chain, "Stub should always be in chain"
    print(f"  ✅ Default chain: {[b.value for b in chain]}")
    
    # Chain with preferred backend
    chain = controller.get_fallback_chain(preferred=Backend.SVD)
    assert chain[0] == Backend.SVD or Backend.SVD in chain
    print(f"  ✅ With preferred: {[b.value for b in chain]}")
    
    # Chain for high quality
    chain = controller.get_fallback_chain(quality_level=QualityLevel.PRODUCTION)
    print(f"  ✅ Production quality: {[b.value for b in chain]}")
    
    return True


def test_quality_metrics():
    """Test quality metrics calculation."""
    print("\n" + "="*60)
    print("TEST 10: Quality Metrics")
    print("="*60)
    
    from agents.production.production_controller import QualityMetrics
    
    metrics = QualityMetrics(
        resolution=(1920, 1080),
        fps=24.0,
        duration_seconds=8.0,
        motion_score=0.8,
        consistency_score=0.7,
        style_match_score=0.9,
    )
    
    assert metrics.overall_score > 0
    assert metrics.passed_all_gates  # No failed gates yet
    
    print(f"  ✅ Resolution: {metrics.resolution}")
    print(f"  ✅ Overall score: {metrics.overall_score:.2f}")
    print(f"  ✅ Passed all gates: {metrics.passed_all_gates}")
    return True


def test_component_inventory():
    """Test production component inventory."""
    print("\n" + "="*60)
    print("PRODUCTION COMPONENT INVENTORY")
    print("="*60)
    
    from agents.production.production_controller import (
        Backend, BackendStatus, QualityLevel
    )
    
    inventory = {
        "Backends": len(Backend),
        "Backend Statuses": len(BackendStatus),
        "Quality Levels": len(QualityLevel),
    }
    
    total = 0
    for name, count in inventory.items():
        print(f"  {name:20}: {count:3}")
        total += count
    
    print(f"\n  {'TOTAL':20}: {total}")
    
    return True


def main():
    """Run all production tests."""
    print("\n" + "#"*60)
    print("# PRODUCTION PIPELINE TEST SUITE")
    print("#"*60)
    
    results = {}
    
    tests = [
        ("backend_definitions", test_backend_definitions),
        ("backend_status", test_backend_status),
        ("quality_levels", test_quality_levels),
        ("quality_gate_creation", test_quality_gate_creation),
        ("quality_gates_config", test_quality_gates_config),
        ("backend_configs", test_backend_configs),
        ("generation_request", test_generation_request),
        ("controller_init", test_controller_initialization),
        ("fallback_chain", test_fallback_chain),
        ("quality_metrics", test_quality_metrics),
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
    print(f"🎉 {passed}/{total} production tests passed!" if passed == total else f"⚠️ {passed}/{total} tests passed")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
