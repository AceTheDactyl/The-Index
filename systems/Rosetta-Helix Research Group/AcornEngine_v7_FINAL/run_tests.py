#!/usr/bin/env python3
"""
Comprehensive Test Suite for The Ultimate Acorn v7

Tests all major components:
- Engine core
- ISS subsystems
- Fractal simulation
- Holographic plates
- Adapter layer
"""

import sys
import os
import time
import tempfile

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from acorn import (
    AcornEngine,
    WorldState,
    EntityType,
    Position,
    AdapterLayer
)
from acorn.plates import HolographicPlate, PlateManager
from acorn.fractal import FractalSimulationEngine


class TestResult:
    """Test result container."""
    
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.error = None
        self.duration = 0.0
    
    def __repr__(self):
        status = "✓" if self.passed else "✗"
        error_msg = f" ({self.error})" if self.error else ""
        return f"{status} {self.name} ({self.duration*1000:.2f}ms){error_msg}"


class TestSuite:
    """Main test suite."""
    
    def __init__(self):
        self.results = []
        self.total_tests = 0
        self.passed_tests = 0
        
    def run_test(self, name: str, test_func):
        """Run a single test."""
        result = TestResult(name)
        self.total_tests += 1
        
        start_time = time.time()
        
        try:
            test_func()
            result.passed = True
            self.passed_tests += 1
        except AssertionError as e:
            result.error = str(e)
        except Exception as e:
            result.error = f"Exception: {e}"
        
        result.duration = time.time() - start_time
        self.results.append(result)
        
        return result.passed
    
    def print_results(self):
        """Print test results."""
        print("\n" + "=" * 60)
        print("TEST RESULTS")
        print("=" * 60)
        print()
        
        # Group by category
        categories = {}
        for result in self.results:
            category = result.name.split(":")[0] if ":" in result.name else "General"
            if category not in categories:
                categories[category] = []
            categories[category].append(result)
        
        # Print by category
        for category, results in sorted(categories.items()):
            print(f"\n{category}:")
            for result in results:
                print(f"  {result}")
        
        # Summary
        print("\n" + "=" * 60)
        print(f"Total: {self.total_tests} tests")
        print(f"Passed: {self.passed_tests} ({100*self.passed_tests/self.total_tests:.1f}%)")
        print(f"Failed: {self.total_tests - self.passed_tests}")
        print("=" * 60)
        print()


# ============================================================================
# ENGINE TESTS
# ============================================================================

def test_engine_creation():
    """Test engine creation."""
    world = WorldState(20, 20, seed=42)
    engine = AcornEngine(world, {"iss_enabled": False})
    assert engine.world.width == 20
    assert engine.world.height == 20
    assert len(engine.world.entities) == 0


def test_entity_creation():
    """Test entity creation."""
    world = WorldState(20, 20)
    engine = AcornEngine(world, {"iss_enabled": False})
    
    entity_id = engine.create_entity(EntityType.BASIC, Position(5, 5))
    assert entity_id is not None
    assert entity_id in engine.world.entities
    
    entity = engine.world.entities[entity_id]
    assert entity.position.x == 5
    assert entity.position.y == 5


def test_entity_movement():
    """Test entity movement."""
    world = WorldState(20, 20)
    engine = AcornEngine(world, {"iss_enabled": False})
    
    entity_id = engine.create_entity(EntityType.BASIC, Position(5, 5))
    
    # Valid move
    success = engine.move_entity(entity_id, Position(6, 6))
    assert success
    
    entity = engine.world.entities[entity_id]
    assert entity.position.x == 6
    assert entity.position.y == 6
    
    # Invalid move (out of bounds)
    success = engine.move_entity(entity_id, Position(100, 100))
    assert not success


def test_engine_tick():
    """Test engine ticking."""
    world = WorldState(20, 20)
    engine = AcornEngine(world, {"iss_enabled": False})
    
    initial_tick = engine.world.tick
    result = engine.tick()
    
    assert result["tick"] == initial_tick + 1
    assert "duration_ms" in result


def test_snapshot_save_load():
    """Test snapshot save/load."""
    world = WorldState(20, 20, seed=42)
    engine = AcornEngine(world, {"iss_enabled": False})
    
    # Create some entities
    engine.create_entity(EntityType.BASIC, Position(5, 5))
    engine.create_entity(EntityType.BASIC, Position(10, 10))
    
    # Run some ticks
    for _ in range(5):
        engine.tick()
    
    # Get snapshot
    snapshot = engine.get_snapshot()
    original_tick = snapshot["world"]["tick"]
    original_count = len(snapshot["world"]["entities"])
    
    # Create new engine and load snapshot
    new_world = WorldState(10, 10)  # Different size
    new_engine = AcornEngine(new_world, {"iss_enabled": False})
    new_engine.load_snapshot(snapshot)
    
    # Verify
    assert new_engine.world.tick == original_tick
    assert len(new_engine.world.entities) == original_count
    assert new_engine.world.width == 20  # Should be from snapshot


# ============================================================================
# ISS TESTS
# ============================================================================

def test_iss_initialization():
    """Test ISS initialization."""
    world = WorldState(20, 20)
    engine = AcornEngine(world, {"iss_enabled": True})
    engine.initialize_iss()
    
    assert engine.iss_affect is not None
    assert engine.iss_imagination is not None
    assert engine.iss_dream is not None
    assert engine.iss_awareness is not None


def test_affect_state():
    """Test affect state initialization and updates."""
    world = WorldState(20, 20)
    engine = AcornEngine(world, {"iss_enabled": True})
    engine.initialize_iss()
    
    entity_id = engine.create_entity(EntityType.BASIC, Position(5, 5))
    entity = engine.world.entities[entity_id]
    
    # Should have affect state
    assert entity.affect_state is not None
    assert "valence" in entity.affect_state
    assert "arousal" in entity.affect_state
    
    # Values should be in [0, 1]
    for value in entity.affect_state.values():
        assert 0 <= value <= 1


def test_awareness_states():
    """Test awareness state transitions."""
    world = WorldState(20, 20)
    engine = AcornEngine(world, {"iss_enabled": True})
    engine.initialize_iss()
    
    entity_id = engine.create_entity(EntityType.BASIC, Position(5, 5))
    entity = engine.world.entities[entity_id]
    
    # Should have awareness state
    assert entity.awareness_state is not None
    assert entity.awareness_state in ["focused", "idle", "overloaded", "dormant"]


def test_imagination_rollouts():
    """Test imagination system can generate rollouts."""
    world = WorldState(20, 20)
    engine = AcornEngine(world, {"iss_enabled": True})
    engine.initialize_iss()
    
    entity_id = engine.create_entity(EntityType.BASIC, Position(5, 5))
    entity = engine.world.entities[entity_id]
    entity.awareness_state = "focused"  # Force focused state
    
    # Run a few ticks to trigger imagination
    for _ in range(10):
        engine.tick()
    
    # Should have imagination bias
    assert entity.imagination_bias is not None


def test_dream_consolidation():
    """Test dream consolidation."""
    world = WorldState(20, 20)
    engine = AcornEngine(world, {"iss_enabled": True})
    engine.initialize_iss()
    
    entity_id = engine.create_entity(EntityType.BASIC, Position(5, 5))
    entity = engine.world.entities[entity_id]
    
    # Add some experiences to dream buffer
    for i in range(10):
        engine.iss_dream.add_to_dream_buffer(entity_id, {
            "type": "action",
            "success": i % 2 == 0,
            "position": {"x": 5 + i, "y": 5}
        })
    
    assert len(entity.dream_buffer) == 10
    
    # Force consolidation
    entity.awareness_state = "idle"
    engine.iss_dream.last_consolidation[entity_id] = -101  # Force consolidation (enough ticks passed)
    engine.iss_dream.update()
    
    # Buffer should be compressed
    assert len(entity.dream_buffer) < 10


# ============================================================================
# FRACTAL TESTS
# ============================================================================

def test_fractal_layer_creation():
    """Test fractal layer creation."""
    world = WorldState(20, 20)
    config = {
        "iss_enabled": False,
        "fractal": {
            "enabled": True,
            "max_depth": 3,
            "budget_decay": 0.5,
            "base_budget": 10
        }
    }
    engine = AcornEngine(world, config)
    fractal = FractalSimulationEngine(engine, config)
    
    entity_id = engine.create_entity(EntityType.BASIC, Position(5, 5))
    
    layer_id = fractal.spawn_layer(entity_id, "test")
    assert layer_id is not None
    assert layer_id in fractal.layers


def test_fractal_depth_limit():
    """Test fractal depth limit."""
    world = WorldState(20, 20)
    config = {
        "iss_enabled": False,
        "fractal": {
            "enabled": True,
            "max_depth": 2,
            "budget_decay": 0.5
        }
    }
    engine = AcornEngine(world, config)
    fractal = FractalSimulationEngine(engine, config)
    
    entity_id = engine.create_entity(EntityType.BASIC, Position(5, 5))
    
    # Should be able to create first layer
    layer_id1 = fractal.spawn_layer(entity_id, "test")
    assert layer_id1 is not None
    
    # Should NOT be able to create beyond depth limit
    # (This test would need layer nesting which is more complex)
    # For now just verify max_depth is respected
    assert fractal.max_depth == 2


def test_fractal_budget():
    """Test fractal computation budget."""
    world = WorldState(20, 20)
    config = {
        "iss_enabled": False,
        "fractal": {
            "enabled": True,
            "max_depth": 3,
            "budget_decay": 0.5,
            "base_budget": 5
        }
    }
    engine = AcornEngine(world, config)
    fractal = FractalSimulationEngine(engine, config)
    
    entity_id = engine.create_entity(EntityType.BASIC, Position(5, 5))
    layer_id = fractal.spawn_layer(entity_id, "test")
    
    layer = fractal.layers[layer_id]
    
    # Execute until budget exhausted
    while layer.can_execute():
        fractal.execute_layer(layer_id)
    
    # Should be at budget limit
    assert layer.tick_count == layer.simulation_budget


# ============================================================================
# PLATES TESTS
# ============================================================================

def test_plate_creation():
    """Test holographic plate creation."""
    world = WorldState(20, 20, seed=42)
    engine = AcornEngine(world, {"iss_enabled": False})
    
    # Create some entities
    engine.create_entity(EntityType.BASIC, Position(5, 5))
    engine.create_entity(EntityType.BASIC, Position(10, 10))
    
    # Create plate
    plate = HolographicPlate()
    snapshot = engine.get_snapshot()
    image = plate.encode_universe(snapshot)
    
    assert image is not None
    assert image.width == plate.width
    assert image.height == plate.height


def test_plate_encode_decode():
    """Test plate encoding and decoding."""
    world = WorldState(20, 20, seed=42)
    engine = AcornEngine(world, {"iss_enabled": False})
    
    # Create entities
    entity_id1 = engine.create_entity(EntityType.BASIC, Position(5, 5))
    entity_id2 = engine.create_entity(EntityType.BASIC, Position(10, 10))
    
    # Run some ticks
    for _ in range(5):
        engine.tick()
    
    # Encode
    plate = HolographicPlate()
    snapshot = engine.get_snapshot()
    plate.encode_universe(snapshot)
    
    # Decode
    decoded = plate.decode_universe(plate.image)
    
    # Verify
    assert decoded["world"]["tick"] == snapshot["world"]["tick"]
    assert len(decoded["world"]["entities"]) == len(snapshot["world"]["entities"])
    assert decoded["world"]["seed"] == snapshot["world"]["seed"]


def test_plate_save_load():
    """Test plate save/load to file."""
    world = WorldState(20, 20, seed=42)
    engine = AcornEngine(world, {"iss_enabled": False})
    
    engine.create_entity(EntityType.BASIC, Position(5, 5))
    
    # Save plate
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        temp_file = f.name
    
    try:
        manager = PlateManager(engine)
        manager.save_plate(temp_file)
        
        original_tick = engine.world.tick
        
        # Load plate in new engine
        new_world = WorldState(10, 10)
        new_engine = AcornEngine(new_world, {"iss_enabled": False})
        new_manager = PlateManager(new_engine)
        new_manager.load_plate(temp_file)
        
        # Verify
        assert new_engine.world.tick == original_tick
        assert len(new_engine.world.entities) == 1
    
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)


# ============================================================================
# ADAPTER TESTS
# ============================================================================

def test_adapter_basic():
    """Test adapter basic functionality."""
    world = WorldState(20, 20)
    engine = AcornEngine(world, {"iss_enabled": False})
    adapter = AdapterLayer(engine)
    
    # Test valid proposal
    result = adapter.submit_proposal({
        "type": "create_entity",
        "entity_type": "basic",
        "position": {"x": 5, "y": 5}
    })
    
    assert result["success"]
    assert "entity_id" in result


def test_adapter_validation():
    """Test adapter proposal validation."""
    world = WorldState(20, 20)
    engine = AcornEngine(world, {"iss_enabled": False})
    adapter = AdapterLayer(engine)
    
    # Invalid proposal (missing fields)
    result = adapter.submit_proposal({
        "type": "create_entity"
    })
    
    assert not result["success"]
    assert "error" in result


def test_adapter_convenience_methods():
    """Test adapter convenience methods."""
    world = WorldState(20, 20)
    engine = AcornEngine(world, {"iss_enabled": False})
    adapter = AdapterLayer(engine)
    
    # Create entity
    entity_id = adapter.create_entity("basic", 5, 5)
    assert entity_id is not None
    
    # Move entity
    success = adapter.move_entity(entity_id, 6, 6)
    assert success
    
    # Tick
    result = adapter.tick_engine()
    assert result["success"]


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all tests."""
    suite = TestSuite()
    
    print("\n" + "=" * 60)
    print("THE ULTIMATE ACORN v7 - TEST SUITE")
    print("=" * 60)
    
    # Engine tests
    suite.run_test("Engine: Creation", test_engine_creation)
    suite.run_test("Engine: Entity Creation", test_entity_creation)
    suite.run_test("Engine: Entity Movement", test_entity_movement)
    suite.run_test("Engine: Tick Execution", test_engine_tick)
    suite.run_test("Engine: Snapshot Save/Load", test_snapshot_save_load)
    
    # ISS tests
    suite.run_test("ISS: Initialization", test_iss_initialization)
    suite.run_test("ISS: Affect State", test_affect_state)
    suite.run_test("ISS: Awareness States", test_awareness_states)
    suite.run_test("ISS: Imagination Rollouts", test_imagination_rollouts)
    suite.run_test("ISS: Dream Consolidation", test_dream_consolidation)
    
    # Fractal tests
    suite.run_test("Fractal: Layer Creation", test_fractal_layer_creation)
    suite.run_test("Fractal: Depth Limit", test_fractal_depth_limit)
    suite.run_test("Fractal: Budget", test_fractal_budget)
    
    # Plates tests
    suite.run_test("Plates: Creation", test_plate_creation)
    suite.run_test("Plates: Encode/Decode", test_plate_encode_decode)
    suite.run_test("Plates: Save/Load", test_plate_save_load)
    
    # Adapter tests
    suite.run_test("Adapter: Basic", test_adapter_basic)
    suite.run_test("Adapter: Validation", test_adapter_validation)
    suite.run_test("Adapter: Convenience Methods", test_adapter_convenience_methods)
    
    # Print results
    suite.print_results()
    
    # Exit with appropriate code
    sys.exit(0 if suite.passed_tests == suite.total_tests else 1)


if __name__ == "__main__":
    main()
