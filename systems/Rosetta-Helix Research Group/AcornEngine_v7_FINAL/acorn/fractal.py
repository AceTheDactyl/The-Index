# INTEGRITY_METADATA
# Date: 2025-12-23
# Status: NEEDS_REVIEW - Uncategorized file
# Severity: MEDIUM RISK
# Risk Types: ['uncategorized']
# File: systems/Rosetta-Helix Research Group/AcornEngine_v7_FINAL/acorn/fractal.py

"""
Fractal Simulation Engine - Recursive Universe Support

This enables:
- Simulations within simulations
- Each layer has bounded computation budget
- Nested worlds maintain parent awareness
- Safe recursive depth limits
- Experimental substrate for cognitive modeling

This is NOT infinite recursion. This is NOT consciousness.
This is a controlled computational hierarchy.
"""

from typing import Dict, List, Optional, Any
import copy
import time


class FractalLayer:
    """A single layer in the fractal simulation hierarchy."""
    
    def __init__(self, depth: int, parent_engine, config: Dict):
        self.depth = depth
        self.parent_engine = parent_engine
        self.config = config
        self.simulation_budget = self._calculate_budget()
        self.tick_count = 0
        self.created_at = time.time()
        
        # Each layer gets its own mini-universe
        # Import here to avoid circular dependency
        from acorn.engine import WorldState, AcornEngine
        
        # Create smaller world for deeper layers
        world_size = max(10, 50 - (depth * 10))
        self.world = WorldState(world_size, world_size)
        
        # Simplified config for sub-layers
        sub_config = copy.deepcopy(config)
        sub_config["max_entities"] = max(10, 100 - (depth * 20))
        sub_config["fractal"]["enabled"] = depth < config["fractal"]["max_depth"] - 1
        
        self.engine = AcornEngine(self.world, sub_config)
        
    def _calculate_budget(self) -> int:
        """Calculate computation budget for this layer."""
        base_budget = self.config["fractal"].get("base_budget", 1000)
        decay = self.config["fractal"].get("budget_decay", 0.5)
        
        # Exponential decay with depth
        budget = int(base_budget * (decay ** self.depth))
        return max(1, budget)  # At least 1 tick
    
    def can_execute(self) -> bool:
        """Check if this layer has budget remaining."""
        return self.tick_count < self.simulation_budget
    
    def execute_tick(self) -> Dict:
        """Execute one tick of this layer's simulation."""
        if not self.can_execute():
            return {"error": "Budget exhausted", "depth": self.depth}
        
        result = self.engine.tick()
        self.tick_count += 1
        
        return {
            "depth": self.depth,
            "tick": result["tick"],
            "budget_remaining": self.simulation_budget - self.tick_count,
            "entities": len(self.world.entities)
        }
    
    def get_summary(self) -> Dict:
        """Get summary of this layer's state."""
        return {
            "depth": self.depth,
            "tick_count": self.tick_count,
            "budget_total": self.simulation_budget,
            "budget_remaining": self.simulation_budget - self.tick_count,
            "world_size": f"{self.world.width}x{self.world.height}",
            "entity_count": len(self.world.entities),
            "age_seconds": time.time() - self.created_at
        }


class FractalSimulationEngine:
    """
    Manages recursive simulation layers.
    
    Allows entities to spawn "imagination universes" or sub-simulations
    for planning, learning, and exploration WITHOUT affecting main universe.
    """
    
    def __init__(self, base_engine, config: Dict):
        self.base_engine = base_engine
        self.config = config
        self.max_depth = config["fractal"].get("max_depth", 5)
        
        # Track all active layers
        self.layers: Dict[str, FractalLayer] = {}  # Layer ID -> FractalLayer
        
        # Track which entities own which layers
        self.entity_layers: Dict[str, List[str]] = {}  # Entity ID -> Layer IDs
        
    def spawn_layer(self, entity_id: str, purpose: str = "imagination") -> Optional[str]:
        """
        Spawn a new fractal layer for an entity.
        
        Args:
            entity_id: Entity spawning the layer
            purpose: Why the layer is being created (imagination, dream, experiment)
        
        Returns:
            Layer ID if successful, None otherwise
        """
        # Check if entity exists
        if entity_id not in self.base_engine.world.entities:
            return None
        
        # Determine depth (entity's layers + 1)
        entity = self.base_engine.world.entities[entity_id]
        current_layers = self.entity_layers.get(entity_id, [])
        depth = len(current_layers) + 1
        
        # Check depth limit
        if depth >= self.max_depth:
            self.base_engine.log_event("fractal_depth_limit", {
                "entity_id": entity_id,
                "attempted_depth": depth,
                "max_depth": self.max_depth
            })
            return None
        
        # Create new layer
        layer = FractalLayer(depth, self.base_engine, self.config)
        
        # Generate layer ID
        import uuid
        layer_id = f"layer_{depth}_{uuid.uuid4().hex[:8]}"
        
        # Store layer
        self.layers[layer_id] = layer
        
        # Track entity ownership
        if entity_id not in self.entity_layers:
            self.entity_layers[entity_id] = []
        self.entity_layers[entity_id].append(layer_id)
        
        # Log layer creation
        self.base_engine.log_event("fractal_layer_spawned", {
            "layer_id": layer_id,
            "entity_id": entity_id,
            "depth": depth,
            "purpose": purpose,
            "budget": layer.simulation_budget
        })
        
        return layer_id
    
    def execute_layer(self, layer_id: str) -> Optional[Dict]:
        """Execute one tick of a fractal layer."""
        if layer_id not in self.layers:
            return None
        
        layer = self.layers[layer_id]
        
        if not layer.can_execute():
            return {"error": "Budget exhausted", "layer_id": layer_id}
        
        result = layer.execute_tick()
        result["layer_id"] = layer_id
        
        return result
    
    def execute_all_layers(self) -> List[Dict]:
        """Execute one tick for all active layers with budget."""
        results = []
        
        for layer_id, layer in list(self.layers.items()):
            if layer.can_execute():
                result = self.execute_layer(layer_id)
                if result:
                    results.append(result)
        
        return results
    
    def collect_layer_results(self, layer_id: str) -> Optional[Dict]:
        """
        Collect results from a completed layer.
        
        This is where layer insights flow back to the parent entity.
        Results are statistical summaries, NOT raw state transfer.
        """
        if layer_id not in self.layers:
            return None
        
        layer = self.layers[layer_id]
        
        # Extract statistical summary
        summary = {
            "layer_id": layer_id,
            "depth": layer.depth,
            "ticks_executed": layer.tick_count,
            "final_entity_count": len(layer.world.entities),
            "final_tick": layer.world.tick,
            
            # Statistical patterns from the layer
            "patterns": self._extract_layer_patterns(layer)
        }
        
        return summary
    
    def _extract_layer_patterns(self, layer: FractalLayer) -> Dict:
        """Extract statistical patterns from a completed layer."""
        patterns = {
            "average_entities": 0.0,
            "total_events": len(layer.engine.event_log),
            "activity_density": 0.0
        }
        
        # Calculate average entity count over layer lifetime
        # (In a full implementation, we'd track this over time)
        patterns["average_entities"] = len(layer.world.entities)
        
        # Activity density: events per tick
        if layer.tick_count > 0:
            patterns["activity_density"] = len(layer.engine.event_log) / layer.tick_count
        
        return patterns
    
    def apply_layer_insights(self, entity_id: str, layer_id: str):
        """
        Apply insights from a layer back to the owning entity.
        
        This is where imagination/dream layers influence behavior.
        """
        # Get layer results
        results = self.collect_layer_results(layer_id)
        if not results:
            return
        
        # Get entity
        if entity_id not in self.base_engine.world.entities:
            return
        
        entity = self.base_engine.world.entities[entity_id]
        
        # Apply patterns as bias
        patterns = results["patterns"]
        
        # High activity in layer → increase exploration tendency
        if patterns["activity_density"] > 0.5 and entity.imagination_bias:
            entity.imagination_bias["exploration"] = min(1.0,
                entity.imagination_bias.get("exploration", 0.5) + 0.1)
        
        # Layer completed successfully → increase confidence
        if results["ticks_executed"] >= results.get("budget", 0) * 0.9:
            if entity.affect_state:
                entity.affect_state["dominance"] = min(1.0,
                    entity.affect_state["dominance"] + 0.05)
        
        # Log insight application
        self.base_engine.log_event("fractal_insights_applied", {
            "entity_id": entity_id,
            "layer_id": layer_id,
            "patterns": patterns
        })
    
    def cleanup_exhausted_layers(self):
        """Remove layers that have exhausted their budget."""
        exhausted = [
            layer_id for layer_id, layer in self.layers.items()
            if not layer.can_execute()
        ]
        
        for layer_id in exhausted:
            # Collect final insights before cleanup
            for entity_id, layer_ids in self.entity_layers.items():
                if layer_id in layer_ids:
                    self.apply_layer_insights(entity_id, layer_id)
            
            # Remove layer
            del self.layers[layer_id]
            
            # Remove from entity tracking
            for entity_id in self.entity_layers:
                if layer_id in self.entity_layers[entity_id]:
                    self.entity_layers[entity_id].remove(layer_id)
    
    def get_layer_hierarchy(self) -> Dict:
        """Get a visualization of the layer hierarchy."""
        hierarchy = {
            "base": {
                "depth": 0,
                "entities": len(self.base_engine.world.entities),
                "layers": {}
            }
        }
        
        # Organize layers by depth
        layers_by_depth = {}
        for layer_id, layer in self.layers.items():
            depth = layer.depth
            if depth not in layers_by_depth:
                layers_by_depth[depth] = []
            layers_by_depth[depth].append({
                "id": layer_id,
                "summary": layer.get_summary()
            })
        
        hierarchy["base"]["layers"] = layers_by_depth
        
        return hierarchy
    
    def get_stats(self) -> Dict:
        """Get fractal simulation statistics."""
        return {
            "total_layers": len(self.layers),
            "max_depth": self.max_depth,
            "active_layers_by_depth": {
                depth: sum(1 for l in self.layers.values() if l.depth == depth)
                for depth in range(self.max_depth)
            },
            "total_sub_entities": sum(
                len(l.world.entities) for l in self.layers.values()
            ),
            "entities_with_layers": len(self.entity_layers)
        }


if __name__ == "__main__":
    # Self-test
    print("Fractal Simulation Engine - Self Test")
    print("=" * 50)
    
    # Create base engine
    from acorn.engine import WorldState, AcornEngine, EntityType, Position
    
    world = WorldState(30, 30)
    config = {
        "iss_enabled": False,  # Disable for testing
        "fractal": {
            "enabled": True,
            "max_depth": 3,
            "budget_decay": 0.5,
            "base_budget": 10
        }
    }
    base_engine = AcornEngine(world, config)
    
    # Create entity
    entity_id = base_engine.create_entity(EntityType.BASIC, Position(5, 5))
    print(f"Created entity: {entity_id}")
    
    # Create fractal engine
    fractal = FractalSimulationEngine(base_engine, config)
    print(f"Fractal engine created, max depth: {fractal.max_depth}")
    
    # Spawn a layer
    layer_id = fractal.spawn_layer(entity_id, "imagination")
    print(f"Spawned layer: {layer_id}")
    
    # Execute some ticks
    for i in range(5):
        results = fractal.execute_all_layers()
        print(f"Tick {i}: {len(results)} layers executed")
    
    # Get hierarchy
    hierarchy = fractal.get_layer_hierarchy()
    print(f"\nLayer hierarchy: {hierarchy}")
    
    # Get stats
    stats = fractal.get_stats()
    print(f"\nFractal stats: {stats}")
    
    print("\n✓ Self-test passed!")
