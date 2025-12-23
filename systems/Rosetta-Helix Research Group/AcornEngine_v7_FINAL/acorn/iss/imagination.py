# INTEGRITY_METADATA
# Date: 2025-12-23
# Status: NEEDS_REVIEW - Uncategorized file
# Severity: MEDIUM RISK
# Risk Types: ['uncategorized']
# File: systems/Rosetta-Helix Research Group/AcornEngine_v7_FINAL/acorn/iss/imagination.py

"""
ISS Imagination System - Counterfactual Rollouts

This is NOT daydreaming. This is NOT creativity.
This is Monte Carlo-style lookahead that:
- Simulates possible future states
- Produces bias vectors
- Does NOT mutate world state
- Bounded computation
"""

from typing import Dict, List, Any, Optional
import copy
import random


class ImaginationSystem:
    """
    Imagination layer of the Internal Simulation Stack.
    
    Performs counterfactual rollouts to explore possible futures.
    Results influence behavior bias, but do NOT mutate world state.
    """
    
    def __init__(self, engine, config: Dict):
        self.engine = engine
        self.config = config
        self.max_rollouts = config.get("max_rollouts", 3)
        self.rollout_depth = config.get("depth", 2)
        self.rollout_cache = {}  # Entity ID -> recent rollouts
        
    def update(self):
        """
        Update imagination for entities.
        Called during ISS pre-tick phase.
        """
        # Only run imagination for entities that need it
        # (e.g., those with high awareness state or specific triggers)
        
        for entity in self.engine.world.entities.values():
            if self._should_run_imagination(entity):
                self._run_entity_imagination(entity)
    
    def _should_run_imagination(self, entity) -> bool:
        """Determine if entity should run imagination this tick."""
        # Run imagination when:
        # - Entity is in focused awareness state
        # - Entity has recent stimulation
        # - Random chance for exploration
        
        if not entity.awareness_state:
            return False
        
        if entity.awareness_state == "focused":
            return True
        
        # Random exploration (10% chance when idle)
        if entity.awareness_state == "idle" and random.random() < 0.1:
            return True
        
        return False
    
    def _run_entity_imagination(self, entity):
        """
        Run imagination rollouts for an entity.
        Generate counterfactual futures and extract bias.
        """
        rollouts = []
        
        # Generate multiple rollouts
        for i in range(self.max_rollouts):
            rollout = self._generate_rollout(entity, depth=self.rollout_depth)
            rollouts.append(rollout)
        
        # Store rollouts (for debugging/analysis)
        self.rollout_cache[entity.id] = rollouts
        
        # Extract bias from rollouts
        bias = self._extract_bias_from_rollouts(entity, rollouts)
        
        # Update entity's imagination bias
        entity.imagination_bias = bias
        
        # Log imagination activity
        self.engine.log_event("imagination_rollout", {
            "entity_id": entity.id,
            "num_rollouts": len(rollouts),
            "bias_extracted": bias
        })
    
    def _generate_rollout(self, entity, depth: int) -> List[Dict]:
        """
        Generate a counterfactual rollout.
        
        This is a simplified simulation of possible future states.
        Does NOT affect actual world state.
        """
        rollout = []
        
        # Create a lightweight copy of relevant world state
        # (We don't need the full world, just entity's local context)
        local_state = {
            "entity_pos": entity.position.to_dict(),
            "nearby_entities": [
                e.to_dict() for e in 
                self.engine.world.get_entities_in_range(entity.position, radius=5)
            ],
            "tick": self.engine.world.tick
        }
        
        current_pos = entity.position
        
        # Simulate forward for 'depth' steps
        for step in range(depth):
            # Imagine possible actions
            possible_actions = self._get_possible_actions(current_pos)
            
            # Choose one randomly (Monte Carlo)
            if possible_actions:
                action = random.choice(possible_actions)
                
                # Simulate action outcome
                outcome = self._simulate_action_outcome(entity, action, local_state)
                
                rollout.append({
                    "step": step,
                    "action": action,
                    "outcome": outcome
                })
                
                # Update simulated position for next step
                if action["type"] == "move":
                    current_pos = type(entity.position)(**action["target"])
        
        return rollout
    
    def _get_possible_actions(self, pos) -> List[Dict]:
        """Get list of possible actions from a position."""
        actions = []
        
        # Movement actions
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            target = type(pos)(pos.x + dx, pos.y + dy)
            if self.engine.world.is_valid_position(target):
                actions.append({
                    "type": "move",
                    "target": target.to_dict()
                })
        
        # Other possible actions could be added here
        # (interact, use item, etc.)
        
        return actions
    
    def _simulate_action_outcome(self, entity, action: Dict, 
                                 local_state: Dict) -> Dict:
        """
        Simulate the outcome of an action.
        This is approximate and does not need to be perfect.
        """
        outcome = {
            "success": True,
            "changes": {}
        }
        
        if action["type"] == "move":
            # Check if movement would be blocked
            target_pos = type(entity.position)(**action["target"])
            
            # Simplified collision check
            would_collide = any(
                e["position"]["x"] == target_pos.x and 
                e["position"]["y"] == target_pos.y
                for e in local_state["nearby_entities"]
                if e["id"] != entity.id
            )
            
            outcome["success"] = not would_collide
            
            if outcome["success"]:
                outcome["changes"]["position"] = action["target"]
        
        return outcome
    
    def _extract_bias_from_rollouts(self, entity, rollouts: List[List[Dict]]) -> Dict[str, float]:
        """
        Extract behavioral bias from imagination rollouts.
        
        This is where imagination influences behavior WITHOUT forcing actions.
        """
        bias = {
            "exploration": 0.5,
            "caution": 0.5,
            "persistence": 0.5
        }
        
        if not rollouts:
            return bias
        
        # Analyze rollouts for patterns
        total_steps = sum(len(r) for r in rollouts)
        successful_steps = sum(
            sum(1 for s in r if s["outcome"]["success"]) 
            for r in rollouts
        )
        
        # Success rate influences caution
        if total_steps > 0:
            success_rate = successful_steps / total_steps
            bias["caution"] = 1.0 - success_rate  # Low success = high caution
        
        # Number of different actions influences exploration
        unique_actions = len(set(
            str(step["action"]) for rollout in rollouts for step in rollout
        ))
        if unique_actions > 0:
            bias["exploration"] = min(1.0, unique_actions / (self.max_rollouts * 2))
        
        # Rollout length influences persistence
        avg_length = total_steps / len(rollouts) if rollouts else 0
        bias["persistence"] = min(1.0, avg_length / self.rollout_depth)
        
        return bias
    
    def get_imagination_summary(self, entity_id: str) -> Dict[str, Any]:
        """Get a summary of recent imagination activity for an entity."""
        if entity_id not in self.rollout_cache:
            return {"error": "No imagination data for entity"}
        
        rollouts = self.rollout_cache[entity_id]
        
        entity = self.engine.world.entities.get(entity_id)
        if not entity:
            return {"error": "Entity not found"}
        
        return {
            "entity_id": entity_id,
            "num_rollouts": len(rollouts),
            "total_steps": sum(len(r) for r in rollouts),
            "current_bias": entity.imagination_bias,
            "rollouts": rollouts[:3]  # First 3 for brevity
        }


if __name__ == "__main__":
    # Self-test
    print("ISS Imagination System - Self Test")
    print("=" * 50)
    
    # Mock engine
    class MockPos:
        def __init__(self, x, y):
            self.x = x
            self.y = y
        def to_dict(self):
            return {"x": self.x, "y": self.y}
    
    class MockEntity:
        def __init__(self):
            self.id = "test_entity"
            self.position = MockPos(5, 5)
            self.awareness_state = "focused"
            self.imagination_bias = {}
    
    class MockWorld:
        tick = 0
        def is_valid_position(self, pos):
            return 0 <= pos.x < 20 and 0 <= pos.y < 20
        def get_entities_in_range(self, pos, radius):
            return []
    
    class MockEngine:
        world = MockWorld()
        event_log = []
        def log_event(self, type, data):
            pass
    
    # Create imagination system
    config = {"max_rollouts": 2, "depth": 2}
    imagination = ImaginationSystem(MockEngine(), config)
    
    # Test rollout generation
    entity = MockEntity()
    rollout = imagination._generate_rollout(entity, depth=2)
    print(f"Generated rollout with {len(rollout)} steps")
    
    # Test bias extraction
    rollouts = [rollout]
    bias = imagination._extract_bias_from_rollouts(entity, rollouts)
    print(f"Extracted bias: {bias}")
    
    print("\n✓ Self-test passed!")
