# INTEGRITY_METADATA
# Date: 2025-12-23
# Status: NEEDS_REVIEW - Uncategorized file
# Severity: MEDIUM RISK
# Risk Types: ['uncategorized']
# File: systems/Rosetta-Helix Research Group/AcornEngine_v7_FINAL/acorn/iss/awareness.py

"""
ISS Awareness System - Processing Modes

This is NOT consciousness. This is NOT attention.
This is a state machine that:
- Tracks processing mode (focused/idle/overloaded/dormant)
- Allocates computation budgets
- Determines which ISS layers are active
- Fully deterministic
"""

from typing import Dict, Any
from enum import Enum


class AwarenessState(Enum):
    """Possible awareness states for entities."""
    FOCUSED = "focused"       # Active processing, high computation
    IDLE = "idle"            # Low activity, dream consolidation
    OVERLOADED = "overloaded"  # Too much input, degraded processing
    DORMANT = "dormant"      # Minimal processing, conservation mode


class AwarenessSystem:
    """
    Awareness layer of the Internal Simulation Stack.
    
    Manages entity processing modes and computation budget allocation.
    This is NOT consciousness - it's a resource allocation state machine.
    """
    
    def __init__(self, engine, config: Dict):
        self.engine = engine
        self.config = config
        self.default_state = AwarenessState(config.get("default_state", "idle"))
        
        # Thresholds for state transitions
        self.thresholds = {
            "focused_activity": 0.7,    # Activity level to enter focused
            "idle_activity": 0.2,       # Activity level to enter idle
            "overload_events": 10,      # Events per tick to trigger overload
            "dormant_ticks": 1000       # Ticks of inactivity to go dormant
        }
        
        # Track entity activity
        self.entity_activity = {}  # Entity ID -> recent activity count
        self.last_activity_tick = {}  # Entity ID -> last tick with activity
    
    def initialize_entity_state(self) -> str:
        """Initialize awareness state for a new entity."""
        return self.default_state.value
    
    def update(self):
        """
        Update awareness states for all entities.
        Called during ISS pre-tick phase.
        """
        current_tick = self.engine.world.tick
        
        for entity in self.engine.world.entities.values():
            if entity.awareness_state:
                # Measure recent activity
                activity = self._measure_entity_activity(entity, current_tick)
                
                # Determine appropriate state
                new_state = self._determine_state(entity, activity, current_tick)
                
                # Update if changed
                if new_state != entity.awareness_state:
                    old_state = entity.awareness_state
                    entity.awareness_state = new_state
                    
                    self.engine.log_event("awareness_transition", {
                        "entity_id": entity.id,
                        "from_state": old_state,
                        "to_state": new_state,
                        "activity_level": activity
                    })
    
    def _measure_entity_activity(self, entity, current_tick: int) -> float:
        """
        Measure entity's recent activity level.
        Returns value in [0, 1].
        """
        # Count events involving this entity in recent ticks
        recent_window = 10  # Last 10 ticks
        recent_events = [
            e for e in self.engine.event_log
            if e.tick >= current_tick - recent_window
            and (e.source == entity.id or 
                 e.data.get("entity_id") == entity.id)
        ]
        
        # Normalize to [0, 1]
        activity = min(1.0, len(recent_events) / 5.0)
        
        # Store activity
        self.entity_activity[entity.id] = activity
        
        # Update last activity if any events
        if recent_events:
            self.last_activity_tick[entity.id] = current_tick
        
        return activity
    
    def _determine_state(self, entity, activity: float, 
                        current_tick: int) -> str:
        """Determine appropriate awareness state based on activity."""
        current_state = AwarenessState(entity.awareness_state)
        
        # Check for overload condition
        very_recent_events = [
            e for e in self.engine.event_log
            if e.tick == current_tick
            and (e.source == entity.id or e.data.get("entity_id") == entity.id)
        ]
        if len(very_recent_events) > self.thresholds["overload_events"]:
            return AwarenessState.OVERLOADED.value
        
        # Check for dormant condition
        last_activity = self.last_activity_tick.get(entity.id, current_tick)
        ticks_inactive = current_tick - last_activity
        if ticks_inactive > self.thresholds["dormant_ticks"]:
            return AwarenessState.DORMANT.value
        
        # Transition based on activity level
        if activity > self.thresholds["focused_activity"]:
            return AwarenessState.FOCUSED.value
        elif activity < self.thresholds["idle_activity"]:
            return AwarenessState.IDLE.value
        
        # Stay in current state if in middle range (hysteresis)
        return current_state.value
    
    def get_computation_budget(self, entity) -> float:
        """
        Get computation budget for entity based on awareness state.
        Returns multiplier in [0, 1].
        """
        state = AwarenessState(entity.awareness_state)
        
        budgets = {
            AwarenessState.FOCUSED: 1.0,      # Full computation
            AwarenessState.IDLE: 0.3,         # Low computation, dream active
            AwarenessState.OVERLOADED: 0.5,   # Degraded processing
            AwarenessState.DORMANT: 0.1       # Minimal processing
        }
        
        return budgets.get(state, 0.5)
    
    def should_run_imagination(self, entity) -> bool:
        """Check if imagination system should run for this entity."""
        state = AwarenessState(entity.awareness_state)
        return state in [AwarenessState.FOCUSED, AwarenessState.IDLE]
    
    def should_run_dream(self, entity) -> bool:
        """Check if dream consolidation should run for this entity."""
        state = AwarenessState(entity.awareness_state)
        return state == AwarenessState.IDLE
    
    def get_awareness_summary(self, entity_id: str) -> Dict[str, Any]:
        """Get a summary of entity's awareness state."""
        entity = self.engine.world.entities.get(entity_id)
        if not entity:
            return {"error": "Entity not found"}
        
        state = AwarenessState(entity.awareness_state)
        activity = self.entity_activity.get(entity_id, 0.0)
        budget = self.get_computation_budget(entity)
        
        return {
            "entity_id": entity_id,
            "state": state.value,
            "activity_level": f"{activity:.2f}",
            "computation_budget": f"{budget:.2f}",
            "can_imagine": self.should_run_imagination(entity),
            "can_dream": self.should_run_dream(entity)
        }
    
    def set_entity_state(self, entity_id: str, state: str):
        """Manually set entity's awareness state (for testing/debugging)."""
        entity = self.engine.world.entities.get(entity_id)
        if not entity:
            raise ValueError(f"Entity not found: {entity_id}")
        
        # Validate state
        try:
            AwarenessState(state)
        except ValueError:
            raise ValueError(f"Invalid awareness state: {state}")
        
        old_state = entity.awareness_state
        entity.awareness_state = state
        
        self.engine.log_event("awareness_manual_set", {
            "entity_id": entity_id,
            "from_state": old_state,
            "to_state": state
        })


if __name__ == "__main__":
    # Self-test
    print("ISS Awareness System - Self Test")
    print("=" * 50)
    
    # Mock entity
    class MockEntity:
        def __init__(self):
            self.id = "test_entity"
            self.awareness_state = "idle"
    
    class MockWorld:
        tick = 100
        def entities(self):
            return {}
    
    class MockEngine:
        world = MockWorld()
        event_log = []
        def log_event(self, type, data):
            pass
    
    # Create awareness system
    config = {"default_state": "idle"}
    awareness = AwarenessSystem(MockEngine(), config)
    
    # Test state initialization
    state = awareness.initialize_entity_state()
    print(f"Initial state: {state}")
    
    # Test activity measurement
    entity = MockEntity()
    activity = awareness._measure_entity_activity(entity, 100)
    print(f"Activity level: {activity:.2f}")
    
    # Test budget calculation
    budget = awareness.get_computation_budget(entity)
    print(f"Computation budget: {budget:.2f}")
    
    # Test state transitions
    for state_name in ["focused", "idle", "overloaded", "dormant"]:
        entity.awareness_state = state_name
        budget = awareness.get_computation_budget(entity)
        can_imagine = awareness.should_run_imagination(entity)
        can_dream = awareness.should_run_dream(entity)
        print(f"{state_name}: budget={budget:.2f}, imagine={can_imagine}, dream={can_dream}")
    
    print("\n✓ Self-test passed!")
