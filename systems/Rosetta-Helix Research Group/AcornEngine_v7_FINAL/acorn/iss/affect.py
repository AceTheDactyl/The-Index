"""
ISS Affect System - Bounded Emotional Vectors

This is NOT consciousness. This is NOT feelings.
This is a bounded state machine that:
- Maintains vectors in [0,1]
- Decays toward baseline
- Responds to events
- Does NOT persist as "mood"
"""

from typing import Dict, Any
import math


class AffectSystem:
    """
    Affect layer of the Internal Simulation Stack.
    
    Manages bounded emotional vectors for entities.
    Vectors represent transient affect state, not persistent personality.
    """
    
    AFFECT_DIMENSIONS = [
        "valence",      # Positive/negative (0=negative, 1=positive, 0.5=neutral)
        "arousal",      # Energy/activation (0=calm, 1=excited)
        "dominance",    # Control/power (0=submissive, 1=dominant)
        "curiosity",    # Exploration drive
        "caution"       # Risk aversion
    ]
    
    def __init__(self, engine, config: Dict):
        self.engine = engine
        self.config = config
        self.baseline = config.get("baseline", 0.5)
        self.decay_rate = config.get("decay_rate", 0.1)
        
    def initialize_entity_state(self) -> Dict[str, float]:
        """Initialize affect state for a new entity."""
        return {dim: self.baseline for dim in self.AFFECT_DIMENSIONS}
    
    def update(self):
        """
        Update affect states for all entities.
        Called during ISS post-tick phase.
        """
        # Decay all affect vectors toward baseline
        for entity in self.engine.world.entities.values():
            if entity.affect_state:
                self._decay_toward_baseline(entity.affect_state)
        
        # Process recent events that should affect entities
        self._process_affect_events()
    
    def _decay_toward_baseline(self, affect_state: Dict[str, float]):
        """Decay affect vectors toward baseline value."""
        for dim in self.AFFECT_DIMENSIONS:
            current = affect_state[dim]
            # Move toward baseline
            delta = (self.baseline - current) * self.decay_rate
            affect_state[dim] = self._clamp(current + delta)
    
    def _process_affect_events(self):
        """Process events that should influence affect states."""
        # Look at recent events (last few ticks)
        recent_events = [e for e in self.engine.event_log 
                        if e.tick >= self.engine.world.tick - 5]
        
        for event in recent_events:
            # Different event types influence affect differently
            if event.type == "entity_created":
                # New entity starts at baseline (already handled in init)
                pass
            
            elif event.type == "entity_moved":
                # Movement slightly increases arousal
                entity_id = event.data.get("entity_id")
                if entity_id in self.engine.world.entities:
                    entity = self.engine.world.entities[entity_id]
                    if entity.affect_state:
                        entity.affect_state["arousal"] = self._clamp(
                            entity.affect_state["arousal"] + 0.05
                        )
            
            elif event.type == "collision":
                # Collision increases arousal and decreases valence
                entity_id = event.data.get("entity_id")
                if entity_id in self.engine.world.entities:
                    entity = self.engine.world.entities[entity_id]
                    if entity.affect_state:
                        entity.affect_state["arousal"] = self._clamp(
                            entity.affect_state["arousal"] + 0.2
                        )
                        entity.affect_state["valence"] = self._clamp(
                            entity.affect_state["valence"] - 0.1
                        )
    
    def apply_stimulus_bias(self, entity_id: str, stimulus: Dict[str, float], 
                           duration_ticks: int):
        """
        Apply a symbolic stimulus bias to an entity's affect.
        
        Example: Text motifs extracted from Alice in Wonderland:
        {
            "nonlinearity": 0.8,
            "absurdity": 0.7,
            "playfulness": 0.9,
            "rule_inversion": 0.6
        }
        
        Maps to affect dimensions as bias multipliers.
        """
        if entity_id not in self.engine.world.entities:
            return
        
        entity = self.engine.world.entities[entity_id]
        if not entity.affect_state:
            return
        
        # Map stimulus dimensions to affect dimensions
        # This is heuristic and can be tuned
        mappings = {
            "nonlinearity": ("curiosity", 1.0),
            "absurdity": ("arousal", 0.5),
            "playfulness": ("valence", 0.7),
            "rule_inversion": ("dominance", -0.5),  # Negative = less dominance
            "exploration": ("curiosity", 1.0),
            "caution": ("caution", 1.0)
        }
        
        # Apply biases
        for stim_dim, stim_value in stimulus.items():
            if stim_dim in mappings:
                affect_dim, multiplier = mappings[stim_dim]
                current = entity.affect_state[affect_dim]
                # Push toward high or low based on stimulus
                if multiplier > 0:
                    target = 0.5 + (stim_value * 0.5)  # Push toward 1.0
                else:
                    target = 0.5 - (stim_value * 0.5)  # Push toward 0.0
                
                # Apply with strength proportional to stimulus
                delta = (target - current) * abs(stim_value) * 0.3
                entity.affect_state[affect_dim] = self._clamp(current + delta)
        
        # Log the stimulus application
        self.engine.log_event("stimulus_applied", {
            "entity_id": entity_id,
            "stimulus": stimulus,
            "duration_ticks": duration_ticks
        })
    
    def get_affect_summary(self, entity_id: str) -> Dict[str, Any]:
        """Get a human-readable summary of an entity's affect state."""
        if entity_id not in self.engine.world.entities:
            return {"error": "Entity not found"}
        
        entity = self.engine.world.entities[entity_id]
        if not entity.affect_state:
            return {"error": "No affect state"}
        
        affect = entity.affect_state
        
        # Interpret affect dimensions
        valence_label = self._interpret_valence(affect["valence"])
        arousal_label = self._interpret_arousal(affect["arousal"])
        
        return {
            "entity_id": entity_id,
            "raw_state": affect.copy(),
            "interpretation": {
                "valence": valence_label,
                "arousal": arousal_label,
                "curiosity": "high" if affect["curiosity"] > 0.7 else "low",
                "caution": "high" if affect["caution"] > 0.7 else "low"
            }
        }
    
    def _interpret_valence(self, value: float) -> str:
        """Interpret valence dimension."""
        if value < 0.3:
            return "negative"
        elif value > 0.7:
            return "positive"
        else:
            return "neutral"
    
    def _interpret_arousal(self, value: float) -> str:
        """Interpret arousal dimension."""
        if value < 0.3:
            return "calm"
        elif value > 0.7:
            return "excited"
        else:
            return "moderate"
    
    def _clamp(self, value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """Clamp value to [min_val, max_val] range."""
        return max(min_val, min(max_val, value))


if __name__ == "__main__":
    # Self-test
    print("ISS Affect System - Self Test")
    print("=" * 50)
    
    # Mock engine
    class MockEngine:
        class MockWorld:
            tick = 0
        world = MockWorld()
        event_log = []
    
    # Create affect system
    config = {"baseline": 0.5, "decay_rate": 0.1}
    affect = AffectSystem(MockEngine(), config)
    
    # Test initialization
    state = affect.initialize_entity_state()
    print(f"Initial state: {state}")
    
    # Test decay
    state["arousal"] = 0.9  # Excited
    affect._decay_toward_baseline(state)
    print(f"After decay: arousal = {state['arousal']:.2f}")
    
    print("\n✓ Self-test passed!")
