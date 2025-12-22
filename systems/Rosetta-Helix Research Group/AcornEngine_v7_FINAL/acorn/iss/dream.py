"""
ISS Dream System - Idle-Time Compression

This is NOT sleep. This is NOT dreaming.
This is statistical consolidation that:
- Compresses recent experiences during idle time
- Extracts patterns
- Updates bias tendencies
- Fully logged and deterministic
"""

from typing import Dict, List, Any
import statistics
import json


class DreamSystem:
    """
    Dream layer of the Internal Simulation Stack.
    
    Compresses experiences during idle time into statistical summaries.
    Results influence long-term behavioral tendencies.
    """
    
    def __init__(self, engine, config: Dict):
        self.engine = engine
        self.config = config
        self.compression_ratio = config.get("compression_ratio", 0.01)
        self.interval_ticks = config.get("interval_ticks", 100)
        self.last_consolidation = {}  # Entity ID -> last consolidation tick
        
    def update(self):
        """
        Update dream consolidation for entities.
        Called during ISS post-tick phase.
        """
        current_tick = self.engine.world.tick
        
        for entity in self.engine.world.entities.values():
            if self._should_consolidate(entity, current_tick):
                self._consolidate_entity_experiences(entity)
                self.last_consolidation[entity.id] = current_tick
    
    def _should_consolidate(self, entity, current_tick: int) -> bool:
        """Determine if entity should consolidate experiences now."""
        # Only consolidate when idle
        if not entity.awareness_state or entity.awareness_state != "idle":
            return False
        
        # Check if enough ticks have passed
        last_tick = self.last_consolidation.get(entity.id, 0)
        if current_tick - last_tick < self.interval_ticks:
            return False
        
        # Check if entity has experience buffer
        if not entity.dream_buffer:
            return False
        
        return True
    
    def _consolidate_entity_experiences(self, entity):
        """
        Consolidate entity's recent experiences into statistical summary.
        
        This is like "memory consolidation" in neuroscience, but WITHOUT
        any claim of consciousness or subjective experience.
        """
        if not entity.dream_buffer:
            return
        
        # Extract patterns from dream buffer
        patterns = self._extract_patterns(entity.dream_buffer)
        
        # Update entity's long-term tendencies based on patterns
        self._update_tendencies(entity, patterns)
        
        # Compress buffer (keep only summary)
        compressed = self._compress_buffer(entity.dream_buffer)
        
        # Replace buffer with compressed version
        entity.dream_buffer = compressed
        
        # Log consolidation
        self.engine.log_event("dream_consolidation", {
            "entity_id": entity.id,
            "patterns_extracted": patterns,
            "compression_ratio": len(compressed) / max(1, len(entity.dream_buffer))
        })
    
    def _extract_patterns(self, buffer: List[Dict]) -> Dict[str, Any]:
        """Extract statistical patterns from experience buffer."""
        patterns = {
            "activity_level": 0.0,
            "success_rate": 0.0,
            "interaction_count": 0,
            "position_variance": 0.0
        }
        
        if not buffer:
            return patterns
        
        # Activity level: How many actions per tick
        actions = [e for e in buffer if e.get("type") == "action"]
        patterns["activity_level"] = len(actions) / len(buffer)
        
        # Success rate: How many actions succeeded
        if actions:
            successes = sum(1 for a in actions if a.get("success", False))
            patterns["success_rate"] = successes / len(actions)
        
        # Interaction count: How many social events
        interactions = [e for e in buffer if e.get("type") == "interaction"]
        patterns["interaction_count"] = len(interactions)
        
        # Position variance: How much movement
        positions = [e.get("position") for e in buffer if e.get("position")]
        if len(positions) > 1:
            x_coords = [p["x"] for p in positions]
            y_coords = [p["y"] for p in positions]
            try:
                patterns["position_variance"] = (
                    statistics.variance(x_coords) + statistics.variance(y_coords)
                ) / 2
            except:
                patterns["position_variance"] = 0.0
        
        return patterns
    
    def _update_tendencies(self, entity, patterns: Dict):
        """Update entity's behavioral tendencies based on patterns."""
        # This is where "learning" happens - but it's just statistical adjustment
        
        # High activity → slight increase in arousal baseline
        if patterns["activity_level"] > 0.7 and entity.affect_state:
            entity.affect_state["arousal"] = min(1.0, 
                entity.affect_state["arousal"] + 0.05)
        
        # Low success rate → increase caution
        if patterns["success_rate"] < 0.3 and entity.imagination_bias:
            entity.imagination_bias["caution"] = min(1.0,
                entity.imagination_bias.get("caution", 0.5) + 0.1)
        
        # High position variance → increase exploration tendency
        if patterns["position_variance"] > 10.0 and entity.imagination_bias:
            entity.imagination_bias["exploration"] = min(1.0,
                entity.imagination_bias.get("exploration", 0.5) + 0.05)
    
    def _compress_buffer(self, buffer: List[Dict]) -> List[Dict]:
        """
        Compress experience buffer using statistical summary.
        Keeps only the most salient events plus a summary.
        """
        if not buffer:
            return []
        
        # Calculate how many events to keep
        target_size = max(1, int(len(buffer) * self.compression_ratio))
        
        # Sort by saliency (events with more data are more salient)
        sorted_buffer = sorted(buffer, 
                              key=lambda e: len(json.dumps(e)), 
                              reverse=True)
        
        # Keep top salient events
        compressed = sorted_buffer[:target_size]
        
        # Add statistical summary
        compressed.append({
            "type": "summary",
            "total_events": len(buffer),
            "compressed_to": target_size,
            "tick_range": [
                buffer[0].get("tick", 0),
                buffer[-1].get("tick", 0)
            ]
        })
        
        return compressed
    
    def add_to_dream_buffer(self, entity_id: str, experience: Dict):
        """Add an experience to entity's dream buffer."""
        entity = self.engine.world.entities.get(entity_id)
        if not entity:
            return
        
        if entity.dream_buffer is None:
            entity.dream_buffer = []
        
        # Add tick timestamp
        experience["tick"] = self.engine.world.tick
        
        entity.dream_buffer.append(experience)
        
        # Limit buffer size to prevent memory explosion
        max_buffer_size = self.interval_ticks * 2
        if len(entity.dream_buffer) > max_buffer_size:
            entity.dream_buffer = entity.dream_buffer[-max_buffer_size:]
    
    def get_dream_summary(self, entity_id: str) -> Dict[str, Any]:
        """Get summary of entity's dream/consolidation state."""
        entity = self.engine.world.entities.get(entity_id)
        if not entity:
            return {"error": "Entity not found"}
        
        buffer_size = len(entity.dream_buffer) if entity.dream_buffer else 0
        last_consolidation = self.last_consolidation.get(entity_id, 0)
        
        return {
            "entity_id": entity_id,
            "buffer_size": buffer_size,
            "last_consolidation_tick": last_consolidation,
            "ticks_since_consolidation": self.engine.world.tick - last_consolidation,
            "consolidation_interval": self.interval_ticks
        }


if __name__ == "__main__":
    # Self-test
    print("ISS Dream System - Self Test")
    print("=" * 50)
    
    # Mock entity
    class MockEntity:
        def __init__(self):
            self.id = "test_entity"
            self.awareness_state = "idle"
            self.dream_buffer = [
                {"type": "action", "success": True, "position": {"x": 5, "y": 5}},
                {"type": "action", "success": True, "position": {"x": 6, "y": 5}},
                {"type": "action", "success": False, "position": {"x": 6, "y": 6}},
                {"type": "interaction", "target": "other_entity"}
            ]
            self.affect_state = {"arousal": 0.5}
            self.imagination_bias = {"caution": 0.5, "exploration": 0.5}
    
    class MockWorld:
        tick = 100
        def entities(self):
            return {}
    
    class MockEngine:
        world = MockWorld()
        event_log = []
        def log_event(self, type, data):
            pass
    
    # Create dream system
    config = {"compression_ratio": 0.01, "interval_ticks": 100}
    dream = DreamSystem(MockEngine(), config)
    
    # Test pattern extraction
    entity = MockEntity()
    patterns = dream._extract_patterns(entity.dream_buffer)
    print(f"Extracted patterns: {patterns}")
    
    # Test compression
    compressed = dream._compress_buffer(entity.dream_buffer)
    print(f"Compressed buffer: {len(entity.dream_buffer)} -> {len(compressed)} events")
    
    print("\n✓ Self-test passed!")
