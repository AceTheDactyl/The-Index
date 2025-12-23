# INTEGRITY_METADATA
# Date: 2025-12-23
# Status: NEEDS_REVIEW - Uncategorized file
# Severity: MEDIUM RISK
# Risk Types: ['uncategorized']
# File: systems/Rosetta-Helix Research Group/AcornEngine_v7_FINAL/acorn/engine.py

"""
The Ultimate Acorn Engine v7 - Core Simulation Engine
Pure headless simulation with ISS integration.

NO GUI DEPENDENCIES. NO ANTHROPOMORPHIZATION.
Entities are state machines. Behavior emerges from rules.
"""

import time
import json
import uuid
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import copy


class EntityType(Enum):
    """Entity types in the simulation."""
    BASIC = "basic"
    PLAYER = "player"
    NPC = "npc"
    STRUCTURE = "structure"
    RESOURCE = "resource"


@dataclass
class Position:
    """2D position in the world."""
    x: int
    y: int
    
    def distance_to(self, other: 'Position') -> float:
        """Calculate distance to another position."""
        return ((self.x - other.x)**2 + (self.y - other.y)**2)**0.5
    
    def to_dict(self) -> Dict:
        return {"x": self.x, "y": self.y}


@dataclass
class Entity:
    """Basic entity in the simulation."""
    id: str
    type: EntityType
    position: Position
    properties: Dict[str, Any]
    
    # ISS state (managed by ISS subsystems)
    affect_state: Optional[Dict[str, float]] = None
    awareness_state: Optional[str] = None
    imagination_bias: Optional[Dict[str, float]] = None
    dream_buffer: Optional[List[Dict]] = None
    
    def to_dict(self) -> Dict:
        """Serialize entity to dictionary."""
        return {
            "id": self.id,
            "type": self.type.value,
            "position": self.position.to_dict(),
            "properties": self.properties,
            "affect_state": self.affect_state,
            "awareness_state": self.awareness_state,
            "imagination_bias": self.imagination_bias,
            "dream_buffer": self.dream_buffer
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Entity':
        """Deserialize entity from dictionary."""
        return cls(
            id=data["id"],
            type=EntityType(data["type"]),
            position=Position(**data["position"]),
            properties=data["properties"],
            affect_state=data.get("affect_state"),
            awareness_state=data.get("awareness_state"),
            imagination_bias=data.get("imagination_bias"),
            dream_buffer=data.get("dream_buffer")
        )


@dataclass
class Event:
    """Event in the simulation log."""
    tick: int
    timestamp: float
    type: str
    data: Dict[str, Any]
    source: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "tick": self.tick,
            "timestamp": self.timestamp,
            "type": self.type,
            "data": self.data,
            "source": self.source
        }


class WorldState:
    """Complete state of the simulated world."""
    
    def __init__(self, width: int, height: int, seed: int = None):
        self.width = width
        self.height = height
        self.seed = seed or int(time.time())
        self.tick = 0
        self.entities: Dict[str, Entity] = {}
        self.tiles: List[List[Dict]] = [[{"type": "grass", "properties": {}} 
                                        for _ in range(width)] 
                                       for _ in range(height)]
        self.metadata = {
            "created": time.time(),
            "version": "7.0.0"
        }
    
    def is_valid_position(self, pos: Position) -> bool:
        """Check if position is within world bounds."""
        return 0 <= pos.x < self.width and 0 <= pos.y < self.height
    
    def is_position_occupied(self, pos: Position, exclude_id: str = None) -> bool:
        """Check if a position is occupied by an entity."""
        for entity_id, entity in self.entities.items():
            if entity_id != exclude_id:
                if entity.position.x == pos.x and entity.position.y == pos.y:
                    return True
        return False
    
    def get_entities_at(self, pos: Position) -> List[Entity]:
        """Get all entities at a specific position."""
        return [e for e in self.entities.values() 
                if e.position.x == pos.x and e.position.y == pos.y]
    
    def get_entities_in_range(self, pos: Position, radius: int) -> List[Entity]:
        """Get entities within a certain radius."""
        return [e for e in self.entities.values() 
                if e.position.distance_to(pos) <= radius]
    
    def to_dict(self) -> Dict:
        """Serialize world state."""
        return {
            "width": self.width,
            "height": self.height,
            "seed": self.seed,
            "tick": self.tick,
            "entities": {eid: e.to_dict() for eid, e in self.entities.items()},
            "tiles": self.tiles,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'WorldState':
        """Deserialize world state."""
        world = cls(data["width"], data["height"], data["seed"])
        world.tick = data["tick"]
        world.entities = {eid: Entity.from_dict(e) 
                         for eid, e in data["entities"].items()}
        world.tiles = data["tiles"]
        world.metadata = data["metadata"]
        return world


class AcornEngine:
    """
    The Ultimate Acorn Engine v7
    
    Pure simulation engine. No GUI. No I/O except through adapters.
    Integrates ISS (Affect, Imagination, Dream, Awareness) as engine-side modules.
    """
    
    def __init__(self, world: WorldState, config: Dict = None):
        self.world = world
        # Merge provided config with defaults
        self.config = self._default_config()
        if config:
            self._merge_config(self.config, config)
        self.event_log: List[Event] = []
        self.running = False
        
        # ISS subsystems (imported lazily to avoid circular deps)
        self.iss_affect = None
        self.iss_imagination = None
        self.iss_dream = None
        self.iss_awareness = None
        
        # Performance tracking
        self.stats = {
            "total_ticks": 0,
            "total_events": 0,
            "entities_created": 0,
            "entities_destroyed": 0
        }
    
    def _default_config(self) -> Dict:
        """Default engine configuration."""
        return {
            "iss_enabled": True,
            "max_entities": 10000,
            "tick_rate": 60,  # Target TPS
            "fractal": {
                "enabled": True,
                "max_depth": 5,
                "budget_decay": 0.5
            },
            "iss": {
                "affect": {
                    "baseline": 0.5,
                    "decay_rate": 0.1
                },
                "imagination": {
                    "max_rollouts": 3,
                    "depth": 2
                },
                "dream": {
                    "compression_ratio": 0.01,
                    "interval_ticks": 100
                },
                "awareness": {
                    "default_state": "idle"
                }
            }
        }
    
    def _merge_config(self, base: Dict, override: Dict):
        """Recursively merge override config into base config."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    def initialize_iss(self):
        """Initialize ISS subsystems. Called once after engine creation."""
        if not self.config["iss_enabled"]:
            return
        
        # Import ISS modules
        from acorn.iss.affect import AffectSystem
        from acorn.iss.imagination import ImaginationSystem
        from acorn.iss.dream import DreamSystem
        from acorn.iss.awareness import AwarenessSystem
        
        self.iss_affect = AffectSystem(self, self.config["iss"]["affect"])
        self.iss_imagination = ImaginationSystem(self, self.config["iss"]["imagination"])
        self.iss_dream = DreamSystem(self, self.config["iss"]["dream"])
        self.iss_awareness = AwarenessSystem(self, self.config["iss"]["awareness"])
        
        self.log_event("iss_initialized", {"subsystems": ["affect", "imagination", "dream", "awareness"]})
    
    def tick(self) -> Dict:
        """
        Execute one simulation tick.
        Returns: Summary of tick execution.
        """
        tick_start = time.time()
        
        # 1. Pre-tick ISS updates
        if self.config["iss_enabled"]:
            self._run_iss_pre_tick()
        
        # 2. Entity updates (physics, behavior, etc.)
        self._update_entities()
        
        # 3. Post-tick ISS updates
        if self.config["iss_enabled"]:
            self._run_iss_post_tick()
        
        # 4. Increment tick counter
        self.world.tick += 1
        self.stats["total_ticks"] += 1
        
        tick_duration = time.time() - tick_start
        
        return {
            "tick": self.world.tick,
            "duration_ms": tick_duration * 1000,
            "entities": len(self.world.entities),
            "events": len(self.event_log)
        }
    
    def _run_iss_pre_tick(self):
        """Run ISS subsystems before main entity update."""
        # Awareness determines processing mode
        if self.iss_awareness:
            self.iss_awareness.update()
        
        # Imagination generates counterfactual rollouts
        if self.iss_imagination:
            self.iss_imagination.update()
    
    def _run_iss_post_tick(self):
        """Run ISS subsystems after main entity update."""
        # Affect processes events and updates emotional vectors
        if self.iss_affect:
            self.iss_affect.update()
        
        # Dream compresses experiences during idle time
        if self.iss_dream:
            self.iss_dream.update()
    
    def _update_entities(self):
        """Update all entities for this tick."""
        # Basic movement and interaction logic
        # (Extended by specific entity types in subclasses)
        for entity in list(self.world.entities.values()):
            # Apply imagination bias to behavior
            if entity.imagination_bias:
                self._apply_imagination_bias(entity)
            
            # Entity-specific update logic would go here
            # This is where behavior emerges from state + rules
            pass
    
    def _apply_imagination_bias(self, entity: Entity):
        """
        Apply imagination rollout results as behavior bias.
        Does NOT mutate world state directly.
        """
        bias = entity.imagination_bias
        if not bias:
            return
        
        # Example: Increased 'exploration' bias makes entity more likely to move
        # Example: Increased 'caution' bias makes entity more likely to avoid risks
        # These are tendencies, not deterministic outcomes
        
        # This is where bias influences the probability distribution
        # of entity actions, but does not force actions
        pass
    
    def create_entity(self, entity_type: EntityType, position: Position, 
                     properties: Dict = None) -> str:
        """Create a new entity in the world."""
        entity_id = f"{entity_type.value}_{uuid.uuid4().hex[:8]}"
        
        # Validate position
        if not self.world.is_valid_position(position):
            raise ValueError(f"Invalid position: {position}")
        
        # Check entity limit
        if len(self.world.entities) >= self.config["max_entities"]:
            raise RuntimeError("Maximum entity limit reached")
        
        # Create entity
        entity = Entity(
            id=entity_id,
            type=entity_type,
            position=position,
            properties=properties or {}
        )
        
        # Initialize ISS state if enabled
        if self.config["iss_enabled"]:
            entity.affect_state = self.iss_affect.initialize_entity_state() if self.iss_affect else None
            entity.awareness_state = self.iss_awareness.initialize_entity_state() if self.iss_awareness else None
            entity.imagination_bias = {}
            entity.dream_buffer = []
        
        self.world.entities[entity_id] = entity
        self.stats["entities_created"] += 1
        
        self.log_event("entity_created", {
            "entity_id": entity_id,
            "type": entity_type.value,
            "position": position.to_dict()
        })
        
        return entity_id
    
    def destroy_entity(self, entity_id: str):
        """Remove an entity from the world."""
        if entity_id not in self.world.entities:
            raise ValueError(f"Entity not found: {entity_id}")
        
        del self.world.entities[entity_id]
        self.stats["entities_destroyed"] += 1
        
        self.log_event("entity_destroyed", {"entity_id": entity_id})
    
    def move_entity(self, entity_id: str, target: Position) -> bool:
        """
        Move an entity to a new position.
        Returns True if successful, False otherwise.
        """
        if entity_id not in self.world.entities:
            return False
        
        entity = self.world.entities[entity_id]
        
        # Validate target position
        if not self.world.is_valid_position(target):
            return False
        
        # Check if target is occupied
        if self.world.is_position_occupied(target, exclude_id=entity_id):
            return False
        
        old_pos = entity.position
        entity.position = target
        
        self.log_event("entity_moved", {
            "entity_id": entity_id,
            "from": old_pos.to_dict(),
            "to": target.to_dict()
        })
        
        return True
    
    def log_event(self, event_type: str, data: Dict, source: str = None):
        """Log an event in the simulation."""
        event = Event(
            tick=self.world.tick,
            timestamp=time.time(),
            type=event_type,
            data=data,
            source=source
        )
        self.event_log.append(event)
        self.stats["total_events"] += 1
    
    def get_snapshot(self) -> Dict:
        """Get a complete snapshot of current world state."""
        return {
            "world": self.world.to_dict(),
            "stats": self.stats.copy(),
            "config": self.config
        }
    
    def load_snapshot(self, snapshot: Dict):
        """Load a world state from a snapshot."""
        self.world = WorldState.from_dict(snapshot["world"])
        self.stats = snapshot.get("stats", self.stats)
        # Note: config is not overwritten, only world state
    
    def process_proposal(self, proposal: Dict) -> Dict:
        """
        Process a proposal from the adapter layer.
        This is the ONLY way external systems should interact with the engine.
        
        Returns: Result dictionary with 'success' and optional 'data' or 'error'.
        """
        proposal_type = proposal.get("type")
        
        if proposal_type == "create_entity":
            try:
                entity_id = self.create_entity(
                    EntityType(proposal["entity_type"]),
                    Position(**proposal["position"]),
                    proposal.get("properties")
                )
                return {"success": True, "entity_id": entity_id}
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        elif proposal_type == "move_entity":
            try:
                success = self.move_entity(
                    proposal["entity_id"],
                    Position(**proposal["target"])
                )
                return {"success": success}
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        elif proposal_type == "destroy_entity":
            try:
                self.destroy_entity(proposal["entity_id"])
                return {"success": True}
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        elif proposal_type == "get_snapshot":
            return {"success": True, "snapshot": self.get_snapshot()}
        
        elif proposal_type == "tick":
            result = self.tick()
            return {"success": True, "tick_result": result}
        
        else:
            return {"success": False, "error": f"Unknown proposal type: {proposal_type}"}
    
    def get_stats(self) -> Dict:
        """Get current engine statistics."""
        return {
            **self.stats,
            "current_tick": self.world.tick,
            "entity_count": len(self.world.entities),
            "event_count": len(self.event_log)
        }


if __name__ == "__main__":
    # Basic self-test
    print("Acorn Engine v7 - Self Test")
    print("=" * 50)
    
    # Create a small test world
    world = WorldState(20, 20, seed=12345)
    engine = AcornEngine(world)
    
    print(f"World created: {world.width}x{world.height}")
    print(f"Seed: {world.seed}")
    
    # Create some entities
    entity_id = engine.create_entity(EntityType.BASIC, Position(5, 5))
    print(f"Entity created: {entity_id}")
    
    # Run a few ticks
    for i in range(5):
        result = engine.tick()
        print(f"Tick {result['tick']}: {result['duration_ms']:.2f}ms, {result['entities']} entities")
    
    # Get stats
    stats = engine.get_stats()
    print("\nFinal stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✓ Self-test passed!")
