# INTEGRITY_METADATA
# Date: 2025-12-23
# Status: NEEDS_REVIEW - Uncategorized file
# Severity: MEDIUM RISK
# Risk Types: ['uncategorized']
# File: systems/Rosetta-Helix Research Group/AcornEngine_v7_FINAL/acorn/adapter.py

"""
Adapter Layer - Proposal-Based GUI-Engine Interface

This is the ONLY way GUIs should communicate with the engine.
All interactions go through validated proposals.

NO direct state mutation. NO bypassing validation.
"""

from typing import Dict, Any, Optional, Callable
import time
import json


class ProposalValidator:
    """Validates proposals before sending to engine."""
    
    VALID_PROPOSAL_TYPES = {
        "tick",
        "create_entity",
        "move_entity",
        "destroy_entity",
        "get_snapshot",
        "get_stats",
        "save_plate",
        "load_plate"
    }
    
    REQUIRED_FIELDS = {
        "create_entity": ["entity_type", "position"],
        "move_entity": ["entity_id", "target"],
        "destroy_entity": ["entity_id"],
        "save_plate": ["filepath"],
        "load_plate": ["filepath"]
    }
    
    @classmethod
    def validate(cls, proposal: Dict) -> tuple[bool, Optional[str]]:
        """
        Validate a proposal.
        
        Returns:
            (valid, error_message)
        """
        # Check type field exists
        if "type" not in proposal:
            return False, "Missing 'type' field"
        
        # Check type is valid
        proposal_type = proposal["type"]
        if proposal_type not in cls.VALID_PROPOSAL_TYPES:
            return False, f"Invalid proposal type: {proposal_type}"
        
        # Check required fields
        if proposal_type in cls.REQUIRED_FIELDS:
            for field in cls.REQUIRED_FIELDS[proposal_type]:
                if field not in proposal:
                    return False, f"Missing required field: {field}"
        
        # Type-specific validation
        if proposal_type == "create_entity":
            if "entity_type" not in proposal or "position" not in proposal:
                return False, "create_entity requires entity_type and position"
            
            if not isinstance(proposal["position"], dict):
                return False, "position must be a dict with x and y"
            
            if "x" not in proposal["position"] or "y" not in proposal["position"]:
                return False, "position must have x and y fields"
        
        elif proposal_type == "move_entity":
            if "entity_id" not in proposal or "target" not in proposal:
                return False, "move_entity requires entity_id and target"
            
            if not isinstance(proposal["target"], dict):
                return False, "target must be a dict with x and y"
            
            if "x" not in proposal["target"] or "y" not in proposal["target"]:
                return False, "target must have x and y fields"
        
        return True, None


class AdapterLayer:
    """
    Adapter between GUI and Engine.
    
    Responsibilities:
    - Validate proposals
    - Route proposals to engine
    - Format responses
    - Log interactions
    - Rate limiting (optional)
    """
    
    def __init__(self, engine):
        self.engine = engine
        self.validator = ProposalValidator()
        
        # Statistics
        self.stats = {
            "proposals_received": 0,
            "proposals_accepted": 0,
            "proposals_rejected": 0,
            "errors": 0
        }
        
        # Optional callbacks for GUI updates
        self.update_callbacks: list[Callable] = []
        
    def submit_proposal(self, proposal: Dict) -> Dict[str, Any]:
        """
        Submit a proposal to the engine.
        
        Args:
            proposal: Dictionary with 'type' and type-specific fields
        
        Returns:
            Response dictionary with 'success' and optional 'data' or 'error'
        """
        self.stats["proposals_received"] += 1
        
        # Validate proposal
        valid, error = self.validator.validate(proposal)
        if not valid:
            self.stats["proposals_rejected"] += 1
            return {
                "success": False,
                "error": f"Validation failed: {error}",
                "timestamp": time.time()
            }
        
        # Process proposal
        try:
            result = self.engine.process_proposal(proposal)
            
            if result.get("success", False):
                self.stats["proposals_accepted"] += 1
            else:
                self.stats["proposals_rejected"] += 1
            
            result["timestamp"] = time.time()
            
            # Notify callbacks
            self._notify_callbacks(proposal, result)
            
            return result
            
        except Exception as e:
            self.stats["errors"] += 1
            return {
                "success": False,
                "error": f"Exception: {str(e)}",
                "timestamp": time.time()
            }
    
    def add_update_callback(self, callback: Callable):
        """Add a callback to be notified of engine updates."""
        self.update_callbacks.append(callback)
    
    def remove_update_callback(self, callback: Callable):
        """Remove an update callback."""
        if callback in self.update_callbacks:
            self.update_callbacks.remove(callback)
    
    def _notify_callbacks(self, proposal: Dict, result: Dict):
        """Notify all registered callbacks of an update."""
        for callback in self.update_callbacks:
            try:
                callback(proposal, result)
            except Exception as e:
                print(f"Callback error: {e}")
    
    def get_world_state(self) -> Dict:
        """Get current world state (convenience method)."""
        result = self.submit_proposal({"type": "get_snapshot"})
        if result.get("success"):
            return result.get("snapshot", {})
        return {}
    
    def tick_engine(self) -> Dict:
        """Execute one engine tick (convenience method)."""
        return self.submit_proposal({"type": "tick"})
    
    def create_entity(self, entity_type: str, x: int, y: int, 
                     properties: Dict = None) -> Optional[str]:
        """
        Create an entity (convenience method).
        
        Returns:
            Entity ID if successful, None otherwise
        """
        proposal = {
            "type": "create_entity",
            "entity_type": entity_type,
            "position": {"x": x, "y": y}
        }
        if properties:
            proposal["properties"] = properties
        
        result = self.submit_proposal(proposal)
        if result.get("success"):
            return result.get("entity_id")
        return None
    
    def move_entity(self, entity_id: str, x: int, y: int) -> bool:
        """
        Move an entity (convenience method).
        
        Returns:
            True if successful, False otherwise
        """
        proposal = {
            "type": "move_entity",
            "entity_id": entity_id,
            "target": {"x": x, "y": y}
        }
        
        result = self.submit_proposal(proposal)
        return result.get("success", False)
    
    def get_adapter_stats(self) -> Dict:
        """Get adapter statistics."""
        return {
            **self.stats,
            "success_rate": (
                self.stats["proposals_accepted"] / max(1, self.stats["proposals_received"])
            ),
            "error_rate": (
                self.stats["errors"] / max(1, self.stats["proposals_received"])
            )
        }


class AdapterLogger:
    """Logs all adapter interactions for debugging and analysis."""
    
    def __init__(self, filepath: str = "adapter_log.jsonl"):
        self.filepath = filepath
        self.enabled = True
    
    def log_interaction(self, proposal: Dict, result: Dict):
        """Log a single adapter interaction."""
        if not self.enabled:
            return
        
        entry = {
            "timestamp": time.time(),
            "proposal": proposal,
            "result": result
        }
        
        try:
            with open(self.filepath, 'a') as f:
                f.write(json.dumps(entry) + '\n')
        except Exception as e:
            print(f"Logger error: {e}")
    
    def enable(self):
        """Enable logging."""
        self.enabled = True
    
    def disable(self):
        """Disable logging."""
        self.enabled = False


if __name__ == "__main__":
    # Self-test
    print("Adapter Layer - Self Test")
    print("=" * 50)
    
    # Create mock engine
    from acorn.engine import WorldState, AcornEngine
    
    world = WorldState(20, 20)
    engine = AcornEngine(world, {"iss_enabled": False})
    
    # Create adapter
    adapter = AdapterLayer(engine)
    print("Adapter created")
    
    # Test valid proposal
    result = adapter.submit_proposal({
        "type": "create_entity",
        "entity_type": "basic",
        "position": {"x": 5, "y": 5}
    })
    print(f"Create entity result: {result}")
    assert result["success"], "Should succeed"
    
    # Test invalid proposal (missing fields)
    result = adapter.submit_proposal({
        "type": "create_entity"
    })
    print(f"Invalid proposal result: {result}")
    assert not result["success"], "Should fail"
    
    # Test convenience methods
    entity_id = adapter.create_entity("basic", 10, 10)
    print(f"Created entity: {entity_id}")
    assert entity_id is not None, "Should return entity ID"
    
    # Test move
    moved = adapter.move_entity(entity_id, 11, 11)
    print(f"Move result: {moved}")
    assert moved, "Should succeed"
    
    # Test tick
    tick_result = adapter.tick_engine()
    print(f"Tick result: {tick_result}")
    assert tick_result["success"], "Should succeed"
    
    # Get stats
    stats = adapter.get_adapter_stats()
    print(f"\nAdapter stats: {stats}")
    
    print("\n✓ Self-test passed!")
