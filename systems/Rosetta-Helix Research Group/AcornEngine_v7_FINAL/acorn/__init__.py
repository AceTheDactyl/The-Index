# INTEGRITY_METADATA
# Date: 2025-12-23
# Status: NEEDS_REVIEW - Uncategorized file
# Severity: MEDIUM RISK
# Risk Types: ['uncategorized']
# File: systems/Rosetta-Helix Research Group/AcornEngine_v7_FINAL/acorn/__init__.py

"""
The Ultimate Acorn v7 - Complete Autonomous Universe Simulator

Core components:
- Engine: Pure simulation logic
- ISS: Internal Simulation Stack (Affect, Imagination, Dream, Awareness)
- Fractal: Recursive simulation layers
- Plates: Holographic PNG memory
- Adapter: Proposal-based GUI interface

This is NOT consciousness. This is computational substrate.
"""

from acorn.engine import (
    AcornEngine,
    WorldState,
    Entity,
    EntityType,
    Position,
    Event
)

from acorn.adapter import AdapterLayer, ProposalValidator
from acorn.plates import HolographicPlate, PlateManager
from acorn.fractal import FractalSimulationEngine, FractalLayer

__version__ = "7.0.0"
__all__ = [
    # Engine
    'AcornEngine',
    'WorldState',
    'Entity',
    'EntityType',
    'Position',
    'Event',
    
    # Adapter
    'AdapterLayer',
    'ProposalValidator',
    
    # Memory
    'HolographicPlate',
    'PlateManager',
    
    # Fractal
    'FractalSimulationEngine',
    'FractalLayer'
]
