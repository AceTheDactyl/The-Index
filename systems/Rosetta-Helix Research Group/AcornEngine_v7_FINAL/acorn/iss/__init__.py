# INTEGRITY_METADATA
# Date: 2025-12-23
# Status: NEEDS_REVIEW - Uncategorized file
# Severity: MEDIUM RISK
# Risk Types: ['uncategorized']
# File: systems/Rosetta-Helix Research Group/AcornEngine_v7_FINAL/acorn/iss/__init__.py

"""
Internal Simulation Stack (ISS) - v7

Four bounded subsystems that add simulation depth WITHOUT consciousness:
- Affect: Bounded emotional vectors
- Imagination: Counterfactual rollouts
- Dream: Idle-time compression
- Awareness: Processing mode states

These are NOT minds. They are NOT feelings. They are state machines.
"""

from acorn.iss.affect import AffectSystem
from acorn.iss.imagination import ImaginationSystem
from acorn.iss.dream import DreamSystem
from acorn.iss.awareness import AwarenessSystem, AwarenessState

__all__ = [
    'AffectSystem',
    'ImaginationSystem',
    'DreamSystem',
    'AwarenessSystem',
    'AwarenessState'
]

__version__ = "7.0.0"
