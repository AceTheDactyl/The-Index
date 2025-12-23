# INTEGRITY_METADATA
# Date: 2025-12-23
# Status: JUSTIFIED - Auto-generated code
# Severity: LOW RISK
# Risk Types: ['generated']
# File: systems/Ace-Systems/reference/rrrr-UCF-v5.0.0/generated/rrrr_lattice_engine/rrrr_lattice_engine/exceptions.py

__all__ = ['RrrrLatticeEngineError', 'ValidationError', 'ProcessingError']

"""rrrr_lattice_engine exceptions."""


class RrrrLatticeEngineError(Exception):
    """Base exception for rrrr_lattice_engine."""
    pass


class ValidationError(RrrrLatticeEngineError):
    """Validation failed."""
    pass


class ProcessingError(RrrrLatticeEngineError):
    """Processing failed."""
    pass
