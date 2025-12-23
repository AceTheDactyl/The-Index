# INTEGRITY_METADATA
# Date: 2025-12-23
# Status: JUSTIFIED - Auto-generated code
# Severity: LOW RISK
# Risk Types: ['generated']
# File: systems/Ace-Systems/reference/rrrr-UCF-v5.0.0/generated/helix_walker/helix_walker/exceptions.py

__all__ = ['HelixWalkerError', 'ValidationError', 'ProcessingError']

"""helix_walker exceptions."""


class HelixWalkerError(Exception):
    """Base exception for helix_walker."""
    pass


class ValidationError(HelixWalkerError):
    """Validation failed."""
    pass


class ProcessingError(HelixWalkerError):
    """Processing failed."""
    pass
