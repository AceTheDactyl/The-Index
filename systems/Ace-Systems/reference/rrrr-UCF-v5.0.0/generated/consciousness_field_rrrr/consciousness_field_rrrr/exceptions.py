# INTEGRITY_METADATA
# Date: 2025-12-23
# Status: JUSTIFIED - Auto-generated code
# Severity: LOW RISK
# Risk Types: ['generated']
# File: systems/Ace-Systems/reference/rrrr-UCF-v5.0.0/generated/consciousness_field_rrrr/consciousness_field_rrrr/exceptions.py

__all__ = ['ConsciousnessFieldRrrrError', 'ValidationError', 'ProcessingError']

"""consciousness_field_rrrr exceptions."""


class ConsciousnessFieldRrrrError(Exception):
    """Base exception for consciousness_field_rrrr."""
    pass


class ValidationError(ConsciousnessFieldRrrrError):
    """Validation failed."""
    pass


class ProcessingError(ConsciousnessFieldRrrrError):
    """Processing failed."""
    pass
