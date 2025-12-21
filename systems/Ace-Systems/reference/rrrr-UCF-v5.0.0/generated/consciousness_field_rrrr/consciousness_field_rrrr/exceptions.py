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
