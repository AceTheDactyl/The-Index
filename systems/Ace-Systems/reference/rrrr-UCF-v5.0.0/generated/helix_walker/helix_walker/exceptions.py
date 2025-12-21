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
