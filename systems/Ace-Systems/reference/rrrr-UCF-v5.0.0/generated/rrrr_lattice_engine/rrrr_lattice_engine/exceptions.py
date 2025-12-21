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
