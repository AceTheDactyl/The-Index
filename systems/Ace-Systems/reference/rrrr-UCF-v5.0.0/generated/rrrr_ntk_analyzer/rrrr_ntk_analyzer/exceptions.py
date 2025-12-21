__all__ = ['RrrrNtkAnalyzerError', 'ValidationError', 'ProcessingError']

"""rrrr_ntk_analyzer exceptions."""


class RrrrNtkAnalyzerError(Exception):
    """Base exception for rrrr_ntk_analyzer."""
    pass


class ValidationError(RrrrNtkAnalyzerError):
    """Validation failed."""
    pass


class ProcessingError(RrrrNtkAnalyzerError):
    """Processing failed."""
    pass
