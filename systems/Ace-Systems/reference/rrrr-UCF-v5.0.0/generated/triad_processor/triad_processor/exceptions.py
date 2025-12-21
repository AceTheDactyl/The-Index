__all__ = ['TriadProcessorError', 'ValidationError', 'ProcessingError']

"""triad_processor exceptions."""


class TriadProcessorError(Exception):
    """Base exception for triad_processor."""
    pass


class ValidationError(TriadProcessorError):
    """Validation failed."""
    pass


class ProcessingError(TriadProcessorError):
    """Processing failed."""
    pass
