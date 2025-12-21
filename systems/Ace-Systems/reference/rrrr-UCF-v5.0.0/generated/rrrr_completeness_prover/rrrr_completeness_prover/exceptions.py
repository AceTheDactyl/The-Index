__all__ = ['RrrrCompletenessProverError', 'ValidationError', 'ProcessingError']

"""rrrr_completeness_prover exceptions."""


class RrrrCompletenessProverError(Exception):
    """Base exception for rrrr_completeness_prover."""
    pass


class ValidationError(RrrrCompletenessProverError):
    """Validation failed."""
    pass


class ProcessingError(RrrrCompletenessProverError):
    """Processing failed."""
    pass
