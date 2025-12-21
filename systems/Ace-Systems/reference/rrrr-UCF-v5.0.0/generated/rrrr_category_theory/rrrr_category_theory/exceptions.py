__all__ = ['RrrrCategoryTheoryError', 'ValidationError', 'ProcessingError']

"""rrrr_category_theory exceptions."""


class RrrrCategoryTheoryError(Exception):
    """Base exception for rrrr_category_theory."""
    pass


class ValidationError(RrrrCategoryTheoryError):
    """Validation failed."""
    pass


class ProcessingError(RrrrCategoryTheoryError):
    """Processing failed."""
    pass
