# INTEGRITY_METADATA
# Date: 2025-12-23
# Status: JUSTIFIED - Auto-generated code
# Severity: LOW RISK
# Risk Types: ['generated']
# File: systems/Ace-Systems/reference/rrrr-UCF-v5.0.0/generated/rrrr_category_theory/rrrr_category_theory/exceptions.py

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
