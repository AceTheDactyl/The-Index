# INTEGRITY_METADATA
# Date: 2025-12-23
# Status: JUSTIFIED - Auto-generated code
# Severity: LOW RISK
# Risk Types: ['generated']
# File: systems/Ace-Systems/reference/rrrr-UCF-v5.0.0/generated/rrrr_completeness_prover/rrrr_completeness_prover/exceptions.py

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
