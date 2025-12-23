# INTEGRITY_METADATA
# Date: 2025-12-23
# Status: JUSTIFIED - Auto-generated code
# Severity: LOW RISK
# Risk Types: ['generated']
# File: systems/Ace-Systems/reference/rrrr-UCF-v5.0.0/generated/rrrr_ntk_analyzer/rrrr_ntk_analyzer/exceptions.py

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
