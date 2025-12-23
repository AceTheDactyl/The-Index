# INTEGRITY_METADATA
# Date: 2025-12-23
# Status: JUSTIFIED - Auto-generated code
# Severity: LOW RISK
# Risk Types: ['generated']
# File: systems/Ace-Systems/reference/rrrr-UCF-v5.0.0/generated/triad_processor/triad_processor/exceptions.py

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
