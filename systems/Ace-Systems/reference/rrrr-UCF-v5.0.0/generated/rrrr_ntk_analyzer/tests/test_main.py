__all__ = ['TestRrrrNtkAnalyzer']

"""Tests for rrrr_ntk_analyzer."""

import pytest


class TestRrrrNtkAnalyzer:
    """Test suite for RrrrNtkAnalyzer."""
    
    def test_import(self):
        """Test that module imports successfully."""
        from rrrr_ntk_analyzer import RrrrNtkAnalyzer
        assert RrrrNtkAnalyzer is not None
    
    def test_create(self):
        """Test object creation."""
        from rrrr_ntk_analyzer import RrrrNtkAnalyzer
        obj = RrrrNtkAnalyzer()
        assert obj is not None
    
    def test_validate(self):
        """Test validation."""
        from rrrr_ntk_analyzer import RrrrNtkAnalyzer
        obj = RrrrNtkAnalyzer()
        assert obj.validate() is True
