__all__ = ['TestTriadProcessor']

"""Tests for triad_processor."""

import pytest


class TestTriadProcessor:
    """Test suite for TriadProcessor."""
    
    def test_import(self):
        """Test that module imports successfully."""
        from triad_processor import TriadProcessor
        assert TriadProcessor is not None
    
    def test_create(self):
        """Test object creation."""
        from triad_processor import TriadProcessor
        obj = TriadProcessor()
        assert obj is not None
    
    def test_validate(self):
        """Test validation."""
        from triad_processor import TriadProcessor
        obj = TriadProcessor()
        assert obj.validate() is True
