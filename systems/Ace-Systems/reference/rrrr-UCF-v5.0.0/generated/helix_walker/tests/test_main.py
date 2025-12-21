__all__ = ['TestHelixWalker']

"""Tests for helix_walker."""

import pytest


class TestHelixWalker:
    """Test suite for HelixWalker."""
    
    def test_import(self):
        """Test that module imports successfully."""
        from helix_walker import HelixWalker
        assert HelixWalker is not None
    
    def test_create(self):
        """Test object creation."""
        from helix_walker import HelixWalker
        obj = HelixWalker()
        assert obj is not None
    
    def test_validate(self):
        """Test validation."""
        from helix_walker import HelixWalker
        obj = HelixWalker()
        assert obj.validate() is True
