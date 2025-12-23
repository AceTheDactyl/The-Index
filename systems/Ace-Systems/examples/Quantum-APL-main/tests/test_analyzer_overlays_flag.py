# INTEGRITY_METADATA
# Date: 2025-12-23
# Status: JUSTIFIED - Test file validates system behavior
# Severity: LOW RISK
# Risk Types: ['test_coverage']
# File: systems/Ace-Systems/examples/Quantum-APL-main/tests/test_analyzer_overlays_flag.py

import os

from src.quantum_apl_python.analyzer import _overlays_enabled


def test_overlays_off_by_default(monkeypatch):
    monkeypatch.delenv("QAPL_ANALYZER_OVERLAYS", raising=False)
    assert _overlays_enabled() is False


def test_overlays_on_when_flag_set(monkeypatch):
    monkeypatch.setenv("QAPL_ANALYZER_OVERLAYS", "1")
    assert _overlays_enabled() is True

