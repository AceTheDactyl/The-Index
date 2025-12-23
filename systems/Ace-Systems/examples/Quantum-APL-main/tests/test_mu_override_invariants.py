# INTEGRITY_METADATA
# Date: 2025-12-23
# Status: JUSTIFIED - Test file validates system behavior
# Severity: LOW RISK
# Risk Types: ['test_coverage']
# File: systems/Ace-Systems/examples/Quantum-APL-main/tests/test_mu_override_invariants.py

import os
from quantum_apl_python.analyzer import QuantumAnalyzer


def test_analyzer_barrier_override_prints_delta(monkeypatch):
    monkeypatch.setenv('QAPL_MU_P', '0.600')
    results = {
        'quantum': {'z': 0.5, 'phi': 0.0, 'entropy': 0.0, 'purity': 1.0},
        'classical': {},
        'history': {},
        'analytics': {}
    }
    az = QuantumAnalyzer(results)
    text = az.summary()
    # When MU_P is overridden off exact, barrier line shows Δ=...
    assert '(Δ=' in text or 'Delta=' in text

