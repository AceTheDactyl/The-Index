#!/usr/bin/env python3

# INTEGRITY_METADATA
# Date: 2025-12-23
# Status: JUSTIFIED - Example code demonstrates usage
# Severity: LOW RISK
# Risk Types: ['documentation']
# File: systems/Ace-Systems/examples/Quantum-APL-main/quantum_apl_bridge.py

"""Backward-compatible entrypoint for the QuantumAPL Python tooling."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if SRC_DIR.exists():  # Make package importable without installation
    sys.path.insert(0, str(SRC_DIR))

from quantum_apl_python.analyzer import QuantumAnalyzer
from quantum_apl_python.cli import main as cli_main
from quantum_apl_python.engine import QuantumAPLEngine
from quantum_apl_python.experiments import QuantumExperiment

__all__ = [
    "QuantumAPLEngine",
    "QuantumAnalyzer",
    "QuantumExperiment",
    "main",
]


def main(argv: list[str] | None = None) -> int:
    """Invoke the CLI entrypoint (retained for compatibility)."""

    return cli_main(argv)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
