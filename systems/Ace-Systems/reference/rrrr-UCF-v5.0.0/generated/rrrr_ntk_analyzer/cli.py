# INTEGRITY_METADATA
# Date: 2025-12-23
# Status: JUSTIFIED - Auto-generated code
# Severity: LOW RISK
# Risk Types: ['generated']
# File: systems/Ace-Systems/reference/rrrr-UCF-v5.0.0/generated/rrrr_ntk_analyzer/cli.py

__all__ = ['main']

"""rrrr_ntk_analyzer CLI."""

import argparse
import sys

from rrrr_ntk_analyzer import create_rrrr_ntk_analyzer


def main():
    parser = argparse.ArgumentParser(description="Neural Tangent Kernel eigenvalue decomposition into 4D lattice products for architecture analysis")
    parser.add_argument("--config", type=str, help="Config file path")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    try:
        instance = create_rrrr_ntk_analyzer()
        instance.run()
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
