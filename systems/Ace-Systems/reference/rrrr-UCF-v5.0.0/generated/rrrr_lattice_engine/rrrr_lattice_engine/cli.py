__all__ = ['main']

"""rrrr_lattice_engine CLI."""

import argparse
import sys

from rrrr_lattice_engine import create_rrrr_lattice_engine


def main():
    parser = argparse.ArgumentParser(description="4D eigenvalue lattice engine implementing R(R)=R multiplicative basis with φ⁻¹, e⁻¹, π⁻¹, √2⁻¹ decomposition")
    parser.add_argument("--config", type=str, help="Config file path")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    try:
        instance = create_rrrr_lattice_engine()
        instance.run()
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
