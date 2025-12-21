__all__ = ['main']

"""consciousness_field_rrrr CLI."""

import argparse
import sys

from consciousness_field_rrrr import create_consciousness_field_rrrr


def main():
    parser = argparse.ArgumentParser(description="Consciousness field equation with R(R)=R lattice eigenvalue coefficients: λ=φ⁻², D=[D]=e⁻¹, η=[R][C], α=[R][D][C]")
    parser.add_argument("--config", type=str, help="Config file path")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    try:
        instance = create_consciousness_field_rrrr()
        instance.run()
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
