__all__ = ['main']

"""rrrr_completeness_prover CLI."""

import argparse
import sys

from rrrr_completeness_prover import create_rrrr_completeness_prover


def main():
    parser = argparse.ArgumentParser(description="Completeness theorem prover for R(R)=R 4D lattice: verifies self-referential operator spectra lie within epsilon of lattice")
    parser.add_argument("--config", type=str, help="Config file path")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    try:
        instance = create_rrrr_completeness_prover()
        instance.run()
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
