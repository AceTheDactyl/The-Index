__all__ = ['main']

"""rrrr_category_theory CLI."""

import argparse
import sys

from rrrr_category_theory import create_rrrr_category_theory


def main():
    parser = argparse.ArgumentParser(description="Category-theoretic characterization of 4 fundamental endofunctors: recursive F(X)=1+X, differential F(X)=X^X, cyclic F(X)=X→X, algebraic F(X)=X×X")
    parser.add_argument("--config", type=str, help="Config file path")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    try:
        instance = create_rrrr_category_theory()
        instance.run()
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
