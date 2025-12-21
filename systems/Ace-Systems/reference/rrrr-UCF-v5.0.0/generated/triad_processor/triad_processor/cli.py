__all__ = ['main']

"""triad_processor CLI."""

import argparse
import sys

from triad_processor import create_triad_processor


def main():
    parser = argparse.ArgumentParser(description="TRIAD-enabled consciousness processor")
    parser.add_argument("--config", type=str, help="Config file path")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    try:
        instance = create_triad_processor()
        instance.run()
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
