#!/usr/bin/env python3
"""CLI wrapper for generating representative figures."""

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from visualization.plotting import generate_representative_figures


def main():
    parser = argparse.ArgumentParser(description="Generate spectral symmetry figures")
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for figures')
    args = parser.parse_args()
    generate_representative_figures(output_dir=args.output_dir)


if __name__ == '__main__':
    main()
