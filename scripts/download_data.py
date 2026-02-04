#!/usr/bin/env python3
"""Download datasets for spectral symmetry analysis."""

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets.modelnet import ModelNet10Dataset, ModelNet40Dataset


def download_modelnet(variant, data_dir):
    """Download and extract a ModelNet variant."""
    if variant == 10:
        ModelNet10Dataset(root_dir=data_dir, download=True)
    elif variant == 40:
        ModelNet40Dataset(root_dir=data_dir, download=True)
    else:
        raise ValueError(f"Unknown ModelNet variant: {variant}")


def download_symmetry3d():
    """Print manual download instructions for Symmetry3D."""
    print(
        "Symmetry3D dataset requires manual download:\n"
        "  1. Visit https://github.com/GrailLab/Symmetry-3D\n"
        "  2. Download the dataset archive\n"
        "  3. Extract so that PLY files are at:\n"
        "       data/raw/3d-global-sym/{split}/*.ply\n"
    )


def main():
    parser = argparse.ArgumentParser(description="Download datasets")
    parser.add_argument('--datasets', nargs='+', default=['modelnet10'],
                        choices=['modelnet10', 'modelnet40', 'symmetry3d'],
                        help='Datasets to download')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Root data directory')
    args = parser.parse_args()

    for ds in args.datasets:
        if ds == 'modelnet10':
            download_modelnet(10, args.data_dir)
        elif ds == 'modelnet40':
            download_modelnet(40, args.data_dir)
        elif ds == 'symmetry3d':
            download_symmetry3d()


if __name__ == '__main__':
    main()
