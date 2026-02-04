#!/usr/bin/env python3
"""CLI for running spectral symmetry analysis across datasets."""

import argparse
import csv
import sys
from pathlib import Path

from joblib import Parallel, delayed
from tqdm import tqdm

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets import (
    DirectoryDataset,
    ModelNet10Dataset,
    ModelNet40Dataset,
    SymmetriaDataset,
    Symmetry3DDataset,
)
from src.metrics import aggregate_results
from src.spectral_core import analyze_spectrum


def analyze_shape(name, points, n_eigs, n_neighbors, threshold):
    """Analyze a single shape. Returns result dict or None on failure."""
    try:
        result = analyze_spectrum(points, n_eigs=n_eigs, n_neighbors=n_neighbors,
                                  threshold=threshold)
        result['name'] = name
        result['n_points'] = len(points)
        return result
    except Exception as e:
        print(f"  Error on {name}: {e}")
        return None


def write_detailed_csv(results, path):
    """Write per-shape, per-eigenvector results to CSV."""
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['shape', 'n_points', 'eig_index', 'eigenvalue',
                         'uncanon_score', 'is_uncanonicalizable',
                         'is_uncanonicalizable_raw', 'multiplicity',
                         'spectral_gap'])
        for r in results:
            mult = r['multiplicity_info']['multiplicity']
            for i in range(len(r['scores'])):
                writer.writerow([
                    r['name'], r['n_points'], i,
                    f"{r['eigenvalues'][i]:.8f}",
                    f"{r['scores'][i]:.6f}",
                    r['uncanonicalizable'][i],
                    r['uncanonicalizable_raw'][i],
                    mult[i],
                    f"{r['spectral_gap']:.8f}",
                ])


def write_summary_table(all_stats, path, threshold_used):
    """Write a markdown summary table comparing datasets."""
    with open(path, 'w') as f:
        f.write("# Spectral Symmetry Analysis — Summary\n\n")
        f.write(f"Threshold: {threshold_used:.4f} (5 / sqrt(N))\n\n")
        f.write("| Dataset | Shapes | Avg Uncanon Ratio | Spectral Gap (mean) "
                "| Fiedler Uncanon Rate |\n")
        f.write("|---------|-------:|------------------:|--------------------:"
                "|---------------------:|\n")
        for s in all_stats:
            f.write(f"| {s.dataset_name} | {s.n_shapes} | {s.avg_uncanon_ratio:.4f} "
                    f"| {s.spectral_gap_mean:.6f} | {s.fiedler_uncanon_rate:.4f} |\n")
        f.write("\nLower uncanonicalizability score \u2192 stronger sign ambiguity.\n")

        # Multiplicity section
        f.write("\n## Eigenvalue Multiplicity\n\n")
        f.write("| Dataset | Shapes | Avg Repeating Eigs | Avg Non-Repeating Eigs |\n")
        f.write("|---------|-------:|-------------------:|-----------------------:|\n")
        for s in all_stats:
            f.write(f"| {s.dataset_name} | {s.n_shapes} | {s.avg_repeating_eigenvalues:.2f} "
                    f"| {s.avg_non_repeating_eigenvalues:.2f} |\n")


def build_dataset(name, args):
    """Construct a dataset object from a name string."""
    if name == 'symmetria':
        return SymmetriaDataset(n_instances=args.n_instances, n_points=args.n_points)
    elif name == 'modelnet10':
        return ModelNet10Dataset(
            root_dir=args.data_dir, split='train',
            max_points=args.n_points, download=args.download,
        )
    elif name == 'modelnet40':
        return ModelNet40Dataset(
            root_dir=args.data_dir, split='train',
            max_points=args.n_points, download=args.download,
        )
    elif name == 'symmetry3d':
        return Symmetry3DDataset(
            root_dir=args.data_dir, split='train', max_points=args.n_points,
        )
    else:
        # Treat as directory path
        return DirectoryDataset(name, max_points=args.n_points)


def run_dataset(dataset, args):
    """Run analysis on an entire dataset, return (results_list, stats)."""
    items = list(dataset)
    print(f"\n=== {dataset} — {len(items)} shapes ===")

    if args.n_jobs == 1:
        results = []
        for name, pts in tqdm(items, desc=str(dataset)):
            r = analyze_shape(name, pts, args.n_eigs, args.n_neighbors, args.threshold)
            if r is not None:
                results.append(r)
    else:
        results = Parallel(n_jobs=args.n_jobs, backend='loky')(
            delayed(analyze_shape)(name, pts, args.n_eigs, args.n_neighbors, args.threshold)
            for name, pts in tqdm(items, desc=str(dataset))
        )
        results = [r for r in results if r is not None]

    dataset_name = getattr(dataset, 'root_dir', type(dataset).__name__)
    stats = aggregate_results(results, dataset_name=str(dataset_name))
    return results, stats


def main():
    parser = argparse.ArgumentParser(description="Spectral symmetry analysis pipeline")
    parser.add_argument('--datasets', nargs='+', default=['symmetria'],
                        help='Dataset names or directory paths '
                             '(symmetria, modelnet10, modelnet40, symmetry3d, or a path)')
    parser.add_argument('--n-eigs', type=int, default=20)
    parser.add_argument('--n-neighbors', type=int, default=12)
    parser.add_argument('--n-points', type=int, default=1024)
    parser.add_argument('--n-instances', type=int, default=10,
                        help='Instances per shape type for Symmetria')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Uncanon threshold (default: 5/sqrt(N))')
    parser.add_argument('--n-jobs', type=int, default=1,
                        help='Parallel workers (1=sequential)')
    parser.add_argument('--output-dir', type=str, default='results')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Root data directory for ModelNet/Symmetry3D')
    parser.add_argument('--download', action='store_true',
                        help='Download datasets if not present')
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_results = []
    all_stats = []

    for ds_name in args.datasets:
        try:
            dataset = build_dataset(ds_name, args)
        except FileNotFoundError as e:
            print(f"Skipping {ds_name}: {e}")
            continue

        results, stats = run_dataset(dataset, args)
        all_results.extend(results)
        all_stats.append(stats)

        print(f"  Shapes analyzed: {stats.n_shapes}")
        print(f"  Avg uncanon ratio: {stats.avg_uncanon_ratio:.4f}")
        print(f"  Spectral gap mean: {stats.spectral_gap_mean:.6f}")
        print(f"  Fiedler uncanon rate: {stats.fiedler_uncanon_rate:.4f}")

    # Determine threshold actually used (grab from first result if auto)
    if all_results:
        threshold_used = all_results[0]['threshold_used']
    else:
        threshold_used = args.threshold or 0.0
    print(f"\n  Threshold used: {threshold_used:.2e}")

    # Write outputs
    write_summary_table(all_stats, out / 'summary_table.md', threshold_used)
    write_detailed_csv(all_results, out / 'detailed_results.csv')
    print(f"\nSummary: {out / 'summary_table.md'}")
    print(f"Details: {out / 'detailed_results.csv'}")


if __name__ == '__main__':
    main()
