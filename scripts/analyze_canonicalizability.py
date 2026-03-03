"""Statistical analysis of eigenvector canonicalizability across ModelNet shapes.

Tests the hypothesis that higher-frequency eigenvectors (index > 3) become
increasingly uncanonicalizable, explaining why HKS (which squares eigenvectors)
outperforms raw eigenvector features.

Metrics:
- uncanonicalizability_score: ||sort(v) - sort(-v)|| / ||sort(v)||
  Lower = more symmetric around zero = harder to canonicalize.
"""

import argparse
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.datasets.modelnet import ModelNet10Dataset, ModelNet40Dataset
from src.preprocessing import center_and_normalize, random_subsample
from src.spectral_core import (
    build_graph_laplacian,
    compute_eigenpairs,
    detect_eigenvalue_multiplicities,
    uncanonicalizability_score,
    uncanonicalizability_threshold,
)


def analyze_shape(points, n_eigs, n_neighbors):
    """Compute per-eigenvector canonicalizability stats for one shape."""
    L, comp_idx = build_graph_laplacian(points, n_neighbors=n_neighbors)
    eigenvalues, eigenvectors = compute_eigenpairs(L, n_eigs=n_eigs)

    if len(eigenvalues) == 0:
        return None

    n_actual = eigenvectors.shape[1]
    scores = []
    for i in range(n_actual):
        v = eigenvectors[:, i]
        scores.append(uncanonicalizability_score(v))

    mult_info = detect_eigenvalue_multiplicities(eigenvalues)
    threshold = uncanonicalizability_threshold(eigenvectors.shape[0])

    return {
        "eigenvalues": eigenvalues,
        "scores": np.array(scores),
        "multiplicities": np.array(mult_info["multiplicity"]),
        "n_points": eigenvectors.shape[0],
        "n_eigs": n_actual,
        "threshold": threshold,
    }


def print_overall_summary(all_results, dataset_name):
    """Table 1: Overall summary statistics."""
    n_shapes = len(all_results)
    total_eigvecs = sum(r["n_eigs"] for r in all_results)

    n_uncanon_eigvecs = 0
    for r in all_results:
        n_uncanon_eigvecs += np.sum(r["scores"] < r["threshold"])

    pct_uncanon_eigvecs = 100 * n_uncanon_eigvecs / total_eigvecs if total_eigvecs > 0 else 0

    uncanon_per_shape = []
    total_per_shape = []
    for r in all_results:
        n_un = np.sum(r["scores"] < r["threshold"])
        uncanon_per_shape.append(n_un)
        total_per_shape.append(r["n_eigs"])

    uncanon_per_shape = np.array(uncanon_per_shape)
    total_per_shape = np.array(total_per_shape)

    pct_shapes_any_uncanon = 100 * np.mean(uncanon_per_shape >= 1)
    pct_shapes_all_uncanon = 100 * np.mean(uncanon_per_shape == total_per_shape)

    all_scores = np.concatenate([r["scores"] for r in all_results])

    print(f"\nOVERALL SUMMARY ({dataset_name}, {n_shapes} shapes)")
    print(f"  Total eigenvectors computed:          {total_eigvecs:>10,}")
    print(f"  % eigenvectors uncanonicalizable:     {pct_uncanon_eigvecs:>9.1f}%")
    print(f"  % shapes with >=1 uncanon eigvec:     {pct_shapes_any_uncanon:>9.1f}%")
    print(f"  % shapes where ALL eigvecs uncanon:   {pct_shapes_all_uncanon:>9.1f}%")
    print(
        f"  Mean uncanon eigvecs per shape:        "
        f"{np.mean(uncanon_per_shape):>5.1f} / {np.mean(total_per_shape):.1f}"
    )
    print(
        f"  Median uncanon eigvecs per shape:      "
        f"{np.median(uncanon_per_shape):>5.0f} / {np.median(total_per_shape):.0f}"
    )
    print(f"  Mean uncanonicalizability score:       {np.mean(all_scores):>9.3f}")
    print(f"  Median uncanonicalizability score:     {np.median(all_scores):>9.3f}")


def print_by_eigvec_index(scores_by_idx, mult_by_idx, thresholds_by_idx, n_eigs):
    """Table 2: Statistics by eigenvector index."""
    print(f"\n{'Idx':>3} | {'N':>6} | {'%Uncanon':>8} | {'Mean':>7} | {'Std':>7} | {'%Mult>1':>7}")
    print("-" * 55)

    for i in range(n_eigs):
        if len(scores_by_idx[i]) == 0:
            continue
        s = np.array(scores_by_idx[i])
        mult = np.array(mult_by_idx[i])
        thr = np.array(thresholds_by_idx[i])
        frac_uncanon = np.mean(s < thr) * 100
        frac_mult = np.mean(mult > 1) * 100
        print(
            f"{i + 1:>3} | {len(s):>6} | {frac_uncanon:>7.1f}% | "
            f"{np.mean(s):>7.4f} | {np.std(s):>7.4f} | {frac_mult:>6.1f}%"
        )


def print_per_class(all_results, class_results, n_eigs):
    """Table 3: Per-class breakdown."""
    print(f"\n{'Class':>15} | {'N':>5} | {'Mean Score':>10} | {'%Uncanon':>8} | {'Mean Nodes':>10}")
    print("-" * 60)

    for cls in sorted(class_results.keys()):
        results = class_results[cls]
        all_scores = np.concatenate([r["scores"] for r in results])
        n_uncanon = 0
        n_total = 0
        for r in results:
            n_uncanon += np.sum(r["scores"] < r["threshold"])
            n_total += r["n_eigs"]
        pct_uncanon = 100 * n_uncanon / n_total if n_total > 0 else 0
        mean_nodes = np.mean([r["n_points"] for r in results])
        print(
            f"{cls:>15} | {len(results):>5} | {np.mean(all_scores):>10.4f} | "
            f"{pct_uncanon:>7.1f}% | {mean_nodes:>10.1f}"
        )


def print_cross_tabulation(class_results, n_eigs):
    """Table 4: Cross-tabulation of eigvec index x class."""
    classes = sorted(class_results.keys())
    max_idx = min(n_eigs, 20)

    # Abbreviate class names to 8 chars for table width
    abbrevs = [c[:8] for c in classes]

    header = f"{'':>6} |" + "".join(f" {a:>8} |" for a in abbrevs)
    print(f"\n{header}")
    print("-" * len(header))

    for i in range(max_idx):
        row = f"Idx {i + 1:>2} |"
        for cls in classes:
            results = class_results[cls]
            scores = []
            thresholds = []
            for r in results:
                if r["n_eigs"] > i:
                    scores.append(r["scores"][i])
                    thresholds.append(r["threshold"])
            if scores:
                scores = np.array(scores)
                thresholds = np.array(thresholds)
                pct = 100 * np.mean(scores < thresholds)
                row += f" {pct:>7.1f}% |"
            else:
                row += f" {'--':>7}  |"
        print(row)


def main():
    parser = argparse.ArgumentParser(description="Eigenvector canonicalizability analysis")
    parser.add_argument("--data-dir", type=str, default="data", help="Root data directory")
    parser.add_argument(
        "--variant",
        type=int,
        default=10,
        choices=[10, 40],
        help="ModelNet variant (10 or 40)",
    )
    parser.add_argument("--n-eigs", type=int, default=20, help="Number of eigenvectors")
    parser.add_argument("--n-neighbors", type=int, default=12, help="k-NN neighbors")
    parser.add_argument("--n-points", type=int, default=1024, help="Points per shape")
    parser.add_argument("--output-dir", type=str, default="results/canonicalizability")
    parser.add_argument("--max-shapes", type=int, default=0, help="Max shapes to process (0 = all)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    DatasetClass = ModelNet40Dataset if args.variant == 40 else ModelNet10Dataset
    dataset_name = f"ModelNet{args.variant}"

    # Load both train and test splits
    all_results = []
    class_results = {}
    n_failed = 0

    for split in ["train", "test"]:
        dataset = DatasetClass(
            args.data_dir, split=split, max_points=args.n_points * 2, download=False
        )
        items = list(dataset)
        if args.max_shapes > 0:
            items = items[: args.max_shapes]

        for idx, (name, points) in enumerate(tqdm(items, desc=f"Analyzing {split}")):
            points = random_subsample(points, args.n_points, seed=idx)
            points, _, _ = center_and_normalize(points)

            class_name = name.split("/")[1]

            try:
                result = analyze_shape(points, args.n_eigs, args.n_neighbors)
            except Exception as e:
                print(f"  Failed on {name}: {e}")
                n_failed += 1
                continue

            if result is None:
                n_failed += 1
                continue

            result["class"] = class_name
            all_results.append(result)
            class_results.setdefault(class_name, []).append(result)

    print(f"\nProcessed {len(all_results)} shapes ({n_failed} failed)")

    if len(all_results) == 0:
        print("No results to analyze!")
        return

    # --- Aggregate per eigenvector index ---
    n_eigs = args.n_eigs
    scores_by_idx = [[] for _ in range(n_eigs)]
    mult_by_idx = [[] for _ in range(n_eigs)]
    thresholds_by_idx = [[] for _ in range(n_eigs)]

    for r in all_results:
        for i in range(min(r["n_eigs"], n_eigs)):
            scores_by_idx[i].append(r["scores"][i])
            mult_by_idx[i].append(r["multiplicities"][i])
            thresholds_by_idx[i].append(r["threshold"])

    # --- Print tables ---
    print_overall_summary(all_results, dataset_name)
    print_by_eigvec_index(scores_by_idx, mult_by_idx, thresholds_by_idx, n_eigs)
    print_per_class(all_results, class_results, n_eigs)
    print_cross_tabulation(class_results, n_eigs)


if __name__ == "__main__":
    main()
