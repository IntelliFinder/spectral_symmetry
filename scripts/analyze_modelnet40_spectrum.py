#!/usr/bin/env python
"""Spectral analysis of ModelNet40: compute statistics and plots for eigenvalue multiplicities and uncanonicalizability."""

import argparse
import json
import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.datasets.modelnet import ModelNet40Dataset  # noqa: E402
from src.preprocessing import center_and_normalize, random_subsample  # noqa: E402
from src.spectral_core import analyze_spectrum  # noqa: E402


def process_shape(name, points, n_points, n_eigs, n_neighbors, seed):
    """Process a single shape and return spectral analysis results."""
    try:
        points = random_subsample(points, n_points, seed=seed)
        points, _, _ = center_and_normalize(points)
        result = analyze_spectrum(points, n_eigs=n_eigs, n_neighbors=n_neighbors)
        return {
            "name": name,
            "scores": result["scores"],
            "uncanonicalizable": result["uncanonicalizable"],
            "multiplicity": result["multiplicity_info"]["multiplicity"],
            "n_repeating": result["multiplicity_info"]["n_repeating"],
            "n_non_repeating": result["multiplicity_info"]["n_non_repeating"],
            "n_eigs_actual": len(result["scores"]),
        }
    except Exception as e:
        return {"name": name, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Analyze ModelNet40 spectral properties")
    parser.add_argument("--data-dir", type=str, default="data", help="Root data directory")
    parser.add_argument("--n-points", type=int, default=512, help="Points per shape")
    parser.add_argument("--n-eigs", type=int, default=20, help="Number of eigenvectors")
    parser.add_argument("--n-neighbors", type=int, default=12, help="k-NN neighbors for graph")
    parser.add_argument("--n-jobs", type=int, default=4, help="Number of parallel jobs")
    parser.add_argument("--download", action="store_true", help="Download ModelNet40 if missing")
    parser.add_argument("--output-dir", type=str, default="results/modelnet40_analysis", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load both train and test splits
    print("Loading ModelNet40 train split...")
    train_ds = ModelNet40Dataset(
        args.data_dir,
        split="train",
        max_points=args.n_points * 2,
        download=args.download,
    )
    print("Loading ModelNet40 test split...")
    test_ds = ModelNet40Dataset(
        args.data_dir,
        split="test",
        max_points=args.n_points * 2,
        download=False,
    )

    # Collect all shapes
    all_shapes = []
    print("Collecting shapes...")
    for name, points in tqdm(train_ds, desc="Train"):
        all_shapes.append((name, points))
    for name, points in tqdm(test_ds, desc="Test"):
        all_shapes.append((name, points))
    print(f"Total shapes: {len(all_shapes)}")

    # Process shapes in parallel
    print(f"Analyzing spectra with {args.n_jobs} parallel jobs...")
    results = Parallel(n_jobs=args.n_jobs, backend="loky")(
        delayed(process_shape)(
            name, points, args.n_points, args.n_eigs, args.n_neighbors, seed=idx
        )
        for idx, (name, points) in enumerate(tqdm(all_shapes, desc="Processing"))
    )

    # Filter out errors
    valid_results = [r for r in results if "error" not in r]
    n_errors = len(results) - len(valid_results)
    if n_errors > 0:
        print(f"Warning: {n_errors} shapes failed processing")

    n_shapes = len(valid_results)
    print(f"Successfully processed {n_shapes} shapes")

    # ─── Compute Statistics ────────────────────────────────────────────────────

    # 1. Multiplicity distribution
    mult_1_counts = []
    mult_2_counts = []
    mult_3plus_counts = []

    for r in valid_results:
        mult = r["multiplicity"]
        mult_1 = sum(1 for m in mult if m == 1)
        mult_2 = sum(1 for m in mult if m == 2)
        mult_3plus = sum(1 for m in mult if m >= 3)
        mult_1_counts.append(mult_1)
        mult_2_counts.append(mult_2)
        mult_3plus_counts.append(mult_3plus)

    avg_mult_1 = float(np.mean(mult_1_counts))
    avg_mult_2 = float(np.mean(mult_2_counts))
    avg_mult_3plus = float(np.mean(mult_3plus_counts))

    # 2. Uncanonicalizable eigenvectors
    uncanon_counts = [sum(r["uncanonicalizable"]) for r in valid_results]
    avg_uncanon = float(np.mean(uncanon_counts))
    std_uncanon = float(np.std(uncanon_counts))
    max_uncanon = int(np.max(uncanon_counts))
    min_uncanon = int(np.min(uncanon_counts))

    # 3. First 10 eigenvectors simple spectrum
    first10_simple_counts = []
    for r in valid_results:
        mult = r["multiplicity"][:10] if len(r["multiplicity"]) >= 10 else r["multiplicity"]
        simple_count = sum(1 for m in mult if m == 1)
        first10_simple_counts.append(simple_count)
    avg_first10_simple = float(np.mean(first10_simple_counts))

    # 4. Per-eigenvector-index scores
    max_eigs = args.n_eigs
    scores_by_index = [[] for _ in range(max_eigs)]
    uncanon_by_index = [[] for _ in range(max_eigs)]

    for r in valid_results:
        for i, score in enumerate(r["scores"]):
            if i < max_eigs:
                scores_by_index[i].append(score)
        for i, uncanon in enumerate(r["uncanonicalizable"]):
            if i < max_eigs:
                uncanon_by_index[i].append(int(uncanon))

    avg_scores_by_index = [float(np.mean(s)) if s else 0.0 for s in scores_by_index]
    std_scores_by_index = [float(np.std(s)) if s else 0.0 for s in scores_by_index]
    avg_uncanon_rate_by_index = [float(np.mean(u)) if u else 0.0 for u in uncanon_by_index]

    # ─── Build Statistics Dictionary ───────────────────────────────────────────

    # Compute dynamic threshold (based on n_points)
    threshold = 5.0 / np.sqrt(args.n_points)

    stats = {
        "dataset": "ModelNet40",
        "n_shapes": n_shapes,
        "n_errors": n_errors,
        "n_points": args.n_points,
        "n_eigs": args.n_eigs,
        "n_neighbors": args.n_neighbors,
        "threshold": float(threshold),
        "multiplicity_distribution": {
            "avg_multiplicity_1": avg_mult_1,
            "avg_multiplicity_2": avg_mult_2,
            "avg_multiplicity_3plus": avg_mult_3plus,
        },
        "uncanonicalizable_eigenvectors": {
            "avg_count": avg_uncanon,
            "std_count": std_uncanon,
            "max_count": max_uncanon,
            "min_count": min_uncanon,
        },
        "first_10_eigenvectors": {
            "avg_simple_spectrum_count": avg_first10_simple,
        },
        "per_eigenvector_index": {
            "avg_uncanonicalizability_score": avg_scores_by_index,
            "std_uncanonicalizability_score": std_scores_by_index,
            "avg_uncanonicalizable_rate": avg_uncanon_rate_by_index,
        },
    }

    # Save statistics
    stats_path = output_dir / "modelnet40_spectral_statistics.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Statistics saved to {stats_path}")

    # ─── Print Summary ─────────────────────────────────────────────────────────

    print("\n" + "=" * 70)
    print("MODELNET40 SPECTRAL ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"  Total shapes analyzed: {n_shapes}")
    print(f"  Points per shape: {args.n_points}")
    print(f"  Eigenvectors computed: {args.n_eigs}")
    print(f"  Uncanonicalizability threshold: {threshold:.4f}")
    print()
    print("Multiplicity Distribution (out of {} eigenvalues):".format(args.n_eigs))
    print(f"  Avg multiplicity-1 (simple): {avg_mult_1:.2f}")
    print(f"  Avg multiplicity-2 (double): {avg_mult_2:.2f}")
    print(f"  Avg multiplicity-3+ (higher): {avg_mult_3plus:.2f}")
    print()
    print("Uncanonicalizable Eigenvectors:")
    print(f"  Average count: {avg_uncanon:.2f}")
    print(f"  Std deviation: {std_uncanon:.2f}")
    print(f"  Maximum count: {max_uncanon}")
    print(f"  Minimum count: {min_uncanon}")
    print()
    print("First 10 Eigenvectors Simple Spectrum:")
    print(f"  Avg multiplicity-1 count: {avg_first10_simple:.2f}")
    print("=" * 70)

    # ─── Generate Plots ────────────────────────────────────────────────────────

    # Plot 1: Eigenvector index vs uncanonicalizability score
    fig, ax = plt.subplots(figsize=(10, 6))
    indices = np.arange(max_eigs)
    ax.errorbar(
        indices,
        avg_scores_by_index,
        yerr=std_scores_by_index,
        fmt="o-",
        capsize=3,
        label="Avg score",
    )
    ax.axhline(y=threshold, color="r", linestyle="--", label=f"Threshold ({threshold:.4f})")
    ax.set_xlabel("Eigenvector Index")
    ax.set_ylabel("Uncanonicalizability Score")
    ax.set_title("ModelNet40: Uncanonicalizability Score by Eigenvector Index")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(indices)

    # Save plot
    plot_path_png = output_dir / "modelnet40_eigvec_score_vs_index.png"
    plot_path_pdf = output_dir / "modelnet40_eigvec_score_vs_index.pdf"
    fig.savefig(plot_path_png, dpi=150, bbox_inches="tight")
    fig.savefig(plot_path_pdf, bbox_inches="tight")
    print(f"Plot saved to {plot_path_png} and {plot_path_pdf}")
    plt.close(fig)

    # Plot 2: Multiplicity distribution bar chart
    fig, ax = plt.subplots(figsize=(8, 6))
    categories = ["Multiplicity 1\n(Simple)", "Multiplicity 2\n(Double)", "Multiplicity 3+\n(Higher)"]
    values = [avg_mult_1, avg_mult_2, avg_mult_3plus]
    bars = ax.bar(categories, values, color=["#2ecc71", "#3498db", "#e74c3c"])
    ax.set_ylabel(f"Average Count (out of {args.n_eigs} eigenvalues)")
    ax.set_title("ModelNet40: Eigenvalue Multiplicity Distribution")
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(
            f"{val:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=12,
        )

    # Save plot
    plot_path_png = output_dir / "modelnet40_multiplicity_distribution.png"
    plot_path_pdf = output_dir / "modelnet40_multiplicity_distribution.pdf"
    fig.savefig(plot_path_png, dpi=150, bbox_inches="tight")
    fig.savefig(plot_path_pdf, bbox_inches="tight")
    print(f"Plot saved to {plot_path_png} and {plot_path_pdf}")
    plt.close(fig)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
