#!/usr/bin/env python
"""Create per-hidden-dim convergence plots and preprocessing time benchmarks.

Part 1: For each hidden_dim, plot test ROC-AUC vs epoch with one line per
canonicalization method (mean +/- std over seeds).

Part 2: Benchmark preprocessing time per canonicalization method on a sample
of moltox21 graphs.

Usage
-----
    python scripts/plot_moltox21_results.py
    python scripts/plot_moltox21_results.py --results-dir results/canonicalization_experiments
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CANONICALIZATIONS = [
    "random_fixed",
    "random_augmented",
    "maxabs",
    "spielman",
    "map",
    "oap",
]
CANON_COLORS = {
    "random_fixed": "#888888",
    "random_augmented": "#bbbbbb",
    "maxabs": "#e67e22",
    "spielman": "#2980b9",
    "map": "#27ae60",
    "oap": "#8e44ad",
}
CANON_LABELS = {
    "random_fixed": "Random (fixed)",
    "random_augmented": "Random (aug)",
    "maxabs": "MaxAbs",
    "spielman": "Spielman",
    "map": "MAP",
    "oap": "OAP",
}


def load_epoch_logs(results_dir):
    """Load epoch logs from exp1_param_efficiency.

    Returns dict: {(canon, hidden_dim): [list of epoch_log arrays]}
    """
    exp_dir = os.path.join(results_dir, "exp1_param_efficiency")
    logs = defaultdict(list)

    for run_dir in sorted(os.listdir(exp_dir)):
        result_path = os.path.join(exp_dir, run_dir, "results.json")
        log_path = os.path.join(exp_dir, run_dir, "epoch_log.json")
        if not os.path.exists(result_path) or not os.path.exists(log_path):
            continue

        with open(result_path) as f:
            r = json.load(f)
        with open(log_path) as f:
            epoch_log = json.load(f)

        canon = r["canonicalization"]
        hdim = r.get("hidden_dim", 256)
        logs[(canon, hdim)].append(epoch_log)

    return logs


def plot_convergence_per_hdim(logs, output_dir):
    """Create one convergence plot per hidden_dim."""
    hidden_dims = sorted(set(h for (_, h) in logs.keys()))

    for hdim in hidden_dims:
        fig, ax = plt.subplots(figsize=(10, 6))

        for canon in CANONICALIZATIONS:
            key = (canon, hdim)
            if key not in logs:
                continue

            run_logs = logs[key]
            max_epochs = max(len(log) for log in run_logs)

            # Build aligned arrays
            test_curves = np.full((len(run_logs), max_epochs), np.nan)
            for i, log in enumerate(run_logs):
                for entry in log:
                    ep = entry["epoch"] - 1
                    if ep < max_epochs:
                        test_curves[i, ep] = entry["test_metric"]

            epochs = np.arange(1, max_epochs + 1)
            mean_test = np.nanmean(test_curves, axis=0)
            std_test = np.nanstd(test_curves, axis=0)

            color = CANON_COLORS.get(canon, "gray")
            label = CANON_LABELS.get(canon, canon)

            ax.plot(epochs, mean_test, label=label, color=color, linewidth=1.5)
            ax.fill_between(
                epochs,
                mean_test - std_test,
                mean_test + std_test,
                alpha=0.15,
                color=color,
            )

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Test ROC-AUC", fontsize=12)
        ax.set_title(f"Convergence: hidden_dim = {hdim}", fontsize=14)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        fname = f"convergence_h{hdim}.pdf"
        fig.savefig(os.path.join(output_dir, fname), dpi=150)
        print(f"  Saved {fname}")
        plt.close(fig)

    # Also create a combined 2x3 or 3x2 grid
    n_dims = len(hidden_dims)
    if n_dims >= 2:
        ncols = min(3, n_dims)
        nrows = (n_dims + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
        if nrows == 1:
            axes = [axes] if ncols == 1 else list(axes)
        else:
            axes = [ax for row in axes for ax in row]

        for idx, hdim in enumerate(hidden_dims):
            ax = axes[idx]
            for canon in CANONICALIZATIONS:
                key = (canon, hdim)
                if key not in logs:
                    continue

                run_logs = logs[key]
                max_epochs = max(len(log) for log in run_logs)
                test_curves = np.full((len(run_logs), max_epochs), np.nan)
                for i, log in enumerate(run_logs):
                    for entry in log:
                        ep = entry["epoch"] - 1
                        if ep < max_epochs:
                            test_curves[i, ep] = entry["test_metric"]

                epochs = np.arange(1, max_epochs + 1)
                mean_test = np.nanmean(test_curves, axis=0)
                std_test = np.nanstd(test_curves, axis=0)

                color = CANON_COLORS.get(canon, "gray")
                label = CANON_LABELS.get(canon, canon)

                ax.plot(epochs, mean_test, label=label, color=color, linewidth=1.2)
                ax.fill_between(
                    epochs,
                    mean_test - std_test,
                    mean_test + std_test,
                    alpha=0.12,
                    color=color,
                )

            ax.set_xlabel("Epoch", fontsize=10)
            ax.set_ylabel("Test ROC-AUC", fontsize=10)
            ax.set_title(f"h = {hdim}", fontsize=12)
            ax.grid(True, alpha=0.3)
            if idx == 0:
                ax.legend(fontsize=7)

        # Hide unused axes
        for idx in range(n_dims, len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle(
            "ogbg-moltox21: Test ROC-AUC vs Epoch by Hidden Dimension",
            fontsize=14,
            y=1.01,
        )
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "convergence_all_hdims.pdf"), dpi=150)
        print("  Saved convergence_all_hdims.pdf")
        plt.close(fig)


def benchmark_preprocessing(output_dir, n_graphs=200):
    """Benchmark preprocessing time per canonicalization method."""
    from scipy.sparse.linalg import eigsh

    from src.spectral_canonicalization import (
        canonicalize,
        canonicalize_maxabs,
        canonicalize_random_fixed,
        spectral_canonicalize,
        spectral_canonicalize_map,
        spectral_canonicalize_oap,
    )

    from scipy.sparse import coo_matrix, diags

    n_eigs = 8

    # Try loading OGB dataset; fall back to synthetic graphs matching moltox21
    dataset = None
    try:
        import torch

        torch.serialization.add_safe_globals([])  # ensure torch is imported
        from ogb.graphproppred import PygGraphPropPredDataset

        dataset = PygGraphPropPredDataset(name="ogbg-moltox21", root="data")
    except Exception:
        pass

    eigenpairs = []
    t0 = time.time()

    if dataset is not None:
        indices = np.random.RandomState(42).choice(
            len(dataset), min(n_graphs, len(dataset)), replace=False
        )
        print(f"\n  Computing eigenpairs for {len(indices)} OGB graphs...")
        for graph_idx in indices:
            data = dataset[int(graph_idx)]
            edge_index = data.edge_index.numpy()
            n = data.num_nodes

            rows = np.concatenate([edge_index[0], edge_index[1]])
            cols = np.concatenate([edge_index[1], edge_index[0]])
            vals = np.ones(len(rows), dtype=np.float64)
            A = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
            A.setdiag(0)
            A.eliminate_zeros()
            deg = np.array(A.sum(axis=1)).flatten()
            L = diags(deg) - A

            try:
                k = min(n_eigs + 1, n - 1)
                if k < 2:
                    continue
                eigenvalues, eigenvectors = eigsh(L, k=k, which="SM", tol=1e-6)
                mask = eigenvalues > 1e-8
                eigenvalues = eigenvalues[mask][:n_eigs]
                eigenvectors = eigenvectors[:, mask][:, :n_eigs]
                if len(eigenvalues) == 0:
                    continue
                eigenpairs.append((eigenvalues, eigenvectors))
            except Exception:
                continue
    else:
        # Synthetic graphs matching moltox21 statistics (15-50 nodes, sparse)
        print(f"\n  OGB load failed; using {n_graphs} synthetic molecular graphs...")
        rng = np.random.RandomState(42)
        for i in range(n_graphs):
            n = rng.randint(15, 50)
            # Random sparse adjacency (avg degree ~3, like molecules)
            n_edges = min(n * 3 // 2, n * (n - 1) // 2)
            edges = set()
            while len(edges) < n_edges:
                u, v = rng.randint(0, n, 2)
                if u != v:
                    edges.add((min(u, v), max(u, v)))
            rows = np.array([e[0] for e in edges] + [e[1] for e in edges])
            cols = np.array([e[1] for e in edges] + [e[0] for e in edges])
            vals = np.ones(len(rows), dtype=np.float64)
            A = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
            deg = np.array(A.sum(axis=1)).flatten()
            L = diags(deg) - A

            try:
                k = min(n_eigs + 1, n - 1)
                if k < 2:
                    continue
                eigenvalues, eigenvectors = eigsh(L, k=k, which="SM", tol=1e-6)
                mask = eigenvalues > 1e-8
                eigenvalues = eigenvalues[mask][:n_eigs]
                eigenvectors = eigenvectors[:, mask][:, :n_eigs]
                if len(eigenvalues) == 0:
                    continue
                eigenpairs.append((eigenvalues, eigenvectors))
            except Exception:
                continue

    eigen_time = time.time() - t0
    print(
        f"  Eigendecomposition: {eigen_time:.2f}s for {len(eigenpairs)} graphs "
        f"({eigen_time / max(len(eigenpairs), 1) * 1000:.2f} ms/graph)"
    )

    # Benchmark each canonicalization method
    methods_to_bench = {
        "none (eigen only)": lambda ev, evec, idx: evec.copy(),
        "random_fixed": lambda ev, evec, idx: canonicalize_random_fixed(evec, idx),
        "maxabs": lambda ev, evec, idx: canonicalize_maxabs(evec),
        "spielman": lambda ev, evec, idx: spectral_canonicalize(evec, ev),
        "map": lambda ev, evec, idx: spectral_canonicalize_map(evec, ev),
        "oap": lambda ev, evec, idx: spectral_canonicalize_oap(evec, ev),
    }

    timing_results = {}
    n_reps = 3  # average over repetitions

    for method_name, fn in methods_to_bench.items():
        times = []
        for rep in range(n_reps):
            t0 = time.time()
            for idx, (ev, evec) in enumerate(eigenpairs):
                fn(ev, evec, idx)
            elapsed = time.time() - t0
            times.append(elapsed)
        mean_t = np.mean(times)
        per_graph_ms = mean_t / len(eigenpairs) * 1000
        timing_results[method_name] = {
            "total_s": float(mean_t),
            "per_graph_ms": float(per_graph_ms),
            "n_graphs": len(eigenpairs),
        }
        print(
            f"  {method_name:25s}: {per_graph_ms:.3f} ms/graph "
            f"({mean_t:.3f}s total for {len(eigenpairs)} graphs)"
        )

    # Add eigendecomposition time
    timing_results["eigendecomposition"] = {
        "total_s": float(eigen_time),
        "per_graph_ms": float(eigen_time / max(len(eigenpairs), 1) * 1000),
        "n_graphs": len(eigenpairs),
    }

    # Save timing data
    with open(os.path.join(output_dir, "preprocessing_timing.json"), "w") as f:
        json.dump(timing_results, f, indent=2)

    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 5))
    methods = list(timing_results.keys())
    per_graph_ms = [timing_results[m]["per_graph_ms"] for m in methods]

    colors = []
    for m in methods:
        if m in CANON_COLORS:
            colors.append(CANON_COLORS[m])
        elif m == "eigendecomposition":
            colors.append("#c0392b")
        else:
            colors.append("#95a5a6")

    bars = ax.bar(range(len(methods)), per_graph_ms, color=colors)
    ax.set_xticks(range(len(methods)))
    labels = [CANON_LABELS.get(m, m.replace("_", " ").title()) for m in methods]
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Time per graph (ms)", fontsize=12)
    ax.set_title("Preprocessing Time per Canonicalization Method\n(ogbg-moltox21)", fontsize=13)

    # Add value labels
    for bar, val in zip(bars, per_graph_ms):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "preprocessing_timing.pdf"), dpi=150)
    print(f"  Saved preprocessing_timing.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Create moltox21 convergence plots and preprocessing benchmarks"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/canonicalization_experiments",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: results_dir/moltox21_plots)",
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Skip preprocessing time benchmark",
    )
    parser.add_argument(
        "--n-benchmark-graphs",
        type=int,
        default=200,
        help="Number of graphs for benchmark",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.results_dir, "moltox21_plots")
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Part 1: Per-hidden-dim convergence plots")
    print("=" * 60)

    logs = load_epoch_logs(args.results_dir)
    print(f"Loaded {sum(len(v) for v in logs.values())} epoch logs")
    if logs:
        plot_convergence_per_hdim(logs, args.output_dir)

    if not args.skip_benchmark:
        print("\n" + "=" * 60)
        print("Part 2: Preprocessing time benchmark")
        print("=" * 60)
        benchmark_preprocessing(args.output_dir, n_graphs=args.n_benchmark_graphs)

    print(f"\nAll outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
