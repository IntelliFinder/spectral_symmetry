#!/usr/bin/env python
"""Analyze how often full Spielman refinement produces finer partitions than partition-only.

For each graph in ogbg-moltox21 and ogbg-molpcba:
  1. Compute eigenpairs (and time the eigendecomposition)
  2. Run initial abs-value row signature partition (Spielman partition)
  3. Run full balanced-block refinement (Spielman full)
  4. Compare block counts — if full produces more blocks, the GF(2) refinement matters

Also benchmarks eigendecomposition time and appends it to the canonicalization runtimes table.

Usage:
    python scripts/spielman_refinement_analysis.py --dataset ogbg-moltox21
    python scripts/spielman_refinement_analysis.py --dataset ogbg-molpcba
    python scripts/spielman_refinement_analysis.py --dataset both
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0, ".")

from src.spectral_canonicalization import _get_abs_row_signature, find_balanced_blocks
from src.spectral_core import compute_eigenpairs


# ── Helpers ──────────────────────────────────────────────────────────────────


def edge_index_to_laplacian(edge_index, num_nodes):
    """Build normalized Laplacian from edge_index (2, E) numpy array."""
    row, col = edge_index[0], edge_index[1]
    # Adjacency
    data = np.ones(len(row), dtype=np.float64)
    A = csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    # Degree
    deg = np.array(A.sum(axis=1)).flatten()
    # Unnormalized Laplacian: L = D - A
    D = csr_matrix((deg, (np.arange(num_nodes), np.arange(num_nodes))), shape=(num_nodes, num_nodes))
    L = D - A
    return L


def initial_partition_blocks(eigenvectors, precision=8):
    """Compute partition-only blocks (abs-value row signature, no GF(2) refinement)."""
    n = eigenvectors.shape[0]
    partition = {}
    for i in range(n):
        sig = _get_abs_row_signature(eigenvectors[i, :], precision)
        if sig not in partition:
            partition[sig] = []
        partition[sig].append(i)
    return [frozenset(indices) for indices in partition.values()]


def analyze_graph(edge_index, num_nodes, k_values):
    """Analyze a single graph across multiple k values.

    Returns dict with per-k results and eigendecomposition time.
    """
    L = edge_index_to_laplacian(edge_index, num_nodes)

    results = {}
    for k in k_values:
        if num_nodes <= k + 1:
            continue

        # Time eigendecomposition
        t0 = time.perf_counter()
        try:
            eigenvalues, eigenvectors = compute_eigenpairs(L, n_eigs=k)
        except Exception:
            continue
        eigdecomp_ms = (time.perf_counter() - t0) * 1000

        if eigenvectors.shape[1] < k:
            continue

        # Time partition-only
        t0 = time.perf_counter()
        blocks_partition = initial_partition_blocks(eigenvectors)
        partition_ms = (time.perf_counter() - t0) * 1000

        # Time full Spielman
        t0 = time.perf_counter()
        blocks_full = find_balanced_blocks(eigenvectors)
        full_ms = (time.perf_counter() - t0) * 1000

        refined = len(blocks_full) > len(blocks_partition)

        results[k] = {
            "n_nodes": num_nodes,
            "k": k,
            "n_blocks_partition": len(blocks_partition),
            "n_blocks_full": len(blocks_full),
            "refined": refined,
            "eigdecomp_ms": eigdecomp_ms,
            "partition_ms": partition_ms,
            "full_ms": full_ms,
        }

    return results


# ── Dataset Loading ──────────────────────────────────────────────────────────


def load_ogb_dataset(dataset_name, data_dir="data"):
    """Load OGB dataset and return list of (edge_index, num_nodes).

    Uses sliced tensor access for speed (avoids per-graph PyG deserialization).
    """
    import torch
    import torch_geometric.data.data
    import torch_geometric.data.storage

    # Allow OGB cached files to load under PyTorch 2.6+
    for cls_name in ["DataEdgeAttr", "DataTensorAttr"]:
        cls = getattr(torch_geometric.data.data, cls_name, None)
        if cls is not None:
            torch.serialization.add_safe_globals([cls])
    for cls_name in ["GlobalStorage"]:
        cls = getattr(torch_geometric.data.storage, cls_name, None)
        if cls is not None:
            torch.serialization.add_safe_globals([cls])

    from ogb.graphproppred import PygGraphPropPredDataset

    dataset = PygGraphPropPredDataset(name=dataset_name, root=data_dir)

    # Fast extraction via internal slices
    data_obj, slices = dataset._data, dataset.slices
    edge_index_all = data_obj.edge_index.numpy()
    edge_slices = slices["edge_index"].numpy()
    node_slices = slices["x"].numpy() if "x" in slices else None

    n_graphs = len(edge_slices) - 1
    graphs = []
    for i in range(n_graphs):
        ei = edge_index_all[:, edge_slices[i] : edge_slices[i + 1]]
        if node_slices is not None:
            num_nodes = int(node_slices[i + 1] - node_slices[i])
        else:
            num_nodes = int(ei.max()) + 1 if ei.size > 0 else 0
        graphs.append((ei, num_nodes))
    return graphs


# ── Main Analysis ────────────────────────────────────────────────────────────


def run_analysis(dataset_name, k_values, output_dir, sample_n=None):
    """Run full analysis on a dataset."""
    print(f"\n{'=' * 70}")
    print(f"Spielman Refinement Analysis: {dataset_name}")
    print(f"k values: {k_values}")
    print(f"{'=' * 70}")

    graphs = load_ogb_dataset(dataset_name)
    n_total = len(graphs)
    print(f"Loaded {n_total} graphs")

    if sample_n is not None and sample_n < n_total:
        rng = np.random.default_rng(42)
        indices = rng.choice(n_total, size=sample_n, replace=False)
        graphs = [graphs[i] for i in sorted(indices)]
        print(f"  Sampled {sample_n} graphs (seed=42)")

    n_graphs = len(graphs)
    all_results = []
    eigdecomp_times = {k: [] for k in k_values}

    for i, (edge_index, num_nodes) in enumerate(graphs):
        if (i + 1) % 1000 == 0 or i == 0:
            print(f"  Processing graph {i + 1}/{n_graphs}...")

        graph_results = analyze_graph(edge_index, num_nodes, k_values)
        for k, r in graph_results.items():
            all_results.append(r)
            eigdecomp_times[k].append(r["eigdecomp_ms"])

    # Aggregate per k
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"Results: {dataset_name}")
    print(f"{'=' * 70}")

    summary_rows = []
    for k in k_values:
        k_results = [r for r in all_results if r["k"] == k]
        if not k_results:
            continue

        n_total = len(k_results)
        n_refined = sum(1 for r in k_results if r["refined"])
        pct_refined = 100.0 * n_refined / n_total if n_total > 0 else 0

        avg_blocks_part = np.mean([r["n_blocks_partition"] for r in k_results])
        avg_blocks_full = np.mean([r["n_blocks_full"] for r in k_results])
        avg_eigdecomp = np.mean([r["eigdecomp_ms"] for r in k_results])
        median_eigdecomp = np.median([r["eigdecomp_ms"] for r in k_results])
        avg_partition = np.mean([r["partition_ms"] for r in k_results])
        avg_full = np.mean([r["full_ms"] for r in k_results])

        # For refined graphs only
        refined_results = [r for r in k_results if r["refined"]]
        avg_extra_blocks = 0
        if refined_results:
            avg_extra_blocks = np.mean(
                [r["n_blocks_full"] - r["n_blocks_partition"] for r in refined_results]
            )

        print(f"\n  k={k}: {n_total} graphs analyzed")
        print(f"    Graphs where full Spielman refines further: {n_refined}/{n_total} ({pct_refined:.2f}%)")
        print(f"    Avg blocks — partition: {avg_blocks_part:.1f}, full: {avg_blocks_full:.1f}")
        if refined_results:
            print(f"    Avg extra blocks when refined: {avg_extra_blocks:.1f}")
        print(f"    Eigendecomposition: mean={avg_eigdecomp:.2f} ms, median={median_eigdecomp:.2f} ms")
        print(f"    Partition time: mean={avg_partition:.3f} ms")
        print(f"    Full Spielman time: mean={avg_full:.3f} ms")

        summary_rows.append({
            "dataset": dataset_name,
            "k": k,
            "n_graphs": n_total,
            "n_refined": n_refined,
            "pct_refined": f"{pct_refined:.2f}",
            "avg_blocks_partition": f"{avg_blocks_part:.1f}",
            "avg_blocks_full": f"{avg_blocks_full:.1f}",
            "avg_extra_blocks_when_refined": f"{avg_extra_blocks:.1f}",
            "eigdecomp_mean_ms": f"{avg_eigdecomp:.2f}",
            "eigdecomp_median_ms": f"{median_eigdecomp:.2f}",
            "partition_mean_ms": f"{avg_partition:.3f}",
            "full_spielman_mean_ms": f"{avg_full:.3f}",
        })

    # Write CSV
    csv_path = os.path.join(output_dir, f"spielman_refinement_{dataset_name.replace('-', '_')}.csv")
    if summary_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\n  Wrote {csv_path}")

    # Write per-graph detail CSV
    detail_csv = os.path.join(
        output_dir, f"spielman_refinement_detail_{dataset_name.replace('-', '_')}.csv"
    )
    if all_results:
        with open(detail_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f"  Wrote {detail_csv}")

    # Write eigendecomposition times for runtime table update
    eigdecomp_csv = os.path.join(
        output_dir, f"eigendecomposition_times_{dataset_name.replace('-', '_')}.csv"
    )
    with open(eigdecomp_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "k", "n_graphs", "mean_ms", "median_ms", "std_ms"])
        for k in k_values:
            times = eigdecomp_times[k]
            if times:
                writer.writerow([
                    dataset_name, k, len(times),
                    f"{np.mean(times):.3f}",
                    f"{np.median(times):.3f}",
                    f"{np.std(times):.3f}",
                ])
    print(f"  Wrote {eigdecomp_csv}")

    return summary_rows, all_results


def main():
    parser = argparse.ArgumentParser(description="Spielman refinement analysis")
    parser.add_argument(
        "--dataset",
        type=str,
        default="both",
        choices=["ogbg-moltox21", "ogbg-molpcba", "both"],
        help="Dataset to analyze",
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[3, 6, 12],
        help="k values (number of eigenvectors)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/nadav_improvements/plots",
        help="Output directory",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Randomly sample N graphs instead of scanning all (for large datasets)",
    )
    args = parser.parse_args()

    datasets = []
    if args.dataset == "both":
        datasets = ["ogbg-moltox21", "ogbg-molpcba"]
    else:
        datasets = [args.dataset]

    all_summary = []
    for ds in datasets:
        summary, _ = run_analysis(ds, args.k, args.output_dir, sample_n=args.sample)
        all_summary.extend(summary)

    # Write combined summary
    if len(datasets) > 1 and all_summary:
        combined_path = os.path.join(args.output_dir, "spielman_refinement_combined.csv")
        with open(combined_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_summary[0].keys())
            writer.writeheader()
            writer.writerows(all_summary)
        print(f"\n  Wrote {combined_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
