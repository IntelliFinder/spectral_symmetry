"""Eigenvector canonicalizability analysis on molecular graphs (OGB).

Tests whether the trend observed on ModelNet10 (higher-frequency eigenvectors
are less canonicalizable) generalizes to a fundamentally different graph domain:
small molecular graphs with explicit bond edges.

Supports ogbg-molpcba (437k graphs) and ogbg-moltox21 (7.8k graphs).
"""

import argparse
import os
import sys

import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.spectral_core import (
    _largest_connected_component,
    compute_eigenpairs,
    detect_eigenvalue_multiplicities,
    uncanonicalizability_score,
    uncanonicalizability_threshold,
)


def edge_index_to_laplacian(edge_index, num_nodes):
    """Convert PyG edge_index [2, E] to combinatorial Laplacian on the LCC.

    Returns (L, n_component) where n_component is the size of the largest
    connected component.
    """
    row = edge_index[0]
    col = edge_index[1]
    # Remove self-loops
    mask = row != col
    row = row[mask]
    col = col[mask]
    # Build sparse adjacency (binary, symmetric)
    data = np.ones(len(row), dtype=np.float64)
    A = sp.csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    A = A.maximum(A.T)  # symmetrize
    A.data[:] = 1.0  # binarize

    # Extract largest connected component
    A, comp_idx = _largest_connected_component(A)
    A = sp.csr_matrix(A)
    n = A.shape[0]

    # Combinatorial Laplacian L = D - A
    degrees = np.array(A.sum(axis=1)).flatten()
    D = sp.diags(degrees)
    L = D - A
    return L, n


def analyze_graph(edge_index, num_nodes, n_eigs):
    """Compute per-eigenvector canonicalizability stats for one molecular graph."""
    L, n_component = edge_index_to_laplacian(edge_index, num_nodes)

    if n_component < 3:
        return None

    # Cap n_eigs: molecules are small, eigsh needs k < n-2
    k = min(n_eigs, n_component - 2)
    if k < 1:
        return None

    eigenvalues, eigenvectors = compute_eigenpairs(L, n_eigs=k)
    if len(eigenvalues) == 0:
        return None

    n_actual = eigenvectors.shape[1]
    scores = []
    for i in range(n_actual):
        v = eigenvectors[:, i]
        scores.append(uncanonicalizability_score(v))

    mult_info = detect_eigenvalue_multiplicities(eigenvalues)
    threshold = uncanonicalizability_threshold(n_component)

    return {
        "eigenvalues": eigenvalues,
        "scores": np.array(scores),
        "multiplicities": np.array(mult_info["multiplicity"]),
        "n_nodes": n_component,
        "n_eigs": n_actual,
        "threshold": threshold,
    }


def print_overall_summary(all_results, dataset_name):
    """Table 1: Overall summary statistics."""
    n_mols = len(all_results)
    total_eigvecs = sum(r["n_eigs"] for r in all_results)

    # Per-eigenvector uncanonicalizable flags
    n_uncanon_eigvecs = 0
    for r in all_results:
        n_uncanon_eigvecs += np.sum(r["scores"] < r["threshold"])

    pct_uncanon_eigvecs = 100 * n_uncanon_eigvecs / total_eigvecs if total_eigvecs > 0 else 0

    # Per-molecule stats
    uncanon_per_mol = []
    total_per_mol = []
    for r in all_results:
        n_un = np.sum(r["scores"] < r["threshold"])
        uncanon_per_mol.append(n_un)
        total_per_mol.append(r["n_eigs"])

    uncanon_per_mol = np.array(uncanon_per_mol)
    total_per_mol = np.array(total_per_mol)

    pct_mols_any_uncanon = 100 * np.mean(uncanon_per_mol >= 1)
    pct_mols_all_uncanon = 100 * np.mean(uncanon_per_mol == total_per_mol)

    all_scores = np.concatenate([r["scores"] for r in all_results])

    print(f"\nOVERALL SUMMARY ({dataset_name}, {n_mols} molecules)")
    print(f"  Total eigenvectors computed:          {total_eigvecs:>10,}")
    print(f"  % eigenvectors uncanonicalizable:     {pct_uncanon_eigvecs:>9.1f}%")
    print(f"  % molecules with >=1 uncanon eigvec:  {pct_mols_any_uncanon:>9.1f}%")
    print(f"  % molecules where ALL eigvecs uncanon: {pct_mols_all_uncanon:>8.1f}%")
    print(
        f"  Mean uncanon eigvecs per molecule:     "
        f"{np.mean(uncanon_per_mol):>5.1f} / {np.mean(total_per_mol):.1f}"
    )
    print(
        f"  Median uncanon eigvecs per molecule:   "
        f"{np.median(uncanon_per_mol):>5.0f} / {np.median(total_per_mol):.0f}"
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


def print_by_graph_size(all_results):
    """Table 3: Statistics by graph size bucket."""
    buckets = [(5, 10), (11, 15), (16, 20), (21, 30), (31, 50), (51, 999)]
    bucket_labels = ["5-10", "11-15", "16-20", "21-30", "31-50", "51+"]

    print(
        f"\n{'Bucket':>8} | {'N Mols':>6} | {'Mean Nodes':>10} | {'Mean #Eigvecs':>13} | "
        f"{'%Eigvecs Uncanon':>16} | {'Mean #Uncanon/Mol':>17} | {'%Mols >=1 Uncanon':>18}"
    )
    print("-" * 105)

    for (lo, hi), label in zip(buckets, bucket_labels):
        bucket_results = [r for r in all_results if lo <= r["n_nodes"] <= hi]
        if not bucket_results:
            print(f"{label:>8} | {'--':>6} |")
            continue

        mean_nodes = np.mean([r["n_nodes"] for r in bucket_results])
        mean_neigs = np.mean([r["n_eigs"] for r in bucket_results])

        n_uncanon_total = 0
        n_eigvecs_total = 0
        uncanon_per_mol = []
        for r in bucket_results:
            n_un = np.sum(r["scores"] < r["threshold"])
            uncanon_per_mol.append(n_un)
            n_uncanon_total += n_un
            n_eigvecs_total += r["n_eigs"]

        pct_eigvecs_uncanon = 100 * n_uncanon_total / n_eigvecs_total if n_eigvecs_total > 0 else 0
        mean_uncanon_per_mol = np.mean(uncanon_per_mol)
        pct_mols_any = 100 * np.mean(np.array(uncanon_per_mol) >= 1)

        print(
            f"{label:>8} | {len(bucket_results):>6} | {mean_nodes:>10.1f} | "
            f"{mean_neigs:>13.1f} | {pct_eigvecs_uncanon:>15.1f}% | "
            f"{mean_uncanon_per_mol:>17.1f} | {pct_mols_any:>17.1f}%"
        )


def print_cross_tabulation(all_results, n_eigs):
    """Table 4: Cross-tabulation of eigvec index x size bucket."""
    buckets = [(5, 10), (11, 15), (16, 20), (21, 30), (31, 50), (51, 999)]
    bucket_labels = ["5-10", "11-15", "16-20", "21-30", "31-50", "51+"]

    # Group results by bucket
    bucket_results = {}
    for (lo, hi), label in zip(buckets, bucket_labels):
        bucket_results[label] = [r for r in all_results if lo <= r["n_nodes"] <= hi]

    max_idx = min(n_eigs, 20)

    header = f"{'':>6} |" + "".join(f" {label:>6} |" for label in bucket_labels)
    print(f"\n{header}")
    print("-" * len(header))

    for i in range(max_idx):
        row = f"Idx {i + 1:>2} |"
        for label in bucket_labels:
            results = bucket_results[label]
            # Collect scores and thresholds for this index from graphs that have it
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
                row += f" {pct:>5.1f}% |"
            else:
                row += f" {'--':>5}  |"
        print(row)


def main():
    parser = argparse.ArgumentParser(
        description="Eigenvector canonicalizability analysis on molecular graphs (OGB)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ogbg-moltox21",
        help="OGB dataset name (ogbg-moltox21 or ogbg-molpcba)",
    )
    parser.add_argument("--data-dir", type=str, default="data", help="Root data directory")
    parser.add_argument("--n-eigs", type=int, default=20, help="Max number of eigenvectors")
    parser.add_argument("--min-nodes", type=int, default=5, help="Skip molecules with fewer nodes")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/canonicalizability_molecules",
        help="Output directory",
    )
    parser.add_argument("--max-graphs", type=int, default=0, help="Max graphs to process (0 = all)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load OGB dataset ---
    # OGB internally uses torch.load without weights_only=False, which fails
    # on PyTorch 2.6+. Patch it before importing.
    import torch

    _orig_torch_load = torch.load
    torch.load = lambda *a, **kw: _orig_torch_load(*a, **{**kw, "weights_only": False})

    from ogb.graphproppred import PygGraphPropPredDataset

    print(f"Loading {args.dataset}...")
    dataset = PygGraphPropPredDataset(name=args.dataset, root=args.data_dir)
    torch.load = _orig_torch_load  # restore
    print(f"  Total graphs: {len(dataset)}")

    n_graphs = len(dataset)
    if args.max_graphs > 0:
        n_graphs = min(n_graphs, args.max_graphs)

    # --- Analyze each graph ---
    all_results = []
    n_skipped_small = 0
    n_failed = 0

    for i in tqdm(range(n_graphs), desc="Analyzing graphs"):
        data = dataset[i]
        num_nodes = data.num_nodes
        if num_nodes < args.min_nodes:
            n_skipped_small += 1
            continue

        edge_index = data.edge_index.numpy()
        try:
            result = analyze_graph(edge_index, num_nodes, args.n_eigs)
        except Exception as e:
            if i < 5:
                print(f"  Failed on graph {i}: {e}")
            n_failed += 1
            continue

        if result is None:
            n_failed += 1
            continue

        all_results.append(result)

    print(
        f"\nProcessed {len(all_results)} graphs "
        f"({n_skipped_small} skipped <{args.min_nodes} nodes, {n_failed} failed)"
    )

    if len(all_results) == 0:
        print("No results to analyze!")
        return

    # --- Aggregate per eigenvector index (ragged) ---
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
    print_overall_summary(all_results, args.dataset)
    print_by_eigvec_index(scores_by_idx, mult_by_idx, thresholds_by_idx, n_eigs)
    print_by_graph_size(all_results)
    print_cross_tabulation(all_results, n_eigs)


if __name__ == "__main__":
    main()
