"""Quick eigenvalue multiplicity audit on ZINC-subset.

Loads ZINC-subset (train+val+test = ~12K graphs), computes Laplacian
eigenpairs, detects eigenvalue multiplicities, and reports what fraction
of graphs have simple spectra (all distinct eigenvalues).

Usage:
    conda activate schnet-angle
    python3 scripts/zinc_multiplicity_audit.py
"""

import csv
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.spectral_core import (
    _largest_connected_component,
    compute_eigenpairs,
    detect_eigenvalue_multiplicities,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_EIGS = 8          # number of non-trivial eigenvectors to compute
RTOL = 1e-3          # relative tolerance for eigenvalue multiplicity
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "audit_results")

# ---------------------------------------------------------------------------
# Laplacian from edge_index (same as spectral_audit.py)
# ---------------------------------------------------------------------------

def edge_index_to_laplacian(edge_index, num_nodes):
    """Convert PyG edge_index [2, E] to combinatorial Laplacian on LCC."""
    row, col = edge_index[0], edge_index[1]
    mask = row != col
    row, col = row[mask], col[mask]

    data = np.ones(len(row), dtype=np.float64)
    A = sp.csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    A = A.maximum(A.T)
    A.data[:] = 1.0

    A, lcc_indices = _largest_connected_component(A)
    A = sp.csr_matrix(A)
    degrees = np.array(A.sum(axis=1)).flatten()
    L = sp.diags(degrees) - A
    return L, lcc_indices


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Patch torch.load for PyTorch 2.6+ compat
    _orig_load = torch.load
    torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, "weights_only": False})

    from torch_geometric.datasets import ZINC

    # Load all three splits
    splits = {}
    for split in ("train", "val", "test"):
        splits[split] = ZINC(root=os.path.join(DATA_DIR, "ZINC"), subset=True, split=split)

    torch.load = _orig_load  # restore

    # Combine all graphs
    all_graphs = []
    for split_name, ds in splits.items():
        print(f"  {split_name}: {len(ds)} graphs")
        for g in ds:
            all_graphs.append(g)

    n_total = len(all_graphs)
    print(f"\nTotal ZINC-subset graphs: {n_total}")
    print(f"Computing {N_EIGS} non-trivial eigenpairs per graph, rtol={RTOL}")
    print()

    # Per-graph stats
    n_simple = 0            # graphs with fully simple spectrum
    n_has_degeneracy = 0    # graphs with at least one degenerate eigenvalue group
    n_failed = 0            # graphs where eigdecomp failed
    n_too_small = 0         # graphs too small for N_EIGS eigenvectors

    graph_sizes = []
    n_degenerate_per_graph = []   # number of eigenvalues in degenerate groups (per graph)
    max_multiplicity_per_graph = []
    n_eigs_computed = []
    min_gap_per_graph = []   # minimum gap between consecutive distinct eigenvalues

    t0 = time.time()

    for i, data in enumerate(tqdm(all_graphs, desc="Multiplicity audit")):
        edge_index = data.edge_index.numpy()
        num_nodes = int(data.num_nodes)
        graph_sizes.append(num_nodes)

        if num_nodes < 3:
            n_too_small += 1
            continue

        try:
            L, lcc_indices = edge_index_to_laplacian(edge_index, num_nodes)
        except Exception:
            n_failed += 1
            continue

        n_lcc = L.shape[0]
        if n_lcc < 3:
            n_too_small += 1
            continue

        k = min(N_EIGS, n_lcc - 2)
        if k < 1:
            n_too_small += 1
            continue

        try:
            eigenvalues, eigenvectors = compute_eigenpairs(L, n_eigs=k)
        except Exception:
            n_failed += 1
            continue

        if len(eigenvalues) == 0:
            n_failed += 1
            continue

        n_eigs_computed.append(len(eigenvalues))

        # Detect multiplicities
        info = detect_eigenvalue_multiplicities(eigenvalues, rtol=RTOL)

        max_mult = max(info["multiplicity"]) if info["multiplicity"] else 1
        max_multiplicity_per_graph.append(max_mult)
        n_degenerate_per_graph.append(info["n_repeating"])

        if info["n_repeating"] == 0:
            n_simple += 1
        else:
            n_has_degeneracy += 1

        # Minimum gap between consecutive eigenvalues
        if len(eigenvalues) > 1:
            gaps = np.diff(np.sort(eigenvalues))
            min_gap_per_graph.append(float(np.min(gaps)))
        else:
            min_gap_per_graph.append(float("inf"))

    elapsed = time.time() - t0

    # ---------------------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------------------
    n_analyzed = n_simple + n_has_degeneracy
    frac_simple = n_simple / n_analyzed if n_analyzed > 0 else 0
    frac_degenerate = n_has_degeneracy / n_analyzed if n_analyzed > 0 else 0

    print("\n" + "=" * 70)
    print("ZINC-SUBSET EIGENVALUE MULTIPLICITY AUDIT")
    print("=" * 70)
    print(f"Total graphs:           {n_total}")
    print(f"Successfully analyzed:  {n_analyzed}")
    print(f"Failed / too small:     {n_failed} / {n_too_small}")
    print(f"Eigenvectors computed:  {N_EIGS} (requested)")
    print(f"Multiplicity rtol:      {RTOL}")
    print(f"Time:                   {elapsed:.1f}s")
    print()
    print(f"--- Spectrum simplicity ---")
    print(f"Simple spectrum (all distinct):   {n_simple:>6d}  ({100*frac_simple:.1f}%)")
    print(f"Has degeneracy (>= 1 repeated):  {n_has_degeneracy:>6d}  ({100*frac_degenerate:.1f}%)")
    print()

    if n_degenerate_per_graph:
        deg_arr = np.array(n_degenerate_per_graph)
        print(f"--- Among all analyzed graphs ---")
        print(f"Mean degenerate eigenvectors per graph:   {deg_arr.mean():.2f}")
        print(f"Median degenerate eigenvectors per graph: {np.median(deg_arr):.1f}")
        print(f"Max degenerate eigenvectors in one graph: {deg_arr.max()}")
        print()

    if max_multiplicity_per_graph:
        mm_arr = np.array(max_multiplicity_per_graph)
        print(f"--- Max eigenvalue multiplicity per graph ---")
        print(f"Mean:   {mm_arr.mean():.2f}")
        print(f"Median: {np.median(mm_arr):.1f}")
        print(f"Max:    {mm_arr.max()}")
        # Distribution
        unique_mults, counts = np.unique(mm_arr, return_counts=True)
        print(f"Distribution of max multiplicity:")
        for m, c in zip(unique_mults, counts):
            print(f"  mult={int(m)}: {c} graphs ({100*c/n_analyzed:.1f}%)")
        print()

    if min_gap_per_graph:
        gap_arr = np.array([g for g in min_gap_per_graph if g < float("inf")])
        if len(gap_arr) > 0:
            print(f"--- Minimum eigenvalue gap (consecutive) ---")
            print(f"Mean:   {gap_arr.mean():.6f}")
            print(f"Median: {np.median(gap_arr):.6f}")
            print(f"5th percentile:  {np.percentile(gap_arr, 5):.6f}")
            print(f"1st percentile:  {np.percentile(gap_arr, 1):.6f}")
            print(f"Min:    {gap_arr.min():.8f}")
            print()

    if graph_sizes:
        gs = np.array(graph_sizes)
        print(f"--- Graph sizes ---")
        print(f"Mean:   {gs.mean():.1f}")
        print(f"Median: {np.median(gs):.1f}")
        print(f"Min:    {gs.min()}, Max: {gs.max()}")
        print()

    if n_eigs_computed:
        ec = np.array(n_eigs_computed)
        print(f"--- Eigenvectors actually computed ---")
        print(f"Mean:   {ec.mean():.1f}")
        print(f"Min:    {ec.min()}, Max: {ec.max()}")
        print(f"Graphs with fewer than {N_EIGS}: {np.sum(ec < N_EIGS)}")
        print()

    # ---------------------------------------------------------------------------
    # Detailed distribution: how many degenerate eigenvectors per graph
    # ---------------------------------------------------------------------------
    if n_degenerate_per_graph:
        print(f"--- Distribution: number of degenerate eigenvectors per graph ---")
        unique_ndegs, counts = np.unique(deg_arr, return_counts=True)
        for nd, c in zip(unique_ndegs, counts):
            print(f"  {int(nd)} degenerate eigvecs: {c} graphs ({100*c/n_analyzed:.1f}%)")
        print()

    # ---------------------------------------------------------------------------
    # Save CSV summary
    # ---------------------------------------------------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "zinc_multiplicity_audit.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "graph_idx", "num_nodes", "n_eigs_computed",
            "n_degenerate", "max_multiplicity", "min_gap"
        ])
        row_idx = 0
        for i in range(n_total):
            if i < len(graph_sizes) and graph_sizes[i] >= 3:
                if row_idx < len(n_degenerate_per_graph):
                    writer.writerow([
                        i,
                        graph_sizes[i],
                        n_eigs_computed[row_idx] if row_idx < len(n_eigs_computed) else "",
                        n_degenerate_per_graph[row_idx],
                        max_multiplicity_per_graph[row_idx],
                        f"{min_gap_per_graph[row_idx]:.8f}" if row_idx < len(min_gap_per_graph) else "",
                    ])
                    row_idx += 1
    print(f"CSV saved to: {csv_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
