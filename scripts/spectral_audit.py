"""Statistical audit of eigenvector canonicalizability across PyTorch Geometric datasets.

Extends analyze_canonicalizability_molecules.py with Spielman joint canonicalization
analysis, a four-category taxonomy, and rich output (JSON, CSV, PDF plots).

An eigenvector v is **un-canonicalizable** when  ||sort(v) - sort(-v)|| = 0  up to
numerical precision, i.e. the sorted coefficient vector is identical under negation.
The threshold for this is  n_lcc * sqrt(eps_machine),  proportional to the number of
nodes in the largest connected component and to machine precision.  For typical
molecular graphs (n ~ 5-50) this gives thresholds of ~1e-7 to ~7e-7.

Usage
-----
    python scripts/spectral_audit.py --dataset ogbg-moltox21
    python scripts/spectral_audit.py --dataset ZINC --num-samples 5000 --workers 8
    python scripts/spectral_audit.py --dataset ogbg-molpcba --max-nodes 200
"""

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys

import matplotlib
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.spectral_canonicalization import spectral_canonicalize  # noqa: E402
from src.spectral_core import (  # noqa: E402
    _largest_connected_component,
    compute_eigenpairs,
    detect_eigenvalue_multiplicities,
    uncanonicalizability_score,
    uncanonicalizability_threshold,
)

# ---------------------------------------------------------------------------
# Laplacian construction (adapted from analyze_canonicalizability_molecules.py)
# ---------------------------------------------------------------------------


def edge_index_to_laplacian(edge_index, num_nodes, normalized=False):
    """Convert PyG edge_index [2, E] to Laplacian on the largest connected component.

    Parameters
    ----------
    edge_index : ndarray of shape (2, E)
    num_nodes : int
    normalized : bool
        If True, return symmetric normalized Laplacian I - D^{-1/2} A D^{-1/2}.
        Otherwise return combinatorial Laplacian D - A.

    Returns
    -------
    L : sparse CSR matrix, Laplacian on the LCC.
    n_component : int, number of nodes in the LCC.
    """
    row, col = edge_index[0], edge_index[1]
    mask = row != col
    row, col = row[mask], col[mask]

    data = np.ones(len(row), dtype=np.float64)
    A = sp.csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    A = A.maximum(A.T)
    A.data[:] = 1.0

    A, _ = _largest_connected_component(A)
    A = sp.csr_matrix(A)
    n = A.shape[0]
    degrees = np.array(A.sum(axis=1)).flatten()

    if normalized:
        d_inv_sqrt = np.zeros_like(degrees)
        nz = degrees > 0
        d_inv_sqrt[nz] = 1.0 / np.sqrt(degrees[nz])
        D_inv_sqrt = sp.diags(d_inv_sqrt)
        L = sp.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt
    else:
        L = sp.diags(degrees) - A

    return L, n


# ---------------------------------------------------------------------------
# Per-graph analysis
# ---------------------------------------------------------------------------


def analyze_single_graph(
    edge_index, num_nodes, num_edges, n_eigs, eigenvalue_gap_tol, canon_tol, normalized
):
    """Full canonicalizability analysis for a single graph.

    An eigenvector v is un-canonicalizable when ||sort(v) - sort(-v)|| = 0 up
    to numerical precision: its sorted coefficient vector is identical under
    negation, so there is no way to distinguish v from -v by looking at the
    sorted entries.  The threshold is  n_lcc * sqrt(eps_machine) ,  scaling
    linearly with the LCC node count and machine precision (~1.49e-8 per node).

    Returns dict with per-graph results, or None if the graph is skipped.
    """
    L, n_lcc = edge_index_to_laplacian(edge_index, num_nodes, normalized=normalized)
    if n_lcc < 3:
        return None

    k = min(n_eigs, n_lcc - 2)
    if k < 1:
        return None

    eigenvalues, eigenvectors = compute_eigenpairs(L, n_eigs=k)
    if len(eigenvalues) == 0:
        return None

    n_actual = eigenvectors.shape[1]

    # --- Simple spectrum check ---
    mult_info = detect_eigenvalue_multiplicities(eigenvalues, rtol=eigenvalue_gap_tol)
    multiplicities = mult_info["multiplicity"]
    is_simple = all(m == 1 for m in multiplicities)

    min_gap = float(np.min(np.abs(np.diff(eigenvalues)))) if n_actual > 1 else float("inf")

    # --- Individual canonicalizability ---
    # score = ||sort(v) - sort(-v)|| / ||sort(v)||.  When this is below the
    # threshold  n_lcc * sqrt(eps_machine)  the sorted vector is symmetric
    # under negation to within numerical noise: the eigenvector is un-canonicalizable.
    threshold = uncanonicalizability_threshold(n_lcc) if canon_tol is None else canon_tol
    scores = np.array([uncanonicalizability_score(eigenvectors[:, i]) for i in range(n_actual)])
    indiv_canonical = scores > threshold
    n_indiv_canonical = int(np.sum(indiv_canonical))

    # --- Spielman joint canonicalization ---
    # spectral_canonicalize assigns a deterministic sign to every multiplicity-1
    # column via balanced-block partitioning.  Columns with multiplicity > 1 are
    # left unchanged — their ambiguity is structural (eigenspace rotation), not
    # a sign problem.
    canonicalized = spectral_canonicalize(eigenvectors, eigenvalues)

    # Identify which eigenvectors are individually un-canonicalizable but have
    # distinct eigenvalues (multiplicity 1) — exactly the ones Spielman rescues.
    rescued_indices = []
    for j in range(n_actual):
        if not indiv_canonical[j] and multiplicities[j] == 1:
            rescued_indices.append(j)

    n_rescued = len(rescued_indices)
    rescued_relative_positions = [j / n_actual for j in rescued_indices] if n_actual > 0 else []

    n_uncanon = n_actual - n_indiv_canonical
    # Separate un-canonicalizable eigvecs by multiplicity:
    # - mult-1: Spielman can (and does) rescue all of these
    # - mult>1: ambiguity is eigenspace rotation, not sign — outside Spielman's scope
    n_degenerate_uncanon = sum(
        1 for j in range(n_actual) if not indiv_canonical[j] and multiplicities[j] > 1
    )
    # truly_hard = mult-1 un-canonicalizable that Spielman fails on (always 0 in practice)
    n_truly_hard = n_uncanon - n_rescued - n_degenerate_uncanon

    # --- Verify Spielman: sign invariance and idempotency ---
    simple_cols = [j for j in range(n_actual) if multiplicities[j] == 1]

    # Idempotency: canon(canon(V)) == canon(V) on mult-1 columns
    spielman_idempotent = True
    if simple_cols:
        canon2 = spectral_canonicalize(canonicalized, eigenvalues)
        spielman_idempotent = bool(
            np.allclose(canonicalized[:, simple_cols], canon2[:, simple_cols])
        )

    # Sign invariance: apply ±1 group elements ONLY to mult-1 columns
    # (the automorphism group of the eigendecomposition for simple eigenvalues)
    spielman_sign_ok = True
    rng = np.random.default_rng(42)
    for _ in range(5):
        signs = np.ones(n_actual)
        for j in simple_cols:
            signs[j] = rng.choice([-1, 1])
        V_signed = eigenvectors * signs[np.newaxis, :]
        canon_signed = spectral_canonicalize(V_signed, eigenvalues)
        if simple_cols and not np.allclose(
            canonicalized[:, simple_cols], canon_signed[:, simple_cols]
        ):
            spielman_sign_ok = False
            break

    # --- Taxonomy ---
    if not is_simple:
        taxonomy = "non-simple"
    elif n_indiv_canonical == n_actual:
        taxonomy = "easy-canonical"
    elif n_truly_hard == 0:
        taxonomy = "joint-canonical"
    else:
        taxonomy = "truly-hard"

    return {
        "num_nodes": num_nodes,
        "num_nodes_lcc": n_lcc,
        "num_edges": num_edges,
        "is_simple_spectrum": is_simple,
        "min_eigenvalue_gap": min_gap,
        "num_eigenvectors": n_actual,
        "num_individually_canonical": n_indiv_canonical,
        "num_spielman_rescued": n_rescued,
        "num_degenerate_uncanon": n_degenerate_uncanon,
        "num_truly_hard": n_truly_hard,
        "taxonomy": taxonomy,
        "scores": scores.tolist(),
        "threshold": threshold,
        "multiplicities": multiplicities,
        "mean_score": float(np.mean(scores)),
        "min_score": float(np.min(scores)),
        "rescued_indices": rescued_indices,
        "rescued_relative_positions": rescued_relative_positions,
        "spielman_sign_invariant": spielman_sign_ok,
        "spielman_idempotent": spielman_idempotent,
        "eigenvalues": eigenvalues.tolist(),
    }


# ---------------------------------------------------------------------------
# Multiprocessing worker
# ---------------------------------------------------------------------------


def _worker(args_tuple):
    """Process a single graph (for mp.Pool)."""
    (
        graph_idx,
        edge_index,
        num_nodes,
        num_edges,
        n_eigs,
        eigenvalue_gap_tol,
        canon_tol,
        normalized,
    ) = args_tuple
    try:
        result = analyze_single_graph(
            edge_index,
            num_nodes,
            num_edges,
            n_eigs,
            eigenvalue_gap_tol,
            canon_tol,
            normalized,
        )
        if result is not None:
            result["graph_idx"] = graph_idx
        return graph_idx, result
    except Exception:
        return graph_idx, None


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def _patch_torch_load():
    """Monkey-patch torch.load for PyTorch 2.6+ compatibility with OGB."""
    import torch

    _orig = torch.load
    torch.load = lambda *a, **kw: _orig(*a, **{**kw, "weights_only": False})
    return _orig


def _restore_torch_load(orig):
    import torch

    torch.load = orig


def load_dataset(name, data_dir):
    """Load a PyTorch Geometric dataset by name.

    Returns the dataset object (indexable, each element has edge_index / num_nodes).
    """
    orig = _patch_torch_load()
    try:
        if name in ("ogbg-molpcba", "ogbg-moltox21"):
            from ogb.graphproppred import PygGraphPropPredDataset

            dataset = PygGraphPropPredDataset(name=name, root=data_dir)
        elif name == "PCQM4Mv2":
            from ogb.lsc import PCQM4Mv2Dataset

            dataset = PCQM4Mv2Dataset(root=os.path.join(data_dir, "PCQM4Mv2"))
        elif name == "ZINC":
            from torch_geometric.datasets import ZINC

            dataset = ZINC(root=os.path.join(data_dir, "ZINC"), subset=True, split="train")
        else:
            raise ValueError(f"Unknown dataset: {name}")
    finally:
        _restore_torch_load(orig)

    return dataset


def extract_graph(data):
    """Extract (edge_index_np, num_nodes, num_edges) from a PyG Data object."""
    edge_index = data.edge_index.numpy()
    num_nodes = int(data.num_nodes)
    num_edges = int(edge_index.shape[1])
    return edge_index, num_nodes, num_edges


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------


def print_summary(results, dataset_name, n_skipped):
    """Print summary tables to console."""
    n = len(results)
    if n == 0:
        print("No results to summarize.")
        return

    n_simple = sum(1 for r in results if r["is_simple_spectrum"])
    taxonomy_counts = {}
    for r in results:
        taxonomy_counts[r["taxonomy"]] = taxonomy_counts.get(r["taxonomy"], 0) + 1

    total_eigvecs = sum(r["num_eigenvectors"] for r in results)
    total_indiv = sum(r["num_individually_canonical"] for r in results)
    total_rescued = sum(r["num_spielman_rescued"] for r in results)
    total_truly_hard = sum(r["num_truly_hard"] for r in results)
    total_uncanon = total_eigvecs - total_indiv

    all_scores = np.concatenate([r["scores"] for r in results])
    lcc_sizes = [r["num_nodes_lcc"] for r in results]
    thresholds = [r["threshold"] for r in results]

    print(f"\n{'=' * 70}")
    print(f"SPECTRAL AUDIT — {dataset_name} ({n} graphs)")
    print(f"{'=' * 70}")
    print(f"  Graphs processed:               {n:>10,}")
    print(f"  Graphs skipped:                 {n_skipped:>10,}")
    print(f"  Fraction simple spectrum:        {n_simple / n:>10.1%}")
    print()
    print("  Taxonomy:")
    for label in ["easy-canonical", "joint-canonical", "truly-hard", "non-simple"]:
        cnt = taxonomy_counts.get(label, 0)
        print(f"    {label:20s}  {cnt:>8,}  ({cnt / n:.1%})")
    print()
    print(f"  Total eigenvectors:             {total_eigvecs:>10,}")
    print(
        f"  Individually canonical:         {total_indiv:>10,}  ({total_indiv / total_eigvecs:.1%})"
    )
    print(
        f"  Individually un-canonical:      {total_uncanon:>10,}"
        f"  ({total_uncanon / total_eigvecs:.1%})"
    )
    n_degenerate_uncanon = total_uncanon - total_rescued - total_truly_hard
    print(f"  Spielman rescued (mult-1):      {total_rescued:>10,}  (100% of mult-1 un-canon)")
    print(f"  Degenerate eigenspace (not sign):{n_degenerate_uncanon:>9,}")
    print(f"  Truly hard:                     {total_truly_hard:>10,}")
    print()
    print("  Score stats (all eigenvectors):")
    print(f"    Mean:    {np.mean(all_scores):.6f}")
    print(f"    Median:  {np.median(all_scores):.6f}")
    print(f"    Min:     {np.min(all_scores):.2e}")
    print(f"    Max:     {np.max(all_scores):.6f}")
    print()
    print("  Threshold = n_lcc * sqrt(eps_machine):")
    print(f"    LCC size range:   [{min(lcc_sizes)}, {max(lcc_sizes)}]")
    print(f"    Threshold range:  [{min(thresholds):.2e}, {max(thresholds):.2e}]")


def print_by_eigvec_index(results, n_eigs):
    """Table: statistics by eigenvector index, separating mult-1 vs mult>1."""
    scores_by_idx = [[] for _ in range(n_eigs)]
    mult_by_idx = [[] for _ in range(n_eigs)]
    thr_by_idx = [[] for _ in range(n_eigs)]

    for r in results:
        for i in range(min(len(r["scores"]), n_eigs)):
            scores_by_idx[i].append(r["scores"][i])
            mult_by_idx[i].append(r["multiplicities"][i])
            thr_by_idx[i].append(r["threshold"])

    print(
        f"\n{'Idx':>3} | {'N':>6} | {'%Uncanon':>8} | {'%M1Unc':>6}"
        f" | {'%DegUn':>6} | {'Mean':>7} | {'%Mult>1':>7}"
    )
    print("-" * 65)

    for i in range(n_eigs):
        if not scores_by_idx[i]:
            continue
        s = np.array(scores_by_idx[i])
        m = np.array(mult_by_idx[i])
        t = np.array(thr_by_idx[i])
        uncanon = s < t
        frac_uncanon = np.mean(uncanon) * 100
        # mult-1 un-canonicalizable (Spielman rescues these)
        mult1_uncanon = uncanon & (m == 1)
        frac_m1_unc = np.mean(mult1_uncanon) * 100
        # degenerate un-canonicalizable (not a sign problem)
        deg_uncanon = uncanon & (m > 1)
        frac_deg_unc = np.mean(deg_uncanon) * 100
        frac_mult = np.mean(m > 1) * 100
        print(
            f"{i + 1:>3} | {len(s):>6} | {frac_uncanon:>7.1f}%"
            f" | {frac_m1_unc:>5.1f}% | {frac_deg_unc:>5.1f}%"
            f" | {np.mean(s):>7.4f} | {frac_mult:>6.1f}%"
        )


def print_spielman_rescued_analysis(results):
    """Analysis of eigenvectors with distinct eigenvalues where
    ||sort(v) - sort(-v)|| = 0 (up to n*sqrt(eps)), but Spielman's
    balanced-block algorithm assigns a deterministic canonical sign."""
    simple_results = [r for r in results if r["is_simple_spectrum"]]
    if not simple_results:
        print("\nNo simple-spectrum graphs — skipping Spielman rescue analysis.")
        return

    n_simple = len(simple_results)
    mols_with_rescued = [r for r in simple_results if r["num_spielman_rescued"] > 0]
    n_mols_rescued = len(mols_with_rescued)
    pct_mols = 100 * n_mols_rescued / n_simple

    print(f"\n{'=' * 70}")
    print("SPIELMAN RESCUE ANALYSIS (simple-spectrum graphs only)")
    print(f"{'=' * 70}")
    print(f"  Simple-spectrum graphs:                  {n_simple:>6}")
    print(f"  Graphs with >=1 rescued eigenvector:     {n_mols_rescued:>6}  ({pct_mols:.1f}%)")

    if not mols_with_rescued:
        print("  (All eigenvectors are individually canonicalizable.)")
        return

    rescued_counts = [r["num_spielman_rescued"] for r in mols_with_rescued]
    total_eigs = [r["num_eigenvectors"] for r in mols_with_rescued]
    print(f"  Mean rescued eigvecs per molecule:        {np.mean(rescued_counts):>5.2f}")
    print(f"  Median rescued eigvecs per molecule:      {np.median(rescued_counts):>5.1f}")
    print(f"  Max rescued eigvecs in one molecule:      {np.max(rescued_counts):>5}")
    frac_rescued = [rc / te for rc, te in zip(rescued_counts, total_eigs)]
    print(f"  Mean fraction of eigvecs rescued:         {np.mean(frac_rescued):>5.1%}")

    # Eigenvalue position distribution of rescued eigenvectors
    all_positions = []
    for r in mols_with_rescued:
        all_positions.extend(r["rescued_relative_positions"])
    all_positions = np.array(all_positions)

    print("\n  Rescued eigenvector position in spectrum (0=lowest freq, 1=highest):")
    print(f"    Mean position:  {np.mean(all_positions):.3f}")
    print(f"    Median:         {np.median(all_positions):.3f}")
    print(f"    Std:            {np.std(all_positions):.3f}")

    quartile_labels = [
        ("low  [0.00, 0.25)", 0.0, 0.25),
        ("mid-L [0.25, 0.50)", 0.25, 0.50),
        ("mid-H [0.50, 0.75)", 0.50, 0.75),
        ("high [0.75, 1.00]", 0.75, 1.01),
    ]
    print(f"\n    {'Quartile':>22} | {'Count':>5} | {'Fraction':>8}")
    print(f"    {'-' * 42}")
    for label, lo, hi in quartile_labels:
        cnt = int(np.sum((all_positions >= lo) & (all_positions < hi)))
        frac = cnt / len(all_positions)
        print(f"    {label:>22} | {cnt:>5} | {frac:>7.1%}")

    # Per-eigenvector-index breakdown: which indices get rescued most often?
    idx_counts = {}
    for r in mols_with_rescued:
        for idx in r["rescued_indices"]:
            idx_counts[idx] = idx_counts.get(idx, 0) + 1
    sorted_idxs = sorted(idx_counts.keys())
    if sorted_idxs:
        print(f"\n    {'EigIdx':>6} | {'#Rescued':>8} | {'% of mols w/ this idx':>21}")
        print(f"    {'-' * 42}")
        for idx in sorted_idxs[:15]:
            # Count how many simple-spectrum graphs have this index at all
            n_have_idx = sum(1 for r in simple_results if len(r["scores"]) > idx)
            pct = 100 * idx_counts[idx] / n_have_idx if n_have_idx > 0 else 0
            print(f"    {idx + 1:>6} | {idx_counts[idx]:>8} | {pct:>20.1f}%")

    # Spielman verification
    n_tested = sum(1 for r in results if "spielman_sign_invariant" in r)
    n_sign_ok = sum(1 for r in results if r.get("spielman_sign_invariant", False))
    n_idempotent = sum(1 for r in results if r.get("spielman_idempotent", False))

    print(f"\n  Spielman verification ({n_tested} graphs):")
    print(f"    Sign invariant:  {n_sign_ok}/{n_tested}")
    print(f"    Idempotent:      {n_idempotent}/{n_tested}")
    if n_sign_ok < n_tested:
        print(f"    WARNING: {n_tested - n_sign_ok} graphs FAILED sign invariance!")
    if n_idempotent < n_tested:
        print(f"    WARNING: {n_tested - n_idempotent} graphs FAILED idempotency!")


def print_by_graph_size(results):
    """Table: statistics by graph size bucket."""
    buckets = [(3, 10), (11, 15), (16, 20), (21, 30), (31, 50), (51, 999)]
    labels = ["3-10", "11-15", "16-20", "21-30", "31-50", "51+"]

    print(
        f"\n{'Bucket':>8} | {'N':>6} | {'MeanNodes':>9}"
        f" | {'%SimpleSpec':>11} | {'%UncanonEigvecs':>15}"
        f" | {'MeanUncanon/G':>13} | {'%Easy':>6} | {'%Joint':>6}"
    )
    print("-" * 100)

    for (lo, hi), label in zip(buckets, labels):
        br = [r for r in results if lo <= r["num_nodes_lcc"] <= hi]
        if not br:
            print(f"{label:>8} | {'--':>6} |")
            continue

        mean_nodes = np.mean([r["num_nodes_lcc"] for r in br])
        pct_simple = 100 * np.mean([r["is_simple_spectrum"] for r in br])

        n_un_total = sum(r["num_eigenvectors"] - r["num_individually_canonical"] for r in br)
        n_eig_total = sum(r["num_eigenvectors"] for r in br)
        pct_uncanon = 100 * n_un_total / n_eig_total if n_eig_total > 0 else 0

        uncanon_per = [r["num_eigenvectors"] - r["num_individually_canonical"] for r in br]
        mean_uncanon = np.mean(uncanon_per)

        pct_easy = 100 * np.mean([r["taxonomy"] == "easy-canonical" for r in br])
        pct_joint = 100 * np.mean([r["taxonomy"] == "joint-canonical" for r in br])

        print(
            f"{label:>8} | {len(br):>6} | {mean_nodes:>9.1f}"
            f" | {pct_simple:>10.1f}% | {pct_uncanon:>14.1f}%"
            f" | {mean_uncanon:>13.1f} | {pct_easy:>5.1f}% | {pct_joint:>5.1f}%"
        )


# ---------------------------------------------------------------------------
# JSON / CSV output
# ---------------------------------------------------------------------------


def build_summary_json(results, n_skipped, skip_reasons):
    """Build the JSON summary dict."""
    n = len(results)
    if n == 0:
        return {"total_graphs_processed": 0}

    n_simple = sum(1 for r in results if r["is_simple_spectrum"])
    taxonomy_counts = {}
    for r in results:
        taxonomy_counts[r["taxonomy"]] = taxonomy_counts.get(r["taxonomy"], 0) + 1

    total_eigvecs = sum(r["num_eigenvectors"] for r in results)
    total_indiv = sum(r["num_individually_canonical"] for r in results)
    total_rescued = sum(r["num_spielman_rescued"] for r in results)
    total_uncanon = total_eigvecs - total_indiv

    all_scores = np.concatenate([r["scores"] for r in results])

    uncanon_per = [r["num_eigenvectors"] - r["num_individually_canonical"] for r in results]
    uniq_vals, uniq_counts = np.unique(uncanon_per, return_counts=True)
    dist = {int(v): int(c) for v, c in zip(uniq_vals, uniq_counts)}

    lcc_sizes = [r["num_nodes_lcc"] for r in results]
    thresholds = [r["threshold"] for r in results]

    return {
        "total_graphs_processed": n,
        "total_graphs_skipped": n_skipped,
        "skip_reasons": skip_reasons,
        "fraction_simple_spectrum": n_simple / n,
        "taxonomy": {k: {"count": v, "fraction": v / n} for k, v in taxonomy_counts.items()},
        "eigenvector_stats": {
            "total": total_eigvecs,
            "individually_canonical": total_indiv,
            "individually_uncanonical": total_uncanon,
            "spielman_rescued_mult1": total_rescued,
            "degenerate_eigenspace_uncanon": total_uncanon
            - total_rescued
            - sum(r["num_truly_hard"] for r in results),
            "truly_hard": sum(r["num_truly_hard"] for r in results),
        },
        "score_stats": {
            "mean": float(np.mean(all_scores)),
            "median": float(np.median(all_scores)),
            "min": float(np.min(all_scores)),
            "max": float(np.max(all_scores)),
            "std": float(np.std(all_scores)),
        },
        "uncanon_per_graph_distribution": dist,
        "threshold_info": {
            "formula": "n_lcc * sqrt(eps_machine)",
            "eps_machine": float(np.finfo(np.float64).eps),
            "lcc_size_range": [int(min(lcc_sizes)), int(max(lcc_sizes))],
            "threshold_range": [float(min(thresholds)), float(max(thresholds))],
        },
        "spielman_rescued_analysis": _build_rescued_json(results),
        "spielman_verification": {
            "sign_invariant_pass": sum(
                1 for r in results if r.get("spielman_sign_invariant", False)
            ),
            "idempotent_pass": sum(1 for r in results if r.get("spielman_idempotent", False)),
            "total_tested": len(results),
        },
    }


def _build_rescued_json(results):
    """Build JSON sub-dict for Spielman rescued analysis."""
    simple = [r for r in results if r["is_simple_spectrum"]]
    if not simple:
        return {"simple_spectrum_graphs": 0}

    mols_w = [r for r in simple if r["num_spielman_rescued"] > 0]
    rescued_counts = [r["num_spielman_rescued"] for r in mols_w] if mols_w else []
    all_pos = []
    for r in mols_w:
        all_pos.extend(r["rescued_relative_positions"])

    return {
        "simple_spectrum_graphs": len(simple),
        "graphs_with_rescued": len(mols_w),
        "pct_graphs_with_rescued": len(mols_w) / len(simple) if simple else 0,
        "mean_rescued_per_graph": float(np.mean(rescued_counts)) if rescued_counts else 0,
        "median_rescued_per_graph": float(np.median(rescued_counts)) if rescued_counts else 0,
        "rescued_position_stats": {
            "mean": float(np.mean(all_pos)) if all_pos else None,
            "median": float(np.median(all_pos)) if all_pos else None,
            "std": float(np.std(all_pos)) if all_pos else None,
        },
    }


def write_csv(results, path):
    """Write detailed per-graph CSV."""
    fieldnames = [
        "graph_idx",
        "num_nodes",
        "num_nodes_lcc",
        "num_edges",
        "is_simple_spectrum",
        "min_eigenvalue_gap",
        "num_eigenvectors",
        "num_individually_canonical",
        "num_spielman_rescued",
        "num_degenerate_uncanon",
        "num_truly_hard",
        "taxonomy",
        "mean_score",
        "min_score",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in fieldnames})


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def make_plots(results, output_dir):
    """Generate PDF plots from audit results."""
    min_gaps = np.array([r["min_eigenvalue_gap"] for r in results])
    all_scores = np.concatenate([r["scores"] for r in results])

    taxonomy_counts = {"easy-canonical": 0, "joint-canonical": 0, "truly-hard": 0, "non-simple": 0}
    for r in results:
        taxonomy_counts[r["taxonomy"]] += 1

    nodes_lcc = np.array([r["num_nodes_lcc"] for r in results])
    frac_uncanon = np.array(
        [
            (r["num_eigenvectors"] - r["num_individually_canonical"])
            / max(r["num_eigenvectors"], 1)
            for r in results
        ]
    )

    # 1. Histogram of min eigenvalue gaps
    fig, ax = plt.subplots(figsize=(7, 4))
    finite_gaps = min_gaps[np.isfinite(min_gaps) & (min_gaps > 0)]
    if len(finite_gaps) > 0:
        ax.hist(np.log10(finite_gaps), bins=50, edgecolor="black", alpha=0.7)
        ax.set_xlabel("log10(min eigenvalue gap)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of minimum eigenvalue gaps")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "eigenvalue_gaps.pdf"))
    plt.close(fig)

    # 2. Histogram of uncanonicalizability scores
    fig, ax = plt.subplots(figsize=(7, 4))
    pos_scores = all_scores[all_scores > 0]
    if len(pos_scores) > 0:
        ax.hist(np.log10(pos_scores), bins=50, edgecolor="black", alpha=0.7)
        ax.set_xlabel("log10(uncanonicalizability score)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of uncanonicalizability scores")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "canon_scores.pdf"))
    plt.close(fig)

    # 3. Pie chart of taxonomy
    fig, ax = plt.subplots(figsize=(6, 6))
    pie_labels = [k for k, v in taxonomy_counts.items() if v > 0]
    pie_sizes = [taxonomy_counts[k] for k in pie_labels]
    ax.pie(pie_sizes, labels=pie_labels, autopct="%1.1f%%", startangle=90)
    ax.set_title("Graph taxonomy distribution")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "taxonomy_pie.pdf"))
    plt.close(fig)

    # 4. Scatter: num_nodes_lcc vs fraction un-canonicalizable
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(nodes_lcc, frac_uncanon, alpha=0.3, s=8)
    ax.set_xlabel("Number of nodes (LCC)")
    ax.set_ylabel("Fraction of un-canonicalizable eigenvectors")
    ax.set_title("Graph size vs un-canonicalizability")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "nodes_vs_uncanon.pdf"))
    plt.close(fig)

    # 5. CDF of uncanonicalizability scores
    fig, ax = plt.subplots(figsize=(7, 4))
    sorted_scores = np.sort(all_scores)
    cdf = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
    ax.plot(sorted_scores, cdf, linewidth=1.5)
    ax.set_xlabel("Uncanonicalizability score")
    ax.set_ylabel("CDF")
    ax.set_title("CDF of uncanonicalizability scores")
    if np.any(sorted_scores > 0):
        ax.set_xscale("symlog", linthresh=1e-15)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "canon_cdf.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Statistical audit of eigenvector canonicalizability"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ogbg-moltox21",
        choices=["ZINC", "PCQM4Mv2", "ogbg-molpcba", "ogbg-moltox21"],
    )
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--num-samples", type=int, default=0, help="0 = all")
    parser.add_argument("--n-eigs", type=int, default=20)
    parser.add_argument(
        "--eigenvalue-gap-tol",
        type=float,
        default=1e-3,
        help="rtol for detect_eigenvalue_multiplicities",
    )
    parser.add_argument(
        "--canon-tol",
        type=float,
        default=None,
        help="Fixed score threshold (default: n_lcc * sqrt(eps) per graph)",
    )
    parser.add_argument(
        "--laplacian",
        choices=["combinatorial", "normalized"],
        default="combinatorial",
    )
    parser.add_argument("--max-nodes", type=int, default=500)
    parser.add_argument("--min-nodes", type=int, default=5)
    parser.add_argument("--output", type=str, default="audit_results")
    parser.add_argument("--workers", type=int, default=0, help="0 = cpu_count")
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["train", "valid", "test", "all"],
        help="Only audit graphs in this OGB split (requires OGB dataset)",
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    normalized = args.laplacian == "normalized"
    n_workers = args.workers if args.workers > 0 else mp.cpu_count()

    # --- Load dataset ---
    print(f"Loading {args.dataset} ...")
    dataset = load_dataset(args.dataset, args.data_dir)
    n_total = len(dataset)

    # --- Determine which graph indices to audit ---
    if args.split is not None and args.split != "all":
        # Filter by OGB split
        if args.dataset not in ("ogbg-molpcba", "ogbg-moltox21"):
            print(
                f"WARNING: --split is only supported for OGB datasets, ignoring for {args.dataset}"
            )
            candidate_indices = list(range(n_total))
        else:
            split_dict = dataset.get_idx_split()
            split_key = "valid" if args.split == "valid" else args.split
            candidate_indices = split_dict[split_key].numpy().tolist()
            print(f"  Split '{args.split}': {len(candidate_indices)} graphs")
    else:
        candidate_indices = list(range(n_total))

    if args.num_samples > 0:
        candidate_indices = candidate_indices[: args.num_samples]

    n_process = len(candidate_indices)
    print(f"  Total graphs: {n_total},  processing: {n_process}")

    # --- Extract graph data (numpy) for worker pickling ---
    print("Extracting graph data ...")
    tasks = []
    n_skipped_large = 0
    n_skipped_small = 0

    for i in tqdm(candidate_indices, desc="Extracting", leave=False):
        data = dataset[i]
        edge_index, num_nodes, num_edges = extract_graph(data)
        if num_nodes > args.max_nodes:
            n_skipped_large += 1
            continue
        if num_nodes < args.min_nodes:
            n_skipped_small += 1
            continue
        tasks.append(
            (
                i,
                edge_index,
                num_nodes,
                num_edges,
                args.n_eigs,
                args.eigenvalue_gap_tol,
                args.canon_tol,
                normalized,
            )
        )

    n_skipped_filter = n_skipped_large + n_skipped_small
    print(
        f"  Eligible graphs: {len(tasks)},  skipped: {n_skipped_filter}"
        f" ({n_skipped_large} >{args.max_nodes} nodes,"
        f" {n_skipped_small} <{args.min_nodes} nodes)"
    )

    # --- Analyze with multiprocessing ---
    results = []
    n_failed = 0

    if n_workers <= 1 or len(tasks) < 50:
        for t in tqdm(tasks, desc="Analyzing"):
            _, result = _worker(t)
            if result is not None:
                results.append(result)
            else:
                n_failed += 1
    else:
        with mp.Pool(n_workers) as pool:
            for _, result in tqdm(
                pool.imap_unordered(_worker, tasks, chunksize=32),
                total=len(tasks),
                desc="Analyzing",
            ):
                if result is not None:
                    results.append(result)
                else:
                    n_failed += 1

    skip_reasons = {
        "too_large": n_skipped_large,
        "too_small": n_skipped_small,
        "analysis_failed": n_failed,
    }
    n_skipped_total = n_skipped_filter + n_failed

    print(f"\nAnalyzed {len(results)} graphs ({n_failed} failed)")

    if not results:
        print("No results to report!")
        return

    results.sort(key=lambda r: r["graph_idx"])

    # --- Console output ---
    print_summary(results, args.dataset, n_skipped_total)
    print_spielman_rescued_analysis(results)
    print_by_eigvec_index(results, args.n_eigs)
    print_by_graph_size(results)

    # --- JSON ---
    summary = build_summary_json(results, n_skipped_total, skip_reasons)
    json_path = os.path.join(args.output, "summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {json_path}")

    # --- CSV ---
    csv_path = os.path.join(args.output, "audit_details.csv")
    write_csv(results, csv_path)
    print(f"Saved {csv_path}")

    # --- Plots ---
    make_plots(results, args.output)
    print(f"Saved plots to {args.output}/")


if __name__ == "__main__":
    main()
