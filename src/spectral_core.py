"""Core spectral analysis for point cloud symmetry detection."""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from sklearn.neighbors import kneighbors_graph


def _largest_connected_component(adjacency):
    """Extract the largest connected component from a sparse adjacency matrix.

    Returns the submatrix and the indices of nodes in the largest component.
    """
    n_components, labels = sp.csgraph.connected_components(adjacency, directed=False)
    if n_components == 1:
        return adjacency, np.arange(adjacency.shape[0])
    component_sizes = np.bincount(labels)
    largest = np.argmax(component_sizes)
    mask = labels == largest
    indices = np.where(mask)[0]
    return adjacency[np.ix_(indices, indices)], indices


def build_graph_laplacian(points, n_neighbors=12):
    """Build a symmetric k-NN graph and return the sparse graph Laplacian L = D - A.

    Parameters
    ----------
    points : ndarray of shape (N, D)
    n_neighbors : int

    Returns
    -------
    L : sparse CSR matrix
    component_indices : ndarray, indices into original point array
    """
    n_points = points.shape[0]
    k = min(n_neighbors, n_points - 1)
    A = kneighbors_graph(points, n_neighbors=k, mode='connectivity', include_self=False)
    A = 0.5 * (A + A.T)  # symmetrize
    A = (A > 0).astype(float)  # binary

    A, component_indices = _largest_connected_component(A)
    A = sp.csr_matrix(A)

    degrees = np.array(A.sum(axis=1)).flatten()
    D = sp.diags(degrees)
    L = D - A
    return L, component_indices


def compute_eigenpairs(L, n_eigs=20):
    """Compute the smallest eigenpairs of a sparse Laplacian.

    Parameters
    ----------
    L : sparse matrix
    n_eigs : int

    Returns
    -------
    eigenvalues : ndarray of shape (n_eigs,)
    eigenvectors : ndarray of shape (N, n_eigs)
    """
    n = L.shape[0]
    k = min(n_eigs, n - 2)  # eigsh needs k < n
    vals, vecs = sla.eigsh(L, k=k, which='SM', tol=1e-8)
    idx = np.argsort(vals)
    return vals[idx], vecs[:, idx]


def uncanonicalizability_score(vec):
    """Compute how close sort(v) is to sort(-v).

    Returns ||sort(v) - sort(-v)|| / ||sort(v)||.
    Lower values indicate sign ambiguity (anti-symmetric distribution).
    """
    v_sorted = np.sort(vec)
    v_neg_sorted = np.sort(-vec)
    norm = np.linalg.norm(v_sorted)
    if norm < 1e-12:
        return 0.0
    return float(np.linalg.norm(v_sorted - v_neg_sorted) / norm)


def detect_eigenvalue_multiplicities(eigenvalues, rtol=1e-3):
    """Group eigenvalues by approximate equality and report multiplicities.

    Parameters
    ----------
    eigenvalues : ndarray of shape (k,)
    rtol : float
        Relative tolerance for considering two eigenvalues equal.
        Two eigenvalues a, b are grouped if |a - b| <= rtol * max(|a|, |b|, 1).

    Returns
    -------
    dict with keys:
        multiplicity : list of int, multiplicity of each eigenvalue's group
        group_indices : list of int, group label for each eigenvalue
        n_repeating : int, number of eigenvalues that belong to a group of size > 1
        n_non_repeating : int, number of eigenvalues that are unique (group size == 1)
    """
    n = len(eigenvalues)
    group_indices = [-1] * n
    current_group = 0

    i = 0
    while i < n:
        group_indices[i] = current_group
        j = i + 1
        while j < n:
            ref = max(abs(eigenvalues[i]), abs(eigenvalues[j]), 1.0)
            if abs(eigenvalues[j] - eigenvalues[i]) <= rtol * ref:
                group_indices[j] = current_group
                j += 1
            else:
                break
        current_group += 1
        i = j

    # Compute per-eigenvalue multiplicity from group sizes
    group_sizes = {}
    for g in group_indices:
        group_sizes[g] = group_sizes.get(g, 0) + 1

    multiplicity = [group_sizes[g] for g in group_indices]
    n_repeating = sum(1 for m in multiplicity if m > 1)
    n_non_repeating = sum(1 for m in multiplicity if m == 1)

    return {
        'multiplicity': multiplicity,
        'group_indices': group_indices,
        'n_repeating': n_repeating,
        'n_non_repeating': n_non_repeating,
    }


def analyze_spectrum(points, n_eigs=20, n_neighbors=12, threshold=None):
    """Full spectral analysis of a point cloud.

    Parameters
    ----------
    points : ndarray of shape (N, D)
    n_eigs : int
    n_neighbors : int
    threshold : float or None. If None, uses 5 / sqrt(n_points) which
        accounts for discretization noise: a perfectly anti-symmetric
        eigenvector on N discrete samples has score ~ O(1/sqrt(N)).

    Returns
    -------
    dict with keys:
        eigenvalues : ndarray
        eigenvectors : ndarray
        scores : list of float, uncanonicalizability score per eigenvector
        uncanonicalizable_raw : list of bool, original score-based classification
        uncanonicalizable : list of bool, corrected (excludes repeated eigenvalues)
        component_indices : ndarray
        spectral_gap : float, lambda_2 - lambda_1
        threshold_used : float
        multiplicity_info : dict from detect_eigenvalue_multiplicities
    """
    L, comp_idx = build_graph_laplacian(points, n_neighbors=n_neighbors)
    eigenvalues, eigenvectors = compute_eigenpairs(L, n_eigs=n_eigs)

    n_points = eigenvectors.shape[0]
    if threshold is None:
        threshold = 5.0 / np.sqrt(n_points)

    scores = [uncanonicalizability_score(eigenvectors[:, i])
              for i in range(eigenvectors.shape[1])]
    uncanonicalizable_raw = [s < threshold for s in scores]

    # Multiplicity detection
    mult_info = detect_eigenvalue_multiplicities(eigenvalues)

    # Corrected uncanonicalizable: exclude eigenvectors belonging to repeated eigenvalues
    # (sign ambiguity from multiplicity is a different phenomenon than from symmetry)
    uncanonicalizable = [
        (raw and mult_info['multiplicity'][i] == 1)
        for i, raw in enumerate(uncanonicalizable_raw)
    ]

    spectral_gap = float(eigenvalues[1] - eigenvalues[0]) if len(eigenvalues) > 1 else 0.0

    return {
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'scores': scores,
        'uncanonicalizable_raw': uncanonicalizable_raw,
        'uncanonicalizable': uncanonicalizable,
        'component_indices': comp_idx,
        'spectral_gap': spectral_gap,
        'threshold_used': threshold,
        'multiplicity_info': mult_info,
    }
