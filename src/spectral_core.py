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


def build_graph_laplacian(points, n_neighbors=12, weighted=False, sigma=None, normalized=False):
    """Build a symmetric k-NN graph and return the sparse graph Laplacian.

    By default computes the combinatorial Laplacian L = D - A.
    When ``normalized=True``, computes the symmetric normalized Laplacian
    L_norm = I - D^{-1/2} A D^{-1/2}.

    Parameters
    ----------
    points : ndarray of shape (N, D)
    n_neighbors : int
    weighted : bool
        If True, use Gaussian kernel weights instead of binary adjacency.
    sigma : float or None
        Bandwidth for Gaussian kernel. If None and weighted=True, uses the
        median of nonzero distances.
    normalized : bool
        If True, return the symmetric normalized Laplacian.

    Returns
    -------
    L : sparse CSR matrix
    component_indices : ndarray, indices into original point array
    """
    n_points = points.shape[0]
    k = min(n_neighbors, n_points - 1)

    if weighted:
        A = kneighbors_graph(points, n_neighbors=k, mode="distance", include_self=False)
        A = sp.csr_matrix(A)
        A = A.maximum(A.T)  # symmetrize via element-wise max (not average)
        # Keep only nonzero structure for the Gaussian kernel
        A = sp.csr_matrix(A)
        if sigma is None:
            sigma = float(np.median(A.data))
        # Apply Gaussian kernel: exp(-d^2 / (2 * sigma^2))
        A.data = np.exp(-(A.data**2) / (2.0 * sigma**2))
    else:
        A = kneighbors_graph(points, n_neighbors=k, mode="connectivity", include_self=False)
        A = 0.5 * (A + A.T)  # symmetrize
        A = (A > 0).astype(float)  # binary

    A, component_indices = _largest_connected_component(A)
    A = sp.csr_matrix(A)

    degrees = np.array(A.sum(axis=1)).flatten()

    if normalized:
        # Symmetric normalized Laplacian: L_norm = I - D^{-1/2} A D^{-1/2}
        d_inv_sqrt = np.zeros_like(degrees)
        nonzero = degrees > 0
        d_inv_sqrt[nonzero] = 1.0 / np.sqrt(degrees[nonzero])
        D_inv_sqrt = sp.diags(d_inv_sqrt)
        L = sp.eye(A.shape[0]) - D_inv_sqrt @ A @ D_inv_sqrt
    else:
        D = sp.diags(degrees)
        L = D - A

    return L, component_indices


def compute_eigenpairs(L, n_eigs=20, trivial_threshold=1e-6):
    """Compute the smallest non-trivial eigenpairs of a sparse Laplacian.

    Following Laplacian Eigenmaps (Belkin & Niyogi 2003), trivial
    eigenvectors (eigenvalue below ``trivial_threshold``) are excluded.
    For a connected graph this removes only the constant eigenvector;
    for near-disconnected graphs it removes all near-zero modes.

    Parameters
    ----------
    L : sparse matrix
    n_eigs : int
        Number of non-trivial eigenpairs to return.
    trivial_threshold : float
        Eigenvalues at or below this value are considered trivial and
        discarded.  Default 1e-6.

    Returns
    -------
    eigenvalues : ndarray of shape (<= n_eigs,)
    eigenvectors : ndarray of shape (N, <= n_eigs)
    """
    n = L.shape[0]
    # Request extras to account for trivial eigenvectors we'll discard
    k = min(n_eigs + 1, n - 2)  # eigsh needs k < n
    vals, vecs = sla.eigsh(L, k=k, which="SM", tol=1e-8)
    idx = np.argsort(vals)
    vals = vals[idx]
    vecs = vecs[:, idx]
    # Skip trivial eigenvectors (eigenvalue <= threshold)
    nontrivial = vals > trivial_threshold
    vals = vals[nontrivial]
    vecs = vecs[:, nontrivial]
    # Return up to n_eigs
    return vals[:n_eigs], vecs[:, :n_eigs]


def compute_hks(eigenvalues, eigenvectors, n_times=16, t_min=None, t_max=None):
    """Compute Heat Kernel Signature for each point.

    HKS is a sign-invariant per-point descriptor:
    ``HKS(i, t) = sum_k exp(-lambda_k * t) * v_k(i)^2``

    Parameters
    ----------
    eigenvalues : ndarray of shape (k,)
        Non-trivial eigenvalues (positive).
    eigenvectors : ndarray of shape (N, k)
        Corresponding eigenvectors.
    n_times : int
        Number of time samples (output feature dimension).
    t_min : float or None
        Minimum time. If None, uses ``4 * ln(10) / eigenvalues[-1]``.
    t_max : float or None
        Maximum time. If None, uses ``4 * ln(10) / eigenvalues[0]``.

    Returns
    -------
    hks : ndarray of shape (N, n_times)
        Heat kernel signature at each point for each time scale.
    """
    if len(eigenvalues) == 0:
        # No non-trivial eigenvalues — return zeros
        N = eigenvectors.shape[0] if eigenvectors.ndim == 2 else 0
        return np.zeros((N, n_times), dtype=np.float32)

    evals = np.clip(eigenvalues, 1e-8, None)
    if t_min is None:
        t_min = 4.0 * np.log(10) / evals[-1]
    if t_max is None:
        t_max = 4.0 * np.log(10) / evals[0]

    times = np.geomspace(t_min, t_max, n_times)  # log-spaced

    # HKS(i, t) = sum_k exp(-lambda_k * t) * v_k(i)^2
    V_sq = eigenvectors**2  # (N, k) -- sign invariant

    hks = np.zeros((eigenvectors.shape[0], n_times), dtype=np.float32)
    for j, t in enumerate(times):
        weights = np.exp(-evals * t)  # (k,)
        hks[:, j] = V_sq @ weights  # (N,)

    return hks


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
        "multiplicity": multiplicity,
        "group_indices": group_indices,
        "n_repeating": n_repeating,
        "n_non_repeating": n_non_repeating,
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
        spectral_gap : float, smallest non-trivial eigenvalue (Fiedler value)
        threshold_used : float
        multiplicity_info : dict from detect_eigenvalue_multiplicities
    """
    L, comp_idx = build_graph_laplacian(points, n_neighbors=n_neighbors)
    eigenvalues, eigenvectors = compute_eigenpairs(L, n_eigs=n_eigs)

    n_points = eigenvectors.shape[0]
    if threshold is None:
        threshold = 5.0 / np.sqrt(n_points)

    scores = [uncanonicalizability_score(eigenvectors[:, i]) for i in range(eigenvectors.shape[1])]
    uncanonicalizable_raw = [s < threshold for s in scores]

    # Multiplicity detection
    mult_info = detect_eigenvalue_multiplicities(eigenvalues)

    # Corrected uncanonicalizable: exclude eigenvectors belonging to repeated eigenvalues
    # (sign ambiguity from multiplicity is a different phenomenon than from symmetry)
    uncanonicalizable = [
        (raw and mult_info["multiplicity"][i] == 1) for i, raw in enumerate(uncanonicalizable_raw)
    ]

    spectral_gap = float(eigenvalues[0]) if len(eigenvalues) > 0 else 0.0

    return {
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "scores": scores,
        "uncanonicalizable_raw": uncanonicalizable_raw,
        "uncanonicalizable": uncanonicalizable,
        "component_indices": comp_idx,
        "spectral_gap": spectral_gap,
        "threshold_used": threshold,
        "multiplicity_info": mult_info,
    }
