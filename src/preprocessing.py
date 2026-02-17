"""Point cloud preprocessing utilities."""

import json
from pathlib import Path

import numpy as np


def center_and_normalize(points):
    """Center a point cloud at the origin and normalize to unit scale.

    Parameters
    ----------
    points : ndarray of shape (N, D)

    Returns
    -------
    normalized : ndarray of shape (N, D)
    centroid : ndarray of shape (D,)
    scale : float
        The maximum distance from the centroid before normalization.
    """
    centroid = points.mean(axis=0)
    centered = points - centroid
    scale = float(np.max(np.linalg.norm(centered, axis=1)))
    if scale < 1e-12:
        return centered, centroid, scale
    normalized = centered / scale
    return normalized, centroid, scale


PCA_SIGN_METHODS = ("majority", "maxabs", "random", "spielman")


def pca_canonicalize(points, sign_method="majority", shape_idx=0):
    """Canonicalize point cloud orientation via PCA eigendecomposition.

    Computes the covariance matrix C = X^T X / N of the centered point cloud,
    eigendecomposes it, and rotates the cloud to align with principal axes.
    Signs are resolved via the chosen method and det(R)=+1 is enforced.

    Parameters
    ----------
    points : ndarray of shape (N, 3), assumed already centered
    sign_method : str
        One of "majority" (majority vote + skewness tie-break),
        "maxabs" (flip so max-abs projection is positive),
        "random" (reproducible random signs seeded by shape_idx),
        "spielman" (Spielman GF(2) canonicalization on projection vectors).
    shape_idx : int
        Index of the shape (used for seeding "random" mode and Spielman).

    Returns
    -------
    canonicalized : ndarray of shape (N, 3)
    """
    N = points.shape[0]
    if N == 0:
        return points.copy()

    # 1. Covariance matrix
    C = points.T @ points / N  # (3, 3)

    # 2. Eigendecompose (eigh returns ascending eigenvalues)
    eigenvalues, eigenvectors = np.linalg.eigh(C)

    # 3. Reorder to descending eigenvalues
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    V = eigenvectors[:, order]  # (3, 3), columns are principal axes

    # 4. Compute projections for sign decisions
    projections = points @ V  # (N, 3)

    # 5. Sign canonicalization
    signs = np.ones(3)
    confidence = np.zeros(3)

    if sign_method == "majority":
        signs, confidence = _pca_sign_majority(projections, N)
    elif sign_method == "maxabs":
        signs, confidence = _pca_sign_maxabs(projections)
    elif sign_method == "random":
        signs, confidence = _pca_sign_random(shape_idx)
    elif sign_method == "spielman":
        signs, confidence = _pca_sign_spielman(projections, eigenvalues)
    else:
        raise ValueError(
            f"Unknown PCA sign method: {sign_method!r}. Choose from {PCA_SIGN_METHODS}"
        )

    # 6. Build rotation matrix R = V * signs
    R = V * signs[np.newaxis, :]  # (3, 3)

    # Enforce det(R) = +1 (proper rotation)
    if np.linalg.det(R) < 0:
        flip_idx = np.argmin(confidence)
        R[:, flip_idx] *= -1

    # 7. Rotate points
    canonicalized = points @ R  # (N, 3)
    return canonicalized


def _pca_sign_majority(projections, N):
    """Majority vote sign resolution with skewness tie-break."""
    signs = np.ones(3)
    confidence = np.zeros(3)
    for j in range(3):
        proj = projections[:, j]
        n_pos = np.sum(proj > 0)
        n_neg = np.sum(proj < 0)
        if n_pos > n_neg:
            signs[j] = 1.0
            confidence[j] = float(n_pos - n_neg)
        elif n_neg > n_pos:
            signs[j] = -1.0
            confidence[j] = float(n_neg - n_pos)
        else:
            skewness = np.mean(proj**3)
            if skewness > 0:
                signs[j] = 1.0
                confidence[j] = abs(skewness)
            elif skewness < 0:
                signs[j] = -1.0
                confidence[j] = abs(skewness)
            else:
                max_idx = np.argmax(np.abs(proj))
                signs[j] = np.sign(proj[max_idx]) if proj[max_idx] != 0 else 1.0
                confidence[j] = 0.0
    return signs, confidence


def _pca_sign_maxabs(projections):
    """Max-absolute-value sign resolution."""
    signs = np.ones(3)
    confidence = np.zeros(3)
    for j in range(3):
        proj = projections[:, j]
        max_idx = np.argmax(np.abs(proj))
        signs[j] = np.sign(proj[max_idx]) if proj[max_idx] != 0 else 1.0
        confidence[j] = np.abs(proj[max_idx])
    return signs, confidence


def _pca_sign_random(shape_idx):
    """Reproducible random sign assignment."""
    signs = np.ones(3)
    confidence = np.zeros(3)
    for j in range(3):
        rng = np.random.RandomState(seed=shape_idx * 1000 + j)
        if rng.random() < 0.5:
            signs[j] = -1.0
    return signs, confidence


def _pca_sign_spielman(projections, eigenvalues):
    """Spielman GF(2) canonicalization applied to PCA projection vectors.

    Treats each column of `projections` (N projections onto a principal axis)
    as an "eigenvector" and applies the Spielman sign convention using the
    PCA eigenvalues.
    """
    from src.spectral_canonicalization import spectral_canonicalize

    # projections: (N, 3) — treat as 3 "eigenvectors" of length N
    canonicalized = spectral_canonicalize(projections, eigenvalues)
    # Determine sign flips by comparing with original
    signs = np.ones(3)
    confidence = np.zeros(3)
    for j in range(3):
        dot = np.dot(canonicalized[:, j], projections[:, j])
        signs[j] = np.sign(dot) if abs(dot) > 1e-12 else 1.0
        confidence[j] = abs(dot)
    return signs, confidence


def random_subsample(points, n_points, seed=None):
    """Randomly subsample a point cloud to exactly n_points.

    Parameters
    ----------
    points : ndarray of shape (N, D)
    n_points : int
    seed : int or None

    Returns
    -------
    ndarray of shape (n_points, D)
    """
    if len(points) <= n_points:
        return points
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(points), n_points, replace=False)
    return points[idx]


def save_processed(points, filepath, metadata=None):
    """Save a point cloud as .npy with an optional JSON sidecar.

    Parameters
    ----------
    points : ndarray
    filepath : str or Path
        Path for the .npy file.
    metadata : dict or None
        If provided, saved as a .json file alongside the .npy.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    np.save(filepath, points)
    if metadata is not None:
        json_path = filepath.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)


def load_processed(filepath):
    """Load a point cloud and optional metadata from .npy + .json sidecar.

    Parameters
    ----------
    filepath : str or Path

    Returns
    -------
    points : ndarray
    metadata : dict or None
    """
    filepath = Path(filepath)
    points = np.load(filepath)
    json_path = filepath.with_suffix(".json")
    metadata = None
    if json_path.exists():
        with open(json_path, "r") as f:
            metadata = json.load(f)
    return points, metadata
