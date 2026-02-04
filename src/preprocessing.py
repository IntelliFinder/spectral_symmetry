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
        json_path = filepath.with_suffix('.json')
        with open(json_path, 'w') as f:
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
    json_path = filepath.with_suffix('.json')
    metadata = None
    if json_path.exists():
        with open(json_path, 'r') as f:
            metadata = json.load(f)
    return points, metadata
