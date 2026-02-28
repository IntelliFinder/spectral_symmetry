"""Concatenated Heat Kernel Features (no summation) for Deep Sets.

Standard HKS **sums** over eigenvectors at each time scale:
    HKS(i, t) = sum_k exp(-lambda_k * t) * v_k(i)^2   ->  (N, T)

This module **keeps each eigenvector's contribution separate**:
    feature(i, k, t) = exp(-lambda_k * t) * v_k(i)^2   ->  (N, K*T)

With K=8 eigenvectors and T=32 time scales this gives 256 features/point
instead of 32.  The model sees which eigenvectors contribute at which
time scales rather than a blended summary.

Non-squared variant with canonicalization:
    feature(i, k, t) = exp(-lambda_k * t) * v_k(i)

Returns ``(features, mask, label)`` -- same 3-tuple interface as
``HKSModelNet``.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.datasets.modelnet import ModelNet10Dataset, ModelNet40Dataset
from src.preprocessing import center_and_normalize, random_subsample
from src.spectral_canonicalization import canonicalize
from src.spectral_core import build_graph_laplacian, compute_eigenpairs

_METHOD_MAP = {"random": "random_fixed"}


def _compute_concat_hks(eigenvalues, eigenvectors, n_times=32):
    """Concatenated heat kernel features (no summation, squared).

    ``feature(i, k, t) = exp(-lambda_k * t) * v_k(i)^2``

    Returns (N, K*T) array where columns are ordered as:
    ``[k0_t0, k0_t1, ..., k0_tT-1, k1_t0, ..., kK-1_tT-1]``
    """
    K = len(eigenvalues)
    N = eigenvectors.shape[0]
    if K == 0:
        return np.zeros((N, 0), dtype=np.float32)

    evals = np.clip(eigenvalues, 1e-8, None)
    t_min = 4.0 * np.log(10) / evals[-1]
    t_max = 4.0 * np.log(10) / evals[0]
    times = np.geomspace(t_min, t_max, n_times)

    V_sq = eigenvectors**2  # (N, K)
    weights = np.exp(-evals[:, None] * times[None, :])  # (K, T)

    result = np.zeros((N, K * n_times), dtype=np.float32)
    for k in range(K):
        result[:, k * n_times : (k + 1) * n_times] = V_sq[:, k : k + 1] * weights[k : k + 1, :]
    return result


def _compute_concat_wes(eigenvalues, eigenvectors, n_times=32):
    """Concatenated weighted eigenvector features (no summation, non-squared).

    ``feature(i, k, t) = exp(-lambda_k * t) * v_k(i)``

    Same structure as ``_compute_concat_hks`` but without squaring.
    Requires eigenvector canonicalization since sign matters.
    """
    K = len(eigenvalues)
    N = eigenvectors.shape[0]
    if K == 0:
        return np.zeros((N, 0), dtype=np.float32)

    evals = np.clip(eigenvalues, 1e-8, None)
    t_min = 4.0 * np.log(10) / evals[-1]
    t_max = 4.0 * np.log(10) / evals[0]
    times = np.geomspace(t_min, t_max, n_times)

    weights = np.exp(-evals[:, None] * times[None, :])  # (K, T)

    result = np.zeros((N, K * n_times), dtype=np.float32)
    for k in range(K):
        result[:, k * n_times : (k + 1) * n_times] = (
            eigenvectors[:, k : k + 1] * weights[k : k + 1, :]
        )
    return result


class ConcatHKSModelNet(Dataset):
    """ModelNet dataset with concatenated heat kernel features (no summation).

    Returns ``(features, mask, label)`` where:

    - ``features``: ``(max_points, coord_dim + n_eigs * n_times)`` float tensor
    - ``mask``: ``(max_points,)`` bool tensor -- True = padded, False = valid.
    - ``label``: int class index.

    Parameters
    ----------
    root : str
        Root data directory.
    dataset : str
        ``"ModelNet10"`` or ``"ModelNet40"``.
    split : str
        ``"train"`` or ``"test"``.
    max_points : int
        Number of points per shape (subsample or zero-pad).
    n_eigs : int
        Number of eigenpairs (K).
    n_neighbors : int
        k-NN neighbors for graph construction.
    n_times : int
        Number of time samples per eigenvector (T).
    weighted : bool
        If True, use Gaussian-kernel weighted graph Laplacian.
    normalized : bool
        If True, use symmetric normalized Laplacian.
    include_xyz : bool
        If True, prepend xyz coordinates to spectral features.
    sigma : float or None
        Bandwidth for Gaussian kernel. Auto-computed if None and
        ``weighted=True``.
    use_squared : bool
        If True (default), compute squared variant (sign-invariant).
        If False, use v_k(i) directly (requires canonicalization).
    canonicalization : str
        Canonicalization method when ``use_squared=False``. One of
        ``"spielman"``, ``"maxabs"``, ``"random"``, ``"none"``.
    """

    def __init__(
        self,
        root,
        dataset="ModelNet10",
        split="train",
        max_points=1024,
        n_eigs=8,
        n_neighbors=30,
        n_times=32,
        weighted=True,
        normalized=True,
        include_xyz=True,
        sigma=None,
        use_squared=True,
        canonicalization="none",
    ):
        super().__init__()
        self.max_points = max_points
        self.n_eigs = n_eigs
        self.n_neighbors = n_neighbors
        self.n_times = n_times
        self.weighted = weighted
        self.normalized = normalized
        self.include_xyz = include_xyz
        self.sigma = sigma
        self.use_squared = use_squared
        self.canonicalization = canonicalization
        self.coord_dim = 3 if include_xyz else 0
        self.spectral_dim = n_eigs * n_times

        if dataset == "ModelNet10":
            variant = 10
        elif dataset == "ModelNet40":
            variant = 40
        else:
            raise ValueError(f"dataset must be 'ModelNet10' or 'ModelNet40', got {dataset!r}")
        self.variant = variant

        if variant == 10:
            base_dataset = ModelNet10Dataset(
                root, split=split, max_points=max_points * 2, download=False
            )
        else:
            base_dataset = ModelNet40Dataset(
                root, split=split, max_points=max_points * 2, download=False
            )

        raw_items = []
        class_names = set()
        for name, points in base_dataset:
            parts = name.split("/")
            class_name = parts[1]
            class_names.add(class_name)
            raw_items.append((name, class_name, points))

        self.classes = sorted(class_names)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.data = []
        n_skipped = 0
        cd = self.coord_dim
        sdim = self.spectral_dim
        desc = f"ConcatHKS {split}"

        for idx, (name, class_name, points) in enumerate(tqdm(raw_items, desc=desc)):
            points = random_subsample(points, max_points, seed=idx)
            points, _, _ = center_and_normalize(points)

            L, comp_idx = build_graph_laplacian(
                points,
                n_neighbors=n_neighbors,
                weighted=weighted,
                sigma=sigma,
                normalized=normalized,
            )

            try:
                eigenvalues, eigenvectors = compute_eigenpairs(L, n_eigs=n_eigs)
            except Exception:
                n_actual = points.shape[0]
                features = np.zeros((max_points, cd + sdim), dtype=np.float32)
                if cd > 0:
                    features[:n_actual, :cd] = points[:n_actual].astype(np.float32)
                mask = np.ones(max_points, dtype=bool)
                mask[:n_actual] = False
                label = self.class_to_idx[class_name]
                self.data.append((features, mask, label))
                n_skipped += 1
                continue

            k_actual = len(eigenvalues)

            if use_squared:
                feat = _compute_concat_hks(eigenvalues, eigenvectors, n_times=n_times)
            else:
                unified_method = _METHOD_MAP.get(canonicalization, canonicalization)
                canon_evecs = canonicalize(
                    eigenvectors, eigenvalues=eigenvalues, method=unified_method, sample_idx=idx
                )
                feat = _compute_concat_wes(eigenvalues, canon_evecs, n_times=n_times)

            n_pts = points.shape[0]
            features = np.zeros((max_points, cd + sdim), dtype=np.float32)
            if cd > 0:
                features[:n_pts, :cd] = points.astype(np.float32)
            # Only fill columns for eigenvectors actually computed
            features[comp_idx, cd : cd + k_actual * n_times] = feat.astype(np.float32)

            mask = np.ones(max_points, dtype=bool)
            mask[:n_pts] = False

            label = self.class_to_idx[class_name]
            self.data.append((features, mask, label))

        if n_skipped > 0:
            print(f"Warning: {n_skipped} shapes had eigensolver failures (using zero features)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features, mask, label = self.data[idx]
        return (
            torch.from_numpy(features).float(),
            torch.from_numpy(mask).bool(),
            label,
        )
