"""HKS (Heat Kernel Signature) ModelNet dataset for Deep Sets.

HKS features are sign-invariant per-point descriptors computed from
Laplacian eigenvectors and eigenvalues. No canonicalization is needed
since HKS uses v_k(i)^2, eliminating sign ambiguity entirely.

When ``use_squared=False``, computes exponentially-weighted eigenvector
sums instead: ``f(i,t) = sum_k exp(-lambda_k * t) * v_k(i)``. This
variant requires eigenvector canonicalization since sign matters.

Returns ``(features, mask, label)`` -- a 3-tuple (no eigenvalues needed
since HKS already incorporates them via the heat kernel weights).
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.datasets.modelnet import ModelNet10Dataset, ModelNet40Dataset
from src.preprocessing import center_and_normalize, random_subsample
from src.spectral_canonicalization import canonicalize
from src.spectral_core import (
    build_graph_laplacian,
    compute_dataset_sigma,
    compute_eigenpairs,
    compute_hks,
)

_METHOD_MAP = {"random": "random_fixed"}


def _compute_weighted_eigvec_sum(eigenvalues, eigenvectors, n_times=16, t_min=None, t_max=None):
    """Compute exponentially-weighted eigenvector sums (non-squared variant).

    ``f(i, t) = sum_k exp(-lambda_k * t) * v_k(i)``

    Uses the same time-scale logic as ``compute_hks`` but without squaring.
    """
    if len(eigenvalues) == 0:
        N = eigenvectors.shape[0] if eigenvectors.ndim == 2 else 0
        return np.zeros((N, n_times), dtype=np.float32)

    evals = np.clip(eigenvalues, 1e-8, None)
    if t_min is None:
        t_min = 4.0 * np.log(10) / evals[-1]
    if t_max is None:
        t_max = 4.0 * np.log(10) / evals[0]

    times = np.geomspace(t_min, t_max, n_times)

    # f(i, t) = sum_k exp(-lambda_k * t) * v_k(i)
    result = np.zeros((eigenvectors.shape[0], n_times), dtype=np.float32)
    for j, t in enumerate(times):
        weights = np.exp(-evals * t)  # (k,)
        result[:, j] = eigenvectors @ weights  # (N,)

    return result


class HKSModelNet(Dataset):
    """ModelNet dataset with Heat Kernel Signature features.

    Returns ``(features, mask, label)`` where:

    - ``features``: ``(max_points, coord_dim + n_times)`` float tensor
      -- optionally xyz concatenated with HKS features.
    - ``mask``: ``(max_points,)`` bool tensor -- True = padded,
      False = valid.
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
        Number of eigenpairs for HKS computation.
    n_neighbors : int
        k-NN neighbors for graph construction.
    n_times : int
        Number of HKS time samples (output feature dimension).
    weighted : bool
        If True, use Gaussian-kernel weighted graph Laplacian.
    normalized : bool
        If True, use symmetric normalized Laplacian.
    include_xyz : bool
        If True, prepend xyz coordinates to HKS features.
    sigma : float or None
        Bandwidth for Gaussian kernel. Auto-computed if None and
        ``weighted=True``.
    use_squared : bool
        If True (default), compute standard HKS using v_k(i)^2.
        If False, compute weighted eigenvector sums using v_k(i) directly
        (requires canonicalization).
    canonicalization : str
        Canonicalization method when ``use_squared=False``. One of
        ``"spielman"``, ``"maxabs"``, ``"random"``, ``"none"``.
        Ignored when ``use_squared=True``.
    """

    def __init__(
        self,
        root,
        dataset="ModelNet10",
        split="train",
        max_points=1024,
        n_eigs=32,
        n_neighbors=30,
        n_times=16,
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

        # Resolve variant from dataset name
        if dataset == "ModelNet10":
            variant = 10
        elif dataset == "ModelNet40":
            variant = 40
        else:
            raise ValueError(f"dataset must be 'ModelNet10' or 'ModelNet40', got {dataset!r}")
        self.variant = variant

        if variant == 10:
            base_dataset = ModelNet10Dataset(
                root,
                split=split,
                max_points=max_points * 2,
                download=False,
            )
        else:
            base_dataset = ModelNet40Dataset(
                root,
                split=split,
                max_points=max_points * 2,
                download=False,
            )

        # Collect all class names first for a sorted label mapping
        raw_items = []
        class_names = set()
        for name, points in base_dataset:
            parts = name.split("/")
            class_name = parts[1]
            class_names.add(class_name)
            raw_items.append((name, class_name, points))

        self.classes = sorted(class_names)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # Compute uniform sigma across the dataset when using weighted Laplacians
        if weighted and sigma is None:
            sample_points = []
            for idx, (name, class_name, points) in enumerate(raw_items[:200]):
                pts = random_subsample(points, max_points, seed=idx)
                pts, _, _ = center_and_normalize(pts)
                sample_points.append(pts)
            sigma = compute_dataset_sigma(sample_points, n_neighbors=n_neighbors)
            self.sigma = sigma

        # Pre-compute HKS features for every shape
        self.data = []
        n_skipped = 0
        desc = f"HKS {split}"
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
                # Eigensolver failure: use zero HKS features
                n_actual = points.shape[0]
                cd = self.coord_dim
                features = np.zeros((max_points, cd + n_times), dtype=np.float32)
                if cd > 0:
                    features[:n_actual, :cd] = points[:n_actual].astype(np.float32)

                mask = np.ones(max_points, dtype=bool)
                mask[:n_actual] = False

                label = self.class_to_idx[class_name]
                self.data.append((features, mask, label))
                n_skipped += 1
                continue

            if use_squared:
                # Standard HKS: sign-invariant via v_k(i)^2
                feat = compute_hks(eigenvalues, eigenvectors, n_times=n_times)
            else:
                # Exponentially-weighted eigenvector sums (sign-sensitive)
                unified_method = _METHOD_MAP.get(canonicalization, canonicalization)
                canon_evecs = canonicalize(
                    eigenvectors, eigenvalues=eigenvalues, method=unified_method, sample_idx=idx
                )
                feat = _compute_weighted_eigvec_sum(eigenvalues, canon_evecs, n_times=n_times)

            n_pts = points.shape[0]
            cd = self.coord_dim

            # Build feature matrix: all points kept, features placed at
            # LCC indices (non-LCC points get zero features but stay valid)
            features = np.zeros((max_points, cd + n_times), dtype=np.float32)
            if cd > 0:
                features[:n_pts, :cd] = points.astype(np.float32)
            features[comp_idx, cd : cd + n_times] = feat.astype(np.float32)

            # Mask: True at padded positions (all n_pts points valid)
            mask = np.ones(max_points, dtype=bool)
            mask[:n_pts] = False

            label = self.class_to_idx[class_name]
            self.data.append((features, mask, label))

        if n_skipped > 0:
            print(f"Warning: {n_skipped} shapes had eigensolver failures (using zero HKS features)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features, mask, label = self.data[idx]
        return (
            torch.from_numpy(features).float(),
            torch.from_numpy(mask).bool(),
            label,
        )
