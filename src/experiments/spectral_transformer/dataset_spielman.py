"""Spielman-canonicalized spectral ModelNet datasets.

Dataset classes that support multiple eigenvector canonicalization strategies:
- "spielman": Spielman-style algebraic canonicalization (requires eigenvalues)
- "maxabs": Legacy max-absolute-value sign convention
- "random": Reproducible random sign flips (seeded by shape index)
- "none": Raw eigensolver output (no sign correction)
"""

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset
from tqdm import tqdm

from src.datasets.modelnet import ModelNet10Dataset, ModelNet40Dataset
from src.preprocessing import center_and_normalize, random_subsample
from src.spectral_canonicalization import canonicalize
from src.spectral_core import build_graph_laplacian, compute_eigenpairs

# Legacy aliases kept for backward compatibility
CANONICALIZATION_CHOICES = ("spielman", "maxabs", "random", "none")

# Map legacy method names to unified dispatcher names
_METHOD_MAP = {"random": "random_fixed"}


def _apply_canonicalization(eigenvectors, eigenvalues, method, shape_idx):
    """Apply the chosen canonicalization to eigenvectors.

    Thin wrapper around ``src.spectral_canonicalization.canonicalize``.
    """
    unified_method = _METHOD_MAP.get(method, method)
    return canonicalize(
        eigenvectors, eigenvalues=eigenvalues, method=unified_method, sample_idx=shape_idx
    )


class SpielmanNodeModelNet(Dataset):
    """ModelNet shapes with eigenvector node features, k-NN distances, and configurable
    canonicalization.

    Nearly identical to ``SpectralNodeModelNet`` but supports multiple
    canonicalization strategies via the ``canonicalization`` parameter and
    preserves eigenvalues for the Spielman method.

    Returns a 4-tuple per item:
    ``(features, dist_matrix, mask, label)`` where:

    - ``features``: ``(n_points, 3 + n_eigs)`` float tensor (xyz || eigvecs).
    - ``dist_matrix``: ``(n_points, n_points)`` float tensor of k-NN distances.
    - ``mask``: ``(n_points,)`` bool tensor, True at padded positions.
    - ``label``: int class index.

    Parameters
    ----------
    root_dir : str
        Root data directory.
    split : str
        'train' or 'test'.
    n_points : int
        Number of points per shape.
    n_eigs : int
        Number of eigenvectors to compute.
    n_neighbors : int
        k-NN neighbors for graph construction.
    download : bool
        If True, download the dataset if missing.
    variant : int
        10 or 40.
    canonicalization : str
        One of "spielman", "maxabs", "random", "none".
    """

    def __init__(
        self,
        root_dir,
        split="train",
        n_points=512,
        n_eigs=16,
        n_neighbors=12,
        download=False,
        variant=10,
        canonicalization="spielman",
        weighted=False,
        sigma=None,
    ):
        super().__init__()
        self.n_points = n_points
        self.n_eigs = n_eigs
        self.n_neighbors = n_neighbors
        self.variant = variant
        self.canonicalization = canonicalization
        self.weighted = weighted
        self.sigma = sigma

        if canonicalization not in CANONICALIZATION_CHOICES:
            raise ValueError(
                f"canonicalization must be one of {CANONICALIZATION_CHOICES}, "
                f"got {canonicalization!r}"
            )

        if variant == 10:
            base_dataset = ModelNet10Dataset(
                root_dir,
                split=split,
                max_points=n_points * 2,
                download=download,
            )
        elif variant == 40:
            base_dataset = ModelNet40Dataset(
                root_dir,
                split=split,
                max_points=n_points * 2,
                download=download,
            )
        else:
            raise ValueError(f"variant must be 10 or 40, got {variant}")

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
        for idx, (name, class_name, points) in enumerate(
            tqdm(raw_items, desc=f"SpielmanNode {split}")
        ):
            points = random_subsample(points, n_points, seed=idx)
            points, _, _ = center_and_normalize(points)

            L, comp_idx = build_graph_laplacian(
                points, n_neighbors=n_neighbors, weighted=weighted, sigma=sigma
            )
            try:
                eigenvalues, eigenvectors = compute_eigenpairs(L, n_eigs=n_eigs)
            except Exception:
                n_skipped += 1
                continue

            # Apply chosen canonicalization
            eigenvectors = _apply_canonicalization(
                eigenvectors, eigenvalues, canonicalization, shape_idx=idx
            )

            pts_cc = points[comp_idx]
            n_actual = pts_cc.shape[0]
            n_eigs_actual = eigenvectors.shape[1]

            # Node features: xyz || eigvecs, padded
            feat_dim = 3 + n_eigs
            features = np.zeros((n_points, feat_dim), dtype=np.float32)
            features[:n_actual, :3] = pts_cc.astype(np.float32)
            features[:n_actual, 3 : 3 + n_eigs_actual] = eigenvectors.astype(np.float32)

            # k-NN distance matrix on connected-component points
            k = min(n_neighbors, n_actual - 1)
            nn_model = NearestNeighbors(n_neighbors=k + 1)
            nn_model.fit(pts_cc)
            distances, indices = nn_model.kneighbors(pts_cc)

            dist_matrix = np.zeros((n_points, n_points), dtype=np.float32)
            for i in range(n_actual):
                for j_idx in range(1, k + 1):
                    j = indices[i, j_idx]
                    dist_matrix[i, j] = distances[i, j_idx]
            dist_matrix = 0.5 * (dist_matrix + dist_matrix.T)

            mask = np.ones(n_points, dtype=bool)
            mask[:n_actual] = False

            label = self.class_to_idx[class_name]
            self.data.append((features, dist_matrix, mask, label))

        if n_skipped > 0:
            print(f"Warning: skipped {n_skipped} shapes due to eigensolver failures")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features, dist_matrix, mask, label = self.data[idx]
        return (
            torch.from_numpy(features),
            torch.from_numpy(dist_matrix),
            torch.from_numpy(mask),
            label,
        )


class SpielmanSpectralModelNet(Dataset):
    """ModelNet shapes with spectral positional encodings and configurable
    canonicalization.

    Nearly identical to ``SpectralModelNet`` but supports multiple
    canonicalization strategies via the ``canonicalization`` parameter.

    Each item is a tuple ``(features, mask, label)`` where:
    - ``features`` is a ``(n_points, 3 + n_eigs)`` float tensor (xyz || eigvecs),
      zero-padded to ``n_points`` along the sequence dimension.
    - ``mask`` is a ``(n_points,)`` bool tensor, True at padded positions.
    - ``label`` is an int class index.

    Parameters
    ----------
    root_dir : str
        Root data directory.
    split : str
        'train' or 'test'.
    n_points : int
        Number of points per shape.
    n_eigs : int
        Number of eigenvectors to compute.
    n_neighbors : int
        k-NN neighbors for graph construction.
    download : bool
        If True, download the dataset if missing.
    variant : int
        10 or 40.
    canonicalization : str
        One of "spielman", "maxabs", "random", "none".
    weighted : bool
        If True, use Gaussian kernel weighted graph Laplacian.
    sigma : float or None
        Bandwidth for Gaussian kernel. Auto-computed if None and weighted=True.
    """

    def __init__(
        self,
        root_dir,
        split="train",
        n_points=512,
        n_eigs=16,
        n_neighbors=12,
        download=False,
        variant=10,
        canonicalization="spielman",
        weighted=False,
        sigma=None,
    ):
        super().__init__()
        self.n_points = n_points
        self.n_eigs = n_eigs
        self.n_neighbors = n_neighbors
        self.variant = variant
        self.canonicalization = canonicalization
        self.weighted = weighted
        self.sigma = sigma

        if canonicalization not in CANONICALIZATION_CHOICES:
            raise ValueError(
                f"canonicalization must be one of {CANONICALIZATION_CHOICES}, "
                f"got {canonicalization!r}"
            )

        if variant == 10:
            base_dataset = ModelNet10Dataset(
                root_dir,
                split=split,
                max_points=n_points * 2,
                download=download,
            )
        elif variant == 40:
            base_dataset = ModelNet40Dataset(
                root_dir,
                split=split,
                max_points=n_points * 2,
                download=download,
            )
        else:
            raise ValueError(f"variant must be 10 or 40, got {variant}")

        # Collect all class names first to build a sorted label mapping
        raw_items = []
        class_names = set()
        for name, points in base_dataset:
            parts = name.split("/")
            class_name = parts[1]
            class_names.add(class_name)
            raw_items.append((name, class_name, points))

        self.classes = sorted(class_names)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # Pre-compute spectral features for every shape
        self.data = []
        n_skipped = 0
        for idx, (name, class_name, points) in enumerate(
            tqdm(raw_items, desc=f"SpielmanSpectral {split}")
        ):
            points = random_subsample(points, n_points, seed=idx)
            points, _, _ = center_and_normalize(points)

            L, comp_idx = build_graph_laplacian(
                points, n_neighbors=n_neighbors, weighted=weighted, sigma=sigma
            )
            try:
                eigenvalues, eigenvectors = compute_eigenpairs(L, n_eigs=n_eigs)
            except Exception:
                n_skipped += 1
                continue

            # Apply chosen canonicalization
            eigenvectors = _apply_canonicalization(
                eigenvectors, eigenvalues, canonicalization, shape_idx=idx
            )

            # Keep only the largest connected component
            pts_cc = points[comp_idx]
            n_actual = pts_cc.shape[0]
            n_eigs_actual = eigenvectors.shape[1]

            # Build feature matrix: xyz || eigvecs, padded to (n_points, 3 + n_eigs)
            feat_dim = 3 + n_eigs
            features = np.zeros((n_points, feat_dim), dtype=np.float32)
            features[:n_actual, :3] = pts_cc.astype(np.float32)
            features[:n_actual, 3 : 3 + n_eigs_actual] = eigenvectors.astype(np.float32)

            # Mask: True at padded positions
            mask = np.ones(n_points, dtype=bool)
            mask[:n_actual] = False

            label = self.class_to_idx[class_name]
            self.data.append((features, mask, label))

        if n_skipped > 0:
            print(f"Warning: skipped {n_skipped} shapes due to eigensolver failures")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features, mask, label = self.data[idx]
        return (
            torch.from_numpy(features),
            torch.from_numpy(mask),
            label,
        )
