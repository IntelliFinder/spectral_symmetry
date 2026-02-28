"""GPS (Global Point Signature) ModelNet dataset for Deep Sets.

GPS features are sign-sensitive per-point descriptors (Rustamov 2007):
``GPS(i, k) = phi_k(i) / sqrt(lambda_k)``

The 1/sqrt(lambda_k) weighting emphasizes lower-frequency eigenvectors.
Unlike HKS, GPS is sign-sensitive so eigenvector canonicalization matters.

Returns ``(features, mask, label)`` -- a 3-tuple matching HKS dataset format.
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
    compute_eigenpairs,
    compute_gps,
)

_METHOD_MAP = {"random": "random_fixed"}


class GPSModelNet(Dataset):
    """ModelNet dataset with Global Point Signature features.

    Returns ``(features, mask, label)`` where:

    - ``features``: ``(max_points, coord_dim + n_eigs)`` float tensor
      -- optionally xyz concatenated with GPS features.
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
        Number of eigenpairs (= GPS feature dimension).
    n_neighbors : int
        k-NN neighbors for graph construction.
    weighted : bool
        If True, use Gaussian-kernel weighted graph Laplacian.
    normalized : bool
        If True, use symmetric normalized Laplacian.
    include_xyz : bool
        If True, prepend xyz coordinates to GPS features.
    sigma : float or None
        Bandwidth for Gaussian kernel. Auto-computed if None and
        ``weighted=True``.
    canonicalization : str
        Canonicalization method. One of
        ``"spielman"``, ``"maxabs"``, ``"random"``, ``"none"``.
    """

    def __init__(
        self,
        root,
        dataset="ModelNet10",
        split="train",
        max_points=1024,
        n_eigs=32,
        n_neighbors=30,
        weighted=True,
        normalized=True,
        include_xyz=True,
        sigma=None,
        canonicalization="maxabs",
    ):
        super().__init__()
        self.max_points = max_points
        self.n_eigs = n_eigs
        self.n_neighbors = n_neighbors
        self.weighted = weighted
        self.normalized = normalized
        self.include_xyz = include_xyz
        self.sigma = sigma
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

        # Pre-compute GPS features for every shape
        self.data = []
        n_skipped = 0
        desc = f"GPS {split}"
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
                # Eigensolver failure: use zero GPS features
                n_actual = points.shape[0]
                cd = self.coord_dim
                features = np.zeros((max_points, cd + n_eigs), dtype=np.float32)
                if cd > 0:
                    features[:n_actual, :cd] = points[:n_actual].astype(np.float32)

                mask = np.ones(max_points, dtype=bool)
                mask[:n_actual] = False

                label = self.class_to_idx[class_name]
                self.data.append((features, mask, label))
                n_skipped += 1
                continue

            # Apply canonicalization (GPS is sign-sensitive)
            unified_method = _METHOD_MAP.get(canonicalization, canonicalization)
            canon_evecs = canonicalize(
                eigenvectors, eigenvalues=eigenvalues, method=unified_method, sample_idx=idx
            )
            feat = compute_gps(eigenvalues, canon_evecs)

            # Pad GPS features to n_eigs columns (eigensolver may return fewer)
            actual_k = feat.shape[1]
            if actual_k < n_eigs:
                feat = np.pad(feat, ((0, 0), (0, n_eigs - actual_k)), mode="constant")

            n_pts = points.shape[0]
            cd = self.coord_dim

            # Build feature matrix: all points kept, features placed at
            # LCC indices (non-LCC points get zero features but stay valid)
            features = np.zeros((max_points, cd + n_eigs), dtype=np.float32)
            if cd > 0:
                features[:n_pts, :cd] = points.astype(np.float32)
            features[comp_idx, cd : cd + n_eigs] = feat.astype(np.float32)

            # Mask: True at padded positions (all n_pts points valid)
            mask = np.ones(max_points, dtype=bool)
            mask[:n_pts] = False

            label = self.class_to_idx[class_name]
            self.data.append((features, mask, label))

        if n_skipped > 0:
            print(f"Warning: {n_skipped} shapes had eigensolver failures (using zero GPS features)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features, mask, label = self.data[idx]
        return (
            torch.from_numpy(features).float(),
            torch.from_numpy(mask).bool(),
            label,
        )
