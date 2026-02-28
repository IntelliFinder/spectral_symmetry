"""Learnable Spectral Weighting ModelNet dataset for Deep Sets.

Returns raw eigenvectors and eigenvalues so the model can learn its own
weighting function (replacing the fixed exponential in HKS/WES).

Returns a 4-tuple ``(features, eigenvalues, mask, label)`` where:

- ``features``: ``(max_points, 3 + K)`` -- xyz (optional) || raw eigvec entries
- ``eigenvalues``: ``(K,)`` -- eigenvalue spectrum (zero-padded if needed)
- ``mask``: ``(max_points,)`` -- True = padded/invalid
- ``label``: int class index

The model is responsible for applying learned weights to eigenvectors.
When ``use_squared=True`` the stored eigvecs are squared (sign-invariant);
when ``use_squared=False`` they are canonicalized and stored raw.
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


class LearnableSpectralModelNet(Dataset):
    """ModelNet dataset returning raw eigenvectors + eigenvalues.

    The eigenvalues are returned alongside features so the model's spectral
    weighting MLP can condition on the full spectrum.

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
        Number of eigenpairs.
    n_neighbors : int
        k-NN neighbors for graph construction.
    weighted : bool
        If True, use Gaussian-kernel weighted graph Laplacian.
    normalized : bool
        If True, use symmetric normalized Laplacian.
    include_xyz : bool
        If True, prepend xyz coordinates to eigenvector features.
    sigma : float or None
        Bandwidth for Gaussian kernel. Auto-computed if None.
    use_squared : bool
        If True, store v_k(i)^2 (sign-invariant).
        If False, store v_k(i) after canonicalization.
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
        n_eigs=8,
        n_neighbors=30,
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
        self.weighted = weighted
        self.normalized = normalized
        self.include_xyz = include_xyz
        self.sigma = sigma
        self.use_squared = use_squared
        self.canonicalization = canonicalization
        self.coord_dim = 3 if include_xyz else 0

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
        desc = f"LearnableSpectral {split}"

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
                # Eigensolver failure: zero features
                n_actual = points.shape[0]
                cd = self.coord_dim
                features = np.zeros((max_points, cd + n_eigs), dtype=np.float32)
                if cd > 0:
                    features[:n_actual, :cd] = points[:n_actual].astype(np.float32)
                evals_out = np.zeros(n_eigs, dtype=np.float32)
                mask = np.ones(max_points, dtype=bool)
                mask[:n_actual] = False
                label = self.class_to_idx[class_name]
                self.data.append((features, evals_out, mask, label))
                n_skipped += 1
                continue

            # Pad eigenvalues to exactly n_eigs (may be fewer if graph is small)
            k_actual = len(eigenvalues)
            evals_out = np.zeros(n_eigs, dtype=np.float32)
            evals_out[:k_actual] = eigenvalues.astype(np.float32)

            # Eigenvector matrix: (N_LCC, k_actual)
            if use_squared:
                evecs_feat = eigenvectors**2  # sign-invariant
            else:
                unified_method = _METHOD_MAP.get(canonicalization, canonicalization)
                evecs_feat = canonicalize(
                    eigenvectors, eigenvalues=eigenvalues, method=unified_method, sample_idx=idx
                )

            n_pts = points.shape[0]
            cd = self.coord_dim
            feat_dim = cd + n_eigs

            # Build feature matrix (max_points, feat_dim)
            features = np.zeros((max_points, feat_dim), dtype=np.float32)
            if cd > 0:
                features[:n_pts, :cd] = points.astype(np.float32)
            # Place eigvec features at LCC point indices, first k_actual columns
            features[comp_idx, cd : cd + k_actual] = evecs_feat.astype(np.float32)

            mask = np.ones(max_points, dtype=bool)
            mask[:n_pts] = False

            label = self.class_to_idx[class_name]
            self.data.append((features, evals_out, mask, label))

        if n_skipped > 0:
            print(f"Warning: {n_skipped} shapes had eigensolver failures (zero eigvec features)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features, eigenvalues, mask, label = self.data[idx]
        return (
            torch.from_numpy(features).float(),
            torch.from_numpy(eigenvalues).float(),
            torch.from_numpy(mask).bool(),
            label,
        )
