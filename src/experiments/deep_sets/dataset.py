"""Deep Sets ModelNet dataset with eigenvalue-scaled spectral features.

Preserves eigenvalues alongside eigenvectors so the Deep Sets model can
scale eigenvectors by eigenvalue-dependent weights (e.g. 1/sqrt(lambda)).
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.datasets.modelnet import ModelNet10Dataset, ModelNet40Dataset
from src.experiments.spectral_transformer.dataset_spielman import (
    CANONICALIZATION_CHOICES,
    _apply_canonicalization,
)
from src.preprocessing import center_and_normalize, random_subsample
from src.spectral_core import build_graph_laplacian, compute_eigenpairs


class DeepSetsModelNet(Dataset):
    """ModelNet dataset for Deep Sets with eigenvalue-scaled spectral features.

    Returns ``(features, eigenvalues, mask, label)`` where:

    - ``features``: ``(max_points, 3 + n_eigs)`` float tensor -- xyz
      concatenated with raw (canonicalized) eigenvectors.
    - ``eigenvalues``: ``(n_eigs,)`` float tensor -- eigenvalues for
      scaling in the model.
    - ``mask``: ``(max_points,)`` bool tensor -- True = padded, False = valid.
    - ``label``: int class index.

    All spectral computation is done once at ``__init__`` and stored in
    memory.

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
        Number of eigenvectors/eigenvalues to compute.
    n_neighbors : int
        k-NN neighbors for graph construction.
    canonicalization : str
        One of ``"spielman"``, ``"maxabs"``, ``"random"``, ``"none"``.
    weighted : bool
        If True, use Gaussian-kernel weighted graph Laplacian.
    sigma : float or None
        Bandwidth for Gaussian kernel. Auto-computed if None and
        ``weighted=True``.
    """

    def __init__(
        self,
        root,
        dataset="ModelNet10",
        split="train",
        max_points=1024,
        n_eigs=8,
        n_neighbors=12,
        canonicalization="spielman",
        weighted=False,
        sigma=None,
        include_xyz=True,
    ):
        super().__init__()
        self.max_points = max_points
        self.n_eigs = n_eigs
        self.n_neighbors = n_neighbors
        self.canonicalization = canonicalization
        self.weighted = weighted
        self.sigma = sigma
        self.include_xyz = include_xyz
        self.coord_dim = 3 if include_xyz else 0

        if canonicalization not in CANONICALIZATION_CHOICES:
            raise ValueError(
                f"canonicalization must be one of {CANONICALIZATION_CHOICES}, "
                f"got {canonicalization!r}"
            )

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

        # Pre-compute spectral features for every shape
        self.data = []
        n_skipped = 0
        for idx, (name, class_name, points) in enumerate(tqdm(raw_items, desc=f"DeepSets {split}")):
            points = random_subsample(points, max_points, seed=idx)
            points, _, _ = center_and_normalize(points)

            L, comp_idx = build_graph_laplacian(
                points,
                n_neighbors=n_neighbors,
                weighted=weighted,
                sigma=sigma,
            )

            try:
                eigenvalues, eigenvectors = compute_eigenpairs(L, n_eigs=n_eigs)
            except Exception:
                # Eigensolver failure: use zero eigenvectors, eigenvalues = 1.0
                # (eigenvalues of 1.0 avoid division-by-zero in 1/sqrt(lambda))
                n_actual = points.shape[0]
                cd = self.coord_dim
                features = np.zeros((max_points, cd + n_eigs), dtype=np.float32)
                if cd > 0:
                    features[:n_actual, :cd] = points[:n_actual].astype(np.float32)

                eigenvalues_out = np.ones(n_eigs, dtype=np.float32)

                mask = np.ones(max_points, dtype=bool)
                mask[:n_actual] = False

                label = self.class_to_idx[class_name]
                self.data.append((features, eigenvalues_out, mask, label))
                n_skipped += 1
                continue

            # Apply chosen canonicalization
            eigenvectors = _apply_canonicalization(
                eigenvectors, eigenvalues, canonicalization, shape_idx=idx
            )

            n_pts = points.shape[0]
            n_eigs_actual = eigenvectors.shape[1]

            # Build feature matrix: ALL points kept, eigvecs placed at
            # LCC indices (non-LCC points get zero eigvecs but stay valid)
            cd = self.coord_dim
            features = np.zeros((max_points, cd + n_eigs), dtype=np.float32)
            if cd > 0:
                features[:n_pts, :cd] = points.astype(np.float32)
            features[comp_idx, cd : cd + n_eigs_actual] = eigenvectors.astype(np.float32)

            # Eigenvalues: pad with 1.0 if fewer than n_eigs were computed
            eigenvalues_out = np.ones(n_eigs, dtype=np.float32)
            eigenvalues_out[:n_eigs_actual] = eigenvalues[:n_eigs_actual].astype(np.float32)

            # Mask: True at padded positions (all n_pts points are valid)
            mask = np.ones(max_points, dtype=bool)
            mask[:n_pts] = False

            label = self.class_to_idx[class_name]
            self.data.append((features, eigenvalues_out, mask, label))

        if n_skipped > 0:
            print(
                f"Warning: {n_skipped} shapes had eigensolver failures "
                f"(using zero eigvecs, eigenvalues=1.0)"
            )

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
