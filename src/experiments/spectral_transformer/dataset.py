"""Spectral ModelNet10 dataset: point clouds with Laplacian eigenvector features."""

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.datasets.modelnet import ModelNet10Dataset
from src.preprocessing import center_and_normalize, random_subsample
from src.spectral_core import build_graph_laplacian, compute_eigenpairs


class SpectralModelNet10(Dataset):
    """ModelNet10 shapes with pre-computed spectral positional encodings.

    Each item is a tuple ``(features, mask, label)`` where:
    - ``features`` is a ``(n_points, 3 + n_eigs)`` float tensor (xyz || eigvecs),
      zero-padded to ``n_points`` along the sequence dimension.
    - ``mask`` is a ``(n_points,)`` bool tensor, True at padded positions.
    - ``label`` is an int class index.

    All spectral computation is done once at init and stored in memory.
    """

    def __init__(
        self,
        root_dir,
        split="train",
        n_points=512,
        n_eigs=16,
        n_neighbors=12,
        download=False,
        canonicalize=False,
    ):
        super().__init__()
        self.n_points = n_points
        self.n_eigs = n_eigs
        self.n_neighbors = n_neighbors
        self.canonicalize = canonicalize

        base_dataset = ModelNet10Dataset(
            root_dir,
            split=split,
            max_points=n_points * 2,
            download=download,
        )

        # Collect all class names first to build a sorted label mapping
        raw_items = []
        class_names = set()
        for name, points in base_dataset:
            # name format: "modelnet10/{class_name}/{file}.off"
            parts = name.split("/")
            class_name = parts[1]
            class_names.add(class_name)
            raw_items.append((name, class_name, points))

        self.classes = sorted(class_names)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # Pre-compute spectral features for every shape
        self.data = []
        n_skipped = 0
        for idx, (name, class_name, points) in enumerate(tqdm(raw_items, desc=f"Spectral {split}")):
            points = random_subsample(points, n_points, seed=idx)
            points, _, _ = center_and_normalize(points)

            L, comp_idx = build_graph_laplacian(points, n_neighbors=n_neighbors)
            try:
                eigenvalues, eigenvectors = compute_eigenpairs(L, n_eigs=n_eigs)
            except Exception:
                n_skipped += 1
                continue

            # Sign canonicalization: flip each eigenvector so its entry sum is non-negative
            if canonicalize:
                signs = np.sign(eigenvectors.sum(axis=0))
                signs[signs == 0] = 1.0
                eigenvectors = eigenvectors * signs

            # Keep only the largest connected component
            pts_cc = points[comp_idx]  # (N', 3)
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
