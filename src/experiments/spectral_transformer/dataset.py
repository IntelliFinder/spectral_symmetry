"""Spectral ModelNet dataset: point clouds with Laplacian eigenvector features."""

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset
from tqdm import tqdm

from src.datasets.modelnet import ModelNet10Dataset, ModelNet40Dataset
from src.preprocessing import center_and_normalize, random_subsample
from src.spectral_core import (
    build_graph_laplacian,
    compute_eigenpairs,
    detect_eigenvalue_multiplicities,
)


class SpectralModelNet(Dataset):
    """ModelNet shapes with pre-computed spectral positional encodings.

    Each item is a tuple ``(features, mask, label)`` where:
    - ``features`` is a ``(n_points, 3 + n_eigs)`` float tensor (xyz || eigvecs),
      zero-padded to ``n_points`` along the sequence dimension.
    - ``mask`` is a ``(n_points,)`` bool tensor, True at padded positions.
    - ``label`` is an int class index.

    All spectral computation is done once at init and stored in memory.

    Parameters
    ----------
    root_dir : str
        Root data directory.
    variant : int
        10 or 40 (default 10).
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
    canonicalize : bool
        If True, canonicalize eigenvector signs.
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
        variant=10,
    ):
        super().__init__()
        self.n_points = n_points
        self.n_eigs = n_eigs
        self.n_neighbors = n_neighbors
        self.canonicalize = canonicalize
        self.variant = variant

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
            # name format: "modelnet{10,40}/{class_name}/{file}.off"
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


class TruncatedSpectralDataset(Dataset):
    """Wrapper that returns only the first k eigenvector columns from a SpectralModelNet.

    Parameters
    ----------
    base_dataset : SpectralModelNet
        The wrapped dataset (must have n_eigs >= k).
    k : int
        Number of eigenvectors to keep.
    use_xyz : bool
        If True, prepend xyz coordinates (columns 0-2). If False, return only
        eigenvector columns, giving shape ``(n_points, k)``.
    """

    def __init__(self, base_dataset, k, use_xyz=False):
        super().__init__()
        self.base_dataset = base_dataset
        self.k = k
        self.use_xyz = use_xyz

    @property
    def classes(self):
        return self.base_dataset.classes

    @property
    def class_to_idx(self):
        return self.base_dataset.class_to_idx

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        features, mask, label = self.base_dataset[idx]
        if self.use_xyz:
            # xyz (cols 0-2) + first k eigenvectors (cols 3 to 3+k)
            features = torch.cat([features[:, :3], features[:, 3:3 + self.k]], dim=1)
        else:
            # only eigenvector columns
            features = features[:, 3:3 + self.k]
        return features, mask, label


class SpectralModelNet10(SpectralModelNet):
    """Backward-compatible alias for SpectralModelNet with variant=10."""

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
        super().__init__(
            root_dir,
            split=split,
            n_points=n_points,
            n_eigs=n_eigs,
            n_neighbors=n_neighbors,
            download=download,
            canonicalize=canonicalize,
            variant=10,
        )


class SpectralModelNet40(SpectralModelNet):
    """Backward-compatible alias for SpectralModelNet with variant=40."""

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
        super().__init__(
            root_dir,
            split=split,
            n_points=n_points,
            n_eigs=n_eigs,
            n_neighbors=n_neighbors,
            download=download,
            canonicalize=canonicalize,
            variant=40,
        )


class PointCloudModelNet(Dataset):
    """ModelNet shapes with xyz features and k-NN distance matrices.

    No spectral computation. Returns a 4-tuple per item:
    ``(features, dist_matrix, mask, label)`` where:
    - ``features`` is a ``(n_points, 3)`` float tensor (xyz coordinates).
    - ``dist_matrix`` is a ``(n_points, n_points)`` float tensor of k-NN distances,
      symmetrized and sparse (only k entries per row non-zero).
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
    n_neighbors : int
        k-NN neighbors for distance matrix.
    download : bool
        If True, download the dataset if missing.
    variant : int
        10 or 40.
    """

    def __init__(
        self,
        root_dir,
        split="train",
        n_points=512,
        n_neighbors=12,
        download=False,
        variant=10,
    ):
        super().__init__()
        self.n_points = n_points
        self.n_neighbors = n_neighbors
        self.variant = variant

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
        for idx, (name, class_name, points) in enumerate(
            tqdm(raw_items, desc=f"PointCloud {split}")
        ):
            points = random_subsample(points, n_points, seed=idx)
            points, _, _ = center_and_normalize(points)

            n_actual = points.shape[0]

            # Build features: xyz only, padded to n_points
            features = np.zeros((n_points, 3), dtype=np.float32)
            features[:n_actual, :3] = points.astype(np.float32)

            # Build k-NN distance matrix
            k = min(n_neighbors, n_actual - 1)
            nn = NearestNeighbors(n_neighbors=k + 1)
            nn.fit(points)
            distances, indices = nn.kneighbors(points)

            dist_matrix = np.zeros((n_points, n_points), dtype=np.float32)
            for i in range(n_actual):
                for j_idx in range(1, k + 1):  # skip self
                    j = indices[i, j_idx]
                    dist_matrix[i, j] = distances[i, j_idx]

            # Symmetrize
            dist_matrix = 0.5 * (dist_matrix + dist_matrix.T)

            # Mask: True at padded positions
            mask = np.ones(n_points, dtype=bool)
            mask[:n_actual] = False

            label = self.class_to_idx[class_name]
            self.data.append((features, dist_matrix, mask, label))

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


class SpectralNodeModelNet(Dataset):
    """ModelNet shapes with canonicalized eigenvector node features and k-NN distance matrices.

    Combines spectral node features (xyz + eigenvectors) with k-NN distance
    matrices for use with distance-modulated attention. Eigenvector signs are
    canonicalized so that the sum of entries is non-negative.

    Returns a 4-tuple per item:
    ``(features, dist_matrix, mask, label)`` where:

    - ``features``: ``(n_points, 3 + n_eigs)`` float tensor (xyz || eigvecs).
    - ``dist_matrix``: ``(n_points, n_points)`` float tensor of k-NN distances.
    - ``mask``: ``(n_points,)`` bool tensor, True at padded positions.
    - ``label``: int class index.
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
    ):
        super().__init__()
        self.n_points = n_points
        self.n_eigs = n_eigs
        self.n_neighbors = n_neighbors
        self.variant = variant

        if variant == 10:
            base_dataset = ModelNet10Dataset(
                root_dir, split=split, max_points=n_points * 2, download=download,
            )
        elif variant == 40:
            base_dataset = ModelNet40Dataset(
                root_dir, split=split, max_points=n_points * 2, download=download,
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
            tqdm(raw_items, desc=f"SpectralNode {split}")
        ):
            points = random_subsample(points, n_points, seed=idx)
            points, _, _ = center_and_normalize(points)

            L, comp_idx = build_graph_laplacian(points, n_neighbors=n_neighbors)
            try:
                eigenvalues, eigenvectors = compute_eigenpairs(L, n_eigs=n_eigs)
            except Exception:
                n_skipped += 1
                continue

            # Canonicalize signs: flip so entry sum is non-negative
            signs = np.sign(eigenvectors.sum(axis=0))
            signs[signs == 0] = 1.0
            eigenvectors = eigenvectors * signs

            pts_cc = points[comp_idx]
            n_actual = pts_cc.shape[0]
            n_eigs_actual = eigenvectors.shape[1]

            # Node features: xyz || eigvecs, padded
            feat_dim = 3 + n_eigs
            features = np.zeros((n_points, feat_dim), dtype=np.float32)
            features[:n_actual, :3] = pts_cc.astype(np.float32)
            features[:n_actual, 3:3 + n_eigs_actual] = eigenvectors.astype(np.float32)

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


class SpectralDistanceModelNet(Dataset):
    """ModelNet shapes with xyz features, k-NN distances, and spectral distance channels.

    Spectral distances are invariant to eigenvector sign flips (multiplicity 1)
    and eigenspace rotations (multiplicity > 1). Each eigenvalue group produces
    one channel via the projection matrix:

    - Multiplicity 1: ``S_k[i,j] = v_k[i] * v_k[j] * exp(-lambda_k)``
    - Multiplicity m: ``S_g[i,j] = (sum_{k in g} v_k[i] * v_k[j]) * exp(-avg_lambda_g)``

    Returns a 5-tuple per item:
    ``(features, dist_matrix, spectral_dists, mask, label)`` where:

    - ``features``: ``(n_points, 3)`` float tensor (xyz).
    - ``dist_matrix``: ``(n_points, n_points)`` float tensor of k-NN distances.
    - ``spectral_dists``: ``(n_points, n_points, n_eigs)`` float tensor of
      spectral distance channels (one per eigenvalue group, zero-padded).
    - ``mask``: ``(n_points,)`` bool tensor, True at padded positions.
    - ``label``: int class index.

    Eigenvectors and eigenvalues are stored compactly; spectral distance
    matrices are computed on-the-fly in ``__getitem__`` to save memory.
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
    ):
        super().__init__()
        self.n_points = n_points
        self.n_eigs = n_eigs
        self.n_neighbors = n_neighbors
        self.variant = variant

        if variant == 10:
            base_dataset = ModelNet10Dataset(
                root_dir, split=split, max_points=n_points * 2, download=download,
            )
        elif variant == 40:
            base_dataset = ModelNet40Dataset(
                root_dir, split=split, max_points=n_points * 2, download=download,
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
            tqdm(raw_items, desc=f"SpectralDist {split}")
        ):
            points = random_subsample(points, n_points, seed=idx)
            points, _, _ = center_and_normalize(points)

            L, comp_idx = build_graph_laplacian(points, n_neighbors=n_neighbors)
            try:
                eigenvalues, eigenvectors = compute_eigenpairs(L, n_eigs=n_eigs)
            except Exception:
                n_skipped += 1
                continue

            pts_cc = points[comp_idx]
            n_actual = pts_cc.shape[0]
            n_eigs_actual = eigenvectors.shape[1]

            # xyz features, padded
            features = np.zeros((n_points, 3), dtype=np.float32)
            features[:n_actual, :3] = pts_cc.astype(np.float32)

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

            # Pre-compute group info for eigenvalue multiplicities
            mult_info = detect_eigenvalue_multiplicities(eigenvalues[:n_eigs_actual])
            group_indices = mult_info["group_indices"]

            # Mask
            mask = np.ones(n_points, dtype=bool)
            mask[:n_actual] = False

            label = self.class_to_idx[class_name]
            self.data.append({
                "features": features,
                "dist_matrix": dist_matrix,
                "eigenvectors": eigenvectors.astype(np.float32),
                "eigenvalues": eigenvalues[:n_eigs_actual].astype(np.float32),
                "group_indices": group_indices,
                "n_actual": n_actual,
                "mask": mask,
                "label": label,
            })

        if n_skipped > 0:
            print(f"Warning: skipped {n_skipped} shapes due to eigensolver failures")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        eigvecs = d["eigenvectors"]
        eigvals = d["eigenvalues"]
        group_indices = d["group_indices"]
        n_actual = d["n_actual"]
        n_eigs_actual = eigvecs.shape[1]

        # Compute spectral distance channels on-the-fly
        spectral_dists = np.zeros(
            (self.n_points, self.n_points, self.n_eigs), dtype=np.float32
        )
        unique_groups = sorted(set(group_indices[:n_eigs_actual]))
        for ch_idx, g in enumerate(unique_groups):
            members = [k for k in range(n_eigs_actual) if group_indices[k] == g]
            V_g = eigvecs[:, members]  # (n_actual, m)
            avg_lambda = np.mean(eigvals[members])
            scale = float(np.exp(-avg_lambda))
            proj = (V_g @ V_g.T) * scale  # (n_actual, n_actual)
            spectral_dists[:n_actual, :n_actual, ch_idx] = proj

        return (
            torch.from_numpy(d["features"]),
            torch.from_numpy(d["dist_matrix"]),
            torch.from_numpy(spectral_dists),
            torch.from_numpy(d["mask"]),
            d["label"],
        )
