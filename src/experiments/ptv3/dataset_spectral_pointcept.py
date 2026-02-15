"""ModelNet40 dataset with normals + pre-computed spectral eigenvectors.

Extends ModelNet40WithNormals with Laplacian eigenvectors computed at init time,
supporting multiple canonicalization strategies (spielman, maxabs, random, none).
"""

import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.experiments.spectral_transformer.dataset_spielman import (
    CANONICALIZATION_CHOICES,
    _apply_canonicalization,
)
from src.spectral_core import build_graph_laplacian, compute_eigenpairs

from .dataset_pointcept import MODELNET40_CLASSES


class ModelNet40SpectralNormals(Dataset):
    """ModelNet40 with surface normals and pre-computed Laplacian eigenvectors.

    Loads from pre-processed .dat files (PointNet++ format) and computes
    eigenvectors at init time for each shape.

    Parameters
    ----------
    data_dir : str
        Path to directory containing the .dat files.
    split : str
        "train" or "test".
    transform : callable, optional
        Augmentation pipeline.
    n_eigs : int
        Number of non-trivial eigenvectors to compute.
    n_neighbors : int
        k-NN neighbors for graph construction.
    canonicalization : str
        One of "spielman", "maxabs", "random", "none".
    weighted : bool
        If True, use Gaussian kernel weighted graph Laplacian.
    sigma : float or None
        Bandwidth for Gaussian kernel. Auto-computed if None and weighted=True.
    """

    def __init__(
        self,
        data_dir,
        split="train",
        transform=None,
        n_eigs=8,
        n_neighbors=12,
        canonicalization="spielman",
        weighted=False,
        sigma=None,
    ):
        super().__init__()
        self.transform = transform
        self.n_eigs = n_eigs
        self.classes = MODELNET40_CLASSES
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        if canonicalization not in CANONICALIZATION_CHOICES:
            raise ValueError(
                f"canonicalization must be one of {CANONICALIZATION_CHOICES}, "
                f"got {canonicalization!r}"
            )

        data_dir = Path(data_dir)
        dat_file = data_dir / f"modelnet40_{split}_1024pts_fps.dat"
        assert dat_file.exists(), f"Data file not found: {dat_file}"

        with open(dat_file, "rb") as f:
            data = pickle.load(f, encoding="latin1")

        self.points_list = data[0]  # list of (1024, 6) arrays
        raw_labels = data[1]
        self.labels = [int(np.array(lb).flatten()[0]) for lb in raw_labels]

        # Pre-compute eigenvectors for all shapes
        self.eigvecs_list = []
        n_failed = 0
        for idx, pts in enumerate(tqdm(self.points_list, desc=f"SpectralNormals {split} eigvecs")):
            xyz = pts[:, :3].astype(np.float64)
            eigvec = np.zeros((pts.shape[0], n_eigs), dtype=np.float32)
            try:
                L, comp_idx = build_graph_laplacian(
                    xyz, n_neighbors=n_neighbors, weighted=weighted, sigma=sigma
                )
                eigenvalues, eigenvectors = compute_eigenpairs(L, n_eigs=n_eigs)
                eigenvectors = _apply_canonicalization(
                    eigenvectors, eigenvalues, canonicalization, shape_idx=idx
                )
                n_actual = eigenvectors.shape[0]
                n_eigs_actual = eigenvectors.shape[1]
                # Place eigvecs at LCC indices, zeros elsewhere
                eigvec[comp_idx[:n_actual], :n_eigs_actual] = eigenvectors.astype(np.float32)
            except Exception:
                n_failed += 1
            self.eigvecs_list.append(eigvec)

        if n_failed > 0:
            print(f"Warning: {n_failed} shapes failed eigensolver, using zero eigvecs")

        print(
            f"ModelNet40SpectralNormals [{split}]: {len(self.points_list)} shapes, "
            f"{len(self.classes)} classes, {n_eigs} eigvecs"
        )

    def __len__(self):
        return len(self.points_list)

    def _get_raw(self, idx):
        """Get raw data dict for a sample (numpy arrays)."""
        pts = self.points_list[idx].copy()  # (1024, 6)
        return {
            "coord": pts[:, :3].astype(np.float32),
            "normal": pts[:, 3:6].astype(np.float32),
            "eigvec": self.eigvecs_list[idx].copy(),
            "label": self.labels[idx],
        }

    def __getitem__(self, idx):
        data = self._get_raw(idx)

        # feat = xyz + normals (6 channels)
        data["feat"] = np.concatenate([data["coord"], data["normal"]], axis=1)

        # Apply transforms (GridSample with eigvec key will subsample eigvecs too)
        if self.transform is not None:
            data = self.transform(data)

        # Convert to tensors
        result = {
            "coord": torch.from_numpy(data["coord"]).float(),
            "feat": torch.from_numpy(data["feat"]).float(),
            "eigvec": torch.from_numpy(data["eigvec"]).float(),
            "label": data["label"],
        }
        if "grid_coord" in data:
            result["grid_coord"] = torch.from_numpy(data["grid_coord"]).int()

        return result


def spectral_pointcept_collate_fn(batch):
    """Collate variable-length point clouds with eigvecs into PTv3's batched format.

    Returns
    -------
    data_dict : dict
        "coord", "feat", "eigvec", "offset", and optionally "grid_coord".
    labels : LongTensor of shape (B,)
    """
    coords = []
    feats = []
    eigvecs = []
    labels = []
    grid_coords = []
    offset = []
    cumulative = 0
    has_grid_coord = "grid_coord" in batch[0]

    for sample in batch:
        n = sample["coord"].shape[0]
        coords.append(sample["coord"])
        feats.append(sample["feat"])
        eigvecs.append(sample["eigvec"])
        labels.append(sample["label"])
        if has_grid_coord:
            grid_coords.append(sample["grid_coord"])
        cumulative += n
        offset.append(cumulative)

    data_dict = {
        "coord": torch.cat(coords, dim=0).float(),
        "feat": torch.cat(feats, dim=0).float(),
        "eigvec": torch.cat(eigvecs, dim=0).float(),
        "offset": torch.tensor(offset, dtype=torch.long),
    }
    if has_grid_coord:
        data_dict["grid_coord"] = torch.cat(grid_coords, dim=0).int()

    labels = torch.tensor(labels, dtype=torch.long)
    return data_dict, labels
