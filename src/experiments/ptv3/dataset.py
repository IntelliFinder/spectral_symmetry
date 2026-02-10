"""PTv3-compatible ModelNet datasets and collate functions."""

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.datasets.modelnet import ModelNet10Dataset, ModelNet40Dataset
from src.preprocessing import center_and_normalize, random_subsample
from src.spectral_core import build_graph_laplacian, compute_eigenpairs
from src.experiments.spectral_transformer.dataset import canonicalize_eigenvectors


class PTv3ModelNet(Dataset):
    """ModelNet point clouds for PTv3 (xyz features only).

    Each item is a dict with:
    - "coord": (n, 3) float32 tensor
    - "feat": (n, 3) float32 tensor (xyz as features)
    - "label": int

    No padding — variable-length samples are batched via ``ptv3_collate_fn``.
    """

    def __init__(self, root_dir, split="train", n_points=512, download=False, variant=10):
        super().__init__()
        self.n_points = n_points

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
        for idx, (name, class_name, points) in enumerate(
            tqdm(raw_items, desc=f"PTv3 {split}")
        ):
            points = random_subsample(points, n_points, seed=idx)
            points, _, _ = center_and_normalize(points)
            label = self.class_to_idx[class_name]
            self.data.append({
                "coord": points.astype(np.float32),
                "feat": points.astype(np.float32),
                "label": label,
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        return {
            "coord": torch.from_numpy(d["coord"]),
            "feat": torch.from_numpy(d["feat"]),
            "label": d["label"],
        }


class PTv3SpectralModelNet(Dataset):
    """ModelNet point clouds for PTv3 with spectral features (xyz + eigvecs).

    Each item is a dict with:
    - "coord": (n, 3) float32 tensor (xyz coordinates)
    - "feat": (n, 3+n_eigs) float32 tensor (xyz || canonicalized eigenvectors)
    - "label": int

    Only points in the largest connected component are kept.
    No padding — variable-length samples are batched via ``ptv3_collate_fn``.
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
        canonicalize=True,
    ):
        super().__init__()
        self.n_points = n_points
        self.n_eigs = n_eigs
        self.n_neighbors = n_neighbors

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
            tqdm(raw_items, desc=f"PTv3Spectral {split}")
        ):
            points = random_subsample(points, n_points, seed=idx)
            points, _, _ = center_and_normalize(points)

            L, comp_idx = build_graph_laplacian(points, n_neighbors=n_neighbors)
            try:
                eigenvalues, eigenvectors = compute_eigenpairs(L, n_eigs=n_eigs)
            except Exception:
                n_skipped += 1
                continue

            if canonicalize:
                eigenvectors = canonicalize_eigenvectors(eigenvectors)

            pts_cc = points[comp_idx]
            n_actual = pts_cc.shape[0]
            n_eigs_actual = eigenvectors.shape[1]

            # Feature: xyz || eigvecs
            feat = np.zeros((n_actual, 3 + n_eigs), dtype=np.float32)
            feat[:, :3] = pts_cc.astype(np.float32)
            feat[:, 3 : 3 + n_eigs_actual] = eigenvectors.astype(np.float32)

            label = self.class_to_idx[class_name]
            self.data.append({
                "coord": pts_cc.astype(np.float32),
                "feat": feat,
                "label": label,
            })

        if n_skipped > 0:
            print(f"Warning: skipped {n_skipped} shapes due to eigensolver failures")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        return {
            "coord": torch.from_numpy(d["coord"]),
            "feat": torch.from_numpy(d["feat"]),
            "label": d["label"],
        }


def ptv3_collate_fn(batch):
    """Collate variable-length point clouds into PTv3's batched format.

    Parameters
    ----------
    batch : list of dicts
        Each with "coord" (n_i, 3), "feat" (n_i, C), "label" int.

    Returns
    -------
    data_dict : dict
        "coord": (N_total, 3), "feat": (N_total, C), "offset": (B,) cumulative.
    labels : LongTensor of shape (B,)
    """
    coords = []
    feats = []
    labels = []
    offset = []
    cumulative = 0

    for sample in batch:
        n = sample["coord"].shape[0]
        coords.append(sample["coord"])
        feats.append(sample["feat"])
        labels.append(sample["label"])
        cumulative += n
        offset.append(cumulative)

    data_dict = {
        "coord": torch.cat(coords, dim=0).float(),
        "feat": torch.cat(feats, dim=0).float(),
        "offset": torch.tensor(offset, dtype=torch.long),
    }
    labels = torch.tensor(labels, dtype=torch.long)

    return data_dict, labels
