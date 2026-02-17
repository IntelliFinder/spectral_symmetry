"""PCA-canonicalized ModelNet dataset for Deep Sets.

Applies PCA-based orientation canonicalization to point clouds so that
shapes are aligned to their principal axes before classification.
No spectral features -- just PCA-aligned xyz coordinates.

Returns ``(features, mask, label)`` -- a 3-tuple compatible with
``HKSDeepSetsClassifier`` (with ``n_times=0``).
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.datasets.modelnet import ModelNet10Dataset, ModelNet40Dataset
from src.preprocessing import center_and_normalize, pca_canonicalize, random_subsample


class PCAModelNet(Dataset):
    """ModelNet dataset with PCA-canonicalized xyz features.

    Returns ``(features, mask, label)`` where:

    - ``features``: ``(max_points, 3)`` float tensor -- PCA-aligned xyz.
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
    sign_method : str
        PCA sign canonicalization method: "majority", "maxabs", "random",
        or "spielman".
    """

    def __init__(
        self,
        root,
        dataset="ModelNet10",
        split="train",
        max_points=1024,
        sign_method="majority",
    ):
        super().__init__()
        self.max_points = max_points
        self.sign_method = sign_method

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

        # Pre-compute PCA-canonicalized features for every shape
        self.data = []
        desc = f"PCA {split}"
        for idx, (name, class_name, points) in enumerate(tqdm(raw_items, desc=desc)):
            points = random_subsample(points, max_points, seed=idx)
            points, _, _ = center_and_normalize(points)

            # PCA canonicalization
            points = pca_canonicalize(points, sign_method=sign_method, shape_idx=idx)

            n_pts = points.shape[0]

            # Build feature matrix: just xyz coordinates
            features = np.zeros((max_points, 3), dtype=np.float32)
            features[:n_pts, :] = points.astype(np.float32)

            # Mask: True at padded positions
            mask = np.ones(max_points, dtype=bool)
            mask[:n_pts] = False

            label = self.class_to_idx[class_name]
            self.data.append((features, mask, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features, mask, label = self.data[idx]
        return (
            torch.from_numpy(features).float(),
            torch.from_numpy(mask).bool(),
            label,
        )
