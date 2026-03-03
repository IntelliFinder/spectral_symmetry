"""ModelNet40 dataset with normals for Pointcept-style PTv3 training.

Supports two data formats:
1. Pre-processed .dat files (pickle, 1024 pts FPS) from PointNet++ repo
2. Raw .txt files from modelnet40_normal_resampled (10k pts with normals)

Both provide 6-channel features: xyz + normals.
"""

import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# ModelNet40 class names in alphabetical order (standard ordering)
MODELNET40_CLASSES = [
    "airplane",
    "bathtub",
    "bed",
    "bench",
    "bookshelf",
    "bottle",
    "bowl",
    "car",
    "chair",
    "cone",
    "cup",
    "curtain",
    "desk",
    "door",
    "dresser",
    "flower_pot",
    "glass_box",
    "guitar",
    "keyboard",
    "lamp",
    "laptop",
    "mantel",
    "monitor",
    "night_stand",
    "person",
    "piano",
    "plant",
    "radio",
    "range_hood",
    "sink",
    "sofa",
    "stairs",
    "stool",
    "table",
    "tent",
    "toilet",
    "tv_stand",
    "vase",
    "wardrobe",
    "xbox",
]


class ModelNet40WithNormals(Dataset):
    """ModelNet40 with surface normals for PTv3 classification.

    Loads from pre-processed .dat files (PointNet++ format):
      pickle list [point_clouds_list, labels_list]
      each point cloud is (1024, 6) float32: xyz + normals.

    Parameters
    ----------
    data_dir : str
        Path to directory containing the .dat files.
    split : str
        "train" or "test".
    transform : callable, optional
        Augmentation pipeline.
    """

    def __init__(self, data_dir, split="train", transform=None):
        super().__init__()
        self.transform = transform
        self.classes = MODELNET40_CLASSES
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        data_dir = Path(data_dir)
        dat_file = data_dir / f"modelnet40_{split}_1024pts_fps.dat"
        assert dat_file.exists(), f"Data file not found: {dat_file}"

        with open(dat_file, "rb") as f:
            data = pickle.load(f, encoding="latin1")

        self.points_list = data[0]  # list of (1024, 6) arrays
        raw_labels = data[1]  # list of (1,) arrays or ints
        self.labels = [int(np.array(lbl).flatten()[0]) for lbl in raw_labels]

        print(
            f"ModelNet40WithNormals [{split}]: {len(self.points_list)} shapes, "
            f"{len(self.classes)} classes"
        )

    def __len__(self):
        return len(self.points_list)

    def _get_raw(self, idx):
        """Get raw data dict for a sample (numpy arrays)."""
        pts = self.points_list[idx].copy()  # (1024, 6)
        return {
            "coord": pts[:, :3].astype(np.float32),
            "normal": pts[:, 3:6].astype(np.float32),
            "label": self.labels[idx],
        }

    def __getitem__(self, idx):
        data = self._get_raw(idx)

        # Apply transforms
        if self.transform is not None:
            data = self.transform(data)

        # feat = xyz + normals (6 channels) — built AFTER augmentation
        data["feat"] = np.concatenate([data["coord"], data["normal"]], axis=1)

        # Convert to tensors
        result = {
            "coord": torch.from_numpy(data["coord"]).float(),
            "feat": torch.from_numpy(data["feat"]).float(),
            "label": data["label"],
        }
        if "grid_coord" in data:
            result["grid_coord"] = torch.from_numpy(data["grid_coord"]).int()

        return result


class ModelNet40NormalResampled(Dataset):
    """ModelNet40 from raw .txt files (modelnet40_normal_resampled).

    Fallback if .dat files unavailable. Each shape is a .txt with N rows
    of 6 comma-separated floats (x, y, z, nx, ny, nz).
    """

    NUM_CLASSES = 40

    def __init__(self, root, split="train", num_points=8192, transform=None):
        super().__init__()
        self.root = Path(root)
        self.num_points = num_points
        self.transform = transform
        self.rng = np.random.RandomState(42)

        shape_ids_file = self.root / f"modelnet40_{split}.txt"
        assert shape_ids_file.exists(), f"Split file not found: {shape_ids_file}"

        with open(shape_ids_file, "r") as f:
            shape_ids = [line.strip() for line in f if line.strip()]

        self.classes = MODELNET40_CLASSES
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples = []
        for shape_id in shape_ids:
            class_name = None
            for c in self.classes:
                if shape_id.startswith(c + "_"):
                    class_name = c
                    break
            if class_name is None:
                continue
            txt_path = self.root / class_name / f"{shape_id}.txt"
            if txt_path.exists():
                self.samples.append((txt_path, self.class_to_idx[class_name]))

        print(f"ModelNet40NormalResampled [{split}]: {len(self.samples)} shapes")

    def __len__(self):
        return len(self.samples)

    def _get_raw(self, idx):
        txt_path, label = self.samples[idx]
        data = np.loadtxt(txt_path, delimiter=",", dtype=np.float32)
        coord, normal = data[:, :3], data[:, 3:6]

        n = coord.shape[0]
        if n >= self.num_points:
            choice = self.rng.choice(n, self.num_points, replace=False)
        else:
            choice = self.rng.choice(n, self.num_points, replace=True)

        return {
            "coord": coord[choice],
            "normal": normal[choice],
            "label": label,
        }

    def __getitem__(self, idx):
        data = self._get_raw(idx)

        if self.transform is not None:
            data = self.transform(data)

        # feat = xyz + normals (6 channels) — built AFTER augmentation
        data["feat"] = np.concatenate([data["coord"], data["normal"]], axis=1)

        result = {
            "coord": torch.from_numpy(data["coord"]).float(),
            "feat": torch.from_numpy(data["feat"]).float(),
            "label": data["label"],
        }
        if "grid_coord" in data:
            result["grid_coord"] = torch.from_numpy(data["grid_coord"]).int()
        return result


def pointcept_collate_fn(batch):
    """Collate variable-length point clouds into PTv3's batched format.

    Returns
    -------
    data_dict : dict
        "coord", "feat", "offset", and optionally "grid_coord".
    labels : LongTensor of shape (B,)
    """
    coords = []
    feats = []
    labels = []
    grid_coords = []
    offset = []
    cumulative = 0
    has_grid_coord = "grid_coord" in batch[0]

    for sample in batch:
        n = sample["coord"].shape[0]
        coords.append(sample["coord"])
        feats.append(sample["feat"])
        labels.append(sample["label"])
        if has_grid_coord:
            grid_coords.append(sample["grid_coord"])
        cumulative += n
        offset.append(cumulative)

    data_dict = {
        "coord": torch.cat(coords, dim=0).float(),
        "feat": torch.cat(feats, dim=0).float(),
        "offset": torch.tensor(offset, dtype=torch.long),
    }
    if has_grid_coord:
        data_dict["grid_coord"] = torch.cat(grid_coords, dim=0).int()

    labels = torch.tensor(labels, dtype=torch.long)
    return data_dict, labels
