"""ModelNet10/40 dataset loaders with optional auto-download."""

import os
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from .base import PointCloudDataset, _parse_off

_URLS = {
    10: "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip",
    40: "http://modelnet.cs.princeton.edu/ModelNet40.zip",
}


class ModelNetDataset(PointCloudDataset):
    """ModelNet dataset (OFF files).

    Parameters
    ----------
    root_dir : str or Path
        Root data directory. Data will be at ``{root_dir}/raw/ModelNet{variant}/``.
    variant : int
        10 or 40.
    split : str
        ``'train'`` or ``'test'``.
    max_points : int
        Subsample to this many points if the mesh has more.
    download : bool
        If True, download and extract the dataset when it is missing.
    """

    def __init__(self, root_dir, variant, split="train", max_points=2048, download=False):
        if variant not in (10, 40):
            raise ValueError(f"variant must be 10 or 40, got {variant}")
        self.root_dir = Path(root_dir)
        self.variant = variant
        self.split = split
        self.max_points = max_points
        self.data_dir = self.root_dir / "raw" / f"ModelNet{variant}"

        if download and not self.data_dir.is_dir():
            self._download_and_extract()

        if not self.data_dir.is_dir():
            raise FileNotFoundError(
                f"ModelNet{variant} not found at {self.data_dir}. "
                f"Pass download=True or run: python scripts/download_data.py "
                f"--datasets modelnet{variant} --data-dir {self.root_dir}"
            )

    def _download_and_extract(self):
        """Download the zip archive and extract it."""
        url = _URLS[self.variant]
        self.root_dir.joinpath("raw").mkdir(parents=True, exist_ok=True)
        zip_path = self.root_dir / "raw" / f"ModelNet{self.variant}.zip"
        print(f"Downloading ModelNet{self.variant} from {url} ...")
        urllib.request.urlretrieve(url, zip_path)
        print(f"Extracting to {self.root_dir / 'raw'} ...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(self.root_dir / "raw")
        os.remove(zip_path)
        print("Done.")

    def __iter__(self):
        class_dirs = sorted(
            p for p in self.data_dir.iterdir() if p.is_dir()
        )
        for class_dir in class_dirs:
            split_dir = class_dir / self.split
            if not split_dir.is_dir():
                continue
            for off_file in sorted(split_dir.glob("*.off")):
                try:
                    points = _parse_off(str(off_file))
                    if len(points) == 0:
                        continue
                    if len(points) > self.max_points:
                        idx = np.random.choice(len(points), self.max_points, replace=False)
                        points = points[idx]
                    name = f"modelnet{self.variant}/{class_dir.name}/{off_file.name}"
                    yield name, points
                except Exception as e:
                    print(f"Warning: skipping {off_file}: {e}")

    def __repr__(self):
        return f"ModelNet{self.variant}Dataset(split={self.split!r})"


class ModelNet10Dataset(ModelNetDataset):
    """Convenience wrapper for ModelNet10."""

    def __init__(self, root_dir, split="train", max_points=2048, download=False):
        super().__init__(
            root_dir, variant=10, split=split, max_points=max_points, download=download,
        )


class ModelNet40Dataset(ModelNetDataset):
    """Convenience wrapper for ModelNet40."""

    def __init__(self, root_dir, split="train", max_points=2048, download=False):
        super().__init__(
            root_dir, variant=40, split=split, max_points=max_points, download=download,
        )
