"""Directory-based point cloud dataset loader."""

from pathlib import Path

import numpy as np

from .base import _PARSERS, PointCloudDataset


class DirectoryDataset(PointCloudDataset):
    """Load point cloud files from a directory."""

    SUPPORTED_EXTENSIONS = set(_PARSERS.keys())

    def __init__(self, root_dir, max_points=2048):
        self.root_dir = Path(root_dir)
        self.max_points = max_points
        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"Directory not found: {root_dir}")

    def __iter__(self):
        files = sorted(
            p for p in self.root_dir.rglob('*')
            if p.suffix.lower() in self.SUPPORTED_EXTENSIONS
        )
        for fpath in files:
            try:
                parser = _PARSERS[fpath.suffix.lower()]
                points = parser(str(fpath))
                if len(points) == 0:
                    continue
                if len(points) > self.max_points:
                    idx = np.random.choice(len(points), self.max_points, replace=False)
                    points = points[idx]
                name = str(fpath.relative_to(self.root_dir))
                yield name, points
            except Exception as e:
                print(f"Warning: skipping {fpath}: {e}")

    def __repr__(self):
        return f"DirectoryDataset({self.root_dir})"
