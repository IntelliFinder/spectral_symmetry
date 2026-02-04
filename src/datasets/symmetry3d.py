"""Symmetry3D dataset (ICCV 2017) — requires manual download."""

from pathlib import Path

import numpy as np

from .base import PointCloudDataset, _parse_ply

_DOWNLOAD_INSTRUCTIONS = """\
Symmetry3D dataset not found at {data_dir}.

This dataset requires manual download:
  1. Visit https://github.com/GrailLab/Symmetry-3D
  2. Download the dataset archive
  3. Extract so that PLY files are at:
       {data_dir}/{{split}}/*.ply
"""


class Symmetry3DDataset(PointCloudDataset):
    """Symmetry3D dataset (ICCV 2017, manual download required).

    Parameters
    ----------
    root_dir : str or Path
        Root data directory. Expects PLY files at
        ``{root_dir}/raw/3d-global-sym/{split}/*.ply``.
    split : str
        Dataset split (e.g. ``'train'``, ``'test'``).
    max_points : int
        Subsample to this many points if the mesh has more.
    """

    def __init__(self, root_dir, split="train", max_points=2048):
        self.root_dir = Path(root_dir)
        self.split = split
        self.max_points = max_points
        self.data_dir = self.root_dir / "raw" / "3d-global-sym"

        if not self.data_dir.is_dir():
            raise FileNotFoundError(
                _DOWNLOAD_INSTRUCTIONS.format(data_dir=self.data_dir)
            )

    def __iter__(self):
        split_dir = self.data_dir / self.split
        if not split_dir.is_dir():
            print(f"Warning: split directory not found: {split_dir}")
            return

        for ply_file in sorted(split_dir.glob("*.ply")):
            try:
                points = _parse_ply(str(ply_file))
                if len(points) == 0:
                    continue
                if len(points) > self.max_points:
                    idx = np.random.choice(len(points), self.max_points, replace=False)
                    points = points[idx]
                name = f"symmetry3d/{self.split}/{ply_file.name}"
                yield name, points
            except Exception as e:
                print(f"Warning: skipping {ply_file}: {e}")

    def __repr__(self):
        return f"Symmetry3DDataset(split={self.split!r})"
