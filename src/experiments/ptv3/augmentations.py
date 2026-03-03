"""Point cloud augmentations matching the Pointcept ModelNet40 config.

Each transform is a callable that operates on a dict with keys:
  "coord": (N, 3) np.ndarray
  "normal": (N, 3) np.ndarray  (optional)
  "feat": (N, C) np.ndarray
  "label": int
"""

import numpy as np


class NormalizeCoord:
    """Center at origin and scale so max radius = 1 (PointNet2-style)."""

    def __call__(self, data):
        coord = data["coord"]
        centroid = coord.mean(axis=0)
        coord = coord - centroid
        max_radius = np.sqrt((coord**2).sum(axis=1)).max()
        if max_radius > 0:
            coord = coord / max_radius
        data["coord"] = coord
        return data


class RandomScale:
    """Per-axis random scaling (anisotropic by default).

    Parameters
    ----------
    scale : tuple of float
        (low, high) range for uniform sampling.
    anisotropic : bool
        If True, sample independently per axis.
    """

    def __init__(self, scale=(0.7, 1.5), anisotropic=True, seed=None):
        self.low, self.high = scale
        self.anisotropic = anisotropic
        self.rng = np.random.RandomState(seed) if seed is not None else np.random

    def __call__(self, data):
        if self.anisotropic:
            s = self.rng.uniform(self.low, self.high, size=3).astype(np.float32)
        else:
            s = self.rng.uniform(self.low, self.high)
            s = np.array([s, s, s], dtype=np.float32)
        data["coord"] = data["coord"] * s
        if "normal" in data:
            # Scale normals by inverse scale to preserve direction, then re-normalize
            normal = data["normal"] / s
            norms = np.linalg.norm(normal, axis=1, keepdims=True)
            norms = np.clip(norms, 1e-8, None)
            data["normal"] = normal / norms
        return data


class RandomShift:
    """Random translation per axis.

    Parameters
    ----------
    shift : list of tuples
        Per-axis (low, high) shift ranges, e.g. [(-0.2, 0.2), (-0.2, 0.2), (0, 0)].
    """

    def __init__(self, shift=None, seed=None):
        if shift is None:
            shift = [(-0.2, 0.2), (-0.2, 0.2), (0, 0)]
        self.shift = shift
        self.rng = np.random.RandomState(seed) if seed is not None else np.random

    def __call__(self, data):
        t = np.array(
            [self.rng.uniform(lo, hi) for lo, hi in self.shift],
            dtype=np.float32,
        )
        data["coord"] = data["coord"] + t
        return data


class GridSample:
    """Voxelize point cloud at given grid resolution.

    In train mode, randomly select one point per occupied voxel.
    In test mode, select the point closest to voxel center.

    Parameters
    ----------
    grid_size : float
        Voxel edge length.
    mode : str
        "train" or "test".
    keys : tuple of str
        Data keys to subsample along with coord.
    """

    def __init__(self, grid_size=0.01, mode="train", keys=("coord", "normal"), seed=None):
        self.grid_size = grid_size
        self.mode = mode
        self.keys = keys
        self.rng = np.random.RandomState(seed) if seed is not None else np.random

    def __call__(self, data):
        coord = data["coord"]
        # Compute voxel indices
        min_coord = coord.min(axis=0)
        grid_coord = ((coord - min_coord) / self.grid_size).astype(np.int32)

        # Hash voxel coordinates to unique IDs
        dims = grid_coord.max(axis=0) + 1
        voxel_id = (
            grid_coord[:, 0].astype(np.int64)
            + grid_coord[:, 1].astype(np.int64) * dims[0]
            + grid_coord[:, 2].astype(np.int64) * dims[0] * dims[1]
        )

        # Find unique voxels
        unique_ids, inverse, counts = np.unique(
            voxel_id,
            return_inverse=True,
            return_counts=True,
        )
        n_voxels = len(unique_ids)

        if self.mode == "train":
            # Vectorized random selection: one point per voxel
            # Sort by voxel, pick random index within each group
            sort_idx = np.argsort(inverse)
            offsets = np.zeros(n_voxels + 1, dtype=np.int64)
            np.cumsum(counts, out=offsets[1:])
            # Random offset within each voxel group
            rand_off = (self.rng.random(n_voxels) * counts).astype(np.int64)
            selected = sort_idx[offsets[:-1] + rand_off]
        else:
            # Vectorized closest-to-center selection
            voxel_center = (grid_coord + 0.5) * self.grid_size + min_coord
            dist_to_center = np.linalg.norm(coord - voxel_center, axis=1)
            # For each voxel, find the member with minimum distance
            sort_idx = np.argsort(inverse)
            offsets = np.zeros(n_voxels + 1, dtype=np.int64)
            np.cumsum(counts, out=offsets[1:])
            selected = np.empty(n_voxels, dtype=np.int64)
            for i in range(n_voxels):
                members = sort_idx[offsets[i] : offsets[i + 1]]
                selected[i] = members[dist_to_center[members].argmin()]

        # Store grid_coord for PTv3
        data["grid_coord"] = grid_coord[selected]

        # Subsample all relevant keys
        for key in self.keys:
            if key in data:
                data[key] = data[key][selected]

        return data


class ShufflePoint:
    """Random permutation of point ordering."""

    def __init__(self, seed=None):
        self.rng = np.random.RandomState(seed) if seed is not None else np.random

    def __call__(self, data):
        n = data["coord"].shape[0]
        perm = self.rng.permutation(n)
        for key in list(data.keys()):
            if isinstance(data[key], np.ndarray) and data[key].shape[0] == n:
                data[key] = data[key][perm]
        return data


class Compose:
    """Chain transforms together."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data
