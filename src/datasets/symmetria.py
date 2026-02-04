"""Procedurally generated symmetric shapes (SymmetriaDataset)."""

import numpy as np

from .base import PointCloudDataset

# ---------------------------------------------------------------------------
# Procedural shape generators
# ---------------------------------------------------------------------------

def _generate_human(n_points, rng):
    """Bilateral-symmetric humanoid figure from stacked ellipsoids."""
    parts = []
    # torso
    torso = rng.randn(n_points // 4, 3) * [0.3, 0.8, 0.2]
    torso[:, 1] += 0.5
    parts.append(torso)
    # head
    head = rng.randn(n_points // 8, 3) * 0.2
    head[:, 1] += 1.6
    parts.append(head)
    # arms (symmetric)
    for sign in [-1, 1]:
        arm = rng.randn(n_points // 8, 3) * [0.1, 0.5, 0.1]
        arm[:, 0] += sign * 0.5
        arm[:, 1] += 0.8
        parts.append(arm)
    # legs (symmetric)
    for sign in [-1, 1]:
        leg = rng.randn(n_points // 8, 3) * [0.12, 0.6, 0.12]
        leg[:, 0] += sign * 0.2
        leg[:, 1] -= 0.4
        parts.append(leg)
    pts = np.vstack(parts)
    return pts[:n_points]


def _generate_airplane(n_points, rng):
    """Bilateral-symmetric airplane from elongated shapes."""
    parts = []
    # fuselage
    fuse = rng.randn(n_points // 3, 3) * [0.1, 0.1, 1.0]
    parts.append(fuse)
    # wings (symmetric)
    for sign in [-1, 1]:
        wing = rng.randn(n_points // 4, 3) * [1.0, 0.02, 0.3]
        wing[:, 0] += sign * 0.6
        parts.append(wing)
    # tail
    tail = rng.randn(n_points // 6, 3) * [0.3, 0.3, 0.05]
    tail[:, 2] -= 1.0
    parts.append(tail)
    pts = np.vstack(parts)
    return pts[:n_points]


def _generate_chair(n_points, rng):
    """Bilateral-symmetric chair."""
    parts = []
    # seat
    seat = rng.randn(n_points // 4, 3) * [0.5, 0.03, 0.5]
    seat[:, 1] += 0.5
    parts.append(seat)
    # backrest
    back = rng.randn(n_points // 5, 3) * [0.5, 0.5, 0.03]
    back[:, 1] += 1.0
    back[:, 2] -= 0.25
    parts.append(back)
    # four legs (pairwise symmetric)
    for sx in [-1, 1]:
        for sz in [-1, 1]:
            leg = rng.randn(n_points // 10, 3) * [0.04, 0.25, 0.04]
            leg[:, 0] += sx * 0.4
            leg[:, 2] += sz * 0.4
            parts.append(leg)
    pts = np.vstack(parts)
    return pts[:n_points]


def _generate_table(n_points, rng):
    """Four-fold symmetric table."""
    parts = []
    # top
    top = rng.randn(n_points // 3, 3) * [0.7, 0.03, 0.7]
    top[:, 1] += 0.8
    parts.append(top)
    # four legs
    for sx in [-1, 1]:
        for sz in [-1, 1]:
            leg = rng.randn(n_points // 6, 3) * [0.04, 0.4, 0.04]
            leg[:, 0] += sx * 0.55
            leg[:, 2] += sz * 0.55
            parts.append(leg)
    pts = np.vstack(parts)
    return pts[:n_points]


def _generate_vase(n_points, rng):
    """Rotationally symmetric vase via surface of revolution."""
    t = rng.uniform(0, 1, n_points)
    theta = rng.uniform(0, 2 * np.pi, n_points)
    # radius profile: bulge in middle, narrow at top and bottom
    r = 0.2 + 0.3 * np.sin(np.pi * t)
    x = r * np.cos(theta)
    z = r * np.sin(theta)
    y = t * 2 - 1  # height from -1 to 1
    return np.column_stack([x, y, z])


_GENERATORS = {
    'human': _generate_human,
    'airplane': _generate_airplane,
    'chair': _generate_chair,
    'table': _generate_table,
    'vase': _generate_vase,
}


class SymmetriaDataset(PointCloudDataset):
    """Procedurally generated symmetric shapes.

    5 shape types × n_instances × len(noise_levels).
    """

    def __init__(self, n_instances=10, n_points=1024, noise_levels=(0.0, 0.01, 0.05), seed=42):
        self.n_instances = n_instances
        self.n_points = n_points
        self.noise_levels = noise_levels
        self.seed = seed

    def __iter__(self):
        rng = np.random.RandomState(self.seed)
        for shape_name, gen_fn in _GENERATORS.items():
            for inst in range(self.n_instances):
                base_points = gen_fn(self.n_points, rng)
                for noise_std in self.noise_levels:
                    pts = base_points.copy()
                    if noise_std > 0:
                        pts += rng.randn(*pts.shape) * noise_std
                    name = f"symmetria/{shape_name}/inst{inst:02d}_noise{noise_std:.3f}"
                    yield name, pts

    def __repr__(self):
        n = len(_GENERATORS) * self.n_instances * len(self.noise_levels)
        return f"SymmetriaDataset({n} shapes)"
