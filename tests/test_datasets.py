"""Tests for dataset classes."""

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets import DirectoryDataset, SymmetriaDataset
from src.datasets.base import _parse_off, _parse_ply


class TestSymmetriaDataset:
    def test_count(self):
        """5 shapes × 2 instances × 1 noise level = 10 items."""
        ds = SymmetriaDataset(n_instances=2, n_points=64, noise_levels=(0.0,), seed=0)
        items = list(ds)
        assert len(items) == 10

    def test_point_shape(self):
        """Each yielded point cloud should have shape (<=n_points, 3)."""
        ds = SymmetriaDataset(n_instances=1, n_points=64, noise_levels=(0.0,), seed=0)
        for name, pts in ds:
            assert pts.ndim == 2
            assert pts.shape[1] == 3
            assert pts.shape[0] <= 64


class TestOFFParser:
    def test_parse_off(self):
        """Parse a minimal OFF file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.off', delete=False) as f:
            f.write("OFF\n3 0 0\n0.0 0.0 0.0\n1.0 0.0 0.0\n0.0 1.0 0.0\n")
            f.flush()
            pts = _parse_off(f.name)
        assert pts.shape == (3, 3)
        np.testing.assert_allclose(pts[1], [1.0, 0.0, 0.0])
        Path(f.name).unlink()

    def test_parse_off_compact_header(self):
        """Parse OFF file with counts on the header line."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.off', delete=False) as f:
            f.write("OFF3 0 0\n0.0 0.0 0.0\n1.0 0.0 0.0\n0.0 1.0 0.0\n")
            f.flush()
            pts = _parse_off(f.name)
        assert pts.shape == (3, 3)
        Path(f.name).unlink()


class TestPLYParser:
    def test_parse_ply_ascii(self):
        """Parse a minimal ASCII PLY file."""
        content = (
            "ply\nformat ascii 1.0\nelement vertex 3\n"
            "property float x\nproperty float y\nproperty float z\n"
            "end_header\n0.0 0.0 0.0\n1.0 0.0 0.0\n0.0 1.0 0.0\n"
        )
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.ply', delete=False) as f:
            f.write(content.encode('ascii'))
            f.flush()
            pts = _parse_ply(f.name)
        assert pts.shape == (3, 3)
        np.testing.assert_allclose(pts[2], [0.0, 1.0, 0.0])
        Path(f.name).unlink()


class TestDirectoryDataset:
    def test_loads_off_files(self):
        """DirectoryDataset should find and parse OFF files in a temp dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            off_path = Path(tmpdir) / "test.off"
            off_path.write_text("OFF\n4 0 0\n0 0 0\n1 0 0\n0 1 0\n0 0 1\n")
            ds = DirectoryDataset(tmpdir, max_points=100)
            items = list(ds)
            assert len(items) == 1
            assert items[0][0] == "test.off"
            assert items[0][1].shape == (4, 3)
