"""Tests for preprocessing module."""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import center_and_normalize, load_processed, random_subsample, save_processed


class TestCenterAndNormalize:
    def test_mean_zero(self):
        """After normalization, mean should be approximately zero."""
        rng = np.random.RandomState(0)
        pts = rng.randn(100, 3) * 5 + np.array([10, 20, 30])
        normalized, centroid, scale = center_and_normalize(pts)
        np.testing.assert_allclose(normalized.mean(axis=0), 0, atol=1e-10)

    def test_max_distance_one(self):
        """After normalization, max distance from origin should be 1."""
        rng = np.random.RandomState(0)
        pts = rng.randn(100, 3) * 5 + np.array([10, 20, 30])
        normalized, centroid, scale = center_and_normalize(pts)
        max_dist = np.max(np.linalg.norm(normalized, axis=1))
        assert max_dist == pytest.approx(1.0, abs=1e-10)


class TestRandomSubsample:
    def test_deterministic_with_seed(self):
        """Same seed should produce identical subsamples."""
        pts = np.arange(300).reshape(100, 3).astype(float)
        s1 = random_subsample(pts, 10, seed=42)
        s2 = random_subsample(pts, 10, seed=42)
        np.testing.assert_array_equal(s1, s2)

    def test_correct_count(self):
        """Output should have exactly n_points rows."""
        pts = np.arange(300).reshape(100, 3).astype(float)
        result = random_subsample(pts, 25, seed=0)
        assert result.shape == (25, 3)

    def test_passthrough_if_small(self):
        """If points <= n_points, return as-is."""
        pts = np.arange(15).reshape(5, 3).astype(float)
        result = random_subsample(pts, 10, seed=0)
        np.testing.assert_array_equal(result, pts)


class TestSaveLoadRoundtrip:
    def test_roundtrip(self):
        """save_processed -> load_processed should preserve data and metadata."""
        rng = np.random.RandomState(0)
        pts = rng.randn(50, 3)
        meta = {"name": "test_shape", "n_original": 200}

        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = Path(tmpdir) / "test.npy"
            save_processed(pts, fpath, metadata=meta)
            loaded_pts, loaded_meta = load_processed(fpath)

        np.testing.assert_array_equal(pts, loaded_pts)
        assert loaded_meta == meta

    def test_roundtrip_no_metadata(self):
        """Roundtrip without metadata should return None for metadata."""
        rng = np.random.RandomState(0)
        pts = rng.randn(20, 3)

        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = Path(tmpdir) / "test.npy"
            save_processed(pts, fpath)
            loaded_pts, loaded_meta = load_processed(fpath)

        np.testing.assert_array_equal(pts, loaded_pts)
        assert loaded_meta is None
