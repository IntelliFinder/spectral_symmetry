"""Tests for src/training.py utilities."""

import random

import numpy as np
import torch
from torch.utils.data import TensorDataset

from src.training import make_train_val_split, seed_everything, worker_init_fn


class TestSeedEverything:
    """Test that seed_everything produces deterministic outputs."""

    def test_numpy_deterministic(self):
        seed_everything(123)
        a = np.random.rand(10)
        seed_everything(123)
        b = np.random.rand(10)
        np.testing.assert_array_equal(a, b)

    def test_torch_deterministic(self):
        seed_everything(123)
        a = torch.randn(10)
        seed_everything(123)
        b = torch.randn(10)
        torch.testing.assert_close(a, b)

    def test_python_random_deterministic(self):
        seed_everything(123)
        a = [random.random() for _ in range(10)]
        seed_everything(123)
        b = [random.random() for _ in range(10)]
        assert a == b

    def test_different_seeds_differ(self):
        seed_everything(1)
        a = np.random.rand(10)
        seed_everything(2)
        b = np.random.rand(10)
        assert not np.array_equal(a, b)


class TestWorkerInitFn:
    """Test worker_init_fn seed derivation."""

    def test_different_workers_different_seeds(self):
        """Different worker IDs produce different numpy random states."""
        torch.manual_seed(42)
        worker_init_fn(0)
        a = np.random.rand(5)

        torch.manual_seed(42)
        worker_init_fn(1)
        b = np.random.rand(5)

        assert not np.array_equal(a, b)

    def test_no_overflow(self):
        """worker_init_fn doesn't crash with large initial seeds."""
        # Simulate a large torch seed close to 2^32 boundary
        torch.manual_seed(2**32 - 1)
        worker_init_fn(15)  # Should not raise


class TestMakeTrainValSplit:
    """Test deterministic train/val splitting."""

    def _make_dataset(self, n=100):
        return TensorDataset(torch.arange(n))

    def test_correct_sizes(self):
        ds = self._make_dataset(100)
        train, val = make_train_val_split(ds, val_fraction=0.2, seed=42)
        assert len(train) == 80
        assert len(val) == 20

    def test_disjoint_indices(self):
        ds = self._make_dataset(100)
        train, val = make_train_val_split(ds, val_fraction=0.2, seed=42)
        train_set = set(train.indices)
        val_set = set(val.indices)
        assert len(train_set & val_set) == 0
        assert len(train_set | val_set) == 100

    def test_deterministic(self):
        ds = self._make_dataset(100)
        t1, v1 = make_train_val_split(ds, val_fraction=0.2, seed=42)
        t2, v2 = make_train_val_split(ds, val_fraction=0.2, seed=42)
        assert t1.indices == t2.indices
        assert v1.indices == v2.indices

    def test_different_seed_different_split(self):
        ds = self._make_dataset(100)
        t1, _ = make_train_val_split(ds, val_fraction=0.2, seed=1)
        t2, _ = make_train_val_split(ds, val_fraction=0.2, seed=2)
        assert t1.indices != t2.indices
