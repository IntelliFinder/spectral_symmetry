"""Tests for the spectral transformer classifier pipeline."""

import numpy as np
import pytest

try:
    import torch

    torch_available = True
except ImportError:
    torch_available = False

skip_no_torch = pytest.mark.skipif(not torch_available, reason="torch not installed")


# ---------------------------------------------------------------------------
# Test spectral feature construction
# ---------------------------------------------------------------------------
@skip_no_torch
class TestSpectralFeatures:
    """Verify the spectral pipeline produces correct feature shapes."""

    def test_spectral_feature_shape(self):
        from src.preprocessing import center_and_normalize, random_subsample
        from src.spectral_core import build_graph_laplacian, compute_eigenpairs

        rng = np.random.RandomState(42)
        points = rng.randn(100, 3).astype(np.float64)

        points = random_subsample(points, 100, seed=0)
        points, _, _ = center_and_normalize(points)

        n_eigs = 8
        L, comp_idx = build_graph_laplacian(points, n_neighbors=12)
        eigenvalues, eigenvectors = compute_eigenpairs(L, n_eigs=n_eigs)

        pts_cc = points[comp_idx]
        n_actual = pts_cc.shape[0]

        # Concatenate xyz with eigenvectors
        features = np.concatenate([pts_cc, eigenvectors], axis=1)
        assert features.shape == (n_actual, 3 + eigenvectors.shape[1])
        assert eigenvectors.shape[1] <= n_eigs

    def test_padding_and_mask(self):
        n_points = 64
        n_eigs = 4
        feat_dim = 3 + n_eigs

        # Simulate a shape with fewer valid points than n_points
        n_actual = 50
        features = np.zeros((n_points, feat_dim), dtype=np.float32)
        features[:n_actual, :3] = np.random.randn(n_actual, 3).astype(np.float32)
        features[:n_actual, 3:] = np.random.randn(n_actual, n_eigs).astype(np.float32)

        mask = np.ones(n_points, dtype=bool)
        mask[:n_actual] = False

        assert features.shape == (n_points, feat_dim)
        assert mask.sum() == n_points - n_actual
        assert (~mask).sum() == n_actual


# ---------------------------------------------------------------------------
# Test model
# ---------------------------------------------------------------------------
@skip_no_torch
class TestSpectralTransformerClassifier:
    """Tests for the transformer model."""

    def _make_model(self, input_dim=19, n_classes=10):
        from src.experiments.spectral_transformer.model import SpectralTransformerClassifier

        return SpectralTransformerClassifier(
            input_dim=input_dim,
            d_model=32,
            nhead=4,
            num_layers=2,
            dim_feedforward=64,
            dropout=0.0,
            n_classes=n_classes,
        )

    def test_forward_shape(self):
        model = self._make_model()
        x = torch.randn(4, 100, 19)
        logits = model(x)
        assert logits.shape == (4, 10)

    def test_forward_with_mask(self):
        model = self._make_model()
        x = torch.randn(4, 100, 19)
        mask = torch.zeros(4, 100, dtype=torch.bool)
        mask[:, 80:] = True  # last 20 positions are padding
        logits = model(x, mask=mask)
        assert logits.shape == (4, 10)

    def test_backward_runs(self):
        model = self._make_model()
        x = torch.randn(4, 100, 19)
        logits = model(x)
        loss = logits.sum()
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"


# ---------------------------------------------------------------------------
# End-to-end test
# ---------------------------------------------------------------------------
@skip_no_torch
class TestEndToEnd:
    """Full pipeline: random points → spectral features → model → logits."""

    def test_pipeline(self):
        from src.experiments.spectral_transformer.model import SpectralTransformerClassifier
        from src.preprocessing import center_and_normalize
        from src.spectral_core import build_graph_laplacian, compute_eigenpairs

        rng = np.random.RandomState(99)
        points = rng.randn(80, 3)
        points, _, _ = center_and_normalize(points)

        n_eigs = 16
        L, comp_idx = build_graph_laplacian(points, n_neighbors=12)
        eigenvalues, eigenvectors = compute_eigenpairs(L, n_eigs=n_eigs)

        pts_cc = points[comp_idx]
        n_actual = pts_cc.shape[0]
        n_eigs_actual = eigenvectors.shape[1]

        n_points = 128
        feat_dim = 3 + n_eigs
        features = np.zeros((n_points, feat_dim), dtype=np.float32)
        features[:n_actual, :3] = pts_cc.astype(np.float32)
        features[:n_actual, 3 : 3 + n_eigs_actual] = eigenvectors.astype(np.float32)

        mask = np.ones(n_points, dtype=bool)
        mask[:n_actual] = False

        x = torch.from_numpy(features).unsqueeze(0)  # (1, 128, 19)
        m = torch.from_numpy(mask).unsqueeze(0)  # (1, 128)

        model = SpectralTransformerClassifier(input_dim=feat_dim, n_classes=10)
        model.eval()
        with torch.no_grad():
            logits = model(x, m)

        assert logits.shape == (1, 10)
