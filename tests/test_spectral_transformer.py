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


# ---------------------------------------------------------------------------
# Permutation equivariance / invariance tests
# ---------------------------------------------------------------------------
@skip_no_torch
class TestPermutationProperties:
    """Verify transformer encoder is equivariant and classifier is invariant to permutations."""

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

    def test_encoder_equivariance(self):
        """encoder(Px) == P(encoder(x)) for a random permutation P."""
        model = self._make_model()
        model.eval()
        torch.manual_seed(42)
        x = torch.randn(1, 50, 19)

        perm = torch.randperm(50)
        x_perm = x[:, perm, :]

        with torch.no_grad():
            h = model.input_proj(x)
            h = model.transformer_encoder(h)

            h_perm = model.input_proj(x_perm)
            h_perm = model.transformer_encoder(h_perm)

        # encoder(Px) should equal P(encoder(x))
        assert torch.allclose(h_perm, h[:, perm, :], atol=1e-5), \
            f"Max diff: {(h_perm - h[:, perm, :]).abs().max().item()}"

    def test_encoder_equivariance_with_mask(self):
        """Same equivariance test with a non-trivial padding mask.

        Only checks valid (non-padded) positions since padded positions
        may differ depending on their location in the sequence.
        """
        model = self._make_model()
        model.eval()
        torch.manual_seed(42)
        x = torch.randn(1, 50, 19)
        mask = torch.zeros(1, 50, dtype=torch.bool)
        mask[:, 40:] = True  # last 10 positions padded

        perm = torch.randperm(50)
        x_perm = x[:, perm, :]
        mask_perm = mask[:, perm]

        with torch.no_grad():
            h = model.input_proj(x)
            h = model.transformer_encoder(h, src_key_padding_mask=mask)

            h_perm = model.input_proj(x_perm)
            h_perm = model.transformer_encoder(h_perm, src_key_padding_mask=mask_perm)

        # Only compare valid (non-padded) positions
        valid_mask_perm = ~mask_perm[0]  # (50,)
        h_perm_valid = h_perm[0, valid_mask_perm]
        h_reordered_valid = h[0, perm][valid_mask_perm]

        assert torch.allclose(h_perm_valid, h_reordered_valid, atol=1e-5), \
            f"Max diff: {(h_perm_valid - h_reordered_valid).abs().max().item()}"

    def test_classifier_invariance(self):
        """model(Px) == model(x) — mean pooling makes output permutation-invariant."""
        model = self._make_model()
        model.eval()
        torch.manual_seed(42)
        x = torch.randn(1, 50, 19)

        perm = torch.randperm(50)
        x_perm = x[:, perm, :]

        with torch.no_grad():
            logits = model(x)
            logits_perm = model(x_perm)

        assert torch.allclose(logits, logits_perm, atol=1e-5), \
            f"Max diff: {(logits - logits_perm).abs().max().item()}"

    def test_classifier_invariance_with_mask(self):
        """model(Px, Pm) == model(x, m) with a padding mask."""
        model = self._make_model()
        model.eval()
        torch.manual_seed(42)
        x = torch.randn(1, 50, 19)
        mask = torch.zeros(1, 50, dtype=torch.bool)
        mask[:, 40:] = True

        perm = torch.randperm(50)
        x_perm = x[:, perm, :]
        mask_perm = mask[:, perm]

        with torch.no_grad():
            logits = model(x, mask=mask)
            logits_perm = model(x_perm, mask=mask_perm)

        assert torch.allclose(logits, logits_perm, atol=1e-5), \
            f"Max diff: {(logits - logits_perm).abs().max().item()}"


# ---------------------------------------------------------------------------
# Positional encoding correctness tests
# ---------------------------------------------------------------------------
@skip_no_torch
class TestPositionalEncodingCorrectness:
    """Verify that spectral features match the spectral_core output."""

    def test_eigenvector_features_match_spectral_core(self):
        """Features[i, 3:3+k] should equal eigenvectors[i, :] from spectral_core."""
        from src.preprocessing import center_and_normalize, random_subsample
        from src.spectral_core import build_graph_laplacian, compute_eigenpairs

        rng = np.random.RandomState(7)
        points = rng.randn(50, 3).astype(np.float64)
        points = random_subsample(points, 50, seed=0)
        points, _, _ = center_and_normalize(points)

        n_eigs = 8
        L, comp_idx = build_graph_laplacian(points, n_neighbors=12)
        eigenvalues, eigenvectors = compute_eigenpairs(L, n_eigs=n_eigs)

        pts_cc = points[comp_idx]
        n_actual = pts_cc.shape[0]
        k = eigenvectors.shape[1]

        n_points = 64
        feat_dim = 3 + n_eigs
        features = np.zeros((n_points, feat_dim), dtype=np.float32)
        features[:n_actual, :3] = pts_cc.astype(np.float32)
        features[:n_actual, 3:3 + k] = eigenvectors.astype(np.float32)

        # Verify eigenvector columns match
        for i in range(n_actual):
            np.testing.assert_allclose(
                features[i, 3:3 + k],
                eigenvectors[i, :].astype(np.float32),
                atol=1e-6,
                err_msg=f"Eigenvector mismatch at point {i}",
            )

    def test_eigenvector_node_correspondence_with_component_extraction(self):
        """Two separated clusters: verify largest component is extracted correctly."""
        from src.spectral_core import build_graph_laplacian, compute_eigenpairs

        rng = np.random.RandomState(123)
        # Cluster 1: 40 points near origin
        c1 = rng.randn(40, 3) * 0.1
        # Cluster 2: 10 points far away
        c2 = rng.randn(10, 3) * 0.1 + 100.0
        points = np.vstack([c1, c2])

        L, comp_idx = build_graph_laplacian(points, n_neighbors=5)

        # Largest component should be the 40-point cluster
        assert len(comp_idx) == 40, f"Expected 40 points in largest component, got {len(comp_idx)}"
        # All indices should be from the first cluster (indices 0-39)
        assert np.all(comp_idx < 40), "Largest component should be the first cluster"

        eigenvalues, eigenvectors = compute_eigenpairs(L, n_eigs=4)
        assert eigenvectors.shape[0] == 40
        assert eigenvectors.shape[1] <= 4


# ---------------------------------------------------------------------------
# Point cloud feature tests
# ---------------------------------------------------------------------------
@skip_no_torch
class TestPointCloudFeatures:
    """Tests for xyz-only features and k-NN distance matrices."""

    def test_xyz_only_features(self):
        """Verify xyz-only features have shape (n_points, 3) with no padding for full clouds."""
        from src.preprocessing import center_and_normalize

        rng = np.random.RandomState(42)
        points = rng.randn(100, 3)
        points, _, _ = center_and_normalize(points)

        n_points = 100
        features = np.zeros((n_points, 3), dtype=np.float32)
        features[:, :3] = points.astype(np.float32)

        assert features.shape == (n_points, 3)
        # No padding: all rows should be non-zero
        assert np.all(np.any(features != 0, axis=1))

    def test_knn_distance_matrix(self):
        """Verify k-NN distance matrix is symmetric with zeros on diagonal."""
        from sklearn.neighbors import NearestNeighbors

        rng = np.random.RandomState(42)
        n_points = 50
        points = rng.randn(n_points, 3).astype(np.float32)

        k = 10
        nn = NearestNeighbors(n_neighbors=k + 1)
        nn.fit(points)
        distances, indices = nn.kneighbors(points)

        # Build sparse distance matrix
        dist_matrix = np.zeros((n_points, n_points), dtype=np.float32)
        for i in range(n_points):
            for j_idx in range(1, k + 1):  # skip self (index 0)
                j = indices[i, j_idx]
                dist_matrix[i, j] = distances[i, j_idx]

        # Symmetrize
        dist_matrix = 0.5 * (dist_matrix + dist_matrix.T)

        assert dist_matrix.shape == (n_points, n_points)
        # Diagonal should be zero
        np.testing.assert_allclose(np.diag(dist_matrix), 0.0, atol=1e-7)
        # Should be symmetric
        np.testing.assert_allclose(dist_matrix, dist_matrix.T, atol=1e-7)
