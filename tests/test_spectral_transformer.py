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


# ---------------------------------------------------------------------------
# Canonicalization tests
# ---------------------------------------------------------------------------
class TestCanonicalization:
    """Verify max-absolute-value sign canonicalization."""

    def test_flips_negative_dominant_entry(self):
        """If the max-abs entry is negative and unique, the eigenvector should be flipped."""
        from src.experiments.spectral_transformer.dataset import canonicalize_eigenvectors

        # Eigenvector where the max-abs entry is negative
        v = np.array([[0.1], [-0.9], [0.2], [0.3]])
        result = canonicalize_eigenvectors(v)
        # Should flip: max-abs is -0.9 at index 1, unique → flip to positive
        assert result[1, 0] > 0, f"Expected positive, got {result[1, 0]}"
        np.testing.assert_allclose(result, -v, atol=1e-12)

    def test_keeps_positive_dominant_entry(self):
        """If the max-abs entry is already positive and unique, no flip."""
        from src.experiments.spectral_transformer.dataset import canonicalize_eigenvectors

        v = np.array([[0.1], [0.9], [0.2], [0.3]])
        result = canonicalize_eigenvectors(v)
        np.testing.assert_allclose(result, v, atol=1e-12)

    def test_no_flip_when_max_abs_repeats(self):
        """If the max-abs value appears more than once, keep eigenvector unchanged."""
        from src.experiments.spectral_transformer.dataset import canonicalize_eigenvectors

        # Two entries with the same absolute value 0.5
        v = np.array([[0.5], [-0.5], [0.1], [0.2]])
        result = canonicalize_eigenvectors(v)
        np.testing.assert_allclose(result, v, atol=1e-12)

    def test_multiple_columns(self):
        """Test canonicalization across multiple eigenvector columns."""
        from src.experiments.spectral_transformer.dataset import canonicalize_eigenvectors

        eigvecs = np.array([
            [0.1, -0.8, 0.5],
            [-0.9, 0.2, -0.5],
            [0.2, 0.1, 0.3],
        ])
        result = canonicalize_eigenvectors(eigvecs)

        # Col 0: max-abs is -0.9 (unique) → flip
        assert result[1, 0] > 0
        # Col 1: max-abs is -0.8 (unique) → flip
        assert result[0, 1] > 0
        # Col 2: max-abs is 0.5 and -0.5 (repeats) → no flip
        np.testing.assert_allclose(result[:, 2], eigvecs[:, 2], atol=1e-12)

    def test_does_not_modify_input(self):
        """Canonicalization should return a copy, not modify in place."""
        from src.experiments.spectral_transformer.dataset import canonicalize_eigenvectors

        v = np.array([[0.1], [-0.9], [0.2]])
        v_orig = v.copy()
        _ = canonicalize_eigenvectors(v)
        np.testing.assert_allclose(v, v_orig, atol=1e-12)

    def test_deterministic(self):
        """Same input always produces the same output (no randomness)."""
        from src.experiments.spectral_transformer.dataset import canonicalize_eigenvectors

        rng = np.random.RandomState(42)
        v = rng.randn(50, 8)
        r1 = canonicalize_eigenvectors(v)
        r2 = canonicalize_eigenvectors(v)
        np.testing.assert_allclose(r1, r2, atol=1e-12)

    def test_idempotent(self):
        """Applying canonicalization twice gives the same result as once."""
        from src.experiments.spectral_transformer.dataset import canonicalize_eigenvectors

        rng = np.random.RandomState(42)
        v = rng.randn(50, 8)
        r1 = canonicalize_eigenvectors(v)
        r2 = canonicalize_eigenvectors(r1)
        np.testing.assert_allclose(r1, r2, atol=1e-12)

    def test_sign_invariance(self):
        """Canonicalizing v and -v should give the same result (when max-abs is unique)."""
        from src.experiments.spectral_transformer.dataset import canonicalize_eigenvectors

        # Single column with a unique max-abs entry
        v = np.array([[0.1], [-0.9], [0.2], [0.3]])
        r_pos = canonicalize_eigenvectors(v)
        r_neg = canonicalize_eigenvectors(-v)
        np.testing.assert_allclose(r_pos, r_neg, atol=1e-12)


# ---------------------------------------------------------------------------
# Laplacian Eigenmaps verification tests
# ---------------------------------------------------------------------------
@skip_no_torch
class TestLaplacianEigenmaps:
    """Verify that positional encodings follow the Laplacian Eigenmaps definition.

    Per Belkin & Niyogi (2003), the embedding for node i should be:
        y_i = (v_2[i], v_3[i], ..., v_{k+1}[i])
    where v_1, v_2, ..., v_n are eigenvectors of L = D - A sorted by ascending
    eigenvalue, and v_1 is the trivial constant eigenvector (eigenvalue 0)
    which is EXCLUDED from the embedding.
    """

    def test_laplacian_is_d_minus_a(self):
        """Verify L = D - A (unnormalized graph Laplacian)."""
        from src.spectral_core import build_graph_laplacian

        rng = np.random.RandomState(42)
        points = rng.randn(60, 3)

        L, comp_idx = build_graph_laplacian(points, n_neighbors=10)
        L_dense = L.toarray()

        # Recover A from L: off-diagonal of L is -A
        A = -L_dense.copy()
        np.fill_diagonal(A, 0)

        # A should be non-negative (binary adjacency)
        assert np.all(A >= 0), "Adjacency should be non-negative"

        # A should be symmetric
        np.testing.assert_allclose(A, A.T, atol=1e-12)

        # D = diag(row sums of A)
        D = np.diag(A.sum(axis=1))

        # L should equal D - A
        np.testing.assert_allclose(L_dense, D - A, atol=1e-12,
                                   err_msg="Laplacian should be D - A")

    def test_laplacian_rows_sum_to_zero(self):
        """Each row of L = D - A should sum to zero."""
        from src.spectral_core import build_graph_laplacian

        rng = np.random.RandomState(42)
        points = rng.randn(60, 3)

        L, _ = build_graph_laplacian(points, n_neighbors=10)
        row_sums = np.array(L.sum(axis=1)).flatten()
        np.testing.assert_allclose(row_sums, 0.0, atol=1e-12,
                                   err_msg="Laplacian rows should sum to zero")

    def test_eigenvalues_sorted_ascending(self):
        """Eigenvalues from compute_eigenpairs should be sorted smallest first."""
        from src.spectral_core import build_graph_laplacian, compute_eigenpairs

        rng = np.random.RandomState(42)
        points = rng.randn(80, 3)

        L, _ = build_graph_laplacian(points, n_neighbors=10)
        eigenvalues, _ = compute_eigenpairs(L, n_eigs=10)

        for i in range(len(eigenvalues) - 1):
            assert eigenvalues[i] <= eigenvalues[i + 1] + 1e-10, \
                f"Eigenvalues not sorted: λ_{i}={eigenvalues[i]} > λ_{i+1}={eigenvalues[i+1]}"

    def test_laplacian_has_zero_eigenvalue(self):
        """For a connected graph, the Laplacian has eigenvalue 0 with constant eigenvector.

        This tests the raw Laplacian (not compute_eigenpairs, which excludes the trivial one).
        """
        import scipy.sparse.linalg as sla

        from src.spectral_core import build_graph_laplacian

        rng = np.random.RandomState(42)
        points = rng.randn(80, 3)

        L, _ = build_graph_laplacian(points, n_neighbors=10)
        vals, vecs = sla.eigsh(L, k=3, which='SM', tol=1e-8)
        vals = np.sort(vals)

        assert abs(vals[0]) < 1e-6, \
            f"Laplacian should have eigenvalue ~0, got {vals[0]}"

    def test_trivial_eigenvector_is_approximately_constant(self):
        """The Laplacian's zero-eigenvalue eigenvector should be approximately constant.

        Tests the raw eigsh output directly (compute_eigenpairs excludes this vector).
        """
        import scipy.sparse.linalg as sla

        from src.spectral_core import build_graph_laplacian

        rng = np.random.RandomState(42)
        points = rng.randn(80, 3)

        L, _ = build_graph_laplacian(points, n_neighbors=10)
        vals, vecs = sla.eigsh(L, k=3, which='SM', tol=1e-8)
        idx = np.argsort(vals)
        v0 = vecs[:, idx[0]]

        v0_normalized = v0 / np.linalg.norm(v0)
        expected_constant = np.ones_like(v0_normalized) / np.sqrt(len(v0_normalized))

        np.testing.assert_allclose(
            np.abs(v0_normalized), np.abs(expected_constant), atol=1e-4,
            err_msg="Trivial eigenvector should be approximately constant"
        )

    def test_eigenvectors_satisfy_eigenvalue_equation(self):
        """Each eigenvector should satisfy Lv = λv."""
        from src.spectral_core import build_graph_laplacian, compute_eigenpairs

        rng = np.random.RandomState(42)
        points = rng.randn(80, 3)

        L, _ = build_graph_laplacian(points, n_neighbors=10)
        eigenvalues, eigenvectors = compute_eigenpairs(L, n_eigs=8)

        L_dense = L.toarray()
        for i in range(eigenvectors.shape[1]):
            v = eigenvectors[:, i]
            lam = eigenvalues[i]
            Lv = L_dense @ v
            np.testing.assert_allclose(
                Lv, lam * v, atol=1e-6,
                err_msg=f"Eigenvector {i} does not satisfy Lv = λv"
            )

    def test_eigenvectors_are_orthonormal(self):
        """Eigenvectors should be orthonormal (V^T V = I)."""
        from src.spectral_core import build_graph_laplacian, compute_eigenpairs

        rng = np.random.RandomState(42)
        points = rng.randn(80, 3)

        L, _ = build_graph_laplacian(points, n_neighbors=10)
        _, eigenvectors = compute_eigenpairs(L, n_eigs=8)

        gram = eigenvectors.T @ eigenvectors
        np.testing.assert_allclose(
            gram, np.eye(eigenvectors.shape[1]), atol=1e-6,
            err_msg="Eigenvectors should be orthonormal"
        )

    def test_node_encoding_is_ith_entry_of_eigenvectors(self):
        """Node i's encoding should be [v_1[i], v_2[i], ..., v_k[i]].

        Verifies that features[i, 3:3+k] == eigenvectors[i, :] as stored
        in the SpectralModelNet dataset pipeline.
        """
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

        # Build features the same way SpectralModelNet does
        n_points = 64
        features = np.zeros((n_points, 3 + n_eigs), dtype=np.float32)
        features[:n_actual, :3] = pts_cc.astype(np.float32)
        features[:n_actual, 3:3 + k] = eigenvectors.astype(np.float32)

        # For each valid node i, features[i, 3:3+k] should be the i-th row of eigenvectors
        for i in range(n_actual):
            for j in range(k):
                assert abs(features[i, 3 + j] - eigenvectors[i, j]) < 1e-6, \
                    f"Node {i}, eigvec {j}: feature={features[i, 3+j]}, expected={eigenvectors[i, j]}"

    def test_compute_eigenpairs_excludes_trivial_eigenvector(self):
        """compute_eigenpairs should exclude the trivial constant eigenvector.

        Per Laplacian Eigenmaps (Belkin & Niyogi 2003), the trivial constant
        eigenvector (eigenvalue 0) is excluded. The first returned eigenvector
        should be the Fiedler vector (eigenvalue > 0).
        """
        from src.spectral_core import build_graph_laplacian, compute_eigenpairs

        rng = np.random.RandomState(42)
        points = rng.randn(80, 3)

        L, _ = build_graph_laplacian(points, n_neighbors=10)
        eigenvalues, eigenvectors = compute_eigenpairs(L, n_eigs=5)

        # The first eigenvalue should be the Fiedler value (> 0), NOT ~0
        assert eigenvalues[0] > 0.01, \
            f"First eigenvalue should be Fiedler value (> 0), got {eigenvalues[0]}"

        # The first eigenvector should NOT be constant
        v0 = eigenvectors[:, 0]
        v0_std = np.std(v0)
        assert v0_std > 0.01, \
            f"First eigenvector should not be constant (std={v0_std})"

        # Should have both positive and negative entries (Fiedler vector)
        assert np.any(v0 > 0) and np.any(v0 < 0), \
            "First eigenvector should be the Fiedler vector with both signs"

        # Should return exactly n_eigs eigenvectors
        assert len(eigenvalues) == 5
        assert eigenvectors.shape[1] == 5

    def test_first_returned_eigenvalue_is_fiedler(self):
        """After excluding trivial, the first eigenvalue should be the Fiedler value (> 0),
        and its eigenvector should have both positive and negative entries."""
        from src.spectral_core import build_graph_laplacian, compute_eigenpairs

        rng = np.random.RandomState(42)
        points = rng.randn(80, 3)

        L, _ = build_graph_laplacian(points, n_neighbors=10)
        eigenvalues, eigenvectors = compute_eigenpairs(L, n_eigs=5)

        # First returned eigenvalue is the Fiedler value (> 0)
        assert eigenvalues[0] > 1e-6, \
            f"Fiedler value should be > 0, got {eigenvalues[0]}"

        # Fiedler vector should have both signs (graph partitioning)
        v_fiedler = eigenvectors[:, 0]
        assert np.any(v_fiedler > 0) and np.any(v_fiedler < 0), \
            "Fiedler vector should have both positive and negative entries"

    def test_all_returned_eigenvalues_positive(self):
        """All returned eigenvalues should be > 0 (trivial excluded)."""
        from src.spectral_core import build_graph_laplacian, compute_eigenpairs

        rng = np.random.RandomState(42)
        points = rng.randn(80, 3)

        L, _ = build_graph_laplacian(points, n_neighbors=10)
        eigenvalues, _ = compute_eigenpairs(L, n_eigs=5)

        assert np.all(eigenvalues > 1e-6), \
            f"All eigenvalues should be > 0 after excluding trivial, got {eigenvalues}"
