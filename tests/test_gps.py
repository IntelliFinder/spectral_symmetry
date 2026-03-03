"""Tests for GPS (Global Point Signature) computation."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.spectral_core import compute_gps


class TestComputeGPS:
    def test_output_shape(self):
        """GPS output should be (N, k)."""
        N, k = 50, 8
        eigenvalues = np.sort(np.random.rand(k) + 0.1)
        eigenvectors = np.random.randn(N, k)
        gps = compute_gps(eigenvalues, eigenvectors)
        assert gps.shape == (N, k), f"Expected ({N}, {k}), got {gps.shape}"

    def test_weighting_first_column_largest_norm(self):
        """First column (smallest eigenvalue) should have largest norm due to 1/sqrt(lambda)."""
        N, k = 100, 5
        eigenvalues = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
        # Use same-magnitude eigenvectors so weighting determines norm
        eigenvectors = np.ones((N, k))
        gps = compute_gps(eigenvalues, eigenvectors)
        col_norms = np.linalg.norm(gps, axis=0)
        # First column should have largest norm
        assert col_norms[0] == col_norms.max(), (
            f"First column norm {col_norms[0]:.4f} should be largest, got {col_norms}"
        )
        # Norms should be monotonically decreasing
        for i in range(k - 1):
            assert col_norms[i] >= col_norms[i + 1], f"Column norms not decreasing: {col_norms}"

    def test_weighting_values(self):
        """GPS values should match phi_k(i) / sqrt(lambda_k) exactly."""
        eigenvalues = np.array([1.0, 4.0])
        eigenvectors = np.array([[2.0, 6.0], [3.0, 8.0]])
        gps = compute_gps(eigenvalues, eigenvectors)
        # Expected: col0 = evec / sqrt(1) = evec, col1 = evec / sqrt(4) = evec/2
        expected = np.array([[2.0, 3.0], [3.0, 4.0]], dtype=np.float32)
        np.testing.assert_allclose(gps, expected, rtol=1e-5)

    def test_empty_eigenvalues(self):
        """Empty eigenvalues should return (N, 0) array."""
        N = 20
        eigenvalues = np.array([])
        eigenvectors = np.zeros((N, 0))
        gps = compute_gps(eigenvalues, eigenvectors)
        assert gps.shape == (N, 0), f"Expected ({N}, 0), got {gps.shape}"

    def test_sign_sensitivity(self):
        """Flipping an eigenvector should negate the corresponding GPS column."""
        N, k = 30, 4
        eigenvalues = np.sort(np.random.rand(k) + 0.1)
        eigenvectors = np.random.randn(N, k)
        gps_orig = compute_gps(eigenvalues, eigenvectors)

        # Flip sign of second eigenvector
        eigvec_flipped = eigenvectors.copy()
        eigvec_flipped[:, 1] *= -1
        gps_flipped = compute_gps(eigenvalues, eigvec_flipped)

        # Column 1 should be negated
        np.testing.assert_allclose(gps_flipped[:, 1], -gps_orig[:, 1], rtol=1e-5)
        # Other columns unchanged
        np.testing.assert_allclose(gps_flipped[:, 0], gps_orig[:, 0], rtol=1e-5)
        np.testing.assert_allclose(gps_flipped[:, 2], gps_orig[:, 2], rtol=1e-5)

    def test_float32_output(self):
        """GPS should return float32."""
        eigenvalues = np.array([1.0, 2.0], dtype=np.float64)
        eigenvectors = np.random.randn(10, 2).astype(np.float64)
        gps = compute_gps(eigenvalues, eigenvectors)
        assert gps.dtype == np.float32, f"Expected float32, got {gps.dtype}"
