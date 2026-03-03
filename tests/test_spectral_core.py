"""Tests for spectral_core module."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.spectral_core import (
    analyze_spectrum,
    build_graph_laplacian,
    compute_dataset_sigma,
    compute_hks,
    detect_eigenvalue_multiplicities,
    uncanonicalizability_score,
)


class TestUncanonicalizabilityScore:
    def test_symmetric_vector_score_near_zero(self):
        """A perfectly anti-symmetric vector should have score near 0."""
        v = np.array([-3, -2, -1, 1, 2, 3], dtype=float)
        score = uncanonicalizability_score(v)
        assert score < 0.01, f"Expected score ~0 for symmetric vec, got {score}"

    def test_asymmetric_vector_score_high(self):
        """A clearly asymmetric vector should have score > 0.5."""
        v = np.array([1, 2, 3, 4, 5, 6], dtype=float)
        score = uncanonicalizability_score(v)
        assert score > 0.5, f"Expected score > 0.5 for asymmetric vec, got {score}"

    def test_zero_vector(self):
        """Zero vector should return 0."""
        v = np.zeros(10)
        assert uncanonicalizability_score(v) == 0.0


class TestMultiplicities:
    def test_all_distinct(self):
        """All distinct eigenvalues should have multiplicity 1."""
        eigs = np.array([0.0, 1.0, 3.0, 6.0, 10.0])
        result = detect_eigenvalue_multiplicities(eigs)
        assert result["multiplicity"] == [1, 1, 1, 1, 1]
        assert result["n_repeating"] == 0
        assert result["n_non_repeating"] == 5

    def test_pair_repeated(self):
        """A pair of nearly equal eigenvalues should be detected."""
        eigs = np.array([0.0, 1.0, 1.0005, 3.0, 5.0])
        result = detect_eigenvalue_multiplicities(eigs)
        assert result["multiplicity"][1] == 2
        assert result["multiplicity"][2] == 2
        assert result["n_repeating"] == 2


class TestLaplacian:
    def test_row_sums_zero(self):
        """Graph Laplacian rows should sum to zero."""
        rng = np.random.RandomState(0)
        points = rng.randn(50, 3)
        L, _ = build_graph_laplacian(points, n_neighbors=5)
        row_sums = np.array(L.sum(axis=1)).flatten()
        np.testing.assert_allclose(row_sums, 0, atol=1e-10)


class TestAnalyzeSpectrum:
    def test_returns_expected_keys(self):
        """analyze_spectrum should return a dict with all expected keys."""
        rng = np.random.RandomState(42)
        points = rng.randn(100, 3)
        result = analyze_spectrum(points, n_eigs=10, n_neighbors=5)
        expected_keys = {
            "eigenvalues",
            "eigenvectors",
            "scores",
            "uncanonicalizable_raw",
            "uncanonicalizable",
            "component_indices",
            "spectral_gap",
            "threshold_used",
            "multiplicity_info",
        }
        assert expected_keys == set(result.keys())

    def test_eigenvalues_sorted(self):
        """Eigenvalues should be in non-decreasing order."""
        rng = np.random.RandomState(42)
        points = rng.randn(100, 3)
        result = analyze_spectrum(points, n_eigs=10, n_neighbors=5)
        eigs = result["eigenvalues"]
        assert np.all(np.diff(eigs) >= -1e-10)


class TestComputeHKS:
    """Tests for Heat Kernel Signature computation."""

    def test_output_shape(self):
        """HKS output has shape (N, n_times)."""
        eigenvalues = np.array([0.5, 1.0, 2.0])
        eigenvectors = np.random.RandomState(42).randn(10, 3)
        hks = compute_hks(eigenvalues, eigenvectors, n_times=16)
        assert hks.shape == (10, 16)

    def test_sign_invariance(self):
        """HKS is invariant to eigenvector sign flips (uses V^2)."""
        eigenvalues = np.array([0.5, 1.0, 2.0])
        rng = np.random.RandomState(42)
        eigenvectors = rng.randn(10, 3)
        hks1 = compute_hks(eigenvalues, eigenvectors, n_times=8)
        # Flip signs of all eigenvectors
        hks2 = compute_hks(eigenvalues, -eigenvectors, n_times=8)
        np.testing.assert_allclose(hks1, hks2, atol=1e-6)

    def test_empty_eigenvalues(self):
        """Returns zeros when no eigenvalues provided."""
        hks = compute_hks(np.array([]), np.zeros((10, 0)), n_times=8)
        assert hks.shape == (10, 8)
        assert np.all(hks == 0)

    def test_float32_output(self):
        """Output is float32."""
        eigenvalues = np.array([1.0, 2.0])
        eigenvectors = np.random.RandomState(42).randn(5, 2)
        hks = compute_hks(eigenvalues, eigenvectors, n_times=4)
        assert hks.dtype == np.float32

    def test_positive_values(self):
        """HKS values should be non-negative (sum of exp * v^2 terms)."""
        eigenvalues = np.array([0.5, 1.0, 2.0])
        eigenvectors = np.random.RandomState(42).randn(10, 3)
        hks = compute_hks(eigenvalues, eigenvectors, n_times=16)
        assert np.all(hks >= 0)


class TestComputeDatasetSigma:
    """Tests for uniform sigma computation."""

    def test_returns_float(self):
        """Returns a single float value."""
        rng = np.random.RandomState(42)
        points_list = [rng.randn(20, 3) for _ in range(5)]
        sigma = compute_dataset_sigma(points_list, n_neighbors=5)
        assert isinstance(sigma, float)
        assert sigma > 0

    def test_empty_list_returns_default(self):
        """Returns 1.0 for empty input."""
        sigma = compute_dataset_sigma([], n_neighbors=5)
        assert sigma == 1.0

    def test_deterministic(self):
        """Same input gives same output."""
        rng = np.random.RandomState(42)
        pts = [rng.randn(20, 3) for _ in range(5)]
        s1 = compute_dataset_sigma(pts, n_neighbors=5)
        s2 = compute_dataset_sigma(pts, n_neighbors=5)
        assert s1 == s2
