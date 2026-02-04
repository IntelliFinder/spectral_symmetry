"""Tests for spectral_core module."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.spectral_core import (
    analyze_spectrum,
    build_graph_laplacian,
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
        assert result['multiplicity'] == [1, 1, 1, 1, 1]
        assert result['n_repeating'] == 0
        assert result['n_non_repeating'] == 5

    def test_pair_repeated(self):
        """A pair of nearly equal eigenvalues should be detected."""
        eigs = np.array([0.0, 1.0, 1.0005, 3.0, 5.0])
        result = detect_eigenvalue_multiplicities(eigs)
        assert result['multiplicity'][1] == 2
        assert result['multiplicity'][2] == 2
        assert result['n_repeating'] == 2


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
            'eigenvalues', 'eigenvectors', 'scores',
            'uncanonicalizable_raw', 'uncanonicalizable',
            'component_indices', 'spectral_gap', 'threshold_used',
            'multiplicity_info',
        }
        assert expected_keys == set(result.keys())

    def test_eigenvalues_sorted(self):
        """Eigenvalues should be in non-decreasing order."""
        rng = np.random.RandomState(42)
        points = rng.randn(100, 3)
        result = analyze_spectrum(points, n_eigs=10, n_neighbors=5)
        eigs = result['eigenvalues']
        assert np.all(np.diff(eigs) >= -1e-10)
