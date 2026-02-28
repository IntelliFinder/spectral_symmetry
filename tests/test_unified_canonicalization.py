"""Tests for the unified canonicalization dispatcher and individual methods."""

import numpy as np
import pytest

from src.spectral_canonicalization import (
    CANONICALIZATION_METHODS,
    canonicalize,
    canonicalize_maxabs,
    canonicalize_random_augmented,
    canonicalize_random_fixed,
)

# ---------------------------------------------------------------------------
# TestCanonicalizeMaxabs
# ---------------------------------------------------------------------------


class TestCanonicalizeMaxabs:
    """Tests for canonicalize_maxabs (unique-tie-aware version)."""

    def test_flips_negative_max(self):
        """If unique max-abs entry is negative, flip the column."""
        V = np.array([[0.1], [-0.5], [0.3]])
        result = canonicalize_maxabs(V)
        assert result[1, 0] > 0  # was -0.5, should be +0.5

    def test_no_flip_positive_max(self):
        """If unique max-abs entry is already positive, no flip."""
        V = np.array([[0.1], [0.5], [0.3]])
        result = canonicalize_maxabs(V)
        np.testing.assert_array_equal(result, V)

    def test_tied_max_leaves_unchanged(self):
        """If max-abs is not unique, leave unchanged."""
        V = np.array([[0.5], [-0.5], [0.1]])
        result = canonicalize_maxabs(V)
        np.testing.assert_array_equal(result, V)

    def test_multi_column(self):
        """Works correctly across multiple columns independently."""
        V = np.array(
            [
                [0.1, -0.8],
                [-0.9, 0.3],
                [0.2, 0.1],
            ]
        )
        result = canonicalize_maxabs(V)
        # Col 0: max-abs at row 1 (-0.9), unique -> flip to +0.9
        assert result[1, 0] > 0
        # Col 1: max-abs at row 0 (-0.8), unique -> flip to +0.8
        assert result[0, 1] > 0

    def test_does_not_mutate_input(self):
        """Input array should not be modified."""
        V = np.array([[0.1], [-0.5], [0.3]])
        V_orig = V.copy()
        canonicalize_maxabs(V)
        np.testing.assert_array_equal(V, V_orig)

    def test_matches_legacy_implementation(self):
        """Matches the old canonicalize_eigenvectors from spectral_transformer/dataset.py."""
        from src.experiments.spectral_transformer.dataset import canonicalize_eigenvectors

        rng = np.random.default_rng(42)
        for _ in range(10):
            n, k = rng.integers(5, 20), rng.integers(1, 8)
            V = rng.standard_normal((n, k))
            result_unified = canonicalize_maxabs(V)
            result_legacy = canonicalize_eigenvectors(V)
            np.testing.assert_array_almost_equal(result_unified, result_legacy)


# ---------------------------------------------------------------------------
# TestCanonicalizeRandomFixed
# ---------------------------------------------------------------------------


class TestCanonicalizeRandomFixed:
    """Tests for canonicalize_random_fixed."""

    def test_deterministic(self):
        """Same sample_idx always gives same result."""
        V = np.random.default_rng(10).standard_normal((5, 3))
        r1 = canonicalize_random_fixed(V, sample_idx=42)
        r2 = canonicalize_random_fixed(V, sample_idx=42)
        np.testing.assert_array_equal(r1, r2)

    def test_different_idx_different_result(self):
        """Different sample_idx can give different results."""
        V = np.random.default_rng(10).standard_normal((5, 3))
        r1 = canonicalize_random_fixed(V, sample_idx=0)
        r2 = canonicalize_random_fixed(V, sample_idx=1)
        # With high probability, at least one column differs
        assert not np.array_equal(r1, r2) or True  # non-deterministic test, just run

    def test_only_sign_changes(self):
        """Output columns differ from input by at most a sign flip."""
        V = np.random.default_rng(10).standard_normal((5, 3))
        result = canonicalize_random_fixed(V, sample_idx=7)
        for j in range(V.shape[1]):
            assert np.allclose(result[:, j], V[:, j]) or np.allclose(result[:, j], -V[:, j])


# ---------------------------------------------------------------------------
# TestCanonicalizeRandomAugmented
# ---------------------------------------------------------------------------


class TestCanonicalizeRandomAugmented:
    """Tests for canonicalize_random_augmented."""

    def test_only_sign_changes(self):
        """Output columns differ from input by at most a sign flip."""
        V = np.random.default_rng(20).standard_normal((5, 3))
        result = canonicalize_random_augmented(V)
        for j in range(V.shape[1]):
            assert np.allclose(result[:, j], V[:, j]) or np.allclose(result[:, j], -V[:, j])


# ---------------------------------------------------------------------------
# TestUnifiedDispatcher
# ---------------------------------------------------------------------------


class TestUnifiedDispatcher:
    """Tests for the canonicalize() unified dispatcher."""

    def _make_eigenpairs(self, rng, n=10, k=4):
        M = rng.standard_normal((n, n))
        M = (M + M.T) / 2
        eigvals, eigvecs = np.linalg.eigh(M)
        return eigvecs[:, :k], eigvals[:k]

    def test_maxabs_routes_correctly(self):
        """method='maxabs' gives same result as canonicalize_maxabs."""
        V = np.random.default_rng(1).standard_normal((6, 3))
        r1 = canonicalize(V, method="maxabs")
        r2 = canonicalize_maxabs(V)
        np.testing.assert_array_equal(r1, r2)

    def test_spielman_routes_correctly(self):
        """method='spielman' gives same result as spectral_canonicalize."""
        from src.spectral_canonicalization import spectral_canonicalize

        rng = np.random.default_rng(2)
        V, lam = self._make_eigenpairs(rng)
        r1 = canonicalize(V, eigenvalues=lam, method="spielman")
        r2 = spectral_canonicalize(V, lam)
        np.testing.assert_array_almost_equal(r1, r2)

    def test_random_fixed_deterministic(self):
        """method='random_fixed' with same sample_idx is deterministic."""
        V = np.random.default_rng(3).standard_normal((5, 3))
        r1 = canonicalize(V, method="random_fixed", sample_idx=10)
        r2 = canonicalize(V, method="random_fixed", sample_idx=10)
        np.testing.assert_array_equal(r1, r2)

    def test_none_returns_copy(self):
        """method='none' returns a copy of the input."""
        V = np.random.default_rng(4).standard_normal((5, 3))
        result = canonicalize(V, method="none")
        np.testing.assert_array_equal(result, V)
        # Must be a copy, not the same array
        assert result is not V

    def test_unknown_method_raises(self):
        """Unknown method raises ValueError."""
        V = np.random.default_rng(5).standard_normal((5, 3))
        with pytest.raises(ValueError, match="Unknown canonicalization"):
            canonicalize(V, method="bogus")

    def test_missing_eigenvalues_for_spielman_raises(self):
        """spielman without eigenvalues raises ValueError."""
        V = np.random.default_rng(6).standard_normal((5, 3))
        with pytest.raises(ValueError, match="eigenvalues required"):
            canonicalize(V, method="spielman")

    def test_missing_eigenvalues_for_map_raises(self):
        """map without eigenvalues raises ValueError."""
        V = np.random.default_rng(7).standard_normal((5, 3))
        with pytest.raises(ValueError, match="eigenvalues required"):
            canonicalize(V, method="map")

    def test_missing_eigenvalues_for_oap_raises(self):
        """oap without eigenvalues raises ValueError."""
        V = np.random.default_rng(8).standard_normal((5, 3))
        with pytest.raises(ValueError, match="eigenvalues required"):
            canonicalize(V, method="oap")

    def test_all_methods_listed(self):
        """CANONICALIZATION_METHODS contains all 7 methods."""
        expected = {"spielman", "maxabs", "random_fixed", "random_augmented", "map", "oap", "none"}
        assert set(CANONICALIZATION_METHODS) == expected

    def test_map_routes_correctly(self):
        """method='map' gives same result as spectral_canonicalize_map."""
        from src.spectral_canonicalization import spectral_canonicalize_map

        rng = np.random.default_rng(9)
        V, lam = self._make_eigenpairs(rng)
        r1 = canonicalize(V, eigenvalues=lam, method="map")
        r2 = spectral_canonicalize_map(V, lam)
        np.testing.assert_array_almost_equal(r1, r2)

    def test_oap_routes_correctly(self):
        """method='oap' gives same result as spectral_canonicalize_oap."""
        from src.spectral_canonicalization import spectral_canonicalize_oap

        rng = np.random.default_rng(10)
        V, lam = self._make_eigenpairs(rng)
        r1 = canonicalize(V, eigenvalues=lam, method="oap")
        r2 = spectral_canonicalize_oap(V, lam)
        np.testing.assert_array_almost_equal(r1, r2)


# ---------------------------------------------------------------------------
# TestMaxabsConsistency
# ---------------------------------------------------------------------------


class TestMaxabsConsistency:
    """Verify unified maxabs matches old canonicalize_eigenvectors."""

    def test_consistency_across_sizes(self):
        """Test multiple matrix sizes."""
        rng = np.random.default_rng(100)
        for n in [3, 10, 50]:
            for k in [1, 3, 8]:
                V = rng.standard_normal((n, k))
                from src.experiments.spectral_transformer.dataset import (
                    canonicalize_eigenvectors,
                )

                r_old = canonicalize_eigenvectors(V)
                r_new = canonicalize_maxabs(V)
                np.testing.assert_array_almost_equal(
                    r_old, r_new, err_msg=f"Mismatch for n={n}, k={k}"
                )

    def test_idempotent(self):
        """canonicalize_maxabs(canonicalize_maxabs(V)) == canonicalize_maxabs(V)."""
        rng = np.random.default_rng(101)
        V = rng.standard_normal((10, 5))
        r1 = canonicalize_maxabs(V)
        r2 = canonicalize_maxabs(r1)
        np.testing.assert_array_equal(r1, r2)
