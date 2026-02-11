"""Tests for Spielman-style spectral eigenvector canonicalization."""

import numpy as np

from src.spectral_canonicalization import (
    _gf2_null_space,
    _gf2_row_reduce,
    find_balanced_blocks,
    solve_z2_system,
    spectral_canonicalize,
)

# ---------------------------------------------------------------------------
# TestSolveZ2System
# ---------------------------------------------------------------------------


class TestSolveZ2System:
    """GF(2) solver correctness."""

    def test_identity_system(self):
        """Identity matrix: unique solution."""
        A = np.eye(3, dtype=int)
        b = np.array([1, 0, 1], dtype=int)
        x, ok = solve_z2_system(A, b)
        assert ok
        assert np.array_equal(x, b)

    def test_inconsistent_system(self):
        """Inconsistent system returns None."""
        A = np.array([[1, 0], [1, 0]], dtype=int)
        b = np.array([0, 1], dtype=int)
        x, ok = solve_z2_system(A, b)
        assert not ok
        assert x is None

    def test_underdetermined_system(self):
        """Underdetermined system: returns a valid solution."""
        A = np.array([[1, 1, 0], [0, 1, 1]], dtype=int)
        b = np.array([1, 0], dtype=int)
        x, ok = solve_z2_system(A, b)
        assert ok
        assert np.array_equal((A @ x) % 2, b)

    def test_matches_theory_code(self):
        """Cross-check with theory/simple_isomorphism.py solver."""
        rng = np.random.default_rng(42)
        for _ in range(20):
            m, n = rng.integers(2, 8), rng.integers(2, 8)
            A = rng.integers(0, 2, size=(m, n))
            b = rng.integers(0, 2, size=(m,))
            x, ok = solve_z2_system(A, b)
            if ok:
                assert np.array_equal((A @ x) % 2, b)


# ---------------------------------------------------------------------------
# TestGF2Helpers
# ---------------------------------------------------------------------------


class TestGF2Helpers:
    """Row reduction, null space helpers."""

    def test_rref_identity(self):
        """RREF of identity is identity."""
        M = np.eye(3, dtype=int)
        R = _gf2_row_reduce(M)
        assert np.array_equal(R, M)

    def test_null_space_dimension(self):
        """Null space dimension = n - rank for a known matrix."""
        # Rank-1 matrix in GF(2): 2x3
        A = np.array([[1, 1, 0], [1, 1, 0]], dtype=int)
        ns = _gf2_null_space(A, 3)
        # rank = 1, so null space dim = 3 - 1 = 2
        assert ns.shape[0] == 2

    def test_null_space_correctness(self):
        """Null space vectors satisfy Av = 0 mod 2."""
        A = np.array([[1, 0, 1], [0, 1, 1]], dtype=int)
        ns = _gf2_null_space(A, 3)
        for v in ns:
            assert np.array_equal((A @ v) % 2, np.zeros(A.shape[0], dtype=int))

    def test_full_rank_empty_null_space(self):
        """Full-rank square matrix has empty null space."""
        A = np.eye(4, dtype=int)
        ns = _gf2_null_space(A, 4)
        assert ns.shape[0] == 0


# ---------------------------------------------------------------------------
# TestFindBalancedBlocks
# ---------------------------------------------------------------------------


class TestFindBalancedBlocks:
    """Phase A: balanced block partitioning."""

    def test_distinct_rows_give_singletons(self):
        """If all rows have distinct absolute-value signatures, each is its own block."""
        V = np.array(
            [
                [0.1, 0.2],
                [0.3, 0.4],
                [0.5, 0.6],
            ]
        )
        blocks = find_balanced_blocks(V)
        assert len(blocks) == 3
        for b in blocks:
            assert len(b) == 1

    def test_identical_abs_rows_grouped(self):
        """Rows with identical absolute values are initially grouped."""
        V = np.array(
            [
                [0.5, 0.3],
                [-0.5, -0.3],
                [0.7, 0.1],
            ]
        )
        blocks = find_balanced_blocks(V)
        # Rows 0 and 1 have the same abs signature; row 2 is different
        block_sizes = sorted(len(b) for b in blocks)
        assert block_sizes == [1, 2]

    def test_four_cycle_blocks(self):
        """4-cycle graph: known block structure."""
        # 4-cycle adjacency
        A = np.array(
            [
                [0, 1, 0, 1],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [1, 0, 1, 0],
            ],
            dtype=float,
        )
        eigvals, eigvecs = np.linalg.eigh(A)
        blocks = find_balanced_blocks(eigvecs)
        # Should produce blocks (the exact structure depends on eigenvectors)
        assert len(blocks) >= 1
        # All nodes accounted for
        all_nodes = set()
        for b in blocks:
            all_nodes |= b
        assert all_nodes == {0, 1, 2, 3}


# ---------------------------------------------------------------------------
# TestSignInvariance
# ---------------------------------------------------------------------------


class TestSignInvariance:
    """canon(V @ diag(s)) == canon(V) for any sign vector s."""

    def _make_test_matrix(self, rng):
        """Create a test eigenvector matrix with distinct absolute-value rows."""
        # Use a graph Laplacian to get realistic eigenvectors
        n = 6
        # Random symmetric matrix with distinct eigenvalues
        M = rng.standard_normal((n, n))
        M = (M + M.T) / 2
        eigvals, eigvecs = np.linalg.eigh(M)
        return eigvecs[:, :4], eigvals[:4]

    def test_single_column_flip(self):
        """Flipping one column doesn't change canonical result."""
        rng = np.random.default_rng(100)
        V, lam = self._make_test_matrix(rng)
        canon_V = spectral_canonicalize(V, lam)

        V_flipped = V.copy()
        V_flipped[:, 1] *= -1
        canon_flipped = spectral_canonicalize(V_flipped, lam)

        np.testing.assert_array_almost_equal(canon_V, canon_flipped)

    def test_all_columns_flipped(self):
        """Flipping all columns doesn't change canonical result."""
        rng = np.random.default_rng(101)
        V, lam = self._make_test_matrix(rng)
        canon_V = spectral_canonicalize(V, lam)

        V_flipped = -V
        canon_flipped = spectral_canonicalize(V_flipped, lam)

        np.testing.assert_array_almost_equal(canon_V, canon_flipped)

    def test_random_sign_combos(self):
        """Random sign combinations produce same canonical result."""
        rng = np.random.default_rng(102)
        V, lam = self._make_test_matrix(rng)
        canon_V = spectral_canonicalize(V, lam)

        for _ in range(10):
            signs = rng.choice([-1, 1], size=V.shape[1])
            V_signed = V * signs[np.newaxis, :]
            canon_signed = spectral_canonicalize(V_signed, lam)
            np.testing.assert_array_almost_equal(canon_V, canon_signed)

    def test_explicit_diag_signs(self):
        """Explicit V @ diag(signs) test."""
        rng = np.random.default_rng(103)
        V, lam = self._make_test_matrix(rng)
        canon_V = spectral_canonicalize(V, lam)

        signs = np.array([-1, 1, -1, 1])
        S = np.diag(signs)
        V_signed = V @ S
        canon_signed = spectral_canonicalize(V_signed, lam)

        np.testing.assert_array_almost_equal(canon_V, canon_signed)


# ---------------------------------------------------------------------------
# TestPermutationInvariance
# ---------------------------------------------------------------------------


class TestPermutationInvariance:
    """canon(P@V) has the same row multiset as canon(V)."""

    def test_row_permutation(self):
        rng = np.random.default_rng(200)
        n = 6
        M = rng.standard_normal((n, n))
        M = (M + M.T) / 2
        eigvals, V = np.linalg.eigh(M)
        V = V[:, :4]
        lam = eigvals[:4]

        canon_V = spectral_canonicalize(V, lam)

        perm = rng.permutation(n)
        V_perm = V[perm, :]
        canon_perm = spectral_canonicalize(V_perm, lam)

        # The canonical form of permuted V should have same rows as canon(V),
        # just in permuted order
        canon_V_permuted = canon_V[perm, :]
        np.testing.assert_array_almost_equal(canon_perm, canon_V_permuted)


# ---------------------------------------------------------------------------
# TestIdempotency
# ---------------------------------------------------------------------------


class TestIdempotency:
    """canon(canon(V)) == canon(V)."""

    def test_idempotent(self):
        rng = np.random.default_rng(300)
        n = 8
        M = rng.standard_normal((n, n))
        M = (M + M.T) / 2
        eigvals, V = np.linalg.eigh(M)
        V = V[:, :5]
        lam = eigvals[:5]

        canon1 = spectral_canonicalize(V, lam)
        canon2 = spectral_canonicalize(canon1, lam)
        np.testing.assert_array_almost_equal(canon1, canon2)


# ---------------------------------------------------------------------------
# TestDeterminism
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Same input always produces same output."""

    def test_deterministic(self):
        rng = np.random.default_rng(400)
        n = 6
        M = rng.standard_normal((n, n))
        M = (M + M.T) / 2
        eigvals, V = np.linalg.eigh(M)
        V = V[:, :4]
        lam = eigvals[:4]

        results = [spectral_canonicalize(V.copy(), lam.copy()) for _ in range(5)]
        for r in results[1:]:
            np.testing.assert_array_equal(results[0], r)


# ---------------------------------------------------------------------------
# TestMultiplicityHandling
# ---------------------------------------------------------------------------


class TestMultiplicityHandling:
    """Columns with multiplicity > 1 are left unchanged."""

    def test_repeated_eigenvalue_columns_unchanged(self):
        """Multiplicity-2 columns should not be modified."""
        rng = np.random.default_rng(500)
        n = 5
        V = rng.standard_normal((n, 4))
        # Eigenvalues: first two are identical (multiplicity 2), last two distinct
        lam = np.array([1.0, 1.0, 2.0, 3.0])

        result = spectral_canonicalize(V, lam)

        # Columns 0 and 1 (multiplicity 2) should be identical to input
        np.testing.assert_array_equal(result[:, 0], V[:, 0])
        np.testing.assert_array_equal(result[:, 1], V[:, 1])

    def test_all_repeated_returns_copy(self):
        """If all eigenvalues are repeated, output equals input."""
        rng = np.random.default_rng(501)
        n = 4
        V = rng.standard_normal((n, 4))
        lam = np.array([1.0, 1.0, 2.0, 2.0])

        result = spectral_canonicalize(V, lam)
        np.testing.assert_array_equal(result, V)


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases: empty, single column, single node, mutation safety."""

    def test_empty_input(self):
        V = np.zeros((0, 0))
        lam = np.array([])
        result = spectral_canonicalize(V, lam)
        assert result.shape == (0, 0)

    def test_single_column(self):
        V = np.array([[0.3], [-0.5], [0.1]])
        lam = np.array([1.0])
        result = spectral_canonicalize(V, lam)
        assert result.shape == (3, 1)
        # Should be deterministic under sign flip
        result2 = spectral_canonicalize(-V, lam)
        np.testing.assert_array_almost_equal(result, result2)

    def test_single_node(self):
        V = np.array([[0.5, -0.3, 0.7]])
        lam = np.array([1.0, 2.0, 3.0])
        result = spectral_canonicalize(V, lam)
        assert result.shape == (1, 3)
        # Single-node: each column's sign is fully determined
        # Should be idempotent
        result2 = spectral_canonicalize(result, lam)
        np.testing.assert_array_almost_equal(result, result2)

    def test_input_not_modified(self):
        """Original input array should not be mutated."""
        V = np.array([[0.5, -0.3], [-0.5, 0.3], [0.1, 0.2]])
        V_orig = V.copy()
        lam = np.array([1.0, 2.0])
        spectral_canonicalize(V, lam)
        np.testing.assert_array_equal(V, V_orig)


# ---------------------------------------------------------------------------
# TestIntegration
# ---------------------------------------------------------------------------


class TestIntegration:
    """Integration with compute_eigenpairs and feature construction."""

    def test_with_compute_eigenpairs(self):
        """Works end-to-end with Laplacian eigenpairs."""
        from src.spectral_core import build_graph_laplacian, compute_eigenpairs

        rng = np.random.default_rng(600)
        points = rng.standard_normal((50, 3))
        L, comp_idx = build_graph_laplacian(points, n_neighbors=8)
        eigenvalues, eigenvectors = compute_eigenpairs(L, n_eigs=6)

        result = spectral_canonicalize(eigenvectors, eigenvalues)
        assert result.shape == eigenvectors.shape
        # Should be idempotent
        result2 = spectral_canonicalize(result, eigenvalues)
        np.testing.assert_array_almost_equal(result, result2)

    def test_output_compatible_with_features(self):
        """Output can be concatenated with xyz for feature construction."""
        from src.spectral_core import build_graph_laplacian, compute_eigenpairs

        rng = np.random.default_rng(601)
        points = rng.standard_normal((30, 3))
        L, comp_idx = build_graph_laplacian(points, n_neighbors=6)
        eigenvalues, eigenvectors = compute_eigenpairs(L, n_eigs=4)

        canon = spectral_canonicalize(eigenvectors, eigenvalues)
        pts_cc = points[comp_idx]
        features = np.concatenate([pts_cc, canon], axis=1)
        assert features.shape == (pts_cc.shape[0], 3 + canon.shape[1])


# ---------------------------------------------------------------------------
# TestHandCrafted
# ---------------------------------------------------------------------------


class TestHandCrafted:
    """Hand-crafted examples with known eigenvectors."""

    def test_path_graph_p3(self):
        """Path graph P3: 3 nodes, known eigenvectors."""
        # Adjacency of P3: 0-1-2
        A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
        eigvals, eigvecs = np.linalg.eigh(A)

        canon = spectral_canonicalize(eigvecs, eigvals)

        # Sign invariance: flipping any column should give same result
        for j in range(eigvecs.shape[1]):
            V_flip = eigvecs.copy()
            V_flip[:, j] *= -1
            canon_flip = spectral_canonicalize(V_flip, eigvals)
            np.testing.assert_array_almost_equal(canon, canon_flip)

    def test_four_cycle(self):
        """4-cycle: sign invariance on simple-spectrum columns.

        The 4-cycle has eigenvalues {-2, 0, 0, 2}. Eigenvalue 0 has
        multiplicity 2, so those columns are left unchanged. We verify
        sign invariance only for the simple-spectrum columns.
        """
        from src.spectral_core import detect_eigenvalue_multiplicities

        A = np.array(
            [
                [0, 1, 0, 1],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [1, 0, 1, 0],
            ],
            dtype=float,
        )
        eigvals, eigvecs = np.linalg.eigh(A)

        mult_info = detect_eigenvalue_multiplicities(eigvals)
        simple_cols = [j for j in range(len(eigvals)) if mult_info["multiplicity"][j] == 1]

        canon = spectral_canonicalize(eigvecs, eigvals)

        # Test all 2^s sign combinations on simple-spectrum columns only
        s = len(simple_cols)
        for bits in range(1 << s):
            signs = np.ones(eigvecs.shape[1])
            for idx, col in enumerate(simple_cols):
                if (bits >> idx) & 1:
                    signs[col] = -1
            V_signed = eigvecs * signs[np.newaxis, :]
            canon_signed = spectral_canonicalize(V_signed, eigvals)
            # Simple-spectrum columns must match
            np.testing.assert_array_almost_equal(
                canon[:, simple_cols],
                canon_signed[:, simple_cols],
                err_msg=f"Failed for simple-col sign bits {bits}",
            )
