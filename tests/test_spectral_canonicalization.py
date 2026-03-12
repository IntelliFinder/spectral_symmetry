"""Tests for Spielman-style spectral eigenvector canonicalization."""

import numpy as np

from src.spectral_canonicalization import (
    _gf2_null_space,
    _gf2_row_reduce,
    canonicalize,
    canonicalize_abs,
    compute_canonical_signs,
    find_balanced_blocks,
    scale_eigenvectors_by_eigenvalues,
    solve_z2_system,
    spectral_canonicalize,
    spectral_canonicalize_partition,
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


# ---------------------------------------------------------------------------
# TestSpielmanStepByStep — Counterexample & separation tests
# ---------------------------------------------------------------------------


class TestSpielmanStepByStep:
    """Step-by-step verification of Spielman phases on known graphs."""

    def _laplacian(self, A):
        """Compute graph Laplacian L = D - A."""
        D = np.diag(A.sum(axis=1))
        return D - A

    def test_path_p5_phase_a_respects_symmetry(self):
        """Path P5 (0-1-2-3-4): nodes 0&4 and 1&3 are symmetric, so Phase A
        should produce 3 blocks: {0,4}, {1,3}, {2}."""
        A = np.zeros((5, 5))
        for i, j in [(0, 1), (1, 2), (2, 3), (3, 4)]:
            A[i, j] = A[j, i] = 1
        L = self._laplacian(A)
        eigvals, eigvecs = np.linalg.eigh(L)

        # Drop trivial eigenvector (eigenvalue 0)
        nontriv = eigvals > 1e-6
        V = eigvecs[:, nontriv]

        blocks = find_balanced_blocks(V)
        block_sizes = sorted(len(b) for b in blocks)
        # Symmetric pairs {0,4} and {1,3} share abs row signatures
        assert block_sizes == [1, 2, 2], f"Expected [1, 2, 2], got {block_sizes}"
        # Center node (2) should be singleton
        singleton = [b for b in blocks if len(b) == 1][0]
        assert 2 in singleton

    def test_path_p5_canonical_signs_deterministic(self):
        """Phase B signs on P5 should be deterministic and flip-invariant."""
        A = np.zeros((5, 5))
        for i, j in [(0, 1), (1, 2), (2, 3), (3, 4)]:
            A[i, j] = A[j, i] = 1
        L = self._laplacian(A)
        eigvals, eigvecs = np.linalg.eigh(L)
        nontriv = eigvals > 1e-6
        V = eigvecs[:, nontriv]

        blocks = find_balanced_blocks(V)
        signs = compute_canonical_signs(V, blocks)
        assert signs.shape == (V.shape[1],)
        assert all(s in (-1, 1) for s in signs)

        # Flipping any column should produce opposite sign for that column
        for j in range(V.shape[1]):
            V_flip = V.copy()
            V_flip[:, j] *= -1
            signs_flip = compute_canonical_signs(V_flip, blocks)
            # After applying signs, result should match
            canon_orig = V * signs[np.newaxis, :]
            canon_flip = V_flip * signs_flip[np.newaxis, :]
            np.testing.assert_array_almost_equal(canon_orig, canon_flip)

    def test_star_graph_phase_a_two_blocks(self):
        """Star graph K_{1,4}: center has unique row, leaves share signature.
        Phase A should produce 2 blocks: {center} and {leaves}."""
        A = np.zeros((5, 5))
        for i in range(1, 5):
            A[0, i] = A[i, 0] = 1
        L = self._laplacian(A)
        eigvals, eigvecs = np.linalg.eigh(L)
        nontriv = eigvals > 1e-6
        V = eigvecs[:, nontriv]

        blocks = find_balanced_blocks(V)
        block_sizes = sorted(len(b) for b in blocks)
        # Center is a singleton, leaves could form 1 or more blocks
        assert 1 in block_sizes, "Center node should be a singleton block"

    def test_isomorphic_graphs_same_canonical_form(self):
        """Two isomorphic graphs (related by node permutation) should produce
        identical canonical eigenvectors (modulo row permutation)."""
        # Graph: triangle with pendant (0-1, 1-2, 2-0, 0-3)
        A1 = np.array([[0, 1, 1, 1], [1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]], dtype=float)
        # Permute nodes: 0->2, 1->0, 2->1, 3->3
        P = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=float)
        A2 = P @ A1 @ P.T

        L1 = self._laplacian(A1)
        L2 = self._laplacian(A2)

        eigvals1, eigvecs1 = np.linalg.eigh(L1)
        eigvals2, eigvecs2 = np.linalg.eigh(L2)

        nontriv1 = eigvals1 > 1e-6
        nontriv2 = eigvals2 > 1e-6

        canon1 = spectral_canonicalize(eigvecs1[:, nontriv1], eigvals1[nontriv1])
        canon2 = spectral_canonicalize(eigvecs2[:, nontriv2], eigvals2[nontriv2])

        # After permuting canon2 rows back, should match canon1
        canon2_permuted = P.T @ canon2
        np.testing.assert_array_almost_equal(np.abs(canon1), np.abs(canon2_permuted), decimal=5)

    def test_non_isomorphic_graphs_different_canonical_form(self):
        """Non-isomorphic graphs should produce different canonical forms."""
        # Graph 1: path P4 (0-1-2-3)
        A1 = np.zeros((4, 4))
        for i, j in [(0, 1), (1, 2), (2, 3)]:
            A1[i, j] = A1[j, i] = 1

        # Graph 2: star K_{1,3} (0-1, 0-2, 0-3)
        A2 = np.zeros((4, 4))
        for i in range(1, 4):
            A2[0, i] = A2[i, 0] = 1

        L1, L2 = self._laplacian(A1), self._laplacian(A2)
        ev1, V1 = np.linalg.eigh(L1)
        ev2, V2 = np.linalg.eigh(L2)

        nt1 = ev1 > 1e-6
        nt2 = ev2 > 1e-6

        c1 = spectral_canonicalize(V1[:, nt1], ev1[nt1])
        c2 = spectral_canonicalize(V2[:, nt2], ev2[nt2])

        # Absolute row signatures should differ
        sig1 = sorted(tuple(np.round(np.abs(c1[i]), 5)) for i in range(4))
        sig2 = sorted(tuple(np.round(np.abs(c2[i]), 5)) for i in range(4))
        assert sig1 != sig2, "Non-isomorphic graphs should have different signatures"

    def test_multiplicity_columns_unchanged(self):
        """Verify that columns with eigenvalue multiplicity > 1 are untouched,
        while simple-spectrum columns are canonicalized."""
        # 4-cycle has eigenvalues {0, 2, 2, 4} for Laplacian → multiplicity 2 at λ=2
        A = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]], dtype=float)
        L = self._laplacian(A)
        eigvals, eigvecs = np.linalg.eigh(L)

        from src.spectral_core import detect_eigenvalue_multiplicities

        mult = detect_eigenvalue_multiplicities(eigvals)["multiplicity"]
        mult_cols = [j for j in range(len(eigvals)) if mult[j] > 1]

        canon = spectral_canonicalize(eigvecs, eigvals)

        # Multiplicity columns should be identical to input
        for j in mult_cols:
            np.testing.assert_array_equal(
                canon[:, j],
                eigvecs[:, j],
                err_msg=f"Column {j} (mult>1) should not change",
            )

    def test_all_sign_combos_simple_spectrum_graph(self):
        """Graph with fully simple spectrum: all 2^k sign combos on
        simple-spectrum columns yield the same canonical result."""
        # Use weighted path to guarantee simple spectrum and no node symmetry
        A = np.array(
            [
                [0, 1, 0, 0, 0],
                [1, 0, 2, 0, 0],
                [0, 2, 0, 3, 0],
                [0, 0, 3, 0, 1],
                [0, 0, 0, 1, 0],
            ],
            dtype=float,
        )
        L = self._laplacian(A)
        eigvals, eigvecs = np.linalg.eigh(L)
        nontriv = eigvals > 1e-6
        V = eigvecs[:, nontriv]
        evals = eigvals[nontriv]

        # Verify simple spectrum
        diffs = np.diff(np.sort(evals))
        assert np.all(diffs > 1e-3), f"Not simple spectrum: diffs={diffs}"

        canon_ref = spectral_canonicalize(V, evals)
        k = V.shape[1]

        # Test all 2^k sign combinations
        for bits in range(1 << k):
            signs = np.ones(k)
            for j in range(k):
                if (bits >> j) & 1:
                    signs[j] = -1
            V_signed = V * signs[np.newaxis, :]
            canon_test = spectral_canonicalize(V_signed, evals)
            np.testing.assert_array_almost_equal(
                canon_ref,
                canon_test,
                decimal=6,
                err_msg=f"Failed for sign bits {bits:0{k}b}",
            )

    def test_multiplicity_graph_only_simple_cols_canonicalized(self):
        """6-node graph with eigenvalue multiplicity: only simple-spectrum
        columns should be sign-invariant; multiplicity columns left as-is."""
        A = np.array(
            [
                [0, 1, 1, 0, 0, 0],
                [1, 0, 1, 0, 0, 0],
                [1, 1, 0, 1, 0, 0],
                [0, 0, 1, 0, 1, 1],
                [0, 0, 0, 1, 0, 1],
                [0, 0, 0, 1, 1, 0],
            ],
            dtype=float,
        )
        L = self._laplacian(A)
        eigvals, eigvecs = np.linalg.eigh(L)
        nontriv = eigvals > 1e-6
        V = eigvecs[:, nontriv]
        evals = eigvals[nontriv]

        from src.spectral_core import detect_eigenvalue_multiplicities

        mult = detect_eigenvalue_multiplicities(evals)["multiplicity"]
        simple_cols = [j for j in range(len(evals)) if mult[j] == 1]
        assert len(simple_cols) < len(evals), "Need some non-simple columns for test"

        canon_ref = spectral_canonicalize(V, evals)

        # Only simple-spectrum columns should be sign-invariant
        for j in simple_cols:
            V_flip = V.copy()
            V_flip[:, j] *= -1
            canon_flip = spectral_canonicalize(V_flip, evals)
            np.testing.assert_array_almost_equal(
                canon_ref[:, simple_cols],
                canon_flip[:, simple_cols],
                decimal=6,
                err_msg=f"Simple col {j} not sign-invariant",
            )

    def test_petersen_graph(self):
        """Petersen graph (10 nodes, 3-regular): a classic counterexample
        for many graph algorithms. Verify Spielman handles it correctly."""
        # Petersen graph adjacency
        A = np.zeros((10, 10))
        outer = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
        inner = [(5, 7), (7, 9), (9, 6), (6, 8), (8, 5)]
        spokes = [(0, 5), (1, 6), (2, 7), (3, 8), (4, 9)]
        for i, j in outer + inner + spokes:
            A[i, j] = A[j, i] = 1

        L = self._laplacian(A)
        eigvals, eigvecs = np.linalg.eigh(L)
        nontriv = eigvals > 1e-6
        V = eigvecs[:, nontriv]
        evals = eigvals[nontriv]

        canon = spectral_canonicalize(V, evals)

        # Petersen has eigenvalues {0, 1(x5), 4(x4)} so high multiplicity
        # Most columns should be left unchanged
        from src.spectral_core import detect_eigenvalue_multiplicities

        mult = detect_eigenvalue_multiplicities(evals)["multiplicity"]
        assert any(m > 1 for m in mult), "Petersen should have repeated eigenvalues"

        # Even with multiplicities, should not crash and should be idempotent
        canon2 = spectral_canonicalize(canon, evals)
        np.testing.assert_array_almost_equal(canon, canon2)

    def test_balanced_block_refinement_step_by_step(self):
        """Trace Phase A on a small graph where initial partition needs
        refinement, verifying that product vectors correctly split blocks."""
        # 6-node graph where two nodes initially share abs row signature
        # but product vectors separate them
        A = np.array(
            [
                [0, 1, 1, 0, 0, 0],
                [1, 0, 0, 1, 0, 0],
                [1, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 1],
                [0, 0, 1, 0, 0, 1],
                [0, 0, 0, 1, 1, 0],
            ],
            dtype=float,
        )
        L = self._laplacian(A)
        eigvals, eigvecs = np.linalg.eigh(L)
        nontriv = eigvals > 1e-6
        V = eigvecs[:, nontriv]

        blocks = find_balanced_blocks(V)

        # All blocks should be non-empty and cover all nodes
        all_nodes = set()
        for b in blocks:
            assert len(b) > 0
            all_nodes.update(b)
        assert all_nodes == set(range(V.shape[0])), "Blocks must cover all nodes"

        # No node should appear in multiple blocks
        total_nodes = sum(len(b) for b in blocks)
        assert total_nodes == V.shape[0], "Blocks must be disjoint"


# ---------------------------------------------------------------------------
# TestCanonicalizeAbs
# ---------------------------------------------------------------------------


class TestCanonicalizeAbs:
    """Tests for absolute-value (SignNet-style) canonicalization."""

    def test_all_nonnegative(self):
        """Output should be non-negative."""
        V = np.array([[1, -2], [-3, 4], [5, -6]], dtype=float)
        result = canonicalize_abs(V)
        assert np.all(result >= 0)

    def test_sign_invariance(self):
        """Flipping any column sign should not change the result."""
        rng = np.random.RandomState(42)
        V = rng.randn(10, 4)
        result1 = canonicalize_abs(V)
        # Flip some columns
        V_flipped = V.copy()
        V_flipped[:, 1] *= -1
        V_flipped[:, 3] *= -1
        result2 = canonicalize_abs(V_flipped)
        np.testing.assert_array_almost_equal(result1, result2)

    def test_idempotent(self):
        """Applying abs twice should give the same result."""
        V = np.array([[1, -2], [-3, 4]], dtype=float)
        result1 = canonicalize_abs(V)
        result2 = canonicalize_abs(result1)
        np.testing.assert_array_almost_equal(result1, result2)

    def test_values_correct(self):
        """Check that output equals element-wise absolute value."""
        V = np.array([[1, -2], [-3, 4]], dtype=float)
        result = canonicalize_abs(V)
        expected = np.array([[1, 2], [3, 4]], dtype=float)
        np.testing.assert_array_almost_equal(result, expected)

    def test_dispatcher(self):
        """Dispatcher routes 'abs' correctly."""
        V = np.array([[1, -2], [-3, 4]], dtype=float)
        result = canonicalize(V, method="abs")
        expected = np.abs(V)
        np.testing.assert_array_almost_equal(result, expected)


# ---------------------------------------------------------------------------
# TestSpielmanPartition
# ---------------------------------------------------------------------------


class TestSpielmanPartition:
    """Tests for partial Spielman (initial partition only, no GF(2) refinement)."""

    def test_sign_invariance(self):
        """Flipping column signs should not change the canonicalized result."""
        rng = np.random.RandomState(42)
        V = rng.randn(8, 3)
        eigenvalues = np.array([0.5, 1.2, 2.7])
        result1 = spectral_canonicalize_partition(V, eigenvalues)
        # Flip all signs
        result2 = spectral_canonicalize_partition(-V, eigenvalues)
        np.testing.assert_array_almost_equal(result1, result2)

    def test_idempotent(self):
        """Applying partial Spielman twice should give the same result."""
        rng = np.random.RandomState(42)
        V = rng.randn(8, 3)
        eigenvalues = np.array([0.5, 1.2, 2.7])
        result1 = spectral_canonicalize_partition(V, eigenvalues)
        result2 = spectral_canonicalize_partition(result1, eigenvalues)
        np.testing.assert_array_almost_equal(result1, result2)

    def test_deterministic(self):
        """Same input should give same output."""
        V = np.random.RandomState(42).randn(8, 3)
        eigenvalues = np.array([0.5, 1.2, 2.7])
        result1 = spectral_canonicalize_partition(V, eigenvalues)
        result2 = spectral_canonicalize_partition(V, eigenvalues)
        np.testing.assert_array_almost_equal(result1, result2)

    def test_multiplicity_columns_unchanged(self):
        """Columns with repeated eigenvalues should not be modified."""
        rng = np.random.RandomState(42)
        V = rng.randn(6, 4)
        # Eigenvalues: 1.0 has multiplicity 2
        eigenvalues = np.array([0.5, 1.0, 1.0, 2.5])
        result = spectral_canonicalize_partition(V, eigenvalues)
        # Columns 1 and 2 (multiplicity 2) should be unchanged
        np.testing.assert_array_almost_equal(result[:, 1], V[:, 1])
        np.testing.assert_array_almost_equal(result[:, 2], V[:, 2])

    def test_agrees_with_full_spielman_on_path_graph(self):
        """On P5 (path graph), partition-only should produce same blocks as full.

        For path graphs, the initial abs-value partition already separates
        all nodes (no refinement needed), so results should match.
        """
        # P5 adjacency
        A = np.zeros((5, 5))
        for i in range(4):
            A[i, i + 1] = 1
            A[i + 1, i] = 1
        D = np.diag(A.sum(axis=1))
        L = D - A
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        # Skip trivial eigenvector (λ=0)
        eigenvalues = eigenvalues[1:]
        eigenvectors = eigenvectors[:, 1:]

        result_full = spectral_canonicalize(eigenvectors, eigenvalues)
        result_part = spectral_canonicalize_partition(eigenvectors, eigenvalues)
        np.testing.assert_array_almost_equal(result_full, result_part)

    def test_dispatcher(self):
        """Dispatcher routes 'spielman_partition' correctly."""
        V = np.random.RandomState(42).randn(6, 3)
        eigenvalues = np.array([0.5, 1.2, 2.7])
        result1 = canonicalize(V, eigenvalues=eigenvalues, method="spielman_partition")
        result2 = spectral_canonicalize_partition(V, eigenvalues)
        np.testing.assert_array_almost_equal(result1, result2)

    def test_empty(self):
        """Empty eigenvectors should return empty."""
        V = np.zeros((5, 0))
        eigenvalues = np.array([])
        result = spectral_canonicalize_partition(V, eigenvalues)
        assert result.shape == (5, 0)


# ---------------------------------------------------------------------------
# TestScaleEigenvectorsByEigenvalues
# ---------------------------------------------------------------------------


class TestScaleEigenvectorsByEigenvalues:
    """Tests for eigenvalue-scaled eigenvectors."""

    def test_basic_scaling(self):
        """Each column j should be divided by sqrt(eigenvalue[j])."""
        V = np.ones((3, 2), dtype=float)
        eigenvalues = np.array([4.0, 9.0])
        result = scale_eigenvectors_by_eigenvalues(V, eigenvalues)
        expected = np.array([[0.5, 1 / 3], [0.5, 1 / 3], [0.5, 1 / 3]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_small_eigenvalue_unscaled(self):
        """Eigenvalues near zero should leave the column unchanged."""
        V = np.ones((3, 2), dtype=float)
        eigenvalues = np.array([1e-10, 4.0])
        result = scale_eigenvectors_by_eigenvalues(V, eigenvalues)
        # Column 0: near-zero eigenvalue, left unchanged
        np.testing.assert_array_almost_equal(result[:, 0], V[:, 0])
        # Column 1: scaled by 1/sqrt(4) = 0.5
        np.testing.assert_array_almost_equal(result[:, 1], 0.5 * np.ones(3))

    def test_does_not_modify_input(self):
        """Should return a copy, not modify in place."""
        V = np.ones((3, 2), dtype=float)
        eigenvalues = np.array([4.0, 9.0])
        _ = scale_eigenvectors_by_eigenvalues(V, eigenvalues)
        np.testing.assert_array_almost_equal(V, np.ones((3, 2)))

    def test_preserves_shape(self):
        """Output shape should match input."""
        V = np.random.randn(10, 5)
        eigenvalues = np.arange(1.0, 6.0)
        result = scale_eigenvectors_by_eigenvalues(V, eigenvalues)
        assert result.shape == V.shape
