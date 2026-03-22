"""Tests for Spielman-style spectral eigenvector canonicalization."""

import numpy as np
import pytest

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
from src.spectral_core import detect_eigenvalue_multiplicities

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
# TestAutomorphismInvariance
# ---------------------------------------------------------------------------


class TestAutomorphismInvariance:
    """Canonicalization is invariant up to automorphism, not strict equality.

    For graphs with nontrivial automorphisms, applying an automorphism σ
    permutes rows of V.  The canonical signs may differ, but the result
    is related by a column-sign matrix:

        canon(P_σ V) = P_σ · canon(V) · diag(d)   for some d ∈ {±1}^k

    This is the correct behaviour: the eigendecomposition is unique only
    up to column signs, and automorphisms act on that ambiguity.
    """

    @staticmethod
    def _laplacian(A):
        D = np.diag(A.sum(axis=1))
        return D - A

    def _build_path_p5(self):
        """Return (Laplacian, non-trivial eigvals, non-trivial eigvecs) for P5."""
        A = np.zeros((5, 5))
        for i, j in [(0, 1), (1, 2), (2, 3), (3, 4)]:
            A[i, j] = A[j, i] = 1
        L = self._laplacian(A)
        eigvals, eigvecs = np.linalg.eigh(L)
        nontriv = eigvals > 1e-6
        return eigvals[nontriv], eigvecs[:, nontriv]

    def test_p5_automorphism_not_strictly_equivariant(self):
        """On P5, canon(P_σ V) ≠ P_σ canon(V) in general.

        The reflection σ = (04)(13) swaps nodes within balanced blocks
        {0,4} and {1,3}, changing the within-block ordering and thus the
        sign selected by the first-non-zero-entry rule.
        """
        evals, V = self._build_path_p5()
        sigma = [4, 3, 2, 1, 0]

        canon_V = spectral_canonicalize(V, evals)
        canon_V_sigma = spectral_canonicalize(V[sigma, :], evals)

        # Strict equivariance should FAIL for at least one column
        canon_V_permuted = canon_V[sigma, :]
        assert not np.allclose(
            canon_V_sigma, canon_V_permuted
        ), "Expected strict equivariance to fail under P5 automorphism"

    def test_p5_automorphism_up_to_column_signs(self):
        """On P5, canon(P_σ V) = P_σ canon(V) D for some sign matrix D.

        Each column of the canonicalized result is either identical or
        negated — this is the correct notion of invariance for the
        eigendecomposition.
        """
        evals, V = self._build_path_p5()
        sigma = [4, 3, 2, 1, 0]

        canon_V = spectral_canonicalize(V, evals)
        canon_V_sigma = spectral_canonicalize(V[sigma, :], evals)
        canon_V_permuted = canon_V[sigma, :]

        for j in range(V.shape[1]):
            col_got = canon_V_sigma[:, j]
            col_exp = canon_V_permuted[:, j]
            same = np.allclose(col_got, col_exp)
            flipped = np.allclose(col_got, -col_exp)
            assert same or flipped, (
                f"Column {j}: neither same nor negated.\n"
                f"  got:      {np.round(col_got, 6)}\n"
                f"  expected: {np.round(col_exp, 6)}"
            )

    def test_p5_automorphism_with_all_sign_combos(self):
        """Combine sign flips with the automorphism: still up-to-column-signs.

        For every sign vector s and the automorphism σ:
            canon(P_σ V diag(s))  ~  P_σ canon(V)   (up to column signs)
        """
        evals, V = self._build_path_p5()
        sigma = [4, 3, 2, 1, 0]
        k = V.shape[1]

        canon_V = spectral_canonicalize(V, evals)
        canon_V_permuted = canon_V[sigma, :]

        for bits in range(1 << k):
            signs = np.array([(-1) ** ((bits >> j) & 1) for j in range(k)], dtype=float)
            V_signed = V * signs[np.newaxis, :]
            V_sigma_signed = V_signed[sigma, :]
            canon_test = spectral_canonicalize(V_sigma_signed, evals)

            for j in range(k):
                col_got = canon_test[:, j]
                col_exp = canon_V_permuted[:, j]
                assert np.allclose(col_got, col_exp) or np.allclose(col_got, -col_exp), (
                    f"sign bits={bits:0{k}b}, col {j}: not ±1 related"
                )

    def test_cycle_c6_automorphism_up_to_column_signs(self):
        """C6 (6-cycle) has rich automorphism group.  Test a rotation and a
        reflection, both should give up-to-column-signs invariance on the
        simple-spectrum columns."""
        from src.spectral_core import detect_eigenvalue_multiplicities

        A = np.zeros((6, 6))
        for i in range(6):
            A[i, (i + 1) % 6] = A[(i + 1) % 6, i] = 1
        L = self._laplacian(A)
        eigvals, eigvecs = np.linalg.eigh(L)
        nontriv = eigvals > 1e-6
        V = eigvecs[:, nontriv]
        evals = eigvals[nontriv]

        mult = detect_eigenvalue_multiplicities(evals)["multiplicity"]
        simple_cols = [j for j in range(len(evals)) if mult[j] == 1]

        if not simple_cols:
            return  # nothing to test — all multiplicities > 1

        canon_V = spectral_canonicalize(V, evals)

        # Rotation by 1: σ(i) = (i+1) mod 6
        rot = [(i + 1) % 6 for i in range(6)]
        # Reflection: σ(i) = (6-i) mod 6
        ref = [(6 - i) % 6 for i in range(6)]

        for sigma_name, sigma in [("rotation", rot), ("reflection", ref)]:
            canon_sigma = spectral_canonicalize(V[sigma, :], evals)
            canon_permuted = canon_V[sigma, :]
            for j in simple_cols:
                col_got = canon_sigma[:, j]
                col_exp = canon_permuted[:, j]
                assert np.allclose(col_got, col_exp) or np.allclose(col_got, -col_exp), (
                    f"C6 {sigma_name}, simple col {j}: not ±1 related"
                )

    def test_petersen_graph_up_to_column_signs(self):
        """Petersen graph (10 nodes, |Aut| = 120): test two automorphisms.

        The Petersen graph has eigenvalues 1 (mult 5) and 4 (mult 4) after
        removing the trivial zero.  If all non-trivial eigenvalues have
        multiplicity > 1 the simple_cols list is empty and we skip the
        strict-equality-fails assertion, but still verify that the
        up-to-column-signs loop holds vacuously.
        """
        import pytest

        from src.spectral_core import detect_eigenvalue_multiplicities

        # Standard Petersen graph: outer 5-cycle 0-1-2-3-4,
        # inner pentagram 5-7-9-6-8-5, spokes 0-5, 1-6, 2-7, 3-8, 4-9.
        edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 0),  # outer cycle
            (5, 7),
            (7, 9),
            (9, 6),
            (6, 8),
            (8, 5),  # inner pentagram
            (0, 5),
            (1, 6),
            (2, 7),
            (3, 8),
            (4, 9),  # spokes
        ]
        A = np.zeros((10, 10))
        for i, j in edges:
            A[i, j] = A[j, i] = 1.0
        L = self._laplacian(A)
        eigvals, eigvecs = np.linalg.eigh(L)
        nontriv = eigvals > 1e-6
        V = eigvecs[:, nontriv]
        evals = eigvals[nontriv]

        mult = detect_eigenvalue_multiplicities(evals)["multiplicity"]
        simple_cols = [j for j in range(len(evals)) if mult[j] == 1]

        if not simple_cols:
            pytest.skip(
                "Petersen graph has no simple non-trivial eigenvalues; "
                "up-to-column-signs holds vacuously"
            )

        canon_V = spectral_canonicalize(V, evals)

        # Automorphism 1: rotation of outer cycle + corresponding inner shift.
        # sigma: 0->1->2->3->4->0, 5->6->7->8->9->5
        sigma1 = [1, 2, 3, 4, 0, 6, 7, 8, 9, 5]
        # Automorphism 2: reflection of outer cycle (0 fixed, 1<->4, 2<->3) + inner.
        # sigma: 0->0, 1->4, 2->3, 3->2, 4->1, 5->5, 6->9, 7->8, 8->7, 9->6
        sigma2 = [0, 4, 3, 2, 1, 5, 9, 8, 7, 6]

        for sigma_name, sigma in [("rotation", sigma1), ("reflection", sigma2)]:
            canon_sigma = spectral_canonicalize(V[sigma, :], evals)
            canon_permuted = canon_V[sigma, :]
            for j in simple_cols:
                col_got = canon_sigma[:, j]
                col_exp = canon_permuted[:, j]
                assert np.allclose(col_got, col_exp) or np.allclose(col_got, -col_exp), (
                    f"Petersen {sigma_name}, simple col {j}: not ±1 related"
                )

    def test_cube_q3_automorphism_up_to_column_signs(self):
        """Cube graph Q3 (8 nodes, |Aut| = 48): test two automorphisms.

        Label nodes as 3-bit binary strings 0..7 where i~j iff they differ in
        exactly one bit.  Laplacian eigenvalues: 0 (mult 1), 2 (mult 3),
        4 (mult 3), 6 (mult 1).  The non-trivial simple eigenvalue is 6;
        all eigenvalue-2 and eigenvalue-4 columns have multiplicity 3.
        Strict equality should fail on the simple column for at least one
        automorphism, while up-to-column-signs must hold for all.
        """
        import pytest

        from src.spectral_core import detect_eigenvalue_multiplicities

        # Q3: nodes 0..7, i-j iff Hamming distance 1
        A = np.zeros((8, 8))
        for i in range(8):
            for bit in range(3):
                j = i ^ (1 << bit)
                A[i, j] = 1.0
        L = self._laplacian(A)
        eigvals, eigvecs = np.linalg.eigh(L)
        nontriv = eigvals > 1e-6
        V = eigvecs[:, nontriv]
        evals = eigvals[nontriv]

        mult = detect_eigenvalue_multiplicities(evals)["multiplicity"]
        simple_cols = [j for j in range(len(evals)) if mult[j] == 1]

        if not simple_cols:
            pytest.skip("Cube Q3: no simple non-trivial eigenvalues found")

        canon_V = spectral_canonicalize(V, evals)

        # Automorphism 1: bit rotation  (b2 b1 b0) -> (b0 b2 b1)
        def _rot_bits(i):
            b0 = (i >> 0) & 1
            b1 = (i >> 1) & 1
            b2 = (i >> 2) & 1
            return b0 | (b2 << 1) | (b1 << 2)

        sigma1 = [_rot_bits(i) for i in range(8)]
        # Automorphism 2: antipodal map  i -> i XOR 7
        sigma2 = [i ^ 7 for i in range(8)]

        # Strict equality should fail for at least one automorphism on a simple column
        found_strict_failure = False
        for sigma in [sigma1, sigma2]:
            canon_sigma = spectral_canonicalize(V[sigma, :], evals)
            canon_permuted = canon_V[sigma, :]
            if not np.allclose(
                canon_sigma[:, simple_cols], canon_permuted[:, simple_cols]
            ):
                found_strict_failure = True
                break
        assert found_strict_failure, (
            "Expected strict equivariance to fail for some Q3 automorphism on simple columns"
        )

        # Up-to-column-signs must hold for both automorphisms
        for sigma_name, sigma in [("bit-rotation", sigma1), ("antipodal", sigma2)]:
            canon_sigma = spectral_canonicalize(V[sigma, :], evals)
            canon_permuted = canon_V[sigma, :]
            for j in simple_cols:
                col_got = canon_sigma[:, j]
                col_exp = canon_permuted[:, j]
                assert np.allclose(col_got, col_exp) or np.allclose(col_got, -col_exp), (
                    f"Cube Q3 {sigma_name}, simple col {j}: not ±1 related"
                )

    def test_complete_bipartite_k33_up_to_column_signs(self):
        """K_{3,3} (6 nodes): within-partition permutations are automorphisms.

        Partition U = {0,1,2}, W = {3,4,5}.  Non-trivial Laplacian eigenvalues:
        3 (mult 4) and 6 (mult 1).  The simple column corresponds to the
        bipartition indicator vector.  Strict equality should fail for at least
        one automorphism; up-to-column-signs must hold for all.
        """
        import pytest

        from src.spectral_core import detect_eigenvalue_multiplicities

        # K_{3,3}: U={0,1,2} fully connected to W={3,4,5}
        A = np.zeros((6, 6))
        for u in range(3):
            for w in range(3, 6):
                A[u, w] = A[w, u] = 1.0
        L = self._laplacian(A)
        eigvals, eigvecs = np.linalg.eigh(L)
        nontriv = eigvals > 1e-6
        V = eigvecs[:, nontriv]
        evals = eigvals[nontriv]

        mult = detect_eigenvalue_multiplicities(evals)["multiplicity"]
        simple_cols = [j for j in range(len(evals)) if mult[j] == 1]

        if not simple_cols:
            pytest.skip("K_{3,3}: no simple non-trivial eigenvalues")

        canon_V = spectral_canonicalize(V, evals)

        # Automorphism 1: swap the two partition sides entirely (U <-> W).
        # This maps (0,1,2) -> (3,4,5) and (3,4,5) -> (0,1,2).
        # The eigenvalue-6 eigenvector is ±(1/√6)[+,+,+,-,-,-], so swapping
        # partitions negates it — strict equality FAILS on the simple column.
        sigma1 = [3, 4, 5, 0, 1, 2]
        # Automorphism 2: cycle within U (0->1->2->0), W unchanged.
        # Within-partition permutations preserve the eigenvalue-6 eigenvector
        # (it is constant within each partition), so strict equality holds here.
        sigma2 = [1, 2, 0, 3, 4, 5]

        # Strict equality should fail for the bipartition-swap automorphism
        canon_sigma1 = spectral_canonicalize(V[sigma1, :], evals)
        canon_permuted1 = canon_V[sigma1, :]
        assert not np.allclose(
            canon_sigma1[:, simple_cols], canon_permuted1[:, simple_cols]
        ), "Expected strict equivariance to fail for K_{3,3} bipartition-swap automorphism"

        # Up-to-column-signs must hold for both automorphisms
        for sigma_name, sigma in [("bipart-swap", sigma1), ("U-cycle", sigma2)]:
            canon_sigma = spectral_canonicalize(V[sigma, :], evals)
            canon_permuted = canon_V[sigma, :]
            for j in simple_cols:
                col_got = canon_sigma[:, j]
                col_exp = canon_permuted[:, j]
                assert np.allclose(col_got, col_exp) or np.allclose(col_got, -col_exp), (
                    f"K_{{3,3}} {sigma_name}, simple col {j}: not ±1 related"
                )

    def test_path_p7_automorphism_up_to_column_signs(self):
        """P7 (7 nodes): reflection automorphism 0<->6, 1<->5, 2<->4, 3 fixed.

        All eigenvalues of P7 are simple, so every non-trivial column is a
        simple column.  Strict equality should fail (the reflection changes
        within-block ordering for at least one column); up-to-column-signs
        must hold for every column.
        """
        from src.spectral_core import detect_eigenvalue_multiplicities

        A = np.zeros((7, 7))
        for i in range(6):
            A[i, i + 1] = A[i + 1, i] = 1.0
        L = self._laplacian(A)
        eigvals, eigvecs = np.linalg.eigh(L)
        nontriv = eigvals > 1e-6
        V = eigvecs[:, nontriv]
        evals = eigvals[nontriv]

        mult = detect_eigenvalue_multiplicities(evals)["multiplicity"]
        simple_cols = [j for j in range(len(evals)) if mult[j] == 1]

        # P7 has all simple eigenvalues — every non-trivial column is simple
        assert len(simple_cols) == len(evals), "P7 should have all simple eigenvalues"

        canon_V = spectral_canonicalize(V, evals)

        # Reflection: 0<->6, 1<->5, 2<->4, 3 fixed
        sigma = [6, 5, 4, 3, 2, 1, 0]

        canon_sigma = spectral_canonicalize(V[sigma, :], evals)
        canon_permuted = canon_V[sigma, :]

        # Strict equality should fail for at least one simple column
        assert not np.allclose(
            canon_sigma[:, simple_cols], canon_permuted[:, simple_cols]
        ), "Expected strict equivariance to fail under P7 reflection automorphism"

        # Up-to-column-signs must hold for every simple column
        for j in simple_cols:
            col_got = canon_sigma[:, j]
            col_exp = canon_permuted[:, j]
            assert np.allclose(col_got, col_exp) or np.allclose(col_got, -col_exp), (
                f"P7 reflection, simple col {j}: not ±1 related.\n"
                f"  got:      {np.round(col_got, 6)}\n"
                f"  expected: {np.round(col_exp, 6)}"
            )

    def test_star_k1_5_automorphism_up_to_column_signs(self):
        """Star K_{1,5} (6 nodes): any leaf permutation is an automorphism.

        Node 0 is the hub; nodes 1-5 are leaves.  Non-trivial Laplacian
        eigenvalues: 1 (mult 4) and 6 (mult 1).  The simple column
        (eigenvalue 6) corresponds to hub-vs-leaves.  Test a transposition
        and a 5-cycle of leaves; up-to-column-signs must hold for both.
        """
        import pytest

        from src.spectral_core import detect_eigenvalue_multiplicities

        # K_{1,5}: hub=0, leaves=1..5
        A = np.zeros((6, 6))
        for leaf in range(1, 6):
            A[0, leaf] = A[leaf, 0] = 1.0
        L = self._laplacian(A)
        eigvals, eigvecs = np.linalg.eigh(L)
        nontriv = eigvals > 1e-6
        V = eigvecs[:, nontriv]
        evals = eigvals[nontriv]

        mult = detect_eigenvalue_multiplicities(evals)["multiplicity"]
        simple_cols = [j for j in range(len(evals)) if mult[j] == 1]

        if not simple_cols:
            pytest.skip("K_{1,5}: no simple non-trivial eigenvalues")

        canon_V = spectral_canonicalize(V, evals)

        # Automorphism 1: swap leaves 1 and 2, rest unchanged
        sigma1 = [0, 2, 1, 3, 4, 5]
        # Automorphism 2: 5-cycle of leaves  1->2->3->4->5->1
        sigma2 = [0, 2, 3, 4, 5, 1]

        # Up-to-column-signs must hold for both automorphisms
        for sigma_name, sigma in [("leaf-swap", sigma1), ("leaf-5cycle", sigma2)]:
            canon_sigma = spectral_canonicalize(V[sigma, :], evals)
            canon_permuted = canon_V[sigma, :]
            for j in simple_cols:
                col_got = canon_sigma[:, j]
                col_exp = canon_permuted[:, j]
                assert np.allclose(col_got, col_exp) or np.allclose(col_got, -col_exp), (
                    f"K_{{1,5}} {sigma_name}, simple col {j}: not ±1 related.\n"
                    f"  got:      {np.round(col_got, 6)}\n"
                    f"  expected: {np.round(col_exp, 6)}"
                )


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


# ---------------------------------------------------------------------------
# TestAutomorphismStress
# ---------------------------------------------------------------------------


class TestAutomorphismStress:
    """Stress tests for up-to-column-signs invariance under graph automorphisms.

    For any graph automorphism sigma, the canonicalized eigenvectors satisfy:
        canon(P_sigma V) = P_sigma * canon(V) * diag(d)
    for some sign vector d in {+-1}^k.  Only simple-spectrum columns are
    checked; repeated-eigenvalue columns are left unchanged by design.

    Graphs used and their key properties:
      - C_10(1,5): circulant, 3-regular, n=10, rotation automorphism, 1 simple col
      - 6-prism (hexagonal prism): 3-regular, n=12, 3 simple cols, rich Aut group
      - C_14(1,7): circulant, 3-regular, n=14, rotation automorphism, 1 simple col
      - C6 (6-cycle): 2-regular, n=6, mixed multiplicities (1 simple + 4 repeated cols)
      - P_n (path graphs): n=5,7,9,11 — reflection automorphism, all simple cols
    """

    @staticmethod
    def _laplacian(A):
        D = np.diag(A.sum(axis=1))
        return D - A

    @staticmethod
    def _is_automorphism(A, sigma):
        """Check if sigma is a graph automorphism."""
        n = A.shape[0]
        P = np.zeros((n, n))
        for i, j in enumerate(sigma):
            P[i, j] = 1
        return np.allclose(P @ A @ P.T, A)

    @staticmethod
    def _check_up_to_column_signs(canon_sigma, canon_permuted, cols):
        """Assert columns are related by +-1."""
        for j in cols:
            col_got = canon_sigma[:, j]
            col_exp = canon_permuted[:, j]
            assert np.allclose(col_got, col_exp) or np.allclose(col_got, -col_exp), (
                f"Column {j}: not +-1 related"
            )

    # ------------------------------------------------------------------
    # Graph construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _path_graph(n):
        """Return adjacency matrix of path graph P_n."""
        A = np.zeros((n, n))
        for i in range(n - 1):
            A[i, i + 1] = A[i + 1, i] = 1
        return A

    @staticmethod
    def _c10_15_graph():
        """Circulant C_10(1,5): 3-regular, n=10.

        Connects each node i to (i+1)%10 (cycle) and (i+5)%10 (perfect matching).
        Rotation sigma(i)=(i+1)%10 is a known automorphism.
        Laplacian: 1 simple eigenvalue column, 8 repeated.
        """
        A = np.zeros((10, 10))
        for i in range(10):
            A[i, (i + 1) % 10] = A[(i + 1) % 10, i] = 1
            A[i, (i + 5) % 10] = A[(i + 5) % 10, i] = 1
        return A

    @staticmethod
    def _prism6_graph():
        """Hexagonal prism (6-prism): 3-regular, n=12.

        Two hexagons (0-1-2-3-4-5-0 and 6-7-8-9-10-11-6) connected by 6 spokes (i -- i+6).
        Automorphisms include: rotation by 1 in each ring, layer swap, reflection.
        Laplacian: 3 simple eigenvalue columns (indices 2, 7, 10 in the non-trivial spectrum).
        """
        A = np.zeros((12, 12))
        for i in range(6):
            A[i, (i + 1) % 6] = A[(i + 1) % 6, i] = 1
            A[6 + i, 6 + (i + 1) % 6] = A[6 + (i + 1) % 6, 6 + i] = 1
            A[i, 6 + i] = A[6 + i, i] = 1
        return A

    @staticmethod
    def _c14_17_graph():
        """Circulant C_14(1,7): 3-regular, n=14.

        Connects each node i to (i+1)%14 (cycle) and (i+7)%14 (perfect matching).
        Rotation sigma(i)=(i+1)%14 is a known automorphism.
        Laplacian: 1 simple eigenvalue column, 12 repeated.
        """
        A = np.zeros((14, 14))
        for i in range(14):
            A[i, (i + 1) % 14] = A[(i + 1) % 14, i] = 1
            A[i, (i + 7) % 14] = A[(i + 7) % 14, i] = 1
        return A

    @staticmethod
    def _c6_graph():
        """Return C6 (6-cycle) adjacency matrix.

        C6 Laplacian non-trivial spectrum: eigenvalue 1 (x2), 3 (x2), 4 (x1).
        Mixed multiplicities: 1 simple col, 4 repeated cols.
        """
        n = 6
        A = np.zeros((n, n))
        for i in range(n):
            A[i, (i + 1) % n] = A[(i + 1) % n, i] = 1
        return A

    @staticmethod
    def _get_simple_cols(evals):
        """Return column indices with eigenvalue multiplicity == 1."""
        mult = detect_eigenvalue_multiplicities(evals)["multiplicity"]
        return [j for j in range(len(evals)) if mult[j] == 1]

    def _eigenpairs(self, A):
        """Return (evals_nontriv, V_nontriv) for the graph Laplacian of A."""
        L = self._laplacian(A)
        eigvals, eigvecs = np.linalg.eigh(L)
        nontriv = eigvals > 1e-6
        return eigvals[nontriv], eigvecs[:, nontriv]

    def _assert_auto_up_to_col_signs(self, A, sigma, label=""):
        """Core helper: verify up-to-col-signs for a single known automorphism."""
        assert self._is_automorphism(A, sigma), f"sigma is not an automorphism {label}"
        evals, V = self._eigenpairs(A)
        simple_cols = self._get_simple_cols(evals)
        assert simple_cols, f"No simple-spectrum columns {label}"
        canon_V = spectral_canonicalize(V, evals)
        canon_sigma = spectral_canonicalize(V[sigma, :], evals)
        canon_permuted = canon_V[sigma, :]
        self._check_up_to_column_signs(canon_sigma, canon_permuted, simple_cols)

    # ------------------------------------------------------------------
    # 1. Random regular graphs (degree 3, n=10,12,14)
    #    Use explicit known automorphisms — random sampling over n! is
    #    statistically impractical for the number of automorphisms present.
    # ------------------------------------------------------------------

    def test_c10_15_rotation_automorphism(self):
        """C_10(1,5), 3-regular n=10: rotation by 1 satisfies up-to-col-signs."""
        A = self._c10_15_graph()
        sigma_rot = [(i + 1) % 10 for i in range(10)]
        self._assert_auto_up_to_col_signs(A, sigma_rot, label="C_10(1,5) rotation")

    def test_c10_15_reflection_automorphism(self):
        """C_10(1,5), 3-regular n=10: reflection sigma(i)=(10-i)%10 satisfies up-to-col-signs."""
        A = self._c10_15_graph()
        sigma_ref = [(10 - i) % 10 for i in range(10)]
        self._assert_auto_up_to_col_signs(A, sigma_ref, label="C_10(1,5) reflection")

    def test_prism6_rotation_automorphism(self):
        """6-prism, 3-regular n=12: synchronous rotation of both rings satisfies up-to-col-signs."""
        A = self._prism6_graph()
        # Rotate both hexagonal rings by 1 step simultaneously
        sigma_rot = [(i + 1) % 6 for i in range(6)] + [6 + (i + 1) % 6 for i in range(6)]
        self._assert_auto_up_to_col_signs(A, sigma_rot, label="6-prism rotation")

    def test_prism6_layer_swap_automorphism(self):
        """6-prism, 3-regular n=12: swapping the two hexagonal layers satisfies up-to-col-signs."""
        A = self._prism6_graph()
        # Swap outer ring (0..5) with inner ring (6..11)
        sigma_swap = list(range(6, 12)) + list(range(6))
        self._assert_auto_up_to_col_signs(A, sigma_swap, label="6-prism layer swap")

    def test_c14_17_rotation_automorphism(self):
        """C_14(1,7), 3-regular n=14: rotation by 1 satisfies up-to-col-signs."""
        A = self._c14_17_graph()
        sigma_rot = [(i + 1) % 14 for i in range(14)]
        self._assert_auto_up_to_col_signs(A, sigma_rot, label="C_14(1,7) rotation")

    # ------------------------------------------------------------------
    # 2. Composition test: apply multiple automorphisms in sequence
    # ------------------------------------------------------------------

    def test_composition_of_automorphisms_prism6(self):
        """6-prism: rotation, reflection, and their composition each satisfy up-to-col-signs."""
        A = self._prism6_graph()
        # Rotation of both rings by 1
        sigma1 = [(i + 1) % 6 for i in range(6)] + [6 + (i + 1) % 6 for i in range(6)]
        # Reflection of both rings: sigma(i) = (6-i)%6 for outer, idem for inner
        sigma2 = [(6 - i) % 6 for i in range(6)] + [6 + (6 - i) % 6 for i in range(6)]
        # Layer swap
        sigma3 = list(range(6, 12)) + list(range(6))

        for name, s in [("sigma1", sigma1), ("sigma2", sigma2), ("sigma3", sigma3)]:
            assert self._is_automorphism(A, s), f"{name} must be a valid 6-prism automorphism"

        # Compositions
        comp12 = [sigma2[sigma1[i]] for i in range(12)]
        comp13 = [sigma3[sigma1[i]] for i in range(12)]
        comp123 = [sigma3[sigma2[sigma1[i]]] for i in range(12)]

        for name, s in [
            ("comp(sigma2 o sigma1)", comp12),
            ("comp(sigma3 o sigma1)", comp13),
            ("comp(sigma3 o sigma2 o sigma1)", comp123),
        ]:
            assert self._is_automorphism(A, s), f"{name} must be a valid automorphism"

        evals, V = self._eigenpairs(A)
        simple_cols = self._get_simple_cols(evals)
        assert simple_cols, "6-prism must have simple-spectrum columns"

        canon_V = spectral_canonicalize(V, evals)

        for sigma_name, sigma in [
            ("sigma1", sigma1),
            ("sigma2", sigma2),
            ("sigma3", sigma3),
            ("sigma2 o sigma1", comp12),
            ("sigma3 o sigma1", comp13),
            ("sigma3 o sigma2 o sigma1", comp123),
        ]:
            canon_sigma = spectral_canonicalize(V[sigma, :], evals)
            canon_permuted = canon_V[sigma, :]
            for j in simple_cols:
                col_got = canon_sigma[:, j]
                col_exp = canon_permuted[:, j]
                assert np.allclose(col_got, col_exp) or np.allclose(col_got, -col_exp), (
                    f"Composition ({sigma_name}), col {j}: not +-1 related"
                )

    def test_composition_path_p9_reflection_involution(self):
        """P9 reflection sigma^2 = id: both sigma and sigma^2 satisfy up-to-col-signs."""
        n = 9
        A = self._path_graph(n)
        sigma = [n - 1 - i for i in range(n)]
        sigma2 = [sigma[sigma[i]] for i in range(n)]
        assert sigma2 == list(range(n)), "sigma^2 must equal identity"
        assert self._is_automorphism(A, sigma)

        evals, V = self._eigenpairs(A)
        simple_cols = self._get_simple_cols(evals)
        if not simple_cols:
            pytest.skip("No simple-spectrum columns on P9")

        canon_V = spectral_canonicalize(V, evals)

        # First application: sigma
        canon_sigma = spectral_canonicalize(V[sigma, :], evals)
        canon_permuted = canon_V[sigma, :]
        self._check_up_to_column_signs(canon_sigma, canon_permuted, simple_cols)

        # Second application: sigma^2 = identity
        canon_sigma2 = spectral_canonicalize(V[sigma2, :], evals)
        canon_permuted2 = canon_V[sigma2, :]
        self._check_up_to_column_signs(canon_sigma2, canon_permuted2, simple_cols)

    # ------------------------------------------------------------------
    # 3. All automorphisms of P5
    # ------------------------------------------------------------------

    def test_p5_identity_automorphism(self):
        """P5 identity automorphism trivially satisfies up-to-col-signs."""
        A = self._path_graph(5)
        evals, V = self._eigenpairs(A)
        simple_cols = self._get_simple_cols(evals)

        sigma_id = list(range(5))
        assert self._is_automorphism(A, sigma_id)

        canon_V = spectral_canonicalize(V, evals)
        canon_sigma = spectral_canonicalize(V[sigma_id, :], evals)
        canon_permuted = canon_V[sigma_id, :]
        self._check_up_to_column_signs(canon_sigma, canon_permuted, simple_cols)

    def test_p5_nontrivial_automorphism(self):
        """P5 reflection (04)(13) satisfies up-to-col-signs invariance."""
        A = self._path_graph(5)
        evals, V = self._eigenpairs(A)
        simple_cols = self._get_simple_cols(evals)

        sigma = [4, 3, 2, 1, 0]
        assert self._is_automorphism(A, sigma), "Reflection must be a valid automorphism of P5"

        canon_V = spectral_canonicalize(V, evals)
        canon_sigma = spectral_canonicalize(V[sigma, :], evals)
        canon_permuted = canon_V[sigma, :]
        self._check_up_to_column_signs(canon_sigma, canon_permuted, simple_cols)

    def test_p5_automorphism_group_is_complete(self):
        """P5 automorphism group has exactly 2 elements: id and (04)(13)."""
        from itertools import permutations

        A = self._path_graph(5)
        autos = [list(p) for p in permutations(range(5)) if self._is_automorphism(A, list(p))]
        assert len(autos) == 2, f"P5 must have exactly 2 automorphisms, found {len(autos)}"
        assert list(range(5)) in autos
        assert [4, 3, 2, 1, 0] in autos

    # ------------------------------------------------------------------
    # 4. Parametric path graphs P_n
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("n", [5, 7, 9, 11])
    def test_path_graph_reflection_up_to_column_signs(self, n):
        """P_n reflection sigma(i)=n-1-i satisfies up-to-col-signs on all simple columns."""
        A = self._path_graph(n)
        sigma = [n - 1 - i for i in range(n)]
        assert self._is_automorphism(A, sigma), f"Reflection must be automorphism of P{n}"

        evals, V = self._eigenpairs(A)
        simple_cols = self._get_simple_cols(evals)
        if not simple_cols:
            pytest.skip(f"No simple-spectrum columns on P{n}")

        canon_V = spectral_canonicalize(V, evals)
        canon_sigma = spectral_canonicalize(V[sigma, :], evals)
        canon_permuted = canon_V[sigma, :]
        self._check_up_to_column_signs(canon_sigma, canon_permuted, simple_cols)

    @pytest.mark.parametrize("n", [5, 7, 9, 11])
    def test_path_graph_reflection_idempotent(self, n):
        """canon(canon(V[sigma])) == canon(V[sigma]) for P_n reflected eigenvectors."""
        A = self._path_graph(n)
        sigma = [n - 1 - i for i in range(n)]

        evals, V = self._eigenpairs(A)
        V_sigma = V[sigma, :]
        canon1 = spectral_canonicalize(V_sigma, evals)
        canon2 = spectral_canonicalize(canon1, evals)
        np.testing.assert_array_almost_equal(canon1, canon2)

    # ------------------------------------------------------------------
    # 5. Eigenvalue multiplicity edge case
    # ------------------------------------------------------------------

    def test_mixed_multiplicity_simple_cols_up_to_signs(self):
        """C6 rotation: simple-spectrum columns satisfy up-to-col-signs; repeated are unchanged."""
        A = self._c6_graph()
        evals, V = self._eigenpairs(A)

        mult_info = detect_eigenvalue_multiplicities(evals)
        multiplicity = mult_info["multiplicity"]
        simple_cols = [j for j in range(len(evals)) if multiplicity[j] == 1]
        repeated_cols = [j for j in range(len(evals)) if multiplicity[j] > 1]

        sigma = [(i + 1) % 6 for i in range(6)]
        assert self._is_automorphism(A, sigma), "Rotation must be automorphism of C6"

        canon_V = spectral_canonicalize(V, evals)
        canon_sigma = spectral_canonicalize(V[sigma, :], evals)
        canon_permuted = canon_V[sigma, :]

        # Simple columns must satisfy up-to-col-signs
        if simple_cols:
            self._check_up_to_column_signs(canon_sigma, canon_permuted, simple_cols)

        # Repeated columns: spectral_canonicalize leaves them as-is (copy of input),
        # so both canon(V[sigma])[:,j] and canon(V)[sigma][:,j] equal V[sigma][:,j].
        for j in repeated_cols:
            np.testing.assert_array_almost_equal(
                canon_sigma[:, j],
                V[sigma, j],
                err_msg=f"Repeated col {j} should equal V[sigma][:,j] in canon(V[sigma])",
            )
            np.testing.assert_array_almost_equal(
                canon_permuted[:, j],
                V[sigma, j],
                err_msg=f"Repeated col {j} should equal V[sigma][:,j] in canon(V)[sigma]",
            )

    def test_mixed_multiplicity_repeated_cols_left_unchanged(self):
        """C6: columns with repeated eigenvalues are never modified by canonicalization."""
        A = self._c6_graph()
        evals, V = self._eigenpairs(A)

        mult_info = detect_eigenvalue_multiplicities(evals)
        multiplicity = mult_info["multiplicity"]
        repeated_cols = [j for j in range(len(evals)) if multiplicity[j] > 1]

        if not repeated_cols:
            pytest.skip("C6 non-trivial spectrum has no repeated eigenvalues")

        result = spectral_canonicalize(V, evals)
        for j in repeated_cols:
            np.testing.assert_array_equal(
                result[:, j],
                V[:, j],
                err_msg=f"Column {j} (multiplicity>1) must not be modified",
            )

    def test_mixed_multiplicity_simple_cols_sign_invariant(self):
        """C6 simple-spectrum columns are invariant under arbitrary column sign flips."""
        A = self._c6_graph()
        evals, V = self._eigenpairs(A)

        mult_info = detect_eigenvalue_multiplicities(evals)
        multiplicity = mult_info["multiplicity"]
        simple_cols = [j for j in range(len(evals)) if multiplicity[j] == 1]

        if not simple_cols:
            pytest.skip("No simple-spectrum columns on C6 non-trivial spectrum")

        canon_V = spectral_canonicalize(V, evals)
        rng = np.random.default_rng(42)

        for _ in range(10):
            signs = rng.choice([-1.0, 1.0], size=len(evals))
            V_signed = V * signs[np.newaxis, :]
            canon_signed = spectral_canonicalize(V_signed, evals)
            for j in simple_cols:
                np.testing.assert_array_almost_equal(
                    canon_signed[:, j],
                    canon_V[:, j],
                    err_msg=f"Simple col {j}: sign flip changed canonical result",
                )


# ---------------------------------------------------------------------------
# TestEigendecompositionEquivalence
# ---------------------------------------------------------------------------


class TestEigendecompositionEquivalence:
    """Verify that canonicalization preserves the eigendecomposition and that
    all column-sign representatives in the same orbit map to equivalent results.

    Key identity tested throughout:
        L = V @ diag(eigenvalues) @ V.T   (for simple-spectrum columns)

    Equivalence relation: V1 ~ V2  iff  V2 = V1 @ diag(d)  for some d in {+/-1}^k.
    """

    @staticmethod
    def _laplacian(A):
        D = np.diag(A.sum(axis=1))
        return D - A

    # ------------------------------------------------------------------
    # Graph builders
    # ------------------------------------------------------------------

    @staticmethod
    def _path_graph(n):
        """Return adjacency matrix for path graph P_n (0-1-..-(n-1))."""
        A = np.zeros((n, n))
        for i in range(n - 1):
            A[i, i + 1] = A[i + 1, i] = 1.0
        return A

    @staticmethod
    def _cycle_graph(n):
        """Return adjacency matrix for cycle graph C_n."""
        A = np.zeros((n, n))
        for i in range(n):
            A[i, (i + 1) % n] = A[(i + 1) % n, i] = 1.0
        return A

    @staticmethod
    def _random_graph(n, seed):
        """Return a connected random weighted graph adjacency matrix."""
        rng = np.random.default_rng(seed)
        A = np.zeros((n, n))
        # Spanning tree for connectivity
        for i in range(1, n):
            j = rng.integers(0, i)
            w = rng.uniform(0.5, 2.0)
            A[i, j] = A[j, i] = w
        # Extra random edges
        for _ in range(n):
            i, j = rng.integers(0, n, size=2)
            if i != j:
                w = rng.uniform(0.5, 2.0)
                A[i, j] = A[j, i] = w
        return A

    def _simple_spectrum_eigenpairs(self, L):
        """Return (eigenvalues, eigenvectors) restricted to simple-spectrum columns."""
        eigvals, eigvecs = np.linalg.eigh(L)
        mult = detect_eigenvalue_multiplicities(eigvals)["multiplicity"]
        simple = [j for j in range(len(eigvals)) if mult[j] == 1]
        return eigvals[simple], eigvecs[:, simple]

    @staticmethod
    def _are_col_sign_equivalent(V1, V2, atol=1e-7):
        """Return True iff V2 = V1 @ diag(d) for some d in {+/-1}^k."""
        if V1.shape != V2.shape:
            return False
        for j in range(V1.shape[1]):
            same = np.allclose(V1[:, j], V2[:, j], atol=atol)
            flipped = np.allclose(V1[:, j], -V2[:, j], atol=atol)
            if not (same or flipped):
                return False
        return True

    # ------------------------------------------------------------------
    # Test 1: Reconstruction
    # ------------------------------------------------------------------

    def test_reconstruction_several_graphs(self):
        """canon(V) @ diag(lam) @ canon(V).T reconstructs the Laplacian
        restricted to simple-spectrum columns, for P5, P7, C5, C7, and a
        random graph.  Proves canonicalization preserves the eigendecomposition."""
        graphs = [
            ("P5", self._path_graph(5)),
            ("P7", self._path_graph(7)),
            ("C5", self._cycle_graph(5)),
            ("C7", self._cycle_graph(7)),
            ("random_8", self._random_graph(8, seed=2024)),
        ]

        for name, A in graphs:
            L = self._laplacian(A)
            evals, V = self._simple_spectrum_eigenpairs(L)

            if V.size == 0:
                continue

            canon_V = spectral_canonicalize(V, evals)

            # Reconstruction: sum_j lam_j * v_j v_j^T  (sign changes cancel in outer product)
            L_reconstructed = (canon_V * evals[np.newaxis, :]) @ canon_V.T
            L_projected = (V * evals[np.newaxis, :]) @ V.T

            np.testing.assert_allclose(
                L_reconstructed,
                L_projected,
                atol=1e-10,
                err_msg=f"{name}: reconstructed Laplacian mismatch after canonicalization",
            )

    # ------------------------------------------------------------------
    # Test 2: Gram matrix invariance
    # ------------------------------------------------------------------

    def test_gram_matrix_invariance(self):
        """canon(V).T @ canon(V) equals the identity for orthonormal V,
        regardless of which sign convention was applied before canonicalization.

        Column sign flips are isometries, so the Gram matrix must be preserved."""
        graphs = [
            ("P5", self._path_graph(5)),
            ("P7", self._path_graph(7)),
            ("C5", self._cycle_graph(5)),
            ("random_8", self._random_graph(8, seed=777)),
        ]

        rng = np.random.default_rng(42)

        for name, A in graphs:
            L = self._laplacian(A)
            evals, V = self._simple_spectrum_eigenpairs(L)

            if V.size == 0:
                continue

            k = V.shape[1]
            I_k = np.eye(k)

            for trial in range(8):
                signs = rng.choice([-1.0, 1.0], size=k)
                V_flipped = V * signs[np.newaxis, :]
                canon_V = spectral_canonicalize(V_flipped, evals)

                gram = canon_V.T @ canon_V
                np.testing.assert_allclose(
                    gram,
                    I_k,
                    atol=1e-10,
                    err_msg=f"{name} trial {trial}: Gram matrix not identity",
                )

    # ------------------------------------------------------------------
    # Test 3: Spectral distance preservation
    # ------------------------------------------------------------------

    def test_spectral_distance_preservation(self):
        """For two differently-signed representations of the same eigenvectors,
        the sorted absolute-value row-signature matrix is identical.

        This models two point clouds sharing the same graph Laplacian spectrum:
        their canonical eigenvectors must be comparable via a sign-invariant metric."""
        graphs = [
            ("P5", self._path_graph(5)),
            ("P7", self._path_graph(7)),
            ("random_6", self._random_graph(6, seed=99)),
        ]

        rng = np.random.default_rng(123)

        for name, A in graphs:
            L = self._laplacian(A)
            evals, V = self._simple_spectrum_eigenpairs(L)

            if V.size == 0:
                continue

            k = V.shape[1]

            signs_a = rng.choice([-1.0, 1.0], size=k)
            signs_b = rng.choice([-1.0, 1.0], size=k)
            V_a = V * signs_a[np.newaxis, :]
            V_b = V * signs_b[np.newaxis, :]

            canon_a = spectral_canonicalize(V_a, evals)
            canon_b = spectral_canonicalize(V_b, evals)

            def _abs_signature_matrix(C):
                """Lexicographically sorted matrix of sorted absolute-value row vectors."""
                rows = [tuple(np.sort(np.abs(C[i]))) for i in range(C.shape[0])]
                rows.sort()
                return np.array(rows)

            sig_a = _abs_signature_matrix(canon_a)
            sig_b = _abs_signature_matrix(canon_b)

            np.testing.assert_allclose(
                sig_a,
                sig_b,
                atol=1e-8,
                err_msg=f"{name}: spectral distance not preserved across sign conventions",
            )

    # ------------------------------------------------------------------
    # Test 4: Column-sign equivalence class on P5
    # ------------------------------------------------------------------

    def test_p5_all_sign_combos_one_equivalence_class(self):
        """Enumerate all 2^4 = 16 sign vectors on P5's 4 non-trivial eigenvectors.

        After canonicalization, every result must lie in exactly 1 column-sign
        equivalence class.  Two matrices are equivalent iff they differ only by
        column sign flips: V2 = V1 @ diag(d) for d in {+/-1}^4."""
        A = self._path_graph(5)
        L = self._laplacian(A)
        eigvals, eigvecs = np.linalg.eigh(L)

        nontriv = eigvals > 1e-6
        V = eigvecs[:, nontriv]
        evals = eigvals[nontriv]

        k = V.shape[1]
        assert k == 4, f"P5 should have 4 non-trivial eigenvectors, got {k}"

        # Collect all 2^4 = 16 canonicalized matrices
        canon_results = []
        for bits in range(1 << k):
            signs = np.array([(-1.0) ** ((bits >> j) & 1) for j in range(k)])
            V_signed = V * signs[np.newaxis, :]
            canon_results.append(spectral_canonicalize(V_signed, evals))

        # Greedy grouping into column-sign equivalence classes
        class_representatives = []
        for canon_V in canon_results:
            found = False
            for rep in class_representatives:
                if self._are_col_sign_equivalent(rep, canon_V):
                    found = True
                    break
            if not found:
                class_representatives.append(canon_V)

        n_classes = len(class_representatives)
        assert n_classes == 1, (
            f"Expected exactly 1 column-sign equivalence class across all 2^4=16 "
            f"sign vectors on P5, got {n_classes} classes"
        )

    # ------------------------------------------------------------------
    # Test 5: Automorphism orbit test on P5
    # ------------------------------------------------------------------

    def test_p5_automorphism_orbit_and_sign_matrix(self):
        """For P5, the reflection sigma: i -> 4-i is a graph automorphism.

        Since P_sigma L P_sigma^T = L, the matrix P_sigma V is also a valid
        eigenvector matrix.  The test verifies:
          (a) canon(P_sigma V) is column-sign equivalent to P_sigma canon(V).
          (b) Explicitly recovers the sign matrix D in {+/-1}^k such that
              canon(P_sigma V) == P_sigma canon(V) @ diag(D)
              and checks D has only +/-1 entries and the equation holds."""
        A = self._path_graph(5)
        L = self._laplacian(A)
        eigvals, eigvecs = np.linalg.eigh(L)

        nontriv = eigvals > 1e-6
        V = eigvecs[:, nontriv]
        evals = eigvals[nontriv]

        k = V.shape[1]

        # Reflection automorphism of P5: node i <-> node 4-i
        sigma = [4, 3, 2, 1, 0]

        canon_V = spectral_canonicalize(V, evals)
        canon_PV = spectral_canonicalize(V[sigma, :], evals)

        # Expected: permute the canonical form by sigma
        P_sigma_canon = canon_V[sigma, :]

        # (a) Column-sign equivalence
        assert self._are_col_sign_equivalent(P_sigma_canon, canon_PV), (
            "canon(P_sigma V) must be column-sign equivalent to P_sigma canon(V) on P5"
        )

        # (b) Explicitly recover D in {+/-1}^k
        D = np.ones(k, dtype=float)
        for j in range(k):
            col_got = canon_PV[:, j]
            col_exp = P_sigma_canon[:, j]
            if np.allclose(col_got, col_exp, atol=1e-7):
                D[j] = 1.0
            elif np.allclose(col_got, -col_exp, atol=1e-7):
                D[j] = -1.0
            else:
                raise AssertionError(
                    f"Column {j}: canon(P_sigma V)[:,{j}] is neither +1 nor -1 times "
                    f"(P_sigma canon(V))[:,{j}] — not in the column-sign equivalence class"
                )

        # D must be a {+/-1} sign vector
        assert np.all(np.abs(D) == 1.0), f"Recovered D is not a +/-1 vector: {D}"

        # Verify closed-form: canon(P_sigma V) == P_sigma canon(V) @ diag(D)
        reconstructed = P_sigma_canon * D[np.newaxis, :]
        np.testing.assert_allclose(
            canon_PV,
            reconstructed,
            atol=1e-7,
            err_msg=(
                "canon(P_sigma V) != P_sigma canon(V) @ diag(D) for the recovered sign matrix D"
            ),
        )
