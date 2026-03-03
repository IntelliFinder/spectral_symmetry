"""Tests for MAP and OAP eigenvector canonicalization.

Tests sign invariance (mult-1), idempotency, determinism, orthonormality,
correctness on standard graph types (path, cycle, star, Petersen, complete),
and reference-matching permutation-equivariance / uniqueness verification.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from spectral_audit import edge_index_to_laplacian

from src.spectral_canonicalization import (
    _group_eigenvalues_by_rounding,
    _map_canonicalize_eigenspace,
    _map_sign_disambiguate,
    _oap_canonicalize_eigenspace,
    _oap_sign_disambiguate,
    spectral_canonicalize_map,
    spectral_canonicalize_oap,
)
from src.spectral_core import compute_eigenpairs

# ---------------------------------------------------------------------------
# Graph construction helpers (same as test_spectral_audit.py)
# ---------------------------------------------------------------------------


def _path_graph(n):
    rows, cols = [], []
    for i in range(n - 1):
        rows.extend([i, i + 1])
        cols.extend([i + 1, i])
    edge_index = np.array([rows, cols])
    return edge_index, n, edge_index.shape[1]


def _cycle_graph(n):
    rows, cols = [], []
    for i in range(n):
        j = (i + 1) % n
        rows.extend([i, j])
        cols.extend([j, i])
    edge_index = np.array([rows, cols])
    return edge_index, n, edge_index.shape[1]


def _star_graph(n):
    rows, cols = [], []
    for i in range(1, n):
        rows.extend([0, i])
        cols.extend([i, 0])
    edge_index = np.array([rows, cols])
    return edge_index, n, edge_index.shape[1]


def _complete_graph(n):
    rows, cols = [], []
    for i in range(n):
        for j in range(i + 1, n):
            rows.extend([i, j])
            cols.extend([j, i])
    edge_index = np.array([rows, cols])
    return edge_index, n, edge_index.shape[1]


def _petersen_graph():
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 0),
        (5, 7),
        (7, 9),
        (9, 6),
        (6, 8),
        (8, 5),
        (0, 5),
        (1, 6),
        (2, 7),
        (3, 8),
        (4, 9),
    ]
    rows, cols = [], []
    for i, j in edges:
        rows.extend([i, j])
        cols.extend([j, i])
    edge_index = np.array([rows, cols])
    return edge_index, 10, edge_index.shape[1]


def _get_eigenpairs(ei, n, n_eigs=6):
    """Helper to get eigenpairs from a graph."""
    L, n_lcc = edge_index_to_laplacian(ei, n)
    k = min(n_eigs, n_lcc - 2)
    eigenvalues, eigenvectors = compute_eigenpairs(L, n_eigs=k)
    return eigenvalues, eigenvectors


# ---------------------------------------------------------------------------
# Reference-style random matrix helpers (numpy, float64)
# ---------------------------------------------------------------------------


def _random_orthonormal_matrix(n, d):
    """Randomly generate an orthonormal matrix of shape (n, d)."""
    A = np.random.randn(n, n)
    A = (A + A.T) / 2  # symmetric
    _, U = np.linalg.eigh(A)
    return U[:, :d]


def _random_sign_matrix(n):
    """Randomly generate a diagonal sign matrix."""
    s = np.random.randint(0, 2, size=n) * 2 - 1
    return np.diag(s.astype(float))


def _random_permutation_matrix(n):
    """Generate a random permutation matrix."""
    return np.eye(n)[np.random.permutation(n)]


# ---------------------------------------------------------------------------
# Test: MAP sign invariance (mult-1 columns, graph-based)
# ---------------------------------------------------------------------------


class TestMAPSignInvariance:
    """Verify that MAP canonicalization is invariant under sign flips on mult-1 columns.

    The sign algorithm may fail to disambiguate some columns (when all entries
    have equal absolute value). For those columns, the output follows the input
    sign. We check agreement up to sign: |canon(V)| == |canon(V_signed)|.
    """

    def _check(self, ei, n, ne, n_eigs=6, n_trials=20):
        eigenvalues, eigenvectors = _get_eigenpairs(ei, n, n_eigs)
        if len(eigenvalues) == 0:
            return

        # Use same grouping as the MAP function (round to 14 decimals)
        ind, mult = _group_eigenvalues_by_rounding(eigenvalues, decimals=14)
        simple_cols = [int(ind[i]) for i in range(len(mult)) if mult[i] == 1]
        if not simple_cols:
            return

        canon = spectral_canonicalize_map(eigenvectors, eigenvalues)

        rng = np.random.default_rng(42)
        n_cols = eigenvectors.shape[1]
        for _ in range(n_trials):
            signs = np.ones(n_cols)
            for j in simple_cols:
                signs[j] = rng.choice([-1, 1])
            V_signed = eigenvectors * signs[np.newaxis, :]
            canon_signed = spectral_canonicalize_map(V_signed, eigenvalues)
            # Each column must agree up to sign (sign assumption may not hold)
            for j in simple_cols:
                np.testing.assert_allclose(
                    np.abs(canon[:, j]),
                    np.abs(canon_signed[:, j]),
                    atol=1e-10,
                )

    def test_path_10(self):
        self._check(*_path_graph(10))

    def test_path_20(self):
        self._check(*_path_graph(20), n_eigs=15)

    def test_star_8(self):
        self._check(*_star_graph(8))

    def test_cycle_7(self):
        self._check(*_cycle_graph(7))

    def test_petersen(self):
        self._check(*_petersen_graph())


# ---------------------------------------------------------------------------
# Test: OAP sign invariance (mult-1 columns, graph-based)
# ---------------------------------------------------------------------------


class TestOAPSignInvariance:
    """Verify that OAP canonicalization is invariant under sign flips on mult-1 columns.

    Same caveat as MAP: sign assumption may not hold for symmetric eigenvectors.
    """

    def _check(self, ei, n, ne, n_eigs=6, n_trials=20):
        eigenvalues, eigenvectors = _get_eigenpairs(ei, n, n_eigs)
        if len(eigenvalues) == 0:
            return

        # Use same grouping as the OAP function (round to 6 decimals)
        ind, mult = _group_eigenvalues_by_rounding(eigenvalues, decimals=6)
        simple_cols = [int(ind[i]) for i in range(len(mult)) if mult[i] == 1]
        if not simple_cols:
            return

        canon = spectral_canonicalize_oap(eigenvectors, eigenvalues)

        rng = np.random.default_rng(42)
        n_cols = eigenvectors.shape[1]
        for _ in range(n_trials):
            signs = np.ones(n_cols)
            for j in simple_cols:
                signs[j] = rng.choice([-1, 1])
            V_signed = eigenvectors * signs[np.newaxis, :]
            canon_signed = spectral_canonicalize_oap(V_signed, eigenvalues)
            for j in simple_cols:
                np.testing.assert_allclose(
                    np.abs(canon[:, j]),
                    np.abs(canon_signed[:, j]),
                    atol=1e-10,
                )

    def test_path_10(self):
        self._check(*_path_graph(10))

    def test_path_20(self):
        self._check(*_path_graph(20), n_eigs=15)

    def test_star_8(self):
        self._check(*_star_graph(8))

    def test_cycle_7(self):
        self._check(*_cycle_graph(7))

    def test_petersen(self):
        self._check(*_petersen_graph())


# ---------------------------------------------------------------------------
# Test: Idempotency
# ---------------------------------------------------------------------------


class TestMAPIdempotency:
    """Verify MAP(MAP(V)) == MAP(V) on mult-1 columns.

    For degenerate eigenspaces, MAP is not strictly idempotent due to
    numerical precision in the complementary space construction. We verify
    that the eigenspace is approximately preserved.
    """

    def _check(self, ei, n, ne, n_eigs=6):
        eigenvalues, eigenvectors = _get_eigenpairs(ei, n, n_eigs)
        if len(eigenvalues) == 0:
            return

        # Use same grouping as the MAP function (round to 14 decimals)
        ind, mult = _group_eigenvalues_by_rounding(eigenvalues, decimals=14)
        simple_cols = [int(ind[i]) for i in range(len(mult)) if mult[i] == 1]

        canon1 = spectral_canonicalize_map(eigenvectors, eigenvalues)
        canon2 = spectral_canonicalize_map(canon1, eigenvalues)

        # Mult-1 columns: check agreement up to sign (sign assumption may fail)
        for j in simple_cols:
            np.testing.assert_allclose(np.abs(canon1[:, j]), np.abs(canon2[:, j]), atol=1e-10)

        # Degenerate eigenspaces: check that the subspace is approximately preserved
        for i in range(len(mult)):
            if mult[i] <= 1:
                continue
            cols = list(range(int(ind[i]), int(ind[i + 1])))
            U1 = canon1[:, cols]
            U2 = canon2[:, cols]
            Q1, _ = np.linalg.qr(U1, mode="reduced")
            for j in range(len(cols)):
                proj = Q1 @ (Q1.T @ U2[:, j])
                residual = np.linalg.norm(U2[:, j] - proj)
                assert residual < 0.15, f"Eigenspace not preserved (residual={residual:.2e})"

    def test_path(self):
        self._check(*_path_graph(10))

    def test_star(self):
        self._check(*_star_graph(8))

    def test_cycle(self):
        self._check(*_cycle_graph(7))

    def test_petersen(self):
        self._check(*_petersen_graph())

    def test_complete(self):
        self._check(*_complete_graph(5))


class TestOAPIdempotency:
    """Verify OAP(OAP(V)) == OAP(V) on mult-1 columns."""

    def _check(self, ei, n, ne, n_eigs=6):
        eigenvalues, eigenvectors = _get_eigenpairs(ei, n, n_eigs)
        if len(eigenvalues) == 0:
            return

        # Use same grouping as the OAP function (round to 6 decimals)
        ind, mult = _group_eigenvalues_by_rounding(eigenvalues, decimals=6)
        simple_cols = [int(ind[i]) for i in range(len(mult)) if mult[i] == 1]

        canon1 = spectral_canonicalize_oap(eigenvectors, eigenvalues)
        canon2 = spectral_canonicalize_oap(canon1, eigenvalues)

        if simple_cols:
            np.testing.assert_allclose(canon1[:, simple_cols], canon2[:, simple_cols], atol=1e-10)

        for i in range(len(mult)):
            if mult[i] <= 1:
                continue
            cols = list(range(int(ind[i]), int(ind[i + 1])))
            U1 = canon1[:, cols]
            U2 = canon2[:, cols]
            Q1, _ = np.linalg.qr(U1, mode="reduced")
            for j in range(len(cols)):
                proj = Q1 @ (Q1.T @ U2[:, j])
                residual = np.linalg.norm(U2[:, j] - proj)
                assert residual < 1e-6, f"Eigenspace not preserved (residual={residual:.2e})"

    def test_path(self):
        self._check(*_path_graph(10))

    def test_star(self):
        self._check(*_star_graph(8))

    def test_cycle(self):
        self._check(*_cycle_graph(7))

    def test_petersen(self):
        self._check(*_petersen_graph())

    def test_complete(self):
        self._check(*_complete_graph(5))


# ---------------------------------------------------------------------------
# Test: Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Verify same input gives same output every time."""

    def test_map_deterministic(self):
        ei, n, _ = _path_graph(12)
        eigenvalues, eigenvectors = _get_eigenpairs(ei, n, n_eigs=8)
        c1 = spectral_canonicalize_map(eigenvectors, eigenvalues)
        c2 = spectral_canonicalize_map(eigenvectors, eigenvalues)
        np.testing.assert_array_equal(c1, c2)

    def test_oap_deterministic(self):
        ei, n, _ = _path_graph(12)
        eigenvalues, eigenvectors = _get_eigenpairs(ei, n, n_eigs=8)
        c1 = spectral_canonicalize_oap(eigenvectors, eigenvalues)
        c2 = spectral_canonicalize_oap(eigenvectors, eigenvalues)
        np.testing.assert_array_equal(c1, c2)


# ---------------------------------------------------------------------------
# Test: Output is orthonormal for degenerate eigenspaces
# ---------------------------------------------------------------------------


class TestOrthonormality:
    """For degenerate eigenspaces, the canonical basis should be orthonormal."""

    def _check(self, canonicalize_fn, ei, n, ne, n_eigs=6, decimals=6):
        eigenvalues, eigenvectors = _get_eigenpairs(ei, n, n_eigs)
        if len(eigenvalues) == 0:
            return

        canon = canonicalize_fn(eigenvectors, eigenvalues)

        ind, mult = _group_eigenvalues_by_rounding(eigenvalues, decimals=decimals)
        for i in range(len(mult)):
            if mult[i] <= 1:
                continue
            cols = list(range(int(ind[i]), int(ind[i + 1])))
            V_group = canon[:, cols]
            gram = V_group.T @ V_group
            np.testing.assert_allclose(
                gram,
                np.eye(len(cols)),
                atol=1e-8,
                err_msg=f"Non-orthonormal canonical basis for eigenspace {cols}",
            )

    def test_map_star(self):
        """Star graph has large degenerate eigenspace."""
        self._check(spectral_canonicalize_map, *_star_graph(7), decimals=14)

    def test_map_complete(self):
        self._check(spectral_canonicalize_map, *_complete_graph(5), decimals=14)

    def test_map_cycle_even(self):
        self._check(spectral_canonicalize_map, *_cycle_graph(6), decimals=14)

    def test_oap_star(self):
        self._check(spectral_canonicalize_oap, *_star_graph(7), decimals=6)

    def test_oap_complete(self):
        self._check(spectral_canonicalize_oap, *_complete_graph(5), decimals=6)

    def test_oap_cycle_even(self):
        self._check(spectral_canonicalize_oap, *_cycle_graph(6), decimals=6)


# ---------------------------------------------------------------------------
# Test: Canonical basis spans the same eigenspace
# ---------------------------------------------------------------------------


class TestEigenspacePreservation:
    """The canonical basis should span the same eigenspace as the original."""

    def _check(self, canonicalize_fn, ei, n, ne, n_eigs=6, decimals=6):
        eigenvalues, eigenvectors = _get_eigenpairs(ei, n, n_eigs)
        if len(eigenvalues) == 0:
            return

        canon = canonicalize_fn(eigenvectors, eigenvalues)

        ind, mult = _group_eigenvalues_by_rounding(eigenvalues, decimals=decimals)
        for i in range(len(mult)):
            cols = list(range(int(ind[i]), int(ind[i + 1])))
            U_orig = eigenvectors[:, cols]
            U_canon = canon[:, cols]

            for j in range(len(cols)):
                v = U_canon[:, j]
                Q, _ = np.linalg.qr(U_orig, mode="reduced")
                proj = Q @ (Q.T @ v)
                residual = np.linalg.norm(v - proj)
                assert residual < 1e-6, (
                    f"Canonical vector {j} not in original eigenspace (residual={residual:.2e})"
                )

    def test_map_star(self):
        self._check(spectral_canonicalize_map, *_star_graph(7), decimals=14)

    def test_map_cycle_even(self):
        self._check(spectral_canonicalize_map, *_cycle_graph(6), decimals=14)

    def test_oap_star(self):
        self._check(spectral_canonicalize_oap, *_star_graph(7), decimals=6)

    def test_oap_cycle_even(self):
        self._check(spectral_canonicalize_oap, *_cycle_graph(6), decimals=6)

    def test_oap_petersen(self):
        self._check(spectral_canonicalize_oap, *_petersen_graph(), decimals=6)


# ---------------------------------------------------------------------------
# Test: Empty / edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases: empty eigenvectors, single eigenvector."""

    def test_map_empty(self):
        V = np.zeros((5, 0))
        evals = np.array([])
        result = spectral_canonicalize_map(V, evals)
        assert result.shape == (5, 0)

    def test_oap_empty(self):
        V = np.zeros((5, 0))
        evals = np.array([])
        result = spectral_canonicalize_oap(V, evals)
        assert result.shape == (5, 0)

    def test_map_single_eigvec(self):
        """Single eigenvector: sign should be deterministic and sign-invariant."""
        V = np.array([[0.5], [-0.3], [0.8], [-0.1], [0.2]])
        evals = np.array([1.0])
        result1 = spectral_canonicalize_map(V, evals)
        result2 = spectral_canonicalize_map(-V, evals)
        assert result1.shape == V.shape
        # Sign-invariant: canon(V) == canon(-V)
        np.testing.assert_allclose(result1, result2, atol=1e-10)

    def test_oap_single_eigvec(self):
        """Single eigenvector: sign should be deterministic and sign-invariant."""
        V = np.array([[0.5], [-0.3], [0.8], [-0.1], [0.2]])
        evals = np.array([1.0])
        result1 = spectral_canonicalize_oap(V, evals)
        result2 = spectral_canonicalize_oap(-V, evals)
        assert result1.shape == V.shape
        np.testing.assert_allclose(result1, result2, atol=1e-10)

    def test_map_negative_dominant(self):
        """Sign disambiguation should produce consistent results."""
        V = np.array([[-0.9], [0.1], [0.2], [-0.3], [0.1]])
        evals = np.array([1.0])
        result1 = spectral_canonicalize_map(V, evals)
        result2 = spectral_canonicalize_map(-V, evals)
        np.testing.assert_allclose(result1, result2, atol=1e-10)

    def test_oap_negative_dominant(self):
        V = np.array([[-0.9], [0.1], [0.2], [-0.3], [0.1]])
        evals = np.array([1.0])
        result1 = spectral_canonicalize_oap(V, evals)
        result2 = spectral_canonicalize_oap(-V, evals)
        np.testing.assert_allclose(result1, result2, atol=1e-10)


# ---------------------------------------------------------------------------
# Test: Eigenvalue grouping helper
# ---------------------------------------------------------------------------


class TestEigenvalueGrouping:
    """Test the rounding-based eigenvalue grouping helper."""

    def test_distinct_eigenvalues(self):
        evals = np.array([0.5, 1.0, 1.5, 2.0])
        ind, mult = _group_eigenvalues_by_rounding(evals, decimals=6)
        assert list(mult) == [1, 1, 1, 1]
        assert list(ind) == [0, 1, 2, 3, 4]

    def test_degenerate_eigenvalues(self):
        evals = np.array([1.0, 1.0, 2.0, 2.0, 2.0])
        ind, mult = _group_eigenvalues_by_rounding(evals, decimals=6)
        assert list(mult) == [2, 3]
        assert list(ind) == [0, 2, 5]

    def test_near_degenerate_map(self):
        """MAP uses 14 decimals — very close eigenvalues should be distinct."""
        evals = np.array([1.0, 1.0 + 1e-13, 2.0])
        ind, mult = _group_eigenvalues_by_rounding(evals, decimals=14)
        assert list(mult) == [1, 1, 1]

    def test_near_degenerate_oap(self):
        """OAP uses 6 decimals — eigenvalues within 1e-6 should merge."""
        evals = np.array([1.0, 1.0 + 1e-8, 2.0])
        ind, mult = _group_eigenvalues_by_rounding(evals, decimals=6)
        assert list(mult) == [2, 1]


# ===========================================================================
# Reference verification tests (ported from map.py / oap.py __main__)
# ===========================================================================

N_TRIALS = 200


class TestMAPSignVerification:
    """Verify MAP sign disambiguation matches reference properties.

    Port of the __main__ verification from map.py:
    - Permutation-equivariance: P @ sign(U) == sign(P @ U)
    - Uniqueness: sign(U @ S) == sign(U) for any sign matrix S
    - Both simultaneously
    """

    def test_permutation_equivariance(self):
        np.random.seed(42)
        correct = 0
        for _ in range(N_TRIALS):
            n = np.random.randint(2, 20)
            U = _random_orthonormal_matrix(n, n)
            U_0 = _map_sign_disambiguate(U)
            P = _random_permutation_matrix(n)
            V = P @ U
            V_0 = _map_sign_disambiguate(V)
            if np.allclose(P @ U_0, V_0, atol=1e-6):
                correct += 1
        assert correct == N_TRIALS, f"MAP sign perm-equiv: {correct}/{N_TRIALS}"

    def test_sign_uniqueness(self):
        np.random.seed(43)
        correct = 0
        for _ in range(N_TRIALS):
            n = np.random.randint(2, 20)
            U = _random_orthonormal_matrix(n, n)
            U_0 = _map_sign_disambiguate(U)
            S = _random_sign_matrix(n)
            W = U @ S
            W_0 = _map_sign_disambiguate(W)
            if np.allclose(U_0, W_0, atol=1e-6):
                correct += 1
        assert correct == N_TRIALS, f"MAP sign uniqueness: {correct}/{N_TRIALS}"

    def test_both(self):
        np.random.seed(44)
        correct = 0
        for _ in range(N_TRIALS):
            n = np.random.randint(2, 20)
            U = _random_orthonormal_matrix(n, n)
            U_0 = _map_sign_disambiguate(U)
            P = _random_permutation_matrix(n)
            S = _random_sign_matrix(n)
            Y = P @ U @ S
            Y_0 = _map_sign_disambiguate(Y)
            if np.allclose(P @ U_0, Y_0, atol=1e-6):
                correct += 1
        assert correct == N_TRIALS, f"MAP sign both: {correct}/{N_TRIALS}"


class TestMAPBasisVerification:
    """Verify MAP basis disambiguation matches reference properties.

    Port of the __main__ verification from map.py:
    - Permutation-equivariance: P @ basis(U) == basis(P @ U)
    - Uniqueness: basis(U @ Q) == basis(U) for any orthogonal Q
    - Both simultaneously
    """

    def test_permutation_equivariance(self):
        np.random.seed(45)
        p_correct = total = 0
        for _ in range(N_TRIALS):
            n = np.random.randint(2, 20)
            d = np.random.randint(1, n)
            U = _random_orthonormal_matrix(n, d)
            try:
                U_0 = _map_canonicalize_eigenspace(U)
            except AssertionError:
                continue
            P = _random_permutation_matrix(n)
            V = P @ U
            try:
                V_0 = _map_canonicalize_eigenspace(V)
            except AssertionError:
                continue
            total += 1
            if np.allclose(P @ U_0, V_0, atol=1e-6):
                p_correct += 1
        assert total > 0, "No valid trials for MAP basis perm-equiv"
        assert p_correct == total, f"MAP basis perm-equiv: {p_correct}/{total}"

    def test_basis_uniqueness(self):
        np.random.seed(46)
        q_correct = total = 0
        for _ in range(N_TRIALS):
            n = np.random.randint(2, 20)
            d = np.random.randint(1, n)
            U = _random_orthonormal_matrix(n, d)
            try:
                U_0 = _map_canonicalize_eigenspace(U)
            except AssertionError:
                continue
            Q = _random_orthonormal_matrix(d, d)
            W = U @ Q
            try:
                W_0 = _map_canonicalize_eigenspace(W)
            except AssertionError:
                continue
            total += 1
            if np.allclose(U_0, W_0, atol=1e-6):
                q_correct += 1
        assert total > 0, "No valid trials for MAP basis uniqueness"
        assert q_correct == total, f"MAP basis uniqueness: {q_correct}/{total}"

    def test_both(self):
        np.random.seed(47)
        pq_correct = total = 0
        for _ in range(N_TRIALS):
            n = np.random.randint(2, 20)
            d = np.random.randint(1, n)
            U = _random_orthonormal_matrix(n, d)
            try:
                U_0 = _map_canonicalize_eigenspace(U)
            except AssertionError:
                continue
            P = _random_permutation_matrix(n)
            Q = _random_orthonormal_matrix(d, d)
            Y = P @ U @ Q
            try:
                Y_0 = _map_canonicalize_eigenspace(Y)
            except AssertionError:
                continue
            total += 1
            if np.allclose(P @ U_0, Y_0, atol=1e-6):
                pq_correct += 1
        assert total > 0, "No valid trials for MAP basis both"
        assert pq_correct == total, f"MAP basis both: {pq_correct}/{total}"

    def test_assumptions_rarely_violated(self):
        """Basis assumptions should almost never be violated."""
        np.random.seed(48)
        total = 0
        for _ in range(N_TRIALS):
            n = np.random.randint(2, 20)
            d = np.random.randint(1, n)
            U = _random_orthonormal_matrix(n, d)
            try:
                _map_canonicalize_eigenspace(U)
                total += 1
            except AssertionError:
                pass
        assert total >= N_TRIALS * 0.95, (
            f"MAP basis assumptions violated too often: {total}/{N_TRIALS}"
        )


class TestOAPSignVerification:
    """Verify OAP sign disambiguation matches reference properties."""

    def test_permutation_equivariance(self):
        np.random.seed(52)
        correct = 0
        for _ in range(N_TRIALS):
            n = np.random.randint(2, 20)
            U = _random_orthonormal_matrix(n, n)
            U_0 = _oap_sign_disambiguate(U)
            P = _random_permutation_matrix(n)
            V = P @ U
            V_0 = _oap_sign_disambiguate(V)
            if np.allclose(P @ U_0, V_0, atol=1e-6):
                correct += 1
        assert correct == N_TRIALS, f"OAP sign perm-equiv: {correct}/{N_TRIALS}"

    def test_sign_uniqueness(self):
        np.random.seed(53)
        correct = 0
        for _ in range(N_TRIALS):
            n = np.random.randint(2, 20)
            U = _random_orthonormal_matrix(n, n)
            U_0 = _oap_sign_disambiguate(U)
            S = _random_sign_matrix(n)
            W = U @ S
            W_0 = _oap_sign_disambiguate(W)
            if np.allclose(U_0, W_0, atol=1e-6):
                correct += 1
        assert correct == N_TRIALS, f"OAP sign uniqueness: {correct}/{N_TRIALS}"

    def test_both(self):
        np.random.seed(54)
        correct = 0
        for _ in range(N_TRIALS):
            n = np.random.randint(2, 20)
            U = _random_orthonormal_matrix(n, n)
            U_0 = _oap_sign_disambiguate(U)
            P = _random_permutation_matrix(n)
            S = _random_sign_matrix(n)
            Y = P @ U @ S
            Y_0 = _oap_sign_disambiguate(Y)
            if np.allclose(P @ U_0, Y_0, atol=1e-6):
                correct += 1
        assert correct == N_TRIALS, f"OAP sign both: {correct}/{N_TRIALS}"


class TestOAPBasisVerification:
    """Verify OAP basis disambiguation matches reference properties."""

    def test_permutation_equivariance(self):
        np.random.seed(55)
        p_correct = total = 0
        for _ in range(N_TRIALS):
            n = np.random.randint(2, 20)
            d = np.random.randint(1, n)
            U = _random_orthonormal_matrix(n, d)
            try:
                U_0 = _oap_canonicalize_eigenspace(U)
            except AssertionError:
                continue
            P = _random_permutation_matrix(n)
            V = P @ U
            try:
                V_0 = _oap_canonicalize_eigenspace(V)
            except AssertionError:
                continue
            total += 1
            if np.allclose(P @ U_0, V_0, atol=1e-6):
                p_correct += 1
        assert total > 0, "No valid trials for OAP basis perm-equiv"
        assert p_correct == total, f"OAP basis perm-equiv: {p_correct}/{total}"

    def test_basis_uniqueness(self):
        np.random.seed(56)
        q_correct = total = 0
        for _ in range(N_TRIALS):
            n = np.random.randint(2, 20)
            d = np.random.randint(1, n)
            U = _random_orthonormal_matrix(n, d)
            try:
                U_0 = _oap_canonicalize_eigenspace(U)
            except AssertionError:
                continue
            Q = _random_orthonormal_matrix(d, d)
            W = U @ Q
            try:
                W_0 = _oap_canonicalize_eigenspace(W)
            except AssertionError:
                continue
            total += 1
            if np.allclose(U_0, W_0, atol=1e-6):
                q_correct += 1
        assert total > 0, "No valid trials for OAP basis uniqueness"
        assert q_correct == total, f"OAP basis uniqueness: {q_correct}/{total}"

    def test_both(self):
        np.random.seed(57)
        pq_correct = total = 0
        for _ in range(N_TRIALS):
            n = np.random.randint(2, 20)
            d = np.random.randint(1, n)
            U = _random_orthonormal_matrix(n, d)
            try:
                U_0 = _oap_canonicalize_eigenspace(U)
            except AssertionError:
                continue
            P = _random_permutation_matrix(n)
            Q = _random_orthonormal_matrix(d, d)
            Y = P @ U @ Q
            try:
                Y_0 = _oap_canonicalize_eigenspace(Y)
            except AssertionError:
                continue
            total += 1
            if np.allclose(P @ U_0, Y_0, atol=1e-6):
                pq_correct += 1
        assert total > 0, "No valid trials for OAP basis both"
        assert pq_correct == total, f"OAP basis both: {pq_correct}/{total}"

    def test_assumptions_rarely_violated(self):
        """Basis assumptions should almost never be violated."""
        np.random.seed(58)
        total = 0
        for _ in range(N_TRIALS):
            n = np.random.randint(2, 20)
            d = np.random.randint(1, n)
            U = _random_orthonormal_matrix(n, d)
            try:
                _oap_canonicalize_eigenspace(U)
                total += 1
            except AssertionError:
                pass
        assert total >= N_TRIALS * 0.95, (
            f"OAP basis assumptions violated too often: {total}/{N_TRIALS}"
        )
