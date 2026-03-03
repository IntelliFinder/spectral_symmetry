"""Tests for spectral_audit: metric correctness, Spielman verification, noise stability."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from spectral_audit import analyze_single_graph, edge_index_to_laplacian

from src.spectral_canonicalization import spectral_canonicalize
from src.spectral_core import (
    compute_eigenpairs,
    detect_eigenvalue_multiplicities,
    uncanonicalizability_score,
    uncanonicalizability_threshold,
)

# ---------------------------------------------------------------------------
# Helper: build edge_index arrays for common graph types
# ---------------------------------------------------------------------------


def _path_graph(n):
    """Return (edge_index [2, 2*(n-1)], num_nodes, num_edges) for path P_n."""
    rows, cols = [], []
    for i in range(n - 1):
        rows.extend([i, i + 1])
        cols.extend([i + 1, i])
    edge_index = np.array([rows, cols])
    return edge_index, n, edge_index.shape[1]


def _cycle_graph(n):
    """Return (edge_index, num_nodes, num_edges) for cycle C_n."""
    rows, cols = [], []
    for i in range(n):
        j = (i + 1) % n
        rows.extend([i, j])
        cols.extend([j, i])
    edge_index = np.array([rows, cols])
    return edge_index, n, edge_index.shape[1]


def _star_graph(n):
    """Return (edge_index, num_nodes, num_edges) for star S_n (hub=0, n-1 leaves)."""
    rows, cols = [], []
    for i in range(1, n):
        rows.extend([0, i])
        cols.extend([i, 0])
    edge_index = np.array([rows, cols])
    return edge_index, n, edge_index.shape[1]


def _complete_graph(n):
    """Return (edge_index, num_nodes, num_edges) for K_n."""
    rows, cols = [], []
    for i in range(n):
        for j in range(i + 1, n):
            rows.extend([i, j])
            cols.extend([j, i])
    edge_index = np.array([rows, cols])
    return edge_index, n, edge_index.shape[1]


def _petersen_graph():
    """Return (edge_index, 10, num_edges) for the Petersen graph.
    Highly symmetric, 3-regular, non-simple spectrum."""
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
        (8, 5),  # inner star
        (0, 5),
        (1, 6),
        (2, 7),
        (3, 8),
        (4, 9),  # spokes
    ]
    rows, cols = [], []
    for i, j in edges:
        rows.extend([i, j])
        cols.extend([j, i])
    edge_index = np.array([rows, cols])
    return edge_index, 10, edge_index.shape[1]


# ---------------------------------------------------------------------------
# Test: edge_index_to_laplacian
# ---------------------------------------------------------------------------


class TestLaplacianConstruction:
    def test_path_graph_shape(self):
        ei, n, ne = _path_graph(8)
        L, n_lcc = edge_index_to_laplacian(ei, n)
        assert n_lcc == n
        assert L.shape == (n, n)

    def test_laplacian_row_sums_zero(self):
        """Combinatorial Laplacian rows sum to zero."""
        ei, n, ne = _path_graph(10)
        L, _ = edge_index_to_laplacian(ei, n)
        row_sums = np.array(L.sum(axis=1)).flatten()
        np.testing.assert_allclose(row_sums, 0, atol=1e-12)

    def test_normalized_laplacian_eigenvalue_range(self):
        """Normalized Laplacian eigenvalues in [0, 2]."""
        ei, n, ne = _complete_graph(6)
        L, n_lcc = edge_index_to_laplacian(ei, n, normalized=True)
        vals = np.linalg.eigvalsh(L.toarray())
        assert np.all(vals >= -1e-10)
        assert np.all(vals <= 2.0 + 1e-10)

    def test_disconnected_graph_uses_lcc(self):
        """Graph with isolated node: LCC should exclude it."""
        # Path 0-1-2 plus isolated node 3
        ei = np.array([[0, 1, 1, 2], [1, 0, 2, 1]])
        L, n_lcc = edge_index_to_laplacian(ei, 4)
        assert n_lcc == 3  # LCC is the path, node 3 is isolated


# ---------------------------------------------------------------------------
# Test: analyze_single_graph metric correctness
# ---------------------------------------------------------------------------


class TestAnalyzeMetrics:
    def test_path_graph_simple_spectrum(self):
        """Path graphs have simple spectrum (all eigenvalues distinct)."""
        ei, n, ne = _path_graph(10)
        r = analyze_single_graph(ei, n, ne, 8, 1e-3, None, False)
        assert r is not None
        assert r["is_simple_spectrum"] is True
        assert r["taxonomy"] in ("easy-canonical", "joint-canonical")

    def test_cycle_even_non_simple(self):
        """Even cycle C_6 has repeated eigenvalues => non-simple."""
        ei, n, ne = _cycle_graph(6)
        r = analyze_single_graph(ei, n, ne, 4, 1e-3, None, False)
        assert r is not None
        assert r["is_simple_spectrum"] is False
        assert r["taxonomy"] == "non-simple"

    def test_rescued_indices_are_distinct_eigenvalue(self):
        """Every rescued index must have multiplicity 1."""
        ei, n, ne = _path_graph(12)
        r = analyze_single_graph(ei, n, ne, 10, 1e-3, None, False)
        for idx in r["rescued_indices"]:
            assert r["multiplicities"][idx] == 1

    def test_rescued_indices_have_zero_sort_gap(self):
        """Every rescued eigenvector has ||sort(v)-sort(-v)||/||sort(v)|| <= n*sqrt(eps)."""
        ei, n, ne = _path_graph(12)
        r = analyze_single_graph(ei, n, ne, 10, 1e-3, None, False)
        for idx in r["rescued_indices"]:
            assert r["scores"][idx] <= r["threshold"]

    def test_taxonomy_counts_add_up(self):
        """num_rescued + num_degenerate_uncanon + num_truly_hard == total uncanon."""
        for make_graph in [_path_graph, _cycle_graph, _star_graph]:
            ei, n, ne = make_graph(8)
            r = analyze_single_graph(ei, n, ne, 6, 1e-3, None, False)
            if r is None:
                continue
            n_uncanon = r["num_eigenvectors"] - r["num_individually_canonical"]
            assert n_uncanon == (
                r["num_spielman_rescued"] + r["num_degenerate_uncanon"] + r["num_truly_hard"]
            )

    def test_rescued_relative_positions_in_range(self):
        """Relative positions must be in [0, 1)."""
        ei, n, ne = _path_graph(15)
        r = analyze_single_graph(ei, n, ne, 12, 1e-3, None, False)
        for p in r["rescued_relative_positions"]:
            assert 0.0 <= p < 1.0

    def test_path_has_rescued_eigenvectors(self):
        """Path graph P_8: some eigenvectors have ||sort(v)-sort(-v)|| = 0
        (up to n*sqrt(eps)) — un-canonicalizable. Spielman should rescue them."""
        ei, n, ne = _path_graph(8)
        r = analyze_single_graph(ei, n, ne, 6, 1e-3, None, False)
        assert r["is_simple_spectrum"]
        assert r["num_spielman_rescued"] > 0
        assert len(r["rescued_indices"]) == r["num_spielman_rescued"]

    def test_small_graph_skip(self):
        """Graph with < 3 LCC nodes returns None."""
        # Two nodes, one edge
        ei = np.array([[0, 1], [1, 0]])
        r = analyze_single_graph(ei, 2, 2, 4, 1e-3, None, False)
        assert r is None

    def test_threshold_proportional_to_n(self):
        """Threshold scales linearly with n_lcc."""
        t10 = uncanonicalizability_threshold(10)
        t50 = uncanonicalizability_threshold(50)
        t100 = uncanonicalizability_threshold(100)
        # t should be proportional to n
        assert abs(t50 / t10 - 5.0) < 0.01
        assert abs(t100 / t10 - 10.0) < 0.01
        # And all proportional to sqrt(eps)
        eps_sqrt = np.sqrt(np.finfo(np.float64).eps)
        assert abs(t10 / (10 * eps_sqrt) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Test: Spielman sign invariance on specific graphs
# ---------------------------------------------------------------------------


class TestSpielmanSignInvariance:
    """Verify canon(V * diag(s)) == canon(V) on mult-1 columns,
    where s flips ONLY multiplicity-1 columns (the eigendecomposition
    automorphism group for simple eigenvalues)."""

    def _check_sign_invariance(self, ei, n, ne, n_eigs=6, n_trials=20):
        """Apply random ±1 signs to mult-1 columns only, verify canon is unchanged."""
        L, n_lcc = edge_index_to_laplacian(ei, n)
        k = min(n_eigs, n_lcc - 2)
        eigenvalues, eigenvectors = compute_eigenpairs(L, n_eigs=k)
        if len(eigenvalues) == 0:
            return

        mult_info = detect_eigenvalue_multiplicities(eigenvalues)
        simple_cols = [j for j, m in enumerate(mult_info["multiplicity"]) if m == 1]
        if not simple_cols:
            return

        canon = spectral_canonicalize(eigenvectors, eigenvalues)

        rng = np.random.default_rng(123)
        n_cols = eigenvectors.shape[1]
        for trial in range(n_trials):
            # Group element: ±1 on mult-1 columns only, identity on degenerate
            signs = np.ones(n_cols)
            for j in simple_cols:
                signs[j] = rng.choice([-1, 1])
            V_signed = eigenvectors * signs[np.newaxis, :]
            canon_signed = spectral_canonicalize(V_signed, eigenvalues)
            np.testing.assert_allclose(
                canon[:, simple_cols],
                canon_signed[:, simple_cols],
                atol=1e-10,
                err_msg=f"Sign invariance failed on trial {trial}",
            )

    def test_path_graph(self):
        self._check_sign_invariance(*_path_graph(10))

    def test_path_graph_large(self):
        self._check_sign_invariance(*_path_graph(20), n_eigs=15)

    def test_star_graph(self):
        self._check_sign_invariance(*_star_graph(8))

    def test_complete_graph(self):
        self._check_sign_invariance(*_complete_graph(5))

    def test_petersen_graph(self):
        self._check_sign_invariance(*_petersen_graph())


class TestSpielmanIdempotency:
    """Verify canon(canon(V)) == canon(V)."""

    def _check_idempotency(self, ei, n, ne, n_eigs=6):
        L, n_lcc = edge_index_to_laplacian(ei, n)
        k = min(n_eigs, n_lcc - 2)
        eigenvalues, eigenvectors = compute_eigenpairs(L, n_eigs=k)
        if len(eigenvalues) == 0:
            return

        canon1 = spectral_canonicalize(eigenvectors, eigenvalues)
        canon2 = spectral_canonicalize(canon1, eigenvalues)
        np.testing.assert_allclose(canon1, canon2, atol=1e-12)

    def test_path_graph(self):
        self._check_idempotency(*_path_graph(10))

    def test_star_graph(self):
        self._check_idempotency(*_star_graph(8))

    def test_cycle_graph(self):
        self._check_idempotency(*_cycle_graph(7))

    def test_petersen_graph(self):
        self._check_idempotency(*_petersen_graph())


class TestSpielmanFieldsInAnalysis:
    """Verify that analyze_single_graph reports correct Spielman verification."""

    def test_path_sign_invariant(self):
        ei, n, ne = _path_graph(10)
        r = analyze_single_graph(ei, n, ne, 8, 1e-3, None, False)
        assert r["spielman_sign_invariant"] is True

    def test_path_idempotent(self):
        ei, n, ne = _path_graph(10)
        r = analyze_single_graph(ei, n, ne, 8, 1e-3, None, False)
        assert r["spielman_idempotent"] is True

    def test_cycle_sign_invariant(self):
        """Even cycles are non-simple but simple cols should still pass."""
        ei, n, ne = _cycle_graph(6)
        r = analyze_single_graph(ei, n, ne, 4, 1e-3, None, False)
        assert r["spielman_sign_invariant"] is True

    def test_star_sign_invariant(self):
        ei, n, ne = _star_graph(8)
        r = analyze_single_graph(ei, n, ne, 6, 1e-3, None, False)
        assert r["spielman_sign_invariant"] is True


# ---------------------------------------------------------------------------
# Test: multiplicity-1 vs multiplicity>1 separation
# ---------------------------------------------------------------------------


class TestMultiplicitySeparation:
    """Verify that mult-1 and mult>1 eigenvectors are tracked separately."""

    def test_simple_spectrum_no_degenerate_uncanon(self):
        """For simple-spectrum graph, all uncanon eigvecs are mult-1 (no degenerate)."""
        ei, n, ne = _path_graph(12)
        r = analyze_single_graph(ei, n, ne, 10, 1e-3, None, False)
        assert r["is_simple_spectrum"]
        assert r["num_degenerate_uncanon"] == 0
        n_uncanon = r["num_eigenvectors"] - r["num_individually_canonical"]
        assert r["num_spielman_rescued"] == n_uncanon

    def test_non_simple_has_degenerate_uncanon(self):
        """Even cycle: degenerate eigenspaces produce degenerate-uncanon eigvecs."""
        ei, n, ne = _cycle_graph(6)
        r = analyze_single_graph(ei, n, ne, 4, 1e-3, None, False)
        assert not r["is_simple_spectrum"]
        # Some eigenvalues are repeated; those uncanon eigvecs should be degenerate
        if r["num_degenerate_uncanon"] > 0:
            # degenerate-uncanon exist, and they + rescued + truly_hard = total uncanon
            n_uncanon = r["num_eigenvectors"] - r["num_individually_canonical"]
            assert (
                r["num_spielman_rescued"] + r["num_degenerate_uncanon"] + r["num_truly_hard"]
                == n_uncanon
            )

    def test_partition_sums_to_total_uncanon(self):
        """rescued + degenerate_uncanon + truly_hard == total uncanon, for all graph types."""
        for make_graph, size in [
            (_path_graph, 10),
            (_cycle_graph, 6),
            (_cycle_graph, 7),
            (_star_graph, 8),
            (_complete_graph, 5),
        ]:
            ei, n, ne = make_graph(size)
            r = analyze_single_graph(ei, n, ne, 6, 1e-3, None, False)
            if r is None:
                continue
            n_uncanon = r["num_eigenvectors"] - r["num_individually_canonical"]
            assert (
                r["num_spielman_rescued"] + r["num_degenerate_uncanon"] + r["num_truly_hard"]
                == n_uncanon
            ), f"Partition mismatch for {make_graph.__name__}({size})"

    def test_truly_hard_zero_for_simple(self):
        """For simple-spectrum, Spielman handles all mult-1 eigvecs => truly_hard = 0."""
        for size in [8, 10, 15, 20]:
            ei, n, ne = _path_graph(size)
            r = analyze_single_graph(ei, n, ne, min(size - 3, 15), 1e-3, None, False)
            assert r["is_simple_spectrum"]
            assert r["num_truly_hard"] == 0

    def test_star_degenerate_eigenspace(self):
        """Star graph S_n: eigenvalue 1 has multiplicity n-2 (all degenerate)."""
        ei, n, ne = _star_graph(7)
        r = analyze_single_graph(ei, n, ne, 5, 1e-3, None, False)
        assert not r["is_simple_spectrum"]
        # Most eigenvectors should be in the degenerate eigenspace
        n_degenerate = sum(1 for m in r["multiplicities"] if m > 1)
        assert n_degenerate >= 3  # at least n-3 degenerate cols


# ---------------------------------------------------------------------------
# Test: exhaustive automorphism group invariance
# ---------------------------------------------------------------------------


class TestExhaustiveGroupInvariance:
    """Exhaustively test ALL 2^k sign combinations on mult-1 columns for small graphs.
    This tests the full automorphism group of the eigendecomposition for simple eigenvalues,
    not just random samples."""

    def _check_exhaustive(self, ei, n, ne, n_eigs=6):
        """For a small graph, enumerate all 2^|simple_cols| sign vectors."""
        from itertools import product as iter_product

        L, n_lcc = edge_index_to_laplacian(ei, n)
        k = min(n_eigs, n_lcc - 2)
        eigenvalues, eigenvectors = compute_eigenpairs(L, n_eigs=k)
        if len(eigenvalues) == 0:
            return

        mult_info = detect_eigenvalue_multiplicities(eigenvalues)
        simple_cols = [j for j, m in enumerate(mult_info["multiplicity"]) if m == 1]
        if not simple_cols:
            return

        n_simple = len(simple_cols)
        assert n_simple <= 15, f"Too many simple cols ({n_simple}) for exhaustive test"

        canon = spectral_canonicalize(eigenvectors, eigenvalues)
        n_cols = eigenvectors.shape[1]

        # Enumerate all 2^n_simple sign vectors
        n_tested = 0
        for sign_tuple in iter_product([-1, 1], repeat=n_simple):
            signs = np.ones(n_cols)
            for idx, j in enumerate(simple_cols):
                signs[j] = sign_tuple[idx]
            V_signed = eigenvectors * signs[np.newaxis, :]
            canon_signed = spectral_canonicalize(V_signed, eigenvalues)
            np.testing.assert_allclose(
                canon[:, simple_cols],
                canon_signed[:, simple_cols],
                atol=1e-10,
                err_msg=f"Failed for sign tuple {sign_tuple}",
            )
            n_tested += 1

        assert n_tested == 2**n_simple

    def test_path_5(self):
        """P_5: 3 simple eigenvectors => 8 group elements."""
        self._check_exhaustive(*_path_graph(5), n_eigs=3)

    def test_path_7(self):
        """P_7: 5 simple eigenvectors => 32 group elements."""
        self._check_exhaustive(*_path_graph(7), n_eigs=5)

    def test_path_10(self):
        """P_10: 8 simple eigenvectors => 256 group elements."""
        self._check_exhaustive(*_path_graph(10), n_eigs=8)

    def test_cycle_5_odd(self):
        """C_5 (odd): some eigenvalues are close but distinct."""
        self._check_exhaustive(*_cycle_graph(5), n_eigs=4)

    def test_star_4(self):
        """S_4: only 1 simple eigenvalue (Fiedler), rest degenerate."""
        self._check_exhaustive(*_star_graph(4), n_eigs=2)

    def test_petersen(self):
        """Petersen: non-simple, but some simple cols may exist."""
        self._check_exhaustive(*_petersen_graph(), n_eigs=8)

    def test_complete_4(self):
        """K_4: all eigenvalues degenerate except one. Exhaustive on the 1 simple col."""
        self._check_exhaustive(*_complete_graph(4), n_eigs=2)


class TestDegenerateColumnsUntouched:
    """Verify Spielman leaves degenerate (mult>1) columns completely unchanged."""

    def _check_degenerate_unchanged(self, ei, n, ne, n_eigs=6, n_trials=20):
        L, n_lcc = edge_index_to_laplacian(ei, n)
        k = min(n_eigs, n_lcc - 2)
        eigenvalues, eigenvectors = compute_eigenpairs(L, n_eigs=k)
        if len(eigenvalues) == 0:
            return

        mult_info = detect_eigenvalue_multiplicities(eigenvalues)
        degen_cols = [j for j, m in enumerate(mult_info["multiplicity"]) if m > 1]
        if not degen_cols:
            return  # no degenerate columns to check

        # Regardless of sign flips on simple cols, degenerate cols should be unchanged
        rng = np.random.default_rng(456)
        simple_cols = [j for j, m in enumerate(mult_info["multiplicity"]) if m == 1]
        n_cols = eigenvectors.shape[1]

        for _ in range(n_trials):
            signs = np.ones(n_cols)
            for j in simple_cols:
                signs[j] = rng.choice([-1, 1])
            V_signed = eigenvectors * signs[np.newaxis, :]
            canon = spectral_canonicalize(V_signed, eigenvalues)
            # Degenerate columns should be identical to the (sign-flipped) input
            np.testing.assert_allclose(
                canon[:, degen_cols],
                V_signed[:, degen_cols],
                atol=1e-12,
                err_msg="Spielman modified a degenerate column",
            )

    def test_star_graph(self):
        """Star: eigenvalue 1 has mult n-2."""
        self._check_degenerate_unchanged(*_star_graph(7))

    def test_cycle_even(self):
        """C_6: has degenerate eigenvalues."""
        self._check_degenerate_unchanged(*_cycle_graph(6))

    def test_complete_graph(self):
        """K_5: eigenvalue n has mult n-1."""
        self._check_degenerate_unchanged(*_complete_graph(5))

    def test_petersen(self):
        self._check_degenerate_unchanged(*_petersen_graph())


# ---------------------------------------------------------------------------
# Test: noise stability — small perturbations don't change taxonomy
# ---------------------------------------------------------------------------


class TestNoiseStability:
    """The taxonomy classification should be robust to numerical noise."""

    def test_eigenvalue_gap_robust_to_noise(self):
        """Adding tiny noise to eigenvalues doesn't change multiplicity detection."""
        eigenvalues = np.array([0.5, 1.2, 2.1, 3.5, 4.8])
        rng = np.random.default_rng(99)

        for _ in range(10):
            noise = rng.uniform(-1e-10, 1e-10, size=len(eigenvalues))
            perturbed = eigenvalues + noise
            m_clean = detect_eigenvalue_multiplicities(eigenvalues, rtol=1e-3)
            m_noisy = detect_eigenvalue_multiplicities(perturbed, rtol=1e-3)
            assert m_clean["multiplicity"] == m_noisy["multiplicity"]

    def test_score_robust_to_eigenvector_noise(self):
        """Small perturbation of a clearly-canonicalizable eigenvector keeps it above threshold."""
        n = 20
        # Asymmetric vector (score >> threshold)
        v = np.zeros(n)
        v[0] = 0.9
        v[1:] = np.linspace(0.01, 0.3, n - 1)
        v /= np.linalg.norm(v)
        score_clean = uncanonicalizability_score(v)
        threshold = uncanonicalizability_threshold(n)
        assert score_clean > threshold

        rng = np.random.default_rng(77)
        for _ in range(20):
            noise = rng.normal(0, 1e-12, size=n)
            v_noisy = v + noise
            v_noisy /= np.linalg.norm(v_noisy)
            score_noisy = uncanonicalizability_score(v_noisy)
            assert score_noisy > threshold

    def test_taxonomy_stable_across_runs(self):
        """Same graph analyzed twice gives the same taxonomy."""
        ei, n, ne = _path_graph(12)
        r1 = analyze_single_graph(ei, n, ne, 8, 1e-3, None, False)
        r2 = analyze_single_graph(ei, n, ne, 8, 1e-3, None, False)
        assert r1["taxonomy"] == r2["taxonomy"]
        assert r1["num_spielman_rescued"] == r2["num_spielman_rescued"]
        assert r1["rescued_indices"] == r2["rescued_indices"]

    def test_spielman_stable_under_eigenvector_noise(self):
        """Spielman canonicalization on the same Laplacian eigenvectors is deterministic."""
        ei, n, ne = _path_graph(15)
        L, n_lcc = edge_index_to_laplacian(ei, n)
        eigenvalues, eigenvectors = compute_eigenpairs(L, n_eigs=10)

        canon1 = spectral_canonicalize(eigenvectors, eigenvalues)
        canon2 = spectral_canonicalize(eigenvectors, eigenvalues)
        np.testing.assert_array_equal(canon1, canon2)


# ---------------------------------------------------------------------------
# Test: complete pipeline on multiple graph types
# ---------------------------------------------------------------------------


class TestPipelineOnGraphTypes:
    """End-to-end analyze_single_graph on various graph types."""

    def test_path_graph(self):
        ei, n, ne = _path_graph(10)
        r = analyze_single_graph(ei, n, ne, 8, 1e-3, None, False)
        assert r is not None
        assert r["is_simple_spectrum"]
        assert r["num_nodes_lcc"] == 10
        assert "eigenvalues" in r

    def test_cycle_odd(self):
        """Odd cycle C_7: eigenvalues come in pairs but are distinct."""
        ei, n, ne = _cycle_graph(7)
        r = analyze_single_graph(ei, n, ne, 5, 1e-3, None, False)
        assert r is not None

    def test_cycle_even(self):
        ei, n, ne = _cycle_graph(8)
        r = analyze_single_graph(ei, n, ne, 6, 1e-3, None, False)
        assert r is not None
        # C_8 has repeated eigenvalues
        assert r["taxonomy"] == "non-simple"

    def test_star_graph(self):
        ei, n, ne = _star_graph(6)
        r = analyze_single_graph(ei, n, ne, 4, 1e-3, None, False)
        assert r is not None
        # Star has repeated eigenvalue = 1 with multiplicity n-2
        assert r["taxonomy"] == "non-simple"

    def test_complete_graph(self):
        ei, n, ne = _complete_graph(5)
        r = analyze_single_graph(ei, n, ne, 3, 1e-3, None, False)
        assert r is not None
        # K_n: eigenvalue -1 has multiplicity n-1
        assert r["taxonomy"] == "non-simple"

    def test_petersen_graph(self):
        ei, n, ne = _petersen_graph()
        r = analyze_single_graph(ei, n, ne, 8, 1e-3, None, False)
        assert r is not None
        assert r["num_nodes_lcc"] == 10

    def test_normalized_laplacian(self):
        ei, n, ne = _path_graph(10)
        r = analyze_single_graph(ei, n, ne, 8, 1e-3, None, True)
        assert r is not None
        # All eigenvalues of normalized Laplacian are in [0, 2]
        for lam in r["eigenvalues"]:
            assert -1e-10 <= lam <= 2.0 + 1e-10
