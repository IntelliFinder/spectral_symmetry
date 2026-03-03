"""Comprehensive tests for uncanonicalizability_score and its threshold logic.

These tests verify that the computational-precision threshold (n * sqrt(eps))
correctly separates genuinely anti-symmetric eigenvectors from generic
zero-mean vectors that happen to have low scores.

Mathematical background
-----------------------
The score computes:

    score = ||sort(v) - sort(-v)|| / ||sort(v)||

Since sort(-v) = -reversed(sort(v)), this simplifies to:

    score = ||sort(v) + reversed(sort(v))|| / ||v||

For a unit vector, ||v|| = 1, so score = ||sort(v) + reversed(sort(v))||.

This measures distributional asymmetry around zero. Perfectly symmetric
distributions (e.g., [-a, a]) yield score = 0.

Key question: Is the threshold 5/sqrt(n) well-calibrated relative to the
score's range, especially for small n?
"""

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.spectral_core import (
    build_graph_laplacian,
    compute_eigenpairs,
    uncanonicalizability_score,
    uncanonicalizability_threshold,
)

# ============================================================
# Section 1: Known easy cases (clearly canonicalizable)
# ============================================================


class TestKnownEasyCases:
    """Vectors that are clearly canonicalizable should have high scores."""

    def test_spike_vector(self):
        """v = [1, 0, 0, ...] is clearly canonicalizable.

        sort(v) = [0, ..., 0, 1]
        sort(-v) = [-1, 0, ..., 0]
        diff = [1, 0, ..., 0, 1]
        ||diff|| = sqrt(2)
        score = sqrt(2) / 1 = 1.414
        """
        n = 100
        v = np.zeros(n)
        v[0] = 1.0
        score = uncanonicalizability_score(v)
        assert abs(score - np.sqrt(2)) < 1e-10, (
            f"Spike vector should have score sqrt(2) ~ 1.414, got {score}"
        )

    def test_all_positive_vector(self):
        """v = [1/sqrt(n), ...] (all same sign) should be maximally asymmetric.

        score = 2.0 for a constant positive vector.
        """
        n = 100
        v = np.ones(n) / np.sqrt(n)
        score = uncanonicalizability_score(v)
        assert abs(score - 2.0) < 1e-10, (
            f"All-positive unit vector should have score 2.0, got {score}"
        )

    def test_mostly_positive_vector(self):
        """A vector with most entries positive should have a high score."""
        n = 100
        v = np.zeros(n)
        v[:90] = 1.0
        v[90:] = -0.1
        v = v / np.linalg.norm(v)
        score = uncanonicalizability_score(v)
        assert score > 1.0, f"Mostly-positive vector should have score > 1.0, got {score}"

    def test_one_dominant_entry(self):
        """v with one large entry and small others should be canonicalizable."""
        n = 50
        v = np.random.RandomState(42).randn(n) * 0.01
        v[0] = 1.0
        v = v / np.linalg.norm(v)
        score = uncanonicalizability_score(v)
        assert score > 1.0, f"One-dominant-entry vector should have score > 1.0, got {score}"


# ============================================================
# Section 2: Known hard cases (uncanonicalizable / ambiguous)
# ============================================================


class TestKnownHardCases:
    """Vectors with symmetric value distributions should have score near 0."""

    def test_perfectly_symmetric(self):
        """v = [-3, -2, -1, 1, 2, 3] / ||v|| is perfectly anti-symmetric.

        sort(v) = [-3, -2, -1, 1, 2, 3]
        sort(-v) = [-3, -2, -1, 1, 2, 3]
        diff = [0, 0, 0, 0, 0, 0]
        score = 0.0
        """
        v = np.array([-3, -2, -1, 1, 2, 3], dtype=float)
        score = uncanonicalizability_score(v)
        assert score < 1e-10, f"Perfectly symmetric vector should have score 0, got {score}"

    def test_two_entry_symmetric(self):
        """v = [1, -1, 0, 0, ...]/sqrt(2) has score = 0 (symmetric around 0)."""
        n = 50
        v = np.zeros(n)
        v[0] = 1.0 / np.sqrt(2)
        v[1] = -1.0 / np.sqrt(2)
        score = uncanonicalizability_score(v)
        assert score < 1e-10, f"[1, -1, 0, ...]/sqrt(2) should have score ~0, got {score}"

    def test_gaussian_symmetric(self):
        """A vector symmetric around zero: v_i, -v_i pairs."""
        n = 50
        half = np.arange(1, n // 2 + 1, dtype=float)
        v = np.concatenate([half, -half])
        v = v / np.linalg.norm(v)
        score = uncanonicalizability_score(v)
        assert score < 1e-10, f"Perfectly paired symmetric vector should have score 0, got {score}"


# ============================================================
# Section 3: Score range analysis -- the critical investigation
# ============================================================


class TestScoreRange:
    """Verify the theoretical range of the score and check threshold calibration."""

    def test_maximum_score_is_two(self):
        """The maximum possible score for a unit vector is 2.0.

        Achieved when all entries have the same sign (constant vector).
        """
        for n in [10, 25, 50, 100, 1024]:
            v = np.ones(n) / np.sqrt(n)
            score = uncanonicalizability_score(v)
            assert abs(score - 2.0) < 1e-10, f"Max score should be 2.0 for n={n}, got {score}"

    def test_minimum_score_is_zero(self):
        """The minimum possible score is 0.0 (perfectly symmetric distribution)."""
        for n in [6, 10, 25, 50, 100]:
            half = np.arange(1, n // 2 + 1, dtype=float)
            v = np.concatenate([half, -half])
            v = v / np.linalg.norm(v)
            score = uncanonicalizability_score(v)
            assert score < 1e-10, f"Min score should be ~0 for n={n}, got {score}"

    def test_spike_vector_score_independent_of_n(self):
        """Score for v=[1,0,...,0] should be sqrt(2) regardless of n."""
        for n in [10, 25, 50, 100, 1024]:
            v = np.zeros(n)
            v[0] = 1.0
            score = uncanonicalizability_score(v)
            assert abs(score - np.sqrt(2)) < 1e-10, (
                f"Spike vector score should be sqrt(2) for n={n}, got {score}"
            )

    def test_threshold_never_exceeds_max_score(self):
        """The adaptive threshold must never exceed the max achievable score.

        The max score for a zero-mean vector is sqrt(2(n-2)/(n-1)).
        The fixed threshold caps at 35% of this, so it's always well below.
        """
        for n in [5, 6, 7, 10, 15, 25, 50, 100, 1024]:
            threshold = uncanonicalizability_threshold(n)
            max_zero_mean = np.sqrt(2.0 * (n - 2) / (n - 1))
            assert threshold <= 0.35 * max_zero_mean + 1e-12, (
                f"For n={n}, threshold={threshold:.3f} exceeds "
                f"35% of max_zero_mean={max_zero_mean:.3f}"
            )

    def test_threshold_spike_vector_never_flagged(self):
        """A spike vector [1, 0, ..., 0] with score sqrt(2) should never be
        flagged as uncanonicalizable for any graph size.
        """
        spike_score = np.sqrt(2)
        for n in [5, 10, 12, 25, 50, 100, 1024]:
            threshold = uncanonicalizability_threshold(n)
            assert threshold < spike_score, (
                f"For n={n}, threshold {threshold:.3f} flags spike vector (score {spike_score:.3f})"
            )

    def test_threshold_matches_formula(self):
        """Threshold should equal n * sqrt(machine_eps) for all n > 2."""
        eps_sqrt = np.sqrt(np.finfo(np.float64).eps)
        for n in [5, 10, 50, 500, 1024, 4096]:
            threshold = uncanonicalizability_threshold(n)
            expected = n * eps_sqrt
            assert abs(threshold - expected) < 1e-15, (
                f"For n={n}, threshold {threshold:.2e} != n*sqrt(eps) {expected:.2e}"
            )


# ============================================================
# Section 4: Eigenvector-specific tests on small graphs
# ============================================================


class TestEigenvectorScores:
    """Test on actual Laplacian eigenvectors from small graphs."""

    @staticmethod
    def _path_graph_laplacian(n):
        """Build the combinatorial Laplacian of a path graph on n nodes."""
        diag = 2.0 * np.ones(n)
        diag[0] = 1.0
        diag[-1] = 1.0
        off = -1.0 * np.ones(n - 1)
        L = sp.diags([off, diag, off], [-1, 0, 1], shape=(n, n), format="csr")
        return L

    @staticmethod
    def _cycle_graph_laplacian(n):
        """Build the combinatorial Laplacian of a cycle graph on n nodes."""
        diag = 2.0 * np.ones(n)
        off = -1.0 * np.ones(n - 1)
        L = sp.diags([off, diag, off], [-1, 0, 1], shape=(n, n), format="csr")
        L = sp.lil_matrix(L)
        L[0, n - 1] = -1.0
        L[n - 1, 0] = -1.0
        return L.tocsr()

    @staticmethod
    def _complete_graph_laplacian(n):
        """Build the combinatorial Laplacian of the complete graph K_n."""
        A = np.ones((n, n)) - np.eye(n)
        D = (n - 1) * np.eye(n)
        L = D - A
        return sp.csr_matrix(L)

    def test_path_graph_small(self):
        """Path graph on 25 nodes: eigenvectors are cosine functions.

        Cosine eigenvectors are NOT symmetric around zero (they have
        a positive mean), so they should have nonzero scores.
        But are the scores above the threshold 5/sqrt(25) = 1.0?
        """
        n = 25
        L = self._path_graph_laplacian(n)
        threshold = uncanonicalizability_threshold(n)  # = 1.0

        vals, vecs = sla.eigsh(L, k=n - 2, which="SM", tol=1e-10)
        idx = np.argsort(vals)
        vals = vals[idx]
        vecs = vecs[:, idx]

        # Skip the trivial (constant) eigenvector
        nontrivial = vals > 1e-6
        vals = vals[nontrivial]
        vecs = vecs[:, nontrivial]

        scores = [uncanonicalizability_score(vecs[:, i]) for i in range(vecs.shape[1])]
        n_below_threshold = sum(1 for s in scores if s < threshold)

        # Record stats for the report
        print(f"\nPath graph n={n}, threshold={threshold:.3f}")
        print(f"  Scores: {[f'{s:.4f}' for s in scores[:10]]}")
        print(f"  {n_below_threshold}/{len(scores)} eigenvectors below threshold")
        print(f"  Max score: {max(scores):.4f}, Min score: {min(scores):.4f}")

        # The Fiedler vector of a path graph is a half-cosine, which is
        # NOT symmetric around zero. Its score should be nonzero.
        # But the key question is: is it above 1.0?
        # If most scores are below 1.0, the threshold is too aggressive.
        assert len(scores) > 0, "Should have at least some eigenvectors"

    def test_cycle_graph_small(self):
        """Cycle graph on 25 nodes: eigenvectors come in sin/cos pairs.

        The cycle graph has exact symmetry, but eigsh returns arbitrary
        linear combinations within degenerate eigenspaces.  These
        combinations are NOT guaranteed to be perfectly anti-symmetric,
        so their scores may be O(0.01–0.1), well above machine precision.

        The computational-precision threshold correctly identifies these
        as NOT exactly uncanonicalizable.  To detect the underlying
        symmetry, eigenvalue multiplicities should be used instead.
        """
        n = 25
        L = self._cycle_graph_laplacian(n)
        threshold = uncanonicalizability_threshold(n)

        vals, vecs = sla.eigsh(L, k=n - 2, which="SM", tol=1e-10)
        idx = np.argsort(vals)
        vals = vals[idx]
        vecs = vecs[:, idx]

        nontrivial = vals > 1e-6
        vals = vals[nontrivial]
        vecs = vecs[:, nontrivial]

        scores = [uncanonicalizability_score(vecs[:, i]) for i in range(vecs.shape[1])]

        print(f"\nCycle graph n={n}, threshold={threshold:.2e}")
        print(f"  Scores: {[f'{s:.4f}' for s in scores[:10]]}")
        print(f"  Max score: {max(scores):.4f}, Min score: {min(scores):.4f}")

        # Scores should be moderate (not near 2.0) since the cycle
        # eigenvectors are at least approximately anti-symmetric.
        median_score = np.median(scores)
        assert median_score < 0.5, (
            f"Cycle graph median score should be < 0.5, got {median_score:.4f}"
        )
        # Most eigenvalues should have multiplicity > 1 (degenerate pairs)
        from src.spectral_core import detect_eigenvalue_multiplicities

        mult_info = detect_eigenvalue_multiplicities(vals)
        n_degenerate = sum(1 for m in mult_info["multiplicity"] if m > 1)
        assert n_degenerate >= 0.8 * len(vals), (
            f"Expected >=80% degenerate eigenvalues in cycle graph, got {n_degenerate}/{len(vals)}"
        )

    def test_path_graph_fiedler_vector(self):
        """The Fiedler vector of a path graph is cos(pi*i/(n-1)).

        This is NOT symmetric around zero (it has a specific shape), so
        it should have a meaningful score. Let's check if the score
        exceeds the threshold for n=25.
        """
        n = 25
        # Analytical Fiedler vector of path graph
        fiedler = np.cos(np.pi * np.arange(n) / (n - 1))
        fiedler = fiedler / np.linalg.norm(fiedler)
        score = uncanonicalizability_score(fiedler)
        threshold = uncanonicalizability_threshold(n)

        print(f"\nPath graph Fiedler vector (n={n})")
        print(f"  Score: {score:.4f}, Threshold: {threshold:.3f}")
        print(f"  Flagged as uncanonicalizable: {score < threshold}")

        # The cosine is roughly symmetric around 0 for the path graph,
        # so the score will be low. This is actually correct behavior
        # for this particular vector.

    def test_eigenvector_scores_on_point_cloud(self):
        """Test on a real point cloud to see typical score distributions.

        For n=1024, threshold = 5/sqrt(1024) ~ 0.156.
        For n=25, threshold = 5/sqrt(25) = 1.0.
        """
        rng = np.random.RandomState(42)

        for n in [25, 100, 1024]:
            points = rng.randn(n, 3)
            L, comp_idx = build_graph_laplacian(points, n_neighbors=min(12, n - 1))
            k = min(10, L.shape[0] - 2)
            if k < 1:
                continue
            eigenvalues, eigenvectors = compute_eigenpairs(L, n_eigs=k)
            if len(eigenvalues) == 0:
                continue

            threshold = uncanonicalizability_threshold(eigenvectors.shape[0])
            scores = [
                uncanonicalizability_score(eigenvectors[:, i]) for i in range(eigenvectors.shape[1])
            ]
            n_below = sum(1 for s in scores if s < threshold)
            pct_below = 100 * n_below / len(scores) if scores else 0

            print(f"\nPoint cloud n={n}, threshold={threshold:.3f}")
            print(f"  Scores: {[f'{s:.4f}' for s in scores]}")
            print(f"  {n_below}/{len(scores)} ({pct_below:.0f}%) below threshold")
            print(f"  Max score: {max(scores):.4f}, Min score: {min(scores):.4f}")


# ============================================================
# Section 5: Threshold calibration analysis
# ============================================================


class TestThresholdCalibration:
    """Test whether the computational-precision threshold is well-calibrated.

    The threshold uses n * sqrt(eps_machine) to flag only eigenvectors whose
    anti-symmetry is exact to within numerical precision.
    """

    def test_threshold_vs_empirical_scores_small_graph(self):
        """On a small graph (n=25), measure how many eigenvectors exceed threshold.

        If nearly all eigenvectors have score < threshold, the threshold
        is miscalibrated for this graph size.
        """
        n = 25
        rng = np.random.RandomState(42)
        points = rng.randn(n, 3)
        L, comp_idx = build_graph_laplacian(points, n_neighbors=min(8, n - 1))
        k = min(10, L.shape[0] - 2)
        eigenvalues, eigenvectors = compute_eigenpairs(L, n_eigs=k)

        threshold = uncanonicalizability_threshold(eigenvectors.shape[0])
        scores = [
            uncanonicalizability_score(eigenvectors[:, i]) for i in range(eigenvectors.shape[1])
        ]

        n_below = sum(1 for s in scores if s < threshold)
        pct_below = 100 * n_below / len(scores) if scores else 0

        print(f"\nSmall graph calibration test (n={n})")
        print(f"  Threshold: {threshold:.3f}")
        print(f"  Scores: {[f'{s:.4f}' for s in scores]}")
        print(f"  {n_below}/{len(scores)} ({pct_below:.0f}%) below threshold")

        # For this test, we just record the data. The key insight is:
        # if pct_below is very high (say > 80%), the threshold is too aggressive.

    def test_random_gaussian_vectors_score_distribution(self):
        """Random Gaussian vectors should have moderate scores (not 0 or 2).

        A random Gaussian vector has no special symmetry, so the score should
        be somewhere in the middle. The threshold should not flag most of them.
        """
        rng = np.random.RandomState(42)
        scores_by_n = {}

        for n in [10, 25, 50, 100, 500, 1024]:
            threshold = uncanonicalizability_threshold(n)
            scores = []
            for trial in range(100):
                v = rng.randn(n)
                v = v / np.linalg.norm(v)
                scores.append(uncanonicalizability_score(v))

            scores = np.array(scores)
            n_below = np.sum(scores < threshold)
            scores_by_n[n] = scores

            print(f"\nRandom Gaussian vectors (n={n}), threshold={threshold:.3f}")
            print(f"  Mean score: {scores.mean():.4f}")
            print(f"  Std score: {scores.std():.4f}")
            print(f"  Min score: {scores.min():.4f}")
            print(f"  Max score: {scores.max():.4f}")
            print(f"  {n_below}/100 ({n_below}%) below threshold")

        # Key insight: for small n, random vectors are more likely to
        # have low scores just by chance, AND the threshold is higher.
        # Double whammy causing over-flagging.

    def test_orthogonal_to_constant_vectors(self):
        """Eigenvectors are orthogonal to the constant vector (zero mean).

        This constraint makes vectors more symmetric around zero,
        pushing scores DOWN. Combined with a high threshold for small n,
        this explains the over-flagging.
        """
        rng = np.random.RandomState(42)

        for n in [10, 25, 50, 100]:
            threshold = uncanonicalizability_threshold(n)
            scores = []
            for trial in range(200):
                v = rng.randn(n)
                # Project out the constant vector to simulate eigenvector constraint
                v = v - v.mean()
                v = v / np.linalg.norm(v)
                scores.append(uncanonicalizability_score(v))

            scores = np.array(scores)
            n_below = np.sum(scores < threshold)

            print(f"\nZero-mean random vectors (n={n}), threshold={threshold:.3f}")
            print(f"  Mean score: {scores.mean():.4f}")
            print(f"  Std score: {scores.std():.4f}")
            print(f"  Min score: {scores.min():.4f}")
            print(f"  Max score: {scores.max():.4f}")
            print(f"  {n_below}/200 ({n_below / 2:.1f}%) below threshold")

    def test_score_maximum_for_zero_mean_vectors(self):
        """What is the maximum score achievable by a zero-mean unit vector?

        The constraint sum(v_i) = 0 means the vector cannot be all-positive,
        limiting the maximum score to less than 2.0.

        The max-score zero-mean unit vector has the form:
        v = [-a, -a, ..., -a, b]  where (n-1)*a = b and ||v|| = 1
        This gives a = 1/sqrt(n*(n-1)), b = (n-1)/sqrt(n*(n-1)) = sqrt((n-1)/n)

        For this vector:
        sort(v) = [-a, ..., -a, b]
        sort(-v) = [-b, a, ..., a]
        diff = [-a + b, ..., -a - a, b + b] -- wait, let me compute properly.

        Actually sort(v) = [-a, -a, ..., -a, b]  (n-1 copies of -a, then b)
        sort(-v) = sort([a, a, ..., a, -b]) = [-b, a, a, ..., a]
        diff = sort(v) - sort(-v) = [-a - (-b), -a - a, ..., -a - a, b - a]
             = [b - a, -2a, ..., -2a, b - a]
             First entry: b - a
             Middle (n-2) entries: -2a
             Last entry: b - a

        ||diff||^2 = 2*(b-a)^2 + (n-2)*4*a^2

        With a = 1/sqrt(n(n-1)) and b = (n-1)*a:
        b - a = (n-2)*a
        ||diff||^2 = 2*(n-2)^2*a^2 + 4*(n-2)*a^2 = (n-2)*a^2 * (2*(n-2) + 4)
                   = (n-2)*a^2 * (2n) = 2n(n-2) * a^2 = 2n(n-2) / (n(n-1)) = 2(n-2)/(n-1)
        ||diff|| = sqrt(2(n-2)/(n-1))
        score = ||diff|| / 1 = sqrt(2(n-2)/(n-1))

        For n=25: score = sqrt(2*23/24) = sqrt(46/24) = sqrt(1.917) ~ 1.384
        Threshold = 5/sqrt(25) = 1.0
        So this extreme vector passes (1.384 > 1.0). But real eigenvectors
        are much more spread out than this extreme case.
        """
        for n in [10, 25, 50, 100, 1024]:
            a = 1.0 / np.sqrt(n * (n - 1))
            b = (n - 1) * a
            v = np.full(n, -a)
            v[-1] = b
            # Verify zero mean and unit norm
            assert abs(v.sum()) < 1e-10, f"Not zero mean for n={n}"
            assert abs(np.linalg.norm(v) - 1.0) < 1e-10, f"Not unit norm for n={n}"

            score = uncanonicalizability_score(v)
            theoretical = np.sqrt(2 * (n - 2) / (n - 1))
            threshold = uncanonicalizability_threshold(n)

            print(f"\nMax-score zero-mean vector (n={n})")
            print(f"  Score: {score:.4f}, Theoretical: {theoretical:.4f}")
            print(f"  Threshold: {threshold:.3f}")
            print(f"  Score > threshold: {score > threshold}")

            assert abs(score - theoretical) < 1e-8, (
                f"Score {score:.6f} != theoretical {theoretical:.6f} for n={n}"
            )


# ============================================================
# Section 6: Edge cases
# ============================================================


class TestEdgeCases:
    """Test edge cases for the uncanonicalizability score."""

    def test_zero_vector(self):
        """Zero vector should return 0."""
        v = np.zeros(10)
        assert uncanonicalizability_score(v) == 0.0

    def test_single_entry(self):
        """A single-entry vector [c] should have score 2.0 (if c > 0).

        sort(v) = [c], sort(-v) = [-c]
        diff = [2c], ||diff|| = 2|c|, ||v|| = |c|
        score = 2.0
        """
        v = np.array([3.0])
        score = uncanonicalizability_score(v)
        assert abs(score - 2.0) < 1e-10, f"Single entry should give score 2.0, got {score}"

    def test_two_equal_opposite_entries(self):
        """v = [a, -a] is perfectly symmetric, score = 0."""
        v = np.array([1.0, -1.0])
        score = uncanonicalizability_score(v)
        assert score < 1e-10, f"[a, -a] should have score 0, got {score}"

    def test_two_same_sign_entries(self):
        """v = [a, a] is maximally asymmetric, score = 2.0."""
        v = np.array([1.0, 1.0])
        score = uncanonicalizability_score(v)
        assert abs(score - 2.0) < 1e-10, f"[a, a] should have score 2.0, got {score}"

    def test_all_equal_magnitude_alternating_sign(self):
        """v = [1, -1, 1, -1, ...] / sqrt(n) is perfectly symmetric."""
        n = 100
        v = np.array([(-1) ** i for i in range(n)], dtype=float) / np.sqrt(n)
        score = uncanonicalizability_score(v)
        assert score < 1e-10, (
            f"Alternating-sign equal-magnitude vector should have score 0, got {score}"
        )

    def test_constant_vector_is_maximally_canonicalizable(self):
        """A constant vector (all same value) has the maximum score of 2.0.

        This is NOT a valid eigenvector (it's the trivial one), but it tests
        the score function at its maximum.
        """
        v = np.ones(50) / np.sqrt(50)
        score = uncanonicalizability_score(v)
        assert abs(score - 2.0) < 1e-10, f"Constant vector should have max score 2.0, got {score}"

    def test_near_zero_norm_vector(self):
        """A very small vector should return 0 (guarded by norm check)."""
        v = np.ones(10) * 1e-15
        score = uncanonicalizability_score(v)
        assert score == 0.0, f"Near-zero vector should return 0.0, got {score}"


# ============================================================
# Section 7: Threshold fix verification
# ============================================================


class TestThresholdFix:
    """Verify the computational-precision threshold avoids over-flagging.

    The old threshold 5/sqrt(n) was miscalibrated: for small n it exceeded
    the maximum achievable score, and even for moderate n it flagged generic
    zero-mean vectors.  The new threshold n * sqrt(eps) only flags vectors
    with score at machine precision.
    """

    def test_threshold_much_smaller_than_max_score(self):
        """The threshold should be negligible relative to the max achievable score."""
        for n in [5, 10, 15, 20, 25, 30, 50, 100, 500, 1024]:
            threshold = uncanonicalizability_threshold(n)
            max_zero_mean_score = np.sqrt(2.0 * (n - 2) / (n - 1))
            ratio = threshold / max_zero_mean_score

            print(
                f"n={n:>4}: threshold={threshold:.2e}, "
                f"max_score={max_zero_mean_score:.3f}, ratio={ratio:.2e}"
            )

            assert ratio < 1e-4, f"For n={n}, threshold/max_score ratio={ratio:.2e} is too large"

    def test_small_graph_not_all_flagged(self):
        """On a small path graph, the precision threshold should flag almost nothing.

        Path graph eigenvectors are cosine functions — their anti-symmetry is
        approximate, not exact, so scores are well above machine precision.
        """
        n = 25
        L = TestEigenvectorScores._path_graph_laplacian(n)
        threshold = uncanonicalizability_threshold(n)

        vals, vecs = sla.eigsh(L, k=n - 2, which="SM", tol=1e-10)
        idx = np.argsort(vals)
        vals = vals[idx]
        vecs = vecs[:, idx]
        nontrivial = vals > 1e-6
        vals = vals[nontrivial]
        vecs = vecs[:, nontrivial]

        scores = [uncanonicalizability_score(vecs[:, i]) for i in range(vecs.shape[1])]
        n_flagged = sum(1 for s in scores if s < threshold)
        pct_flagged = 100 * n_flagged / len(scores)

        print(f"\nPath graph n={n}, threshold={threshold:.3f}")
        print(f"Flagged: {n_flagged}/{len(scores)} ({pct_flagged:.0f}%)")
        for i, s in enumerate(scores):
            flag = "FLAGGED" if s < threshold else "ok"
            print(f"  Eigvec {i + 1}: score={s:.4f} [{flag}]")

        # With the fix, not all eigenvectors should be flagged
        assert pct_flagged < 100, f"Expected <100% flagged on path graph, got {pct_flagged:.0f}%"

    def test_max_score_zero_mean_vector_passes(self):
        """The maximally-asymmetric zero-mean vector must NOT be flagged.

        v = [-a, ..., -a, b] with zero mean and unit norm has score
        sqrt(2(n-2)/(n-1)) ~ 1.384 for n=25.  The precision-based threshold
        is many orders of magnitude below this.
        """
        for n in [10, 25, 50]:
            a = 1.0 / np.sqrt(n * (n - 1))
            b = (n - 1) * a
            v = np.full(n, -a)
            v[-1] = b

            score = uncanonicalizability_score(v)
            threshold = uncanonicalizability_threshold(n)

            assert score > threshold, (
                f"For n={n}, max zero-mean vector (score={score:.4f}) should "
                f"pass threshold={threshold:.3f}"
            )

    def test_complete_graph_eigenvectors(self):
        """Complete graph K_n has all eigenvalues equal to n (multiplicity n-1).

        The eigenvectors are genuinely uncanonicalizable due to the full
        eigenspace degeneracy, not because of sign ambiguity. The multiplicity
        detector should catch these.
        """
        n = 25
        L = TestEigenvectorScores._complete_graph_laplacian(n)
        threshold = uncanonicalizability_threshold(n)

        vals, vecs = sla.eigsh(L, k=n - 2, which="SM", tol=1e-10)
        idx = np.argsort(vals)
        vals = vals[idx]
        vecs = vecs[:, idx]
        nontrivial = vals > 1e-6
        vals = vals[nontrivial]
        vecs = vecs[:, nontrivial]

        scores = [uncanonicalizability_score(vecs[:, i]) for i in range(vecs.shape[1])]
        n_flagged = sum(1 for s in scores if s < threshold)

        print(f"\nComplete graph K_{n}, threshold={threshold:.3f}")
        print(f"  Eigenvalues: {vals[:5]} (should all be {n})")
        print(f"  Scores: {[f'{s:.4f}' for s in scores[:5]]}")
        print(f"  Flagged by score: {n_flagged}/{len(scores)}")
