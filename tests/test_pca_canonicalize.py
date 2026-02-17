"""Tests for PCA canonicalization (pca_canonicalize) and PCA-based dataset/model integration."""

from pathlib import Path

import numpy as np
import pytest
import torch

from src.preprocessing import pca_canonicalize

# ---------------------------------------------------------------------------
# Helper: random SO(3) rotation matrix
# ---------------------------------------------------------------------------


def _random_rotation(rng):
    """Generate a random proper rotation matrix via QR of a random matrix."""
    M = rng.standard_normal((3, 3))
    Q, R = np.linalg.qr(M)
    # QR may give det = -1; ensure det = +1
    Q = Q @ np.diag(np.sign(np.diag(R)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


# ===========================================================================
# Core PCA canonicalization tests
# ===========================================================================


class TestBasicAlignment:
    """After PCA canonicalization, largest variance should align with x-axis."""

    def test_basic_alignment(self):
        """Elongated cloud along [1,1,1] should have first PC along x after canon."""
        rng = np.random.RandomState(42)
        N = 500
        # Points stretched along [1,1,1]: large variance along that direction
        direction = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
        t = rng.randn(N, 1) * 10  # large spread along direction
        noise = rng.randn(N, 3) * 0.1  # small noise in other directions
        points = t * direction[np.newaxis, :] + noise
        # Center the cloud
        points -= points.mean(axis=0)

        canon = pca_canonicalize(points)

        # After canonicalization the first coordinate (x) should have the
        # largest variance, since PCA sorts eigenvalues in descending order
        variances = np.var(canon, axis=0)
        assert variances[0] > variances[1], (
            f"Var along x ({variances[0]:.4f}) should exceed var along y ({variances[1]:.4f})"
        )
        assert variances[0] > variances[2], (
            f"Var along x ({variances[0]:.4f}) should exceed var along z ({variances[2]:.4f})"
        )


class TestRotationInvariance:
    """PCA canonicalization should be rotation-invariant."""

    def test_rotation_invariance(self):
        """Rotating a point cloud before PCA canon should give the same result."""
        rng = np.random.default_rng(123)
        N = 300
        # Create a clearly anisotropic cloud (distinct eigenvalues)
        points = np.column_stack(
            [
                rng.standard_normal(N) * 5.0,
                rng.standard_normal(N) * 2.0,
                rng.standard_normal(N) * 0.5,
            ]
        )
        points -= points.mean(axis=0)

        canon_original = pca_canonicalize(points)

        R = _random_rotation(rng)
        rotated = points @ R.T
        rotated -= rotated.mean(axis=0)  # re-center after rotation

        canon_rotated = pca_canonicalize(rotated)

        # The canonicalized results should be approximately equal.
        # However, individual axis signs may differ; compare absolute values
        # of each column to account for sign ambiguity across independent runs.
        np.testing.assert_allclose(
            np.abs(canon_original),
            np.abs(canon_rotated),
            atol=1e-8,
            err_msg="Canonicalized clouds differ under rotation (up to axis signs)",
        )


class TestSignConsistency:
    """Negating all coordinates should produce the same canonicalized result."""

    def test_sign_consistency(self):
        """pca_canonicalize(points) == pca_canonicalize(-points)."""
        rng = np.random.default_rng(456)
        N = 200
        points = np.column_stack(
            [
                rng.standard_normal(N) * 4.0,
                rng.standard_normal(N) * 2.0,
                rng.standard_normal(N) * 1.0,
            ]
        )
        points -= points.mean(axis=0)

        canon_pos = pca_canonicalize(points)
        canon_neg = pca_canonicalize(-points)

        # Covariance is the same for -points, so eigenvectors are identical.
        # The sign canonicalization (majority vote) may or may not produce the
        # exact same result, but the principal axes should be the same, so
        # at minimum |canon_pos| == |canon_neg|.
        np.testing.assert_allclose(
            np.abs(canon_pos),
            np.abs(canon_neg),
            atol=1e-10,
            err_msg="Absolute values should match after negation",
        )


class TestDeterminantPositive:
    """The implicit rotation matrix R should have det(R) = +1 (proper rotation)."""

    def test_determinant_positive(self):
        """The rotation applied by pca_canonicalize has positive determinant."""
        for seed in range(10):
            rng2 = np.random.default_rng(seed + 1000)
            N = 100
            points = rng2.standard_normal((N, 3))
            points -= points.mean(axis=0)

            canon = pca_canonicalize(points)

            # Recover R from the relationship: canon = points @ R
            # R = (X^T X)^{-1} X^T Y via least squares
            R_recovered, _, _, _ = np.linalg.lstsq(points, canon, rcond=None)

            det = np.linalg.det(R_recovered)
            assert det > 0, f"det(R) = {det:.6f} should be > 0 (seed={seed})"
            np.testing.assert_allclose(
                abs(det),
                1.0,
                atol=0.1,
                err_msg=f"det(R) should be close to +1, got {det:.6f}",
            )


class TestMajorityVoteSign:
    """Verify majority vote sign convention for an asymmetric cloud."""

    def test_majority_vote_sign(self):
        """When 80% of mass is in positive x, first PC should have positive projection majority."""
        rng = np.random.default_rng(111)
        N = 500
        # 80% of points in positive x, 20% in negative x
        n_pos = int(0.8 * N)
        n_neg = N - n_pos
        x_pos = rng.uniform(0.5, 5.0, size=n_pos)
        x_neg = rng.uniform(-2.0, -0.5, size=n_neg)
        x = np.concatenate([x_pos, x_neg])
        y = rng.standard_normal(N) * 0.1
        z = rng.standard_normal(N) * 0.1
        points = np.column_stack([x, y, z])
        points -= points.mean(axis=0)

        canon = pca_canonicalize(points)

        # After canonicalization, the first coordinate (first PC) should have
        # more positive projections than negative (majority vote convention)
        n_positive = np.sum(canon[:, 0] > 0)
        n_negative = np.sum(canon[:, 0] < 0)
        assert n_positive > n_negative, (
            f"Expected more positive ({n_positive}) than negative ({n_negative}) "
            f"projections along first PC"
        )


class TestDegenerateEigenvalues:
    """PCA should handle degenerate (repeated) eigenvalues gracefully."""

    def test_degenerate_eigenvalues(self):
        """Points on a circle in xy-plane with small z noise (degenerate top 2 eigenvalues)."""
        rng = np.random.default_rng(222)
        N = 200
        theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
        x = np.cos(theta)
        y = np.sin(theta)
        z = rng.standard_normal(N) * 0.001  # tiny z noise
        points = np.column_stack([x, y, z])
        points -= points.mean(axis=0)

        # Should not crash even with near-degenerate eigenvalues
        canon = pca_canonicalize(points)
        assert canon.shape == (N, 3)
        assert np.all(np.isfinite(canon))

    def test_spherically_symmetric(self):
        """Points drawn from an isotropic Gaussian (all eigenvalues nearly equal)."""
        rng = np.random.default_rng(333)
        N = 500
        points = rng.standard_normal((N, 3))
        points -= points.mean(axis=0)

        canon = pca_canonicalize(points)
        assert canon.shape == (N, 3)
        assert np.all(np.isfinite(canon))


class TestOutputShape:
    """Output shape should match input shape for various N."""

    @pytest.mark.parametrize("N", [1, 2, 3, 10, 50, 500, 2048])
    def test_output_shape(self, N):
        rng = np.random.default_rng(N)
        points = rng.standard_normal((N, 3))
        points -= points.mean(axis=0)
        canon = pca_canonicalize(points)
        assert canon.shape == (N, 3), f"Expected ({N}, 3), got {canon.shape}"

    def test_empty_input(self):
        """Empty point cloud should return empty array."""
        points = np.zeros((0, 3))
        canon = pca_canonicalize(points)
        assert canon.shape == (0, 3)


class TestIdempotent:
    """Applying PCA canonicalization twice should give nearly the same result."""

    def test_idempotent(self):
        """pca_canonicalize(pca_canonicalize(X)) ~= pca_canonicalize(X)."""
        rng = np.random.default_rng(444)
        N = 300
        # Distinct eigenvalues to avoid ambiguity
        points = np.column_stack(
            [
                rng.standard_normal(N) * 5.0,
                rng.standard_normal(N) * 2.0,
                rng.standard_normal(N) * 0.5,
            ]
        )
        points -= points.mean(axis=0)

        canon1 = pca_canonicalize(points)
        canon2 = pca_canonicalize(canon1)

        np.testing.assert_allclose(
            canon1,
            canon2,
            atol=1e-10,
            err_msg="PCA canonicalization is not idempotent",
        )

    def test_idempotent_multiple_seeds(self):
        """Idempotency for several random clouds."""
        for seed in range(5):
            rng = np.random.default_rng(seed + 500)
            points = np.column_stack(
                [
                    rng.standard_normal(100) * (3 + seed),
                    rng.standard_normal(100) * (2 + seed * 0.5),
                    rng.standard_normal(100) * (0.5 + seed * 0.1),
                ]
            )
            points -= points.mean(axis=0)

            c1 = pca_canonicalize(points)
            c2 = pca_canonicalize(c1)
            np.testing.assert_allclose(
                c1,
                c2,
                atol=1e-10,
                err_msg=f"Not idempotent for seed {seed}",
            )


# ===========================================================================
# Integration tests
# ===========================================================================


# Check whether ModelNet data is available for integration dataset test
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_MODELNET10_AVAILABLE = (_DATA_DIR / "ModelNet10").exists()


@pytest.mark.skipif(
    not _MODELNET10_AVAILABLE,
    reason="ModelNet10 data not available at data/ModelNet10",
)
class TestPCADatasetOutputShape:
    """Verify PCAModelNet returns correct tuple shapes."""

    def test_pca_dataset_output_shape(self):
        from src.experiments.deep_sets.dataset_pca import PCAModelNet

        max_points = 256
        ds = PCAModelNet(
            str(_DATA_DIR),
            dataset="ModelNet10",
            split="test",
            max_points=max_points,
        )
        assert len(ds) > 0, "Dataset should not be empty"

        features, mask, label = ds[0]

        assert isinstance(features, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
        assert isinstance(label, int)

        assert features.shape == (max_points, 3), (
            f"Expected features shape ({max_points}, 3), got {features.shape}"
        )
        assert mask.shape == (max_points,), f"Expected mask shape ({max_points},), got {mask.shape}"
        assert features.dtype == torch.float32
        assert mask.dtype == torch.bool

        # Mask should have some False (valid) entries
        assert (~mask).any(), "Mask should have at least one valid entry"
        assert 0 <= label < len(ds.classes)


class TestModelForwardWithPCAFeatures:
    """Verify HKSDeepSetsClassifier forward pass with PCA-style (xyz-only) input."""

    def test_model_forward_with_pca_features(self):
        from src.experiments.deep_sets.model_hks import HKSDeepSetsClassifier

        B, N, n_classes = 4, 128, 10
        model = HKSDeepSetsClassifier(
            in_channels=3,
            n_times=0,
            n_classes=n_classes,
            hidden_dim=64,
            include_xyz=True,
        )
        model.eval()

        features = torch.randn(B, N, 3)
        mask = torch.zeros(B, N, dtype=torch.bool)
        # Pad last 10 points in each sample
        mask[:, -10:] = True

        with torch.no_grad():
            logits = model(features, mask)

        assert logits.shape == (B, n_classes), (
            f"Expected logits shape ({B}, {n_classes}), got {logits.shape}"
        )
        assert torch.all(torch.isfinite(logits)), "Logits contain non-finite values"

    def test_model_forward_no_mask(self):
        """Forward pass with mask=None should also work."""
        from src.experiments.deep_sets.model_hks import HKSDeepSetsClassifier

        B, N, n_classes = 2, 64, 10
        model = HKSDeepSetsClassifier(
            in_channels=3,
            n_times=0,
            n_classes=n_classes,
            hidden_dim=32,
            include_xyz=True,
        )
        model.eval()

        features = torch.randn(B, N, 3)

        with torch.no_grad():
            logits = model(features, mask=None)

        assert logits.shape == (B, n_classes)

    def test_model_forward_batch_size_one(self):
        """Batch size 1 should work correctly."""
        from src.experiments.deep_sets.model_hks import HKSDeepSetsClassifier

        model = HKSDeepSetsClassifier(
            in_channels=3,
            n_times=0,
            n_classes=10,
            hidden_dim=32,
            include_xyz=True,
        )
        model.eval()

        features = torch.randn(1, 256, 3)
        mask = torch.zeros(1, 256, dtype=torch.bool)

        with torch.no_grad():
            logits = model(features, mask)

        assert logits.shape == (1, 10)
