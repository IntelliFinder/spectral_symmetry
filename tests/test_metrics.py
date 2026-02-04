"""Tests for metrics module."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metrics import aggregate_results


class TestAggregateResults:
    def test_empty_results(self):
        """Empty results list should return zeroed statistics."""
        stats = aggregate_results([], dataset_name="empty")
        assert stats.n_shapes == 0
        assert stats.avg_uncanon_ratio == 0.0
        assert stats.spectral_gap_mean == 0.0
        assert stats.fiedler_uncanon_rate == 0.0

    def test_mock_results(self):
        """Aggregate with known mock data should produce correct values."""
        results = [
            {
                'scores': [0.0, 0.02, 0.5],
                'uncanonicalizable': [True, True, False],
                'spectral_gap': 0.1,
                'multiplicity_info': {
                    'n_repeating': 0,
                    'n_non_repeating': 3,
                },
            },
            {
                'scores': [0.0, 0.4, 0.6],
                'uncanonicalizable': [True, False, False],
                'spectral_gap': 0.2,
                'multiplicity_info': {
                    'n_repeating': 2,
                    'n_non_repeating': 1,
                },
            },
        ]
        stats = aggregate_results(results, dataset_name="test")
        assert stats.n_shapes == 2
        assert stats.spectral_gap_mean == pytest.approx(0.15)
        # First result: 2/3 uncanon, second: 1/3 uncanon => avg = 0.5
        assert stats.avg_uncanon_ratio == pytest.approx(0.5, abs=0.01)
        # Only first result has fiedler (idx 1) uncanonicalizable
        assert stats.fiedler_uncanon_rate == pytest.approx(0.5)
        assert stats.avg_repeating_eigenvalues == pytest.approx(1.0)
        assert stats.avg_non_repeating_eigenvalues == pytest.approx(2.0)
