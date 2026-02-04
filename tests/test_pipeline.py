"""End-to-end pipeline test: SymmetriaDataset -> analyze -> aggregate."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets import SymmetriaDataset
from src.metrics import aggregate_results
from src.spectral_core import analyze_spectrum


class TestPipeline:
    def test_end_to_end(self):
        """Full pipeline: generate shapes, analyze spectra, aggregate results."""
        ds = SymmetriaDataset(n_instances=1, n_points=100, noise_levels=(0.0,), seed=42)
        results = []
        for name, pts in ds:
            result = analyze_spectrum(pts, n_eigs=10, n_neighbors=5)
            result['name'] = name
            results.append(result)

        # Should have 5 shapes (1 instance × 1 noise level × 5 types)
        assert len(results) == 5

        stats = aggregate_results(results, dataset_name="symmetria_test")
        assert stats.n_shapes == 5
        assert stats.spectral_gap_mean > 0
        assert 0 <= stats.avg_uncanon_ratio <= 1
        assert 0 <= stats.fiedler_uncanon_rate <= 1
