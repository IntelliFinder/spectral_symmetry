"""Aggregation metrics and statistics for spectral symmetry analysis."""

from dataclasses import dataclass

import numpy as np

from .spectral_core import uncanonicalizability_score  # noqa: F401 — re-exported


@dataclass
class DatasetStatistics:
    """Aggregate statistics across a dataset."""
    dataset_name: str
    n_shapes: int = 0
    avg_uncanon_ratio: float = 0.0
    spectral_gap_mean: float = 0.0
    fiedler_uncanon_rate: float = 0.0
    avg_repeating_eigenvalues: float = 0.0
    avg_non_repeating_eigenvalues: float = 0.0


def aggregate_results(results_list, dataset_name="unknown"):
    """Compute aggregate statistics from a list of per-shape analysis results.

    Parameters
    ----------
    results_list : list of dict
        Each dict has keys: name, scores, uncanonicalizable, spectral_gap
    dataset_name : str

    Returns
    -------
    DatasetStatistics
    """
    if not results_list:
        return DatasetStatistics(dataset_name=dataset_name)

    n_shapes = len(results_list)
    uncanon_ratios = []
    spectral_gaps = []
    fiedler_uncanon_count = 0
    repeating_counts = []
    non_repeating_counts = []

    for r in results_list:
        uncanon = r['uncanonicalizable']
        if len(uncanon) > 0:
            uncanon_ratios.append(sum(uncanon) / len(uncanon))
        spectral_gaps.append(r['spectral_gap'])
        # Fiedler vector is eigenvector index 1 (skip trivial constant eigenvector 0)
        if len(uncanon) > 1 and uncanon[1]:
            fiedler_uncanon_count += 1
        # Multiplicity stats
        mult_info = r.get('multiplicity_info')
        if mult_info is not None:
            repeating_counts.append(mult_info['n_repeating'])
            non_repeating_counts.append(mult_info['n_non_repeating'])

    return DatasetStatistics(
        dataset_name=dataset_name,
        n_shapes=n_shapes,
        avg_uncanon_ratio=float(np.mean(uncanon_ratios)) if uncanon_ratios else 0.0,
        spectral_gap_mean=float(np.mean(spectral_gaps)),
        fiedler_uncanon_rate=fiedler_uncanon_count / n_shapes if n_shapes > 0 else 0.0,
        avg_repeating_eigenvalues=float(np.mean(repeating_counts)) if repeating_counts else 0.0,
        avg_non_repeating_eigenvalues=(
            float(np.mean(non_repeating_counts)) if non_repeating_counts else 0.0
        ),
    )
