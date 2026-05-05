"""Appendix B (Augmentation vs. Canonization on ogbg-molpcba) — code package."""

from .spectral_core import (
    compute_eigenpairs,
    detect_eigenvalue_multiplicities,
)

__all__ = [
    "compute_eigenpairs",
    "detect_eigenvalue_multiplicities",
]
