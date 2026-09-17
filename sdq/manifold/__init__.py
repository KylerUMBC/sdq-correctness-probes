"""sdq.manifold — analytic manifold geometry for hidden-state analysis.

Exports:
    ManifoldExtractor    — PCA-based encoder (drop-in for MultiScaleConvEncoder)
    intrinsic_dimension  — min k for threshold variance
    spectral_profile     — variance explained, effective rank, spectral entropy
    helix_score          — ring-uniformity score ∈ [0, 1]
    subspace_overlap     — mean squared cosine of principal angles ∈ [0, 1]
    ProcrustesTransport  — orthogonal Procrustes alignment
    ProcrustesResult     — dataclass returned by ProcrustesTransport
"""

from sdq.manifold.extractor import ManifoldExtractor
from sdq.manifold.geometry import (
    intrinsic_dimension,
    spectral_profile,
    helix_score,
    subspace_overlap,
)
from sdq.manifold.procrustes import ProcrustesTransport, ProcrustesResult

__all__ = [
    "ManifoldExtractor",
    "intrinsic_dimension",
    "spectral_profile",
    "helix_score",
    "subspace_overlap",
    "ProcrustesTransport",
    "ProcrustesResult",
]
