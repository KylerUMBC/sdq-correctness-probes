"""Loss functions for SDQ training.

Total objective (§10.7):
    L = L_rec + λ_sem·L_sem + λ_trans·L_trans + λ_cycle·L_cycle + λ_gauge·L_gauge + λ_event·L_event

Each loss targets trajectory equivalence under local transport,
not proxy clustering or answer-only similarity.
"""

from sdq.losses.reconstruction import reconstruction_loss
from sdq.losses.semantic_consistency import semantic_consistency_loss
from sdq.losses.transport_consistency import transport_consistency_loss
from sdq.losses.cycle import cycle_loss
from sdq.losses.gauge_regularization import gauge_regularization_loss
from sdq.losses.event_consistency import event_consistency_loss
from sdq.losses.total import SDQLoss
from sdq.losses.triplet import triplet_semantic_loss, batch_triplet_loss
from sdq.losses.velocity import latent_velocity_loss, latent_curvature_loss, latent_velocity_cosine_loss
from sdq.losses.composition import composition_loss, batch_composition_loss
from sdq.losses.retrieval import infonce_retrieval_loss, hard_retrieval_loss
from sdq.losses.trajectory import trajectory_shape_loss, temporal_contrastive_loss, phase_transition_loss

__all__ = [
    "reconstruction_loss",
    "semantic_consistency_loss",
    "transport_consistency_loss",
    "cycle_loss",
    "gauge_regularization_loss",
    "event_consistency_loss",
    "SDQLoss",
    "triplet_semantic_loss",
    "batch_triplet_loss",
    "latent_velocity_loss",
    "latent_curvature_loss",
    "latent_velocity_cosine_loss",
    "composition_loss",
    "batch_composition_loss",
    "infonce_retrieval_loss",
    "hard_retrieval_loss",
    "trajectory_shape_loss",
    "temporal_contrastive_loss",
    "phase_transition_loss",
]
