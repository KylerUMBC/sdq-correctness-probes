"""Latent semantic trajectory encoder, dynamics, decoder, and gauge model.

Components (§9.2):
    A. Encoder:    h_{t-k:t+k} -> z_t  (temporal window -> latent)
    B. Dynamics:   z_{t+1} = z_t + f(z_t)  (latent evolution)
    C. Decoder:    R(z_t, u_t) -> h_hat_t  (reconstruct observed)
    D. Gauge:      surface-form context u_t
"""

from sdq.latent.encoder import TemporalConvEncoder, TemporalTransformerEncoder, MultiScaleConvEncoder
from sdq.latent.dynamics import LatentODE, LatentGRU
from sdq.latent.regime_dynamics import RegimeSwitchingDynamics
from sdq.latent.tube_dynamics import TubeAwareDynamics, TubeGeometry
from sdq.latent.decoder import ObservationDecoder, ResidualDecoder
from sdq.latent.gauge_model import GaugeEmbedding, GaugeEncoder, TemporalGaugeEncoder

__all__ = [
    "TemporalConvEncoder",
    "TemporalTransformerEncoder",
    "MultiScaleConvEncoder",
    "LatentODE",
    "LatentGRU",
    "RegimeSwitchingDynamics",
    "TubeAwareDynamics",
    "TubeGeometry",
    "ObservationDecoder",
    "ResidualDecoder",
    "GaugeEmbedding",
    "GaugeEncoder",
    "TemporalGaugeEncoder",
]
