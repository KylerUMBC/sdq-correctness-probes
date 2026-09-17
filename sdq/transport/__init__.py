"""Local gauge transport operators between trajectories.

The transport module learns G_t^{i->j}: operators that map local
motion in trajectory i to corresponding local motion in trajectory j.

These operators are:
    - smooth over time
    - near identity unless needed
    - low-rank or low-complexity
    - compositional where possible
"""

from sdq.transport.local_operator import LocalTransportField, TransportResult
from sdq.transport.low_rank_maps import LowRankTransport
from sdq.transport.cycle_consistency import (
    inversion_error,
    composition_error,
    cycle_consistency_metrics,
)
from sdq.transport.composition import compose_transports, invert_transport
from sdq.transport.prototypes import TransportPrototypes, PrototypeTransportResult, prototype_regularization_loss

__all__ = [
    "LocalTransportField",
    "TransportResult",
    "LowRankTransport",
    "inversion_error",
    "composition_error",
    "cycle_consistency_metrics",
    "compose_transports",
    "invert_transport",
    "TransportPrototypes",
    "PrototypeTransportResult",
    "prototype_regularization_loss",
]
