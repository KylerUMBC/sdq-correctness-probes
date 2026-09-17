"""Temporal risk model: forecast bad-basin entry from recent hidden-state dynamics.

Takes a window of (semantic state, drift state) pairs and predicts:
  - risk: P(bad outcome within next k steps)
  - recoverability: P(current drift will self-correct)
  - basin: predicted next basin type (optional)

The main SDQ early-warning output is the risk score r_t.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class RiskOutput:
    """Output of the temporal risk model at each timestep."""

    risk: Tensor             # [B, T] or [T]  P(bad outcome ahead)
    recoverability: Tensor   # [B, T] or [T]  P(drift is self-correcting)
    risk_logits: Tensor      # [B, T] or [T]  raw logits for risk head
    recover_logits: Tensor   # [B, T] or [T]  raw logits for recoverability head


class TemporalRiskModel(nn.Module):
    """GRU-based temporal model: (s_t, p_t) windows -> risk scores.

    Architecture:
      1. Input projection fuses semantic + drift streams
      2. Multi-layer GRU processes the temporal sequence
      3. Two output heads predict risk and recoverability

    The model is causal: predictions at time t only depend on
    observations up to and including time t.
    """

    def __init__(
        self,
        semantic_dim: int,
        drift_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_attention: bool = False,
    ):
        super().__init__()
        self.semantic_dim = semantic_dim
        self.drift_dim = drift_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_attention = use_attention

        input_dim = semantic_dim + drift_dim

        self.input_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
        )

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        if use_attention:
            self.attn = nn.MultiheadAttention(
                hidden_dim, num_heads=4, batch_first=True, dropout=dropout,
            )
            self.attn_norm = nn.LayerNorm(hidden_dim)

        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.recover_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        semantic: Tensor,
        drift: Tensor,
        hidden: Tensor | None = None,
    ) -> tuple[RiskOutput, Tensor]:
        """Run the temporal risk model.

        Args:
            semantic: [B, T, semantic_dim] or [T, semantic_dim]
            drift: [B, T, drift_dim] or [T, drift_dim]
            hidden: Optional initial GRU hidden state [num_layers, B, hidden_dim]

        Returns:
            (RiskOutput, final_hidden_state)
        """
        unbatched = semantic.dim() == 2
        if unbatched:
            semantic = semantic.unsqueeze(0)
            drift = drift.unsqueeze(0)

        B, T, _ = semantic.shape

        x = torch.cat([semantic, drift], dim=-1)  # [B, T, input_dim]
        x = self.input_proj(x)  # [B, T, hidden_dim]

        gru_out, h_n = self.gru(x, hidden)  # [B, T, hidden_dim], [num_layers, B, hidden_dim]

        if self.use_attention:
            # Causal self-attention mask
            mask = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
            )
            attn_out, _ = self.attn(gru_out, gru_out, gru_out, attn_mask=mask)
            gru_out = self.attn_norm(gru_out + attn_out)

        risk_logits = self.risk_head(gru_out).squeeze(-1)      # [B, T]
        recover_logits = self.recover_head(gru_out).squeeze(-1)  # [B, T]

        output = RiskOutput(
            risk=torch.sigmoid(risk_logits),
            recoverability=torch.sigmoid(recover_logits),
            risk_logits=risk_logits,
            recover_logits=recover_logits,
        )

        if unbatched:
            output = RiskOutput(
                risk=output.risk.squeeze(0),
                recoverability=output.recoverability.squeeze(0),
                risk_logits=output.risk_logits.squeeze(0),
                recover_logits=output.recover_logits.squeeze(0),
            )

        return output, h_n

    def predict_risk(
        self,
        semantic: Tensor,
        drift: Tensor,
    ) -> Tensor:
        """Convenience: return just the risk scores.

        Args:
            semantic: [B, T, semantic_dim] or [T, semantic_dim]
            drift: [B, T, drift_dim] or [T, drift_dim]

        Returns:
            [B, T] or [T] risk probabilities.
        """
        output, _ = self.forward(semantic, drift)
        return output.risk
