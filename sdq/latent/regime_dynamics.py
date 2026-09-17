"""Regime-switching latent dynamics: per-regime velocity nets + gated mixture.

v5.3: Gumbel-softmax during training (tau annealed), argmax routing at eval.
Each regime has its own MLP; gate predicts mixture / task-family alignment.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RegimeSwitchingDynamics(nn.Module):
    """Gated mixture of per-regime velocity fields f_k(z).

    f(z) = sum_k w_k * f_k(z) with w from gate (Gumbel-softmax in train, one-hot at eval).
    """

    def __init__(
        self,
        latent_dim: int,
        num_regimes: int = 6,
        intermediate_dim: int | None = None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_regimes = num_regimes
        inter = intermediate_dim or latent_dim * 2

        self.gate = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, 128),
            nn.GELU(),
            nn.Linear(128, num_regimes),
        )
        self.velocity_nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(latent_dim),
                    nn.Linear(latent_dim, inter),
                    nn.GELU(),
                    nn.Linear(inter, latent_dim),
                )
                for _ in range(num_regimes)
            ]
        )
        for net in self.velocity_nets:
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)

    def gate_logits(self, z: torch.Tensor) -> torch.Tensor:
        """z: [*, D_z] -> [*, K] logits."""
        return self.gate(z)

    def gate_probs(self, z: torch.Tensor) -> torch.Tensor:
        """Softmax gate probabilities (for metrics / KL / supervision diagnostics)."""
        return F.softmax(self.gate_logits(z), dim=-1)

    def velocity_field(
        self,
        z: torch.Tensor,
        tau: float | None = None,
    ) -> torch.Tensor:
        """Mixture velocity f(z). Training + tau -> Gumbel-softmax; eval -> argmax one-hot."""
        squeezed = z.ndim == 1
        if squeezed:
            z = z.unsqueeze(0)

        logits = self.gate_logits(z)
        if self.training:
            if tau is not None:
                gate_w = F.gumbel_softmax(logits, tau=tau, dim=-1, hard=False)
            else:
                gate_w = F.softmax(logits, dim=-1)
        else:
            idx = logits.argmax(dim=-1)
            gate_w = F.one_hot(idx, num_classes=self.num_regimes).to(
                dtype=logits.dtype, device=logits.device,
            )

        vels = [self.velocity_nets[r](z) for r in range(self.num_regimes)]
        vel_stack = torch.stack(vels, dim=-2)  # [*, K, D]
        out = (gate_w.unsqueeze(-1) * vel_stack).sum(dim=-2)

        if squeezed:
            out = out.squeeze(0)
        return out

    def step(self, z_t: torch.Tensor, tau: float | None = None) -> torch.Tensor:
        return z_t + self.velocity_field(z_t, tau=tau)

    def rollout(self, z_0: torch.Tensor, T: int, tau: float | None = None) -> torch.Tensor:
        states = [z_0]
        z = z_0
        for _ in range(T - 1):
            z = self.step(z, tau=tau)
            states.append(z)
        return torch.stack(states)

    def multi_step_predict(
        self,
        z_t: torch.Tensor,
        num_substeps: int = 10,
        tau: float | None = None,
    ) -> torch.Tensor:
        z = z_t
        for _ in range(num_substeps):
            z = z + self.velocity_field(z, tau=tau)
        return z

    def forward(
        self,
        z: torch.Tensor,
        num_substeps: int = 10,
        tau: float | None = None,
    ) -> torch.Tensor:
        z_curr = z[:-1]
        for _ in range(num_substeps):
            z_curr = z_curr + self.velocity_field(z_curr, tau=tau)
        return z_curr

    def dynamics_consistency_loss(
        self,
        z: torch.Tensor,
        num_substeps: int = 10,
        tau: float | None = None,
    ) -> torch.Tensor:
        predicted = self.forward(z, num_substeps, tau=tau)
        actual = z[1:]
        return (predicted - actual).pow(2).sum(dim=-1).mean()

    def velocity_direction_loss(
        self,
        z: torch.Tensor,
        num_substeps: int = 10,
        tau: float | None = None,
    ) -> torch.Tensor:
        if z.shape[0] < 2:
            return torch.tensor(0.0, device=z.device, dtype=z.dtype)
        dz_true = z[1:] - z[:-1]
        predicted = self.forward(z, num_substeps, tau=tau)
        dz_pred = predicted - z[:-1]
        cos = F.cosine_similarity(dz_pred, dz_true, dim=-1)
        return (1.0 - cos).mean()

    def velocity_magnitude_loss(
        self,
        z: torch.Tensor,
        num_substeps: int = 10,
        tau: float | None = None,
    ) -> torch.Tensor:
        if z.shape[0] < 2:
            return torch.tensor(0.0, device=z.device, dtype=z.dtype)
        dz_true = z[1:] - z[:-1]
        predicted = self.forward(z, num_substeps, tau=tau)
        dz_pred = predicted - z[:-1]
        mag_true = dz_true.norm(dim=-1)
        mag_pred = dz_pred.norm(dim=-1)
        return ((mag_pred - mag_true).pow(2) / (mag_true.pow(2) + 1e-6)).mean()

    def velocity_contrastive_loss(
        self,
        z_points: torch.Tensor,
        family_labels: torch.Tensor,
        temperature: float = 0.1,
        tau: float | None = None,
    ) -> torch.Tensor:
        vel = self.velocity_field(z_points, tau=tau)
        v_norm = F.normalize(vel, dim=-1)
        sim = v_norm @ v_norm.T / temperature

        mask = family_labels.unsqueeze(0) == family_labels.unsqueeze(1)
        mask.fill_diagonal_(False)

        self_mask = torch.eye(len(z_points), dtype=torch.bool, device=z_points.device)
        logits = sim.clone()
        logits[self_mask] = float("-inf")
        log_denom = torch.logsumexp(logits, dim=1)

        pos_mask = mask.float()
        n_pos = pos_mask.sum(dim=1)
        has_pos = n_pos > 0

        if not has_pos.any():
            return torch.tensor(0.0, device=z_points.device, dtype=z_points.dtype)

        log_prob = sim - log_denom.unsqueeze(1)
        loss = -(log_prob * pos_mask).sum(dim=1) / n_pos.clamp(min=1)
        return loss[has_pos].mean()

    def contraction_loss(
        self,
        z_points: torch.Tensor,
        perturbation_scale: float = 0.1,
        target_ratio: float = 0.95,
        tau: float | None = None,
    ) -> torch.Tensor:
        delta = torch.randn_like(z_points)
        delta = delta / delta.norm(dim=-1, keepdim=True) * perturbation_scale

        gz = z_points + self.velocity_field(z_points, tau=tau)
        gz_d = (z_points + delta) + self.velocity_field(z_points + delta, tau=tau)

        ratio = (gz_d - gz).norm(dim=-1) / delta.norm(dim=-1)
        return F.relu(ratio - target_ratio).mean()

    def terminal_velocity_loss(
        self,
        z_terminals: torch.Tensor,
        margin: float = 0.5,
        tau: float | None = None,
    ) -> torch.Tensor:
        """Penalize large ||f(z)|| at trajectory endpoints."""
        vel = self.velocity_field(z_terminals, tau=tau)
        return F.relu(vel.norm(dim=-1) - margin).mean()

    def gate_supervised_loss(
        self,
        z: torch.Tensor,
        task_family_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-entropy: gate logits vs task-family index (0..K-1)."""
        logits = self.gate_logits(z)
        return F.cross_entropy(logits, task_family_labels.long())

    def regime_consistency_loss(
        self,
        z_a: torch.Tensor,
        z_b: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        pa = self.gate_probs(z_a)
        pb = self.gate_probs(z_b)
        kl_ab = (pa * (pa.clamp_min(eps).log() - pb.clamp_min(eps).log())).sum(dim=-1)
        kl_ba = (pb * (pb.clamp_min(eps).log() - pa.clamp_min(eps).log())).sum(dim=-1)
        return 0.5 * (kl_ab + kl_ba).mean()

    def gate_entropy_loss(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        p = self.gate_probs(z)
        ent = -(p * (p.clamp_min(1e-8).log())).sum(dim=-1)
        mean_p = p.mean(dim=0)
        batch_ent = -(mean_p * (mean_p.clamp_min(1e-8).log())).sum()
        return ent.mean() - batch_ent
