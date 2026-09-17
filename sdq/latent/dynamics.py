"""Latent semantic dynamics model.

Captures the temporal evolution of the latent semantic state z_t.
If the SDQ decomposition is correct, the latent dynamics should be
shared across semantically equivalent prompts (after alignment),
while surface differences are handled by gauge/transport.

Two implementations:
    1. LatentODE  — multi-step Euler: z_{t+1} = z_t + Σ f(z^(k))
    2. LatentGRU  — GRU over latent sequence (captures longer dependencies)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentODE(nn.Module):
    """Multi-step residual dynamics (neural ODE with Euler solver).

    Instead of predicting z_{t+1} from z_t in one big step, uses K
    small Euler sub-steps. This keeps the velocity field f(z) at
    moderate magnitudes (~||dz||/K), making fixed points achievable.

    z^(0) = z_t
    z^(k+1) = z^(k) + f(z^(k))   for k = 0, ..., K-1
    z_{t+1} ≈ z^(K)
    """

    def __init__(
        self,
        latent_dim: int,
        intermediate_dim: int | None = None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        inter = intermediate_dim or latent_dim * 2

        self.norm = nn.LayerNorm(latent_dim)
        self.f = nn.Sequential(
            nn.Linear(latent_dim, inter),
            nn.GELU(),
            nn.Linear(inter, latent_dim),
        )

        # Initialize near zero so early dynamics ≈ identity
        nn.init.zeros_(self.f[-1].weight)
        nn.init.zeros_(self.f[-1].bias)

    def velocity_field(self, z: torch.Tensor) -> torch.Tensor:
        """Evaluate the velocity field f(z).

        Args:
            z: [D_z] or [B, D_z] or [T, D_z] latent state(s).

        Returns:
            f(z): same shape — the "force" driving dynamics.
        """
        return self.f(self.norm(z))

    def step(self, z_t: torch.Tensor) -> torch.Tensor:
        """One Euler step: z_{t+1} = z_t + f(z_t).

        Args:
            z_t: [D_z] or [B, D_z] current latent state.

        Returns:
            z_{t+1}: same shape as input.
        """
        return z_t + self.velocity_field(z_t)

    def multi_step_predict(
        self, z_t: torch.Tensor, num_substeps: int = 10,
    ) -> torch.Tensor:
        """Predict z_{t+1} via K Euler sub-steps.

        Each sub-step: z^(k+1) = z^(k) + f(z^(k))
        Returns z^(K) as the prediction for z_{t+1}.

        Args:
            z_t: [D_z] or [B, D_z] starting state.
            num_substeps: K, number of Euler steps.

        Returns:
            Predicted next state, same shape as z_t.
        """
        z = z_t
        for _ in range(num_substeps):
            z = z + self.velocity_field(z)
        return z

    def rollout(self, z_0: torch.Tensor, T: int) -> torch.Tensor:
        """Roll out dynamics for T single steps from initial state.

        Uses single Euler steps (not multi-step). Suitable for
        attractor analysis and intervention experiments.

        Args:
            z_0: [D_z] or [N, D_z] initial latent state(s).
            T: number of steps to generate (including z_0).

        Returns:
            [T, ...] predicted latent trajectory.
        """
        states = [z_0]
        z = z_0
        for _ in range(T - 1):
            z = self.step(z)
            states.append(z)
        return torch.stack(states)

    def forward(
        self, z: torch.Tensor, num_substeps: int = 10,
    ) -> torch.Tensor:
        """Predict next states via multi-step Euler for each position.

        For each t, predicts z_{t+1} by running K sub-steps from z_t.
        Vectorized: all positions are batched through the substeps.

        Args:
            z: [T, D_z] or [B, D_z] latent trajectory / batch of starts.
            num_substeps: K Euler sub-steps per observed interval.

        Returns:
            [T-1, D_z] predicted next states (when input is trajectory).
        """
        z_curr = z[:-1]  # [T-1, D_z] — all starting positions as a batch
        for _ in range(num_substeps):
            z_curr = z_curr + self.velocity_field(z_curr)
        return z_curr

    def dynamics_consistency_loss(
        self, z: torch.Tensor, num_substeps: int = 10,
    ) -> torch.Tensor:
        """How well multi-step dynamics predict actual latent evolution.

        L = mean_t ||z_{t+1} - rollout_K(z_t)||^2

        Args:
            z: [T, D_z] latent trajectory (from encoder).
            num_substeps: K Euler sub-steps per interval.

        Returns:
            Scalar loss.
        """
        predicted = self.forward(z, num_substeps)  # [T-1, D_z]
        actual = z[1:]  # [T-1, D_z]
        return (predicted - actual).pow(2).sum(dim=-1).mean()

    def velocity_direction_loss(
        self, z: torch.Tensor, num_substeps: int = 10,
    ) -> torch.Tensor:
        """Cosine dissimilarity between multi-step displacement and truth.

        L = mean_t (1 - cos(rollout_K(z_t) - z_t, z_{t+1} - z_t))

        Scale-invariant — ensures the flow field points in the right
        direction even if magnitude is off.

        Args:
            z: [T, D_z] latent trajectory (from encoder).
            num_substeps: K Euler sub-steps per interval.

        Returns:
            Scalar loss in [0, 2].
        """
        if z.shape[0] < 2:
            return torch.tensor(0.0, device=z.device, dtype=z.dtype)
        dz_true = z[1:] - z[:-1]  # [T-1, D_z]
        predicted = self.forward(z, num_substeps)  # [T-1, D_z]
        dz_pred = predicted - z[:-1]  # [T-1, D_z]
        cos = F.cosine_similarity(dz_pred, dz_true, dim=-1)  # [T-1]
        return (1.0 - cos).mean()

    def velocity_magnitude_loss(
        self, z: torch.Tensor, num_substeps: int = 10,
    ) -> torch.Tensor:
        """Penalize magnitude mismatch between predicted and true displacement.

        L = mean_t (||dz_pred|| - ||dz_true||)^2 / (||dz_true||^2 + eps)

        Complements velocity_direction_loss: direction loss handles angle,
        this handles scale.

        Args:
            z: [T, D_z] latent trajectory (from encoder).
            num_substeps: K Euler sub-steps per interval.

        Returns:
            Scalar loss.
        """
        if z.shape[0] < 2:
            return torch.tensor(0.0, device=z.device, dtype=z.dtype)
        dz_true = z[1:] - z[:-1]  # [T-1, D_z]
        predicted = self.forward(z, num_substeps)  # [T-1, D_z]
        dz_pred = predicted - z[:-1]  # [T-1, D_z]
        mag_true = dz_true.norm(dim=-1)  # [T-1]
        mag_pred = dz_pred.norm(dim=-1)  # [T-1]
        return ((mag_pred - mag_true).pow(2) / (mag_true.pow(2) + 1e-6)).mean()

    def terminal_contraction_loss(
        self, z_terminals: torch.Tensor, margin: float = 0.01,
    ) -> torch.Tensor:
        """Encourage near-zero velocity at trajectory endpoints.

        L = mean_i ReLU(||f(z_T^i)|| - margin)

        With multi-step training, ||f|| ≈ ||dz||/K ≈ 1.5, so going
        to near-zero at terminals is achievable.

        Args:
            z_terminals: [N, D_z] terminal states of trajectories.
            margin: velocity norm below which no penalty is applied.

        Returns:
            Scalar loss.
        """
        vel = self.velocity_field(z_terminals)  # [N, D_z]
        return F.relu(vel.norm(dim=-1) - margin).mean()

    def velocity_coherence_loss(
        self,
        z_a: torch.Tensor,
        z_b: torch.Tensor,
    ) -> torch.Tensor:
        """Penalize velocity direction mismatch for same-family pairs.

        L = mean(1 - cos(f(z_a), f(z_b)))

        Forces same-family trajectories to have parallel flow fields,
        creating coherent tubes.

        Args:
            z_a: [N, D_z] points from trajectory A (same family as B).
            z_b: [N, D_z] corresponding points from trajectory B.

        Returns:
            Scalar loss in [0, 2].
        """
        vel_a = self.velocity_field(z_a)  # [N, D_z]
        vel_b = self.velocity_field(z_b)  # [N, D_z]
        cos = F.cosine_similarity(vel_a, vel_b, dim=-1)  # [N]
        return (1.0 - cos).mean()

    def velocity_contrastive_loss(
        self,
        z_points: torch.Tensor,
        family_labels: torch.Tensor,
        temperature: float = 0.1,
    ) -> torch.Tensor:
        """InfoNCE contrastive loss on velocity directions.

        For each point, same-family points are positives,
        different-family points are negatives. Applied to
        L2-normalized velocity vectors.

        Args:
            z_points: [N, D_z] latent points.
            family_labels: [N] integer family IDs.
            temperature: softmax temperature.

        Returns:
            Scalar loss.
        """
        vel = self.velocity_field(z_points)          # [N, D_z]
        v_norm = F.normalize(vel, dim=-1)            # [N, D_z]
        sim = v_norm @ v_norm.T / temperature        # [N, N]

        # Same-family mask (exclude self-pairs)
        mask = family_labels.unsqueeze(0) == family_labels.unsqueeze(1)
        mask.fill_diagonal_(False)

        # Numerically stable log-softmax over all non-self entries
        self_mask = torch.eye(len(z_points), dtype=torch.bool, device=z_points.device)
        logits = sim.clone()
        logits[self_mask] = float('-inf')
        log_denom = torch.logsumexp(logits, dim=1)  # [N]

        # Mean log-prob of positives
        pos_mask = mask.float()
        n_pos = pos_mask.sum(dim=1)
        has_pos = n_pos > 0

        if not has_pos.any():
            return torch.tensor(0.0, device=z_points.device, dtype=z_points.dtype)

        log_prob = sim - log_denom.unsqueeze(1)      # [N, N]
        loss = -(log_prob * pos_mask).sum(dim=1) / n_pos.clamp(min=1)
        return loss[has_pos].mean()

    def contraction_loss(
        self,
        z_points: torch.Tensor,
        perturbation_scale: float = 0.1,
        target_ratio: float = 0.95,
    ) -> torch.Tensor:
        """Penalize expansion of the single-step map g(z) = z + f(z).

        Samples random perturbations δ around z_points and penalizes
        the expansion ratio ||g(z+δ) - g(z)|| / ||δ|| when it exceeds
        target_ratio. Enforcing ratio < 1 makes fixed points stable
        (contracting Jacobian) and causes intervention perturbations
        to decay over time.

        Args:
            z_points: [N, D_z] points to enforce contraction at.
            perturbation_scale: magnitude of random perturbations.
            target_ratio: maximum allowed expansion ratio (< 1 = contracting).

        Returns:
            Scalar loss.
        """
        delta = torch.randn_like(z_points)
        delta = delta / delta.norm(dim=-1, keepdim=True) * perturbation_scale

        gz = z_points + self.velocity_field(z_points)
        gz_d = (z_points + delta) + self.velocity_field(z_points + delta)

        ratio = (gz_d - gz).norm(dim=-1) / delta.norm(dim=-1)
        return F.relu(ratio - target_ratio).mean()


class LatentGRU(nn.Module):
    """GRU-based latent dynamics for longer-range dependencies.

    Useful when semantic reasoning has non-local temporal structure
    that simple residual dynamics can't capture.
    """

    def __init__(
        self,
        latent_dim: int,
        num_layers: int = 1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.gru = nn.GRU(
            input_size=latent_dim,
            hidden_size=latent_dim,
            num_layers=num_layers,
            batch_first=False,
        )
        self.output_proj = nn.Linear(latent_dim, latent_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Predict next states for a sequence.

        Args:
            z: [T, D_z] latent trajectory.

        Returns:
            [T-1, D_z] predicted next latent states.
        """
        # GRU expects [T, batch, D]
        out, _ = self.gru(z[:-1].unsqueeze(1))  # [T-1, 1, D_z]
        out = out.squeeze(1)  # [T-1, D_z]
        return self.output_proj(out)

    def dynamics_consistency_loss(self, z: torch.Tensor) -> torch.Tensor:
        """Same interface as LatentODE."""
        predicted = self.forward(z)
        actual = z[1:]
        return (predicted - actual).pow(2).sum(dim=-1).mean()
