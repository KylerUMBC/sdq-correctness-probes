"""Tube-aware dual dynamics: nominal predictor + recovery corrector.

The nominal field models on-manifold forward motion (same architecture as
RegimeSwitchingDynamics: gated mixture of per-regime velocity MLPs).

The recovery field learns to correct off-tube states back toward valid
reasoning trajectories. It uses family-conditioned tube context and
predicts a bounded tube-coordinate correction: a dominant transverse pull
back toward the tube plus a small along-tube phase adjustment. This avoids
learning a constant correction floor near already-valid states while still
allowing small forward re-phasing inside the correct basin.

When tube geometry is loaded via :meth:`set_tube_geometry`, rollout uses a
predictor-corrector step at each tick:
    z_nom = z + f_nominal(z)
    z_next = z_nom + f_recovery(z_nom, local_tube_context)

When tube geometry is absent, rollout uses only the nominal field.

v6.3 adds a learned ``progress_head`` (continuous progress in ``(0,1)``),
an optional ``family_router`` for explicit semantic-family selection when
``num_tube_families > 0``, and feeds both into recovery context blending.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Tube Geometry: precomputed tube structure for nearest-anchor queries
# ---------------------------------------------------------------------------

class TubeGeometry:
    """Precomputed family-tube structure for nearest-anchor lookups.

    Stores mean-path anchor points and unit tangent vectors for every
    family, stacked into flat tensors for efficient batch queries.
    """

    def __init__(
        self,
        anchors: Tensor,
        tangents: Tensor,
        family_ids: Tensor,
        progress_ids: Tensor,
        device: torch.device,
    ):
        self.anchors = anchors.to(device)
        self.tangents = tangents.to(device)
        self.family_ids = family_ids.to(device)
        self.progress_ids = progress_ids.to(device)
        self.device = device
        self._family_to_anchor_indices: dict[int, Tensor] = {}
        if self.family_ids.numel() > 0:
            for fam in torch.unique(self.family_ids).tolist():
                self._family_to_anchor_indices[int(fam)] = (
                    self.family_ids == fam
                ).nonzero(as_tuple=True)[0]

    @classmethod
    def build(
        cls,
        latent_trajectories: dict[str, Tensor],
        families: dict[str, list[str]],
        family_to_idx: dict[str, int],
        device: torch.device,
    ) -> TubeGeometry:
        """Build geometry from per-family latent trajectories.

        For each family with >= 2 members, computes the mean path and
        unit tangent vectors along it.  Every timestep becomes an anchor.
        """
        all_anchors: list[Tensor] = []
        all_tangents: list[Tensor] = []
        all_fam_ids: list[int] = []
        all_progress_ids: list[int] = []

        for fam_name, pids in families.items():
            if fam_name not in family_to_idx:
                continue
            fam_idx = family_to_idx[fam_name]

            trajs = [
                latent_trajectories[p]
                for p in pids
                if p in latent_trajectories
            ]
            if len(trajs) < 2:
                continue

            T_min = min(z.shape[0] for z in trajs)
            stacked = torch.stack([z[:T_min] for z in trajs])  # [N, T, D]
            mean_path = stacked.mean(dim=0)  # [T, D]

            dz = mean_path[1:] - mean_path[:-1]  # [T-1, D]
            t_norms = dz.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            tangents = dz / t_norms  # [T-1, D]

            for t in range(T_min):
                all_anchors.append(mean_path[t])
                t_idx = min(t, tangents.shape[0] - 1)
                all_tangents.append(tangents[t_idx])
                all_fam_ids.append(fam_idx)
                all_progress_ids.append(t)

        if not all_anchors:
            D = next(iter(latent_trajectories.values())).shape[-1]
            return cls(
                torch.zeros(0, D),
                torch.zeros(0, D),
                torch.zeros(0, dtype=torch.long),
                torch.zeros(0, dtype=torch.long),
                device,
            )

        return cls(
            torch.stack(all_anchors),
            torch.stack(all_tangents),
            torch.tensor(all_fam_ids, dtype=torch.long),
            torch.tensor(all_progress_ids, dtype=torch.long),
            device,
        )

    # ----- queries -----

    @torch.no_grad()
    def nearest_context(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """Nearest anchor and its tangent for each point in *z*.

        Args:
            z: ``[*, D]`` query points.

        Returns:
            ``(anchor, tangent)`` each ``[*, D]``.
        """
        if self.anchors.shape[0] == 0:
            return torch.zeros_like(z), torch.zeros_like(z)

        orig_shape = z.shape
        z_flat = z.reshape(-1, z.shape[-1])

        dists = torch.cdist(z_flat, self.anchors)
        nearest = dists.argmin(dim=-1)

        anchor = self.anchors[nearest].reshape(orig_shape)
        tangent = self.tangents[nearest].reshape(orig_shape)
        return anchor, tangent

    @torch.no_grad()
    def nearest_context_with_family(
        self, z: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Like :meth:`nearest_context` but also returns family and progress."""
        if self.anchors.shape[0] == 0:
            return (
                torch.zeros_like(z),
                torch.zeros_like(z),
                torch.zeros(
                    z.shape[:-1], dtype=torch.long, device=z.device,
                ),
                torch.zeros(
                    z.shape[:-1], dtype=torch.long, device=z.device,
                ),
            )

        orig_shape = z.shape
        z_flat = z.reshape(-1, z.shape[-1])

        dists = torch.cdist(z_flat, self.anchors)
        nearest = dists.argmin(dim=-1)

        anchor = self.anchors[nearest].reshape(orig_shape)
        tangent = self.tangents[nearest].reshape(orig_shape)
        fam = self.family_ids[nearest].reshape(orig_shape[:-1])
        progress = self.progress_ids[nearest].reshape(orig_shape[:-1])
        return anchor, tangent, fam, progress

    @torch.no_grad()
    def nearest_context_for_family(
        self,
        z: Tensor,
        family_ids: Tensor,
        progress_hint: Tensor | None = None,
        progress_window: int | None = None,
        backward_window: int | None = None,
        forward_window: int | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Nearest context restricted to a target family and optional window.

        Args:
            z: query points ``[*, D]``.
            family_ids: family index per point ``[*]``.
            progress_hint: optional timestep index per point ``[*]``.
            progress_window: symmetric local window around ``progress_hint``.
            backward_window: explicit backward window override.
            forward_window: explicit forward window override.

        Returns:
            ``(anchor, tangent, progress)``.
        """
        if self.anchors.shape[0] == 0:
            return (
                torch.zeros_like(z),
                torch.zeros_like(z),
                torch.zeros(z.shape[:-1], dtype=torch.long, device=z.device),
            )

        orig_shape = z.shape
        z_flat = z.reshape(-1, z.shape[-1])
        fam_flat = family_ids.reshape(-1).to(self.device)
        hint_flat = (
            progress_hint.reshape(-1).to(self.device)
            if progress_hint is not None
            else None
        )

        anchor_out = torch.zeros_like(z_flat)
        tangent_out = torch.zeros_like(z_flat)
        progress_out = torch.zeros(
            z_flat.shape[0], dtype=torch.long, device=self.device,
        )

        if progress_window is not None:
            if backward_window is None:
                backward_window = progress_window
            if forward_window is None:
                forward_window = progress_window
        if backward_window is None:
            backward_window = 0
        if forward_window is None:
            forward_window = 0

        for i in range(z_flat.shape[0]):
            fam = int(fam_flat[i].item())
            fam_indices = self._family_to_anchor_indices.get(fam)
            if fam_indices is None or fam_indices.numel() == 0:
                continue

            allowed = fam_indices
            if hint_flat is not None:
                fam_progress = self.progress_ids[fam_indices]
                lo = hint_flat[i] - backward_window
                hi = hint_flat[i] + forward_window
                mask = (fam_progress >= lo) & (fam_progress <= hi)
                if mask.any():
                    allowed = fam_indices[mask]

            dists = torch.cdist(
                z_flat[i : i + 1], self.anchors[allowed],
            ).squeeze(0)
            chosen = allowed[dists.argmin()]
            anchor_out[i] = self.anchors[chosen]
            tangent_out[i] = self.tangents[chosen]
            progress_out[i] = self.progress_ids[chosen]

        return (
            anchor_out.reshape(orig_shape),
            tangent_out.reshape(orig_shape),
            progress_out.reshape(orig_shape[:-1]),
        )

    @torch.no_grad()
    def nearest_context_excluding_family(
        self,
        z: Tensor,
        family_ids: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Nearest context from any family except the provided one."""
        if self.anchors.shape[0] == 0:
            return (
                torch.zeros_like(z),
                torch.zeros_like(z),
                torch.zeros(z.shape[:-1], dtype=torch.long, device=z.device),
                torch.zeros(z.shape[:-1], dtype=torch.long, device=z.device),
            )

        orig_shape = z.shape
        z_flat = z.reshape(-1, z.shape[-1])
        fam_flat = family_ids.reshape(-1).to(self.device)

        anchor_out = torch.zeros_like(z_flat)
        tangent_out = torch.zeros_like(z_flat)
        fam_out = torch.zeros(
            z_flat.shape[0], dtype=torch.long, device=self.device,
        )
        progress_out = torch.zeros(
            z_flat.shape[0], dtype=torch.long, device=self.device,
        )

        for i in range(z_flat.shape[0]):
            mask = self.family_ids != fam_flat[i]
            if not mask.any():
                continue
            allowed = mask.nonzero(as_tuple=True)[0]
            dists = torch.cdist(
                z_flat[i : i + 1], self.anchors[allowed],
            ).squeeze(0)
            chosen = allowed[dists.argmin()]
            anchor_out[i] = self.anchors[chosen]
            tangent_out[i] = self.tangents[chosen]
            fam_out[i] = self.family_ids[chosen]
            progress_out[i] = self.progress_ids[chosen]

        return (
            anchor_out.reshape(orig_shape),
            tangent_out.reshape(orig_shape),
            fam_out.reshape(orig_shape[:-1]),
            progress_out.reshape(orig_shape[:-1]),
        )

    @torch.no_grad()
    def contextualize(
        self,
        z: Tensor,
        family_ids: Tensor | None = None,
        progress_hint: Tensor | None = None,
        progress_window: int | None = None,
        backward_window: int | None = None,
        forward_window: int | None = None,
    ) -> dict[str, Tensor]:
        """Return a rich local tube context for each point in ``z``."""
        if family_ids is None:
            anchor, tangent, fam, progress = self.nearest_context_with_family(z)
        else:
            anchor, tangent, progress = self.nearest_context_for_family(
                z,
                family_ids=family_ids,
                progress_hint=progress_hint,
                progress_window=progress_window,
                backward_window=backward_window,
                forward_window=forward_window,
            )
            fam = family_ids.reshape(z.shape[:-1]).to(z.device)

        displacement = z - anchor
        along = (displacement * tangent).sum(dim=-1, keepdim=True)
        transverse = displacement - along * tangent
        transverse_norm = transverse.norm(dim=-1)
        return {
            "anchor": anchor,
            "tangent": tangent,
            "family_ids": fam,
            "progress": progress,
            "displacement": displacement,
            "along": along,
            "transverse": transverse,
            "transverse_norm": transverse_norm,
        }

    @torch.no_grad()
    def distance_to_tube(self, z: Tensor) -> Tensor:
        """Transverse distance from each point to nearest tube anchor.

        Returns:
            ``[*]`` scalar distances (transverse component only).
        """
        return self.contextualize(z)["transverse_norm"]

    @torch.no_grad()
    def distance_to_other_family_tube(
        self,
        z: Tensor,
        family_ids: Tensor,
    ) -> Tensor:
        """Transverse distance to the nearest wrong-family tube."""
        anchor, tangent, _, _ = self.nearest_context_excluding_family(
            z, family_ids,
        )
        disp = z - anchor
        along = (disp * tangent).sum(dim=-1, keepdim=True)
        transverse = disp - along * tangent
        return transverse.norm(dim=-1)


# ---------------------------------------------------------------------------
# TubeAwareDynamics
# ---------------------------------------------------------------------------

class TubeAwareDynamics(nn.Module):
    """Dual dynamics with nominal rollout field and recovery corrector.

    Nominal dynamics use the same gated-mixture architecture as
    ``RegimeSwitchingDynamics`` (per-regime MLPs, Gumbel-softmax gate).

    The recovery network takes ``(z, transverse_displacement)`` and
    outputs a correction vector.  It is zero-initialised so it starts
    as a no-op and can be trained independently of the nominal field.
    """

    def __init__(
        self,
        latent_dim: int,
        num_regimes: int = 6,
        intermediate_dim: int | None = None,
        num_tube_families: int = 0,
        progress_coord_cap: int = 256,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_regimes = num_regimes
        self.num_tube_families = int(num_tube_families)
        self.progress_coord_cap = int(progress_coord_cap)
        inter = intermediate_dim or latent_dim * 2

        # ---- nominal dynamics (gated mixture) ----
        self.gate = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, 128),
            nn.GELU(),
            nn.Linear(128, num_regimes),
        )
        self.velocity_nets = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, inter),
                nn.GELU(),
                nn.Linear(inter, latent_dim),
            )
            for _ in range(num_regimes)
        ])
        for net in self.velocity_nets:
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)

        # Continuous progress coordinate in [0, 1] (used in recovery input + hints)
        self.progress_head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, inter // 2),
            nn.GELU(),
            nn.Linear(inter // 2, 1),
        )
        nn.init.zeros_(self.progress_head[-1].bias)

        # Explicit semantic-family router for off-tube recovery (optional)
        if self.num_tube_families > 0:
            self.family_router = nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, inter),
                nn.GELU(),
                nn.Linear(inter, self.num_tube_families),
            )
        else:
            self.family_router = None

        # ---- recovery corrector ----
        recovery_input_dim = latent_dim * 2 + 1 + 1  # + progress scalar
        self.recovery_net = nn.Sequential(
            nn.LayerNorm(recovery_input_dim),
            nn.Linear(recovery_input_dim, inter),
            nn.GELU(),
            nn.Linear(inter, inter),
            nn.GELU(),
            nn.Linear(inter, latent_dim),
        )
        nn.init.zeros_(self.recovery_net[-1].weight)
        nn.init.zeros_(self.recovery_net[-1].bias)
        self.recovery_gain = nn.Sequential(
            nn.LayerNorm(recovery_input_dim),
            nn.Linear(recovery_input_dim, inter // 2),
            nn.GELU(),
            nn.Linear(inter // 2, 1),
        )
        nn.init.zeros_(self.recovery_gain[-1].weight)
        nn.init.constant_(self.recovery_gain[-1].bias, -12.0)
        self.recovery_along = nn.Sequential(
            nn.LayerNorm(recovery_input_dim),
            nn.Linear(recovery_input_dim, inter // 2),
            nn.GELU(),
            nn.Linear(inter // 2, 1),
        )
        nn.init.zeros_(self.recovery_along[-1].weight)
        nn.init.zeros_(self.recovery_along[-1].bias)

        # ---- tube geometry (set externally) ----
        self._tube_geo: TubeGeometry | None = None

    # ----- tube geometry management -----

    def set_tube_geometry(self, geo: TubeGeometry | None) -> None:
        """Attach or detach tube geometry for recovery-augmented rollout."""
        self._tube_geo = geo

    @property
    def has_tube_geometry(self) -> bool:
        return self._tube_geo is not None

    # ----- gate helpers -----

    def gate_logits(self, z: Tensor) -> Tensor:
        """``[*, D] -> [*, K]`` raw logits."""
        return self.gate(z)

    def gate_probs(self, z: Tensor) -> Tensor:
        """Softmax gate probabilities."""
        return F.softmax(self.gate_logits(z), dim=-1)

    # ----- nominal dynamics -----

    def velocity_field(
        self, z: Tensor, tau: float | None = None,
    ) -> Tensor:
        """Gated mixture velocity ``f(z)``.

        Training + *tau* → Gumbel-softmax; eval → argmax one-hot.
        """
        squeezed = z.ndim == 1
        if squeezed:
            z = z.unsqueeze(0)

        logits = self.gate_logits(z)
        if self.training:
            if tau is not None:
                gate_w = F.gumbel_softmax(
                    logits, tau=tau, dim=-1, hard=False,
                )
            else:
                gate_w = F.softmax(logits, dim=-1)
        else:
            idx = logits.argmax(dim=-1)
            gate_w = F.one_hot(
                idx, num_classes=self.num_regimes,
            ).to(dtype=logits.dtype, device=logits.device)

        vels = [self.velocity_nets[r](z) for r in range(self.num_regimes)]
        vel_stack = torch.stack(vels, dim=-2)  # [*, K, D]
        out = (gate_w.unsqueeze(-1) * vel_stack).sum(dim=-2)

        if squeezed:
            out = out.squeeze(0)
        return out

    # ----- recovery dynamics -----

    def progress_scalar(self, z: Tensor) -> Tensor:
        """Predict normalized trajectory progress in ``(0, 1)`` for each row of *z*."""
        squeezed = z.ndim == 1
        if squeezed:
            z = z.unsqueeze(0)
        ps = torch.sigmoid(self.progress_head(z))
        if squeezed:
            ps = ps.squeeze(0)
        return ps

    def family_router_logits(self, z: Tensor) -> Tensor:
        """Logits over tube semantic families; requires ``num_tube_families > 0``."""
        if self.family_router is None:
            raise RuntimeError("family_router is not configured (num_tube_families=0)")
        squeezed = z.ndim == 1
        if squeezed:
            z = z.unsqueeze(0)
        logits = self.family_router(z)
        if squeezed:
            logits = logits.squeeze(0)
        return logits

    def recovery_field(
        self,
        z: Tensor,
        tube_anchor: Tensor,
        tube_tangent: Tensor,
    ) -> Tensor:
        """Recovery correction from transverse displacement.

        Decomposes ``z - anchor`` into along-tube and transverse parts,
        feeds ``(z, transverse)`` to the recovery MLP.
        """
        squeezed = z.ndim == 1
        if squeezed:
            z = z.unsqueeze(0)
            tube_anchor = tube_anchor.unsqueeze(0)
            tube_tangent = tube_tangent.unsqueeze(0)

        displacement = z - tube_anchor
        along = (displacement * tube_tangent).sum(dim=-1, keepdim=True)
        transverse = displacement - along * tube_tangent

        transverse_norm = transverse.norm(dim=-1, keepdim=True)
        prog_feat = self.progress_scalar(z)
        if prog_feat.ndim == 1:
            prog_feat = prog_feat.unsqueeze(-1)
        elif prog_feat.shape[-1] != 1:
            prog_feat = prog_feat.reshape(*prog_feat.shape[:-1], 1)
        net_input = torch.cat([z, transverse, transverse_norm, prog_feat], dim=-1)
        raw_correction = self.recovery_net(net_input)
        raw_correction = raw_correction - (
            raw_correction * tube_tangent
        ).sum(dim=-1, keepdim=True) * tube_tangent
        base_direction = raw_correction - transverse
        base_direction = base_direction / base_direction.norm(
            dim=-1, keepdim=True,
        ).clamp(min=1e-8)
        gain = 1.5 * torch.sigmoid(self.recovery_gain(net_input))
        along_gain = 0.25 * torch.tanh(self.recovery_along(net_input))
        transverse_correction = gain * transverse_norm * base_direction
        along_correction = along_gain * transverse_norm * tube_tangent
        correction = transverse_correction + along_correction

        if squeezed:
            correction = correction.squeeze(0)
        return correction

    # ----- combined step / rollout -----

    def predict_step(self, z: Tensor, tau: float | None = None) -> Tensor:
        """Alias for the nominal dynamics step."""
        return self.nominal_step(z, tau=tau)

    def recover_step(
        self,
        z: Tensor,
        family_ids: Tensor | None = None,
        progress_hint: Tensor | None = None,
        progress_window: int | None = None,
        backward_window: int | None = None,
        forward_window: int | None = None,
        return_context: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Apply only the recovery corrector at the current state."""
        if self._tube_geo is None:
            if return_context:
                zero_ctx = {
                    "anchor": torch.zeros_like(z),
                    "tangent": torch.zeros_like(z),
                    "family_ids": (
                        torch.zeros(
                            z.shape[:-1], dtype=torch.long, device=z.device,
                        )
                    ),
                    "progress": torch.zeros(
                        z.shape[:-1], dtype=torch.long, device=z.device,
                    ),
                    "transverse": torch.zeros_like(z),
                    "transverse_norm": torch.zeros(
                        z.shape[:-1], device=z.device, dtype=z.dtype,
                    ),
                    "wrong_family_distance": torch.zeros(
                        z.shape[:-1], device=z.device, dtype=z.dtype,
                    ),
                    "correction": torch.zeros_like(z),
                }
                return z, zero_ctx
            return z

        cap = max(self.progress_coord_cap, 1)
        ps = self.progress_scalar(z)
        ph = (ps.squeeze(-1) * float(cap - 1)).long().clamp(0, cap - 1)
        if ps.ndim == 0:
            ph = ph.reshape(())

        if family_ids is None:
            if self.family_router is not None:
                pred_family = self.family_router_logits(z).argmax(dim=-1)
                if progress_hint is None:
                    progress_hint = ph
            else:
                _, _, pred_family, inferred_progress = (
                    self._tube_geo.nearest_context_with_family(z)
                )
                if progress_hint is None:
                    progress_hint = inferred_progress
        else:
            pred_family = family_ids.to(z.device)
            if progress_hint is None:
                progress_hint = ph

        hint = progress_hint.reshape(pred_family.shape).long()
        blended = ((hint.float() + ph.reshape(hint.shape).float()) * 0.5).long().clamp(
            0, cap - 1,
        )

        ctx = self._tube_geo.contextualize(
            z,
            family_ids=pred_family,
            progress_hint=blended,
            progress_window=progress_window,
            backward_window=backward_window,
            forward_window=forward_window,
        )
        correction = self.recovery_field(z, ctx["anchor"], ctx["tangent"])
        z_next = z + correction
        if return_context:
            ctx["correction"] = correction
            ctx["wrong_family_distance"] = self._tube_geo.distance_to_other_family_tube(
                z_next, pred_family.reshape(z.shape[:-1]),
            )
            return z_next, ctx
        return z_next

    def step(
        self,
        z: Tensor,
        tau: float | None = None,
        family_ids: Tensor | None = None,
        progress_hint: Tensor | None = None,
        progress_window: int | None = None,
        backward_window: int | None = None,
        forward_window: int | None = None,
    ) -> Tensor:
        """Predictor-corrector step: nominal advance then recovery."""
        z_pred = z + self.velocity_field(z, tau=tau)
        return self.recover_step(
            z_pred,
            family_ids=family_ids,
            progress_hint=progress_hint,
            progress_window=progress_window,
            backward_window=backward_window,
            forward_window=forward_window,
        )

    def nominal_step(self, z: Tensor, tau: float | None = None) -> Tensor:
        """Nominal-only step (no recovery)."""
        return z + self.velocity_field(z, tau=tau)

    def rollout(
        self,
        z_0: Tensor,
        T: int,
        tau: float | None = None,
        family_ids: Tensor | None = None,
        progress_hint: Tensor | None = None,
        progress_window: int | None = None,
        backward_window: int | None = None,
        forward_window: int | None = None,
    ) -> Tensor:
        """Roll out *T* steps (recovery applied when tube geometry is set)."""
        return self.rollout_recovering(
            z_0,
            T,
            tau=tau,
            family_ids=family_ids,
            progress_hint=progress_hint,
            progress_window=progress_window,
            backward_window=backward_window,
            forward_window=forward_window,
        )

    def rollout_nominal(
        self, z_0: Tensor, T: int, tau: float | None = None,
    ) -> Tensor:
        """Roll out the nominal dynamics only."""
        states = [z_0]
        z = z_0
        for _ in range(T - 1):
            z = self.nominal_step(z, tau=tau)
            states.append(z)
        return torch.stack(states)

    def rollout_recovering(
        self,
        z_0: Tensor,
        T: int,
        tau: float | None = None,
        family_ids: Tensor | None = None,
        progress_hint: Tensor | None = None,
        progress_window: int | None = None,
        backward_window: int | None = None,
        forward_window: int | None = None,
    ) -> Tensor:
        """Roll out predictor-corrector dynamics with local tube context."""
        states = [z_0]
        z = z_0
        current_progress = progress_hint
        for _ in range(T - 1):
            z_pred = self.nominal_step(z, tau=tau)
            if self._tube_geo is None:
                z = z_pred
            else:
                z, ctx = self.recover_step(
                    z_pred,
                    family_ids=family_ids,
                    progress_hint=current_progress,
                    progress_window=progress_window,
                    backward_window=backward_window,
                    forward_window=forward_window,
                    return_context=True,
                )
                if current_progress is None:
                    current_progress = ctx["progress"] + 1
                else:
                    current_progress = (
                        torch.maximum(ctx["progress"], current_progress) + 1
                    )
            states.append(z)
        return torch.stack(states)

    def multi_step_predict(
        self,
        z_t: Tensor,
        num_substeps: int = 10,
        tau: float | None = None,
    ) -> Tensor:
        """Nominal-only multi-step prediction (for dynamics MSE)."""
        z = z_t
        for _ in range(num_substeps):
            z = z + self.velocity_field(z, tau=tau)
        return z

    def forward(
        self,
        z: Tensor,
        num_substeps: int = 10,
        tau: float | None = None,
    ) -> Tensor:
        """Nominal-only forward (start→target prediction)."""
        z_curr = z[:-1]
        for _ in range(num_substeps):
            z_curr = z_curr + self.velocity_field(z_curr, tau=tau)
        return z_curr

    # ----- loss helpers (nominal) -----

    def dynamics_consistency_loss(
        self, z: Tensor, num_substeps: int = 10, tau: float | None = None,
    ) -> Tensor:
        predicted = self.forward(z, num_substeps, tau=tau)
        actual = z[1:]
        return (predicted - actual).pow(2).sum(dim=-1).mean()

    def velocity_direction_loss(
        self, z: Tensor, num_substeps: int = 10, tau: float | None = None,
    ) -> Tensor:
        if z.shape[0] < 2:
            return torch.tensor(0.0, device=z.device, dtype=z.dtype)
        dz_true = z[1:] - z[:-1]
        predicted = self.forward(z, num_substeps, tau=tau)
        dz_pred = predicted - z[:-1]
        cos = F.cosine_similarity(dz_pred, dz_true, dim=-1)
        return (1.0 - cos).mean()

    def velocity_magnitude_loss(
        self, z: Tensor, num_substeps: int = 10, tau: float | None = None,
    ) -> Tensor:
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
        z_points: Tensor,
        family_labels: Tensor,
        temperature: float = 0.1,
        tau: float | None = None,
    ) -> Tensor:
        vel = self.velocity_field(z_points, tau=tau)
        v_norm = F.normalize(vel, dim=-1)
        sim = v_norm @ v_norm.T / temperature

        mask = family_labels.unsqueeze(0) == family_labels.unsqueeze(1)
        mask.fill_diagonal_(False)

        self_mask = torch.eye(
            len(z_points), dtype=torch.bool, device=z_points.device,
        )
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

    def terminal_velocity_loss(
        self,
        z_terminals: Tensor,
        margin: float = 0.5,
        tau: float | None = None,
    ) -> Tensor:
        vel = self.velocity_field(z_terminals, tau=tau)
        return F.relu(vel.norm(dim=-1) - margin).mean()

    # ----- gate supervision losses -----

    def gate_supervised_loss(
        self, z: Tensor, task_family_labels: Tensor,
    ) -> Tensor:
        logits = self.gate_logits(z)
        return F.cross_entropy(logits, task_family_labels.long())

    def regime_consistency_loss(
        self, z_a: Tensor, z_b: Tensor, eps: float = 1e-8,
    ) -> Tensor:
        pa = self.gate_probs(z_a)
        pb = self.gate_probs(z_b)
        kl_ab = (pa * (pa.clamp_min(eps).log() - pb.clamp_min(eps).log())).sum(dim=-1)
        kl_ba = (pb * (pb.clamp_min(eps).log() - pa.clamp_min(eps).log())).sum(dim=-1)
        return 0.5 * (kl_ab + kl_ba).mean()

    def gate_entropy_loss(self, z: Tensor) -> Tensor:
        p = self.gate_probs(z)
        ent = -(p * (p.clamp_min(1e-8).log())).sum(dim=-1)
        mean_p = p.mean(dim=0)
        batch_ent = -(mean_p * (mean_p.clamp_min(1e-8).log())).sum()
        return ent.mean() - batch_ent
