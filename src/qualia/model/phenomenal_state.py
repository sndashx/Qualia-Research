"""PhenomenalState: a first-class structured representation of the system's subjective state.

The state is a tuple of three components:
1. Global workspace vector (GWT-style broadcast slot).
2. Self-model vector (attention-schema / higher-order representation).
3. Phenomenal payload (a structured dict of named, introspectable slots).

The module exposes:
- encode(percepts): updates the state from incoming percepts.
- introspect(): returns a structured dict representation of the current state.
- modulate_action(action): gates an action vector by the current phenomenal state.
- reset_state(): clears recurrent state to the initial value (useful for batch boundaries).

Everything is a torch.nn.Module so the state is fully differentiable and trainable
end-to-end. The live state tensors are kept as autograd-tracked tensors so that
``modulate_action`` participates in the backward graph; ``introspect`` detaches
its return value so callers receive a plain, JSON-serializable snapshot.
Determinism w.r.t. seed is achieved by relying only on torch ops with the
caller-controlled RNG state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

PAYLOAD_KEYS: tuple[str, ...] = ("arousal", "valence", "content", "agency", "confidence")


@dataclass
class PhenomenalStateRecord:
    """Structured, JSON-serializable snapshot of a PhenomenalState."""

    workspace: list[float]
    self_model: list[float]
    payload: dict[str, list[float] | float]
    step: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "workspace": list(self.workspace),
            "self_model": list(self.self_model),
            "payload": {
                k: list(v) if isinstance(v, (list, tuple)) else float(v)
                for k, v in self.payload.items()
            },
            "step": int(self.step),
        }


class PhenomenalState(nn.Module):
    """Maintains a structured, differentiable "subjective state".

    Args:
        percept_dim: Dimensionality of incoming percept vectors.
        workspace_dim: Dimensionality of the global workspace vector.
        self_model_dim: Dimensionality of the self-model / attention-schema vector.
        payload_slots: Number of continuous slots per named payload key.
        payload_keys: Names of the categorical slots carried in the payload dict.
    """

    def __init__(
        self,
        percept_dim: int,
        workspace_dim: int = 32,
        self_model_dim: int = 32,
        payload_slots: int = 16,
        payload_keys: tuple[str, ...] = PAYLOAD_KEYS,
    ) -> None:
        super().__init__()
        if workspace_dim <= 0 or self_model_dim <= 0 or payload_slots <= 0:
            raise ValueError("workspace_dim, self_model_dim, payload_slots must be positive")
        if not payload_keys:
            raise ValueError("payload_keys must be non-empty")

        self.percept_dim = percept_dim
        self.workspace_dim = workspace_dim
        self.self_model_dim = self_model_dim
        self.payload_slots = payload_slots
        self.payload_keys: tuple[str, ...] = tuple(payload_keys)

        self.percept_encoder = nn.Sequential(
            nn.Linear(percept_dim, workspace_dim + self_model_dim),
            nn.Tanh(),
        )
        self.workspace_cell = nn.GRUCell(workspace_dim, workspace_dim)
        self.self_model_cell = nn.GRUCell(self_model_dim, self_model_dim)
        self.self_model_query = nn.Linear(workspace_dim, self_model_dim)

        self.payload_projs = nn.ModuleDict(
            {
                key: nn.Linear(self_model_dim + workspace_dim, payload_slots)
                for key in self.payload_keys
            }
        )

        self.gate_net = nn.Sequential(
            nn.Linear(workspace_dim + self_model_dim, 1),
            nn.Sigmoid(),
        )

        self.register_buffer("_step", torch.zeros((), dtype=torch.long))
        self._init_live_state()

    def _init_live_state(self) -> None:
        device = self.payload_projs[next(iter(self.payload_keys))].weight.device
        self._workspace_live: Tensor = torch.zeros(self.workspace_dim, device=device)
        self._self_model_live: Tensor = torch.zeros(self.self_model_dim, device=device)
        self._payload_live: dict[str, Tensor] = {
            key: torch.zeros(self.payload_slots, device=device) for key in self.payload_keys
        }

    @property
    def workspace(self) -> Tensor:
        return self._workspace_live

    @property
    def self_model(self) -> Tensor:
        return self._self_model_live

    def payload(self) -> dict[str, Tensor]:
        return {key: self._payload_live[key] for key in self.payload_keys}

    def reset_state(self) -> None:
        """Reset the recurrent state to zero. Safe to call between episodes."""
        self._workspace_live = torch.zeros_like(self._workspace_live)
        self._self_model_live = torch.zeros_like(self._self_model_live)
        self._payload_live = {
            key: torch.zeros_like(tensor) for key, tensor in self._payload_live.items()
        }
        self._step.zero_()

    def _initial_state(self, batch_size: int, device: torch.device) -> tuple[Tensor, Tensor]:
        ws = torch.zeros(batch_size, self.workspace_dim, device=device)
        sm = torch.zeros(batch_size, self.self_model_dim, device=device)
        return ws, sm

    def encode(self, percepts: Tensor) -> dict[str, Any]:
        """Update the state from a batch of percepts.

        Args:
            percepts: ``(B, percept_dim)`` tensor of incoming percepts.

        Returns:
            Dict with updated ``workspace``, ``self_model`` and ``payload`` tensors.
        """
        if percepts.dim() != 2 or percepts.shape[-1] != self.percept_dim:
            raise ValueError(
                f"percepts must have shape (B, {self.percept_dim}); got {tuple(percepts.shape)}"
            )
        batch_size = percepts.shape[0]
        device = percepts.device

        encoded = self.percept_encoder(percepts)
        workspace_input, self_model_input = torch.split(
            encoded, [self.workspace_dim, self.self_model_dim], dim=-1
        )

        h_w, h_s = self._initial_state(batch_size, device)
        h_w = self.workspace_cell(workspace_input, h_w)
        query = self.self_model_query(h_w)
        h_s = self.self_model_cell(self_model_input + query, h_s)

        ws_summary = h_w.mean(dim=0)
        sm_summary = h_s.mean(dim=0)
        joint = torch.cat([ws_summary, sm_summary], dim=-1)

        self._workspace_live = ws_summary
        self._self_model_live = sm_summary
        self._payload_live = {key: self.payload_projs[key](joint) for key in self.payload_keys}
        self._step.add_(1)

        return {
            "workspace": h_w,
            "self_model": h_s,
            "payload": {
                key: self.payload_projs[key](torch.cat([h_w, h_s], dim=-1))
                for key in self.payload_keys
            },
        }

    def introspect(self) -> dict[str, Any]:
        """Return a structured dict describing the current phenomenal state."""
        payload = {key: tensor.detach().clone() for key, tensor in self._payload_live.items()}
        return PhenomenalStateRecord(
            workspace=self._workspace_live.detach().cpu().tolist(),
            self_model=self._self_model_live.detach().cpu().tolist(),
            payload={k: v.cpu().tolist() for k, v in payload.items()},
            step=int(self._step.item()),
        ).to_dict()

    def modulate_action(self, action: Tensor) -> Tensor:
        """Gate an action vector by the current phenomenal state.

        The gate is computed from the *current* live state tensors, so gradients
        flow from the gate back through the encode computation that produced
        the state. ``introspect`` will report values consistent with what the
        gate sees.
        """
        if action.dim() < 1:
            raise ValueError("action must have at least one dimension")
        if action.shape[-1] != self.workspace_dim + self.self_model_dim:
            raise ValueError(
                f"action last dim must be {self.workspace_dim + self.self_model_dim}; "
                f"got {action.shape[-1]}"
            )
        state = torch.cat([self._workspace_live, self._self_model_live], dim=-1)
        gate_in = state.unsqueeze(0).expand(action.shape[0], -1)
        gate = self.gate_net(gate_in)
        return action * gate


__all__ = ["PhenomenalState", "PhenomenalStateRecord", "PAYLOAD_KEYS"]
