"""Higher-order self-model: produces introspective reports and an attention schema.

Given a ``PhenomenalState``, the ``SelfModel`` produces:

* A structured self-report with named slots (``arousal``, ``valence``, ``confidence``,
  ``content``, ``agency``) — a higher-order, differentiable description of "what
  the system is reporting about its own state".
* An attention schema — a compressed vector representing what the system
  believes it is attending to about itself.

Losses:

* ``report_consistency_loss(prev_state, curr_state)`` — the self-report at time
  ``t`` should be predictable from the phenomenal state at time ``t-1``. We
  train a small predictor head so its prediction of the current report matches
  the model's own self-report; this is the higher-order analogue of
  "the agent can predict its own next report".
* ``contrastive_report_loss(state_a, state_b, temperature)`` — a contrastive
  objective pulling self-reports for the same phenomenal state together and
  pushing reports for distinct states apart.

The module is intentionally small and dependency-free beyond ``torch`` so it
runs on CPU in unit tests.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

REPORT_SLOT_NAMES: tuple[str, ...] = ("arousal", "valence", "confidence", "content", "agency")


@dataclass
class SelfReport:
    """Structured self-report with named slots.

    Each slot is a 1-D tensor of length ``slot_dim``. The dataclass is a plain
    container — values remain differentiable when sourced from a module that
    is connected to the autograd graph.
    """

    slots: dict[str, Tensor]

    def __post_init__(self) -> None:
        missing = [name for name in REPORT_SLOT_NAMES if name not in self.slots]
        if missing:
            raise ValueError(f"missing report slots: {missing}")
        extra = [name for name in self.slots if name not in REPORT_SLOT_NAMES]
        if extra:
            raise ValueError(f"unexpected report slots: {extra}")

    def __getitem__(self, key: str) -> Tensor:
        return self.slots[key]

    def as_dict(self) -> dict[str, Tensor]:
        return dict(self.slots)

    def vector(self, slot_dim: int) -> Tensor:
        """Concatenate slots in canonical order into a single vector.

        Args:
            slot_dim: Expected dimensionality of each slot; validated against
                every slot's trailing dimension to catch mismatches between
                a report produced by one model and a consumer (loss, snapshot,
                downstream head) built for a different ``slot_dim``.
        """
        if slot_dim <= 0:
            raise ValueError(f"slot_dim must be positive; got {slot_dim}")
        for name in REPORT_SLOT_NAMES:
            slot = self.slots[name]
            if slot.dim() < 1 or slot.shape[-1] != slot_dim:
                raise ValueError(
                    f"slot '{name}' last dim must be {slot_dim}; got {tuple(slot.shape)}"
                )
        return torch.cat([self.slots[name] for name in REPORT_SLOT_NAMES], dim=-1)

    def to_snapshot(self) -> dict[str, list[float]]:
        """Detach and move to CPU for serialization-friendly output."""
        return {name: tensor.detach().cpu().tolist() for name, tensor in self.slots.items()}


@dataclass
class SelfModelOutput:
    """Bundle of (report, attention_schema) produced by a ``SelfModel`` forward pass."""

    report: SelfReport
    attention_schema: Tensor
    state_repr: Tensor

    def as_dict(self) -> dict[str, Any]:
        return {
            "report": self.report.to_snapshot(),
            "attention_schema": self.attention_schema.detach().cpu().tolist(),
            "state_repr": self.state_repr.detach().cpu().tolist(),
        }


class SelfModel(nn.Module):
    """Higher-order module mapping a phenomenal state to a structured self-report.

    Args:
        workspace_dim: Dimensionality of the workspace vector in the phenomenal state.
        self_model_dim: Dimensionality of the self-model vector in the phenomenal state.
        slot_dim: Dimensionality of each named slot in the self-report.
        attention_schema_dim: Dimensionality of the attention schema vector.
        slot_names: Names of the slots; must match the keys of the phenomenal payload.
        hidden_dim: Width of the shared MLP trunk.
    """

    def __init__(
        self,
        workspace_dim: int,
        self_model_dim: int,
        slot_dim: int = 8,
        attention_schema_dim: int = 16,
        slot_names: tuple[str, ...] = REPORT_SLOT_NAMES,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        if workspace_dim <= 0 or self_model_dim <= 0:
            raise ValueError("workspace_dim and self_model_dim must be positive")
        if slot_dim <= 0 or attention_schema_dim <= 0 or hidden_dim <= 0:
            raise ValueError("slot_dim, attention_schema_dim, hidden_dim must be positive")
        if tuple(slot_names) != REPORT_SLOT_NAMES:
            raise ValueError(f"slot_names must equal {REPORT_SLOT_NAMES} (got {tuple(slot_names)})")

        self.workspace_dim = workspace_dim
        self.self_model_dim = self_model_dim
        self.slot_dim = slot_dim
        self.attention_schema_dim = attention_schema_dim
        self.slot_names: tuple[str, ...] = tuple(slot_names)
        self.hidden_dim = hidden_dim

        state_dim = workspace_dim + self_model_dim + len(self.slot_names)
        self.state_dim = state_dim

        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.slot_heads = nn.ModuleDict(
            OrderedDict((name, nn.Linear(hidden_dim, slot_dim)) for name in self.slot_names)
        )

        self.attention_schema_head = nn.Sequential(
            nn.Linear(hidden_dim, attention_schema_dim),
            nn.Tanh(),
        )

        self.report_predictor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, len(self.slot_names) * slot_dim),
        )

    def state_representation(
        self,
        workspace: Tensor,
        self_model_vec: Tensor,
        payload: dict[str, Tensor],
    ) -> Tensor:
        """Build the input vector consumed by the report heads.

        Returns a tensor of shape ``(state_dim,)`` — workspace + self-model
        concatenated with a single summary scalar per payload slot.

        ``payload`` must be provided so the returned vector matches the
        ``state_dim`` the module was constructed with; an empty payload dict
        yields a zero summary for each named slot.
        """
        if workspace.dim() != 1 or self_model_vec.dim() != 1:
            raise ValueError(
                "workspace and self_model_vec must be 1-D tensors; "
                f"got {tuple(workspace.shape)} and {tuple(self_model_vec.shape)}"
            )
        if workspace.shape[-1] != self.workspace_dim:
            raise ValueError(
                f"workspace last dim must be {self.workspace_dim}; got {workspace.shape[-1]}"
            )
        if self_model_vec.shape[-1] != self.self_model_dim:
            raise ValueError(
                f"self_model_vec last dim must be {self.self_model_dim}; "
                f"got {self_model_vec.shape[-1]}"
            )

        parts: list[Tensor] = [workspace, self_model_vec]
        summaries: list[Tensor] = []
        for name in self.slot_names:
            slot = payload.get(name)
            if slot is None:
                summaries.append(
                    torch.zeros(1, dtype=workspace.dtype, device=workspace.device)
                )
                continue
            if slot.dim() != 1:
                raise ValueError(f"payload slot '{name}' must be 1-D; got {tuple(slot.shape)}")
            summaries.append(slot.mean().unsqueeze(0))
        parts.append(torch.cat(summaries, dim=-1))
        return torch.cat(parts, dim=-1)

    def forward(
        self,
        workspace: Tensor,
        self_model_vec: Tensor,
        payload: dict[str, Tensor] | None = None,
    ) -> SelfModelOutput:
        if payload is None:
            payload = {}
        state_repr = self.state_representation(workspace, self_model_vec, payload)
        trunk_out = self.trunk(state_repr)
        slots = OrderedDict((name, self.slot_heads[name](trunk_out)) for name in self.slot_names)
        attention_schema = self.attention_schema_head(trunk_out)
        return SelfModelOutput(
            report=SelfReport(slots=slots),
            attention_schema=attention_schema,
            state_repr=state_repr,
        )

    def forward_from_state(self, state: Any) -> SelfModelOutput:
        """Convenience wrapper that pulls ``workspace``/``self_model``/``payload`` off a
        ``PhenomenalState`` (or any object with those attributes/methods)."""
        workspace = state.workspace
        self_model_vec = state.self_model
        payload = state.payload() if callable(getattr(state, "payload", None)) else None
        return self.forward(workspace, self_model_vec, payload)


def report_consistency_loss(
    model: SelfModel,
    prev_workspace: Tensor,
    prev_self_model: Tensor,
    prev_payload: dict[str, Tensor],
    curr_report: SelfReport,
) -> Tensor:
    """Loss encouraging the current self-report to be predictable from the prior state.

    Computes MSE between:
      - the predictor head's estimate of the next self-report given the prior
        phenomenal state, and
      - the actual self-report produced by the model at the current step.

    Both sides are differentiable; gradients flow through the predictor head and
    back into the report heads via the ``curr_report`` side.
    """
    state_repr = model.state_representation(prev_workspace, prev_self_model, prev_payload)
    predicted = model.report_predictor(state_repr)
    predicted_slots = predicted.view(len(model.slot_names), model.slot_dim)
    target = torch.stack([curr_report[name] for name in model.slot_names], dim=0)
    return torch.mean((predicted_slots - target) ** 2)


def contrastive_report_loss(
    reports: list[SelfReport],
    slot_dim: int,
    positive_index: int = 0,
    temperature: float = 0.1,
) -> Tensor:
    """InfoNCE-style contrastive loss over a batch of self-reports.

    For the anchor ``reports[0]``, treats the report at ``positive_index`` (in
    ``reports[1:]``) as the positive and the remaining reports as negatives.
    This matches the bead spec's "distinct phenomenal states produce distinct
    reports" property: a positive pair comes from the same state, negatives
    come from different states.

    Args:
        reports: Sequence of ``SelfReport`` objects; ``reports[0]`` is the anchor.
        slot_dim: Dimensionality of each slot in the report.
        positive_index: Index into ``reports[1:]`` of the positive partner.
        temperature: Softmax temperature (smaller = sharper).

    Returns:
        Scalar tensor with the contrastive loss.
    """
    if len(reports) < 2:
        raise ValueError("contrastive_report_loss requires at least 2 reports")
    if not (0 <= positive_index < len(reports) - 1):
        raise ValueError(f"positive_index must be in [0, {len(reports) - 2}]; got {positive_index}")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    vectors = torch.stack([r.vector(slot_dim) for r in reports], dim=0)
    anchor = vectors[0:1]
    others = vectors[1:]

    anchor_norm = torch.nn.functional.normalize(anchor, dim=-1)
    others_norm = torch.nn.functional.normalize(others, dim=-1)
    logits = torch.matmul(anchor_norm, others_norm.transpose(0, 1)).squeeze(0) / temperature
    target = torch.tensor(positive_index, dtype=torch.long, device=logits.device)
    return torch.nn.functional.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))


def contrastive_report_loss_pairwise(
    anchors: list[SelfReport],
    partners: list[SelfReport],
    same_state: list[bool],
    slot_dim: int,
    temperature: float = 0.1,
) -> Tensor:
    """Symmetric contrastive loss for paired (anchor, partner, same?) tuples.

    For each triple, push the partner closer when ``same_state`` is True and
    farther apart when False. Uses cosine-similarity logits scaled by
    ``1/temperature``.
    """
    if not (len(anchors) == len(partners) == len(same_state)):
        raise ValueError("anchors, partners, same_state must have the same length")
    if len(anchors) == 0:
        raise ValueError("need at least one (anchor, partner) pair")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    a = torch.stack([r.vector(slot_dim) for r in anchors], dim=0)
    p = torch.stack([r.vector(slot_dim) for r in partners], dim=0)
    a_n = torch.nn.functional.normalize(a, dim=-1)
    p_n = torch.nn.functional.normalize(p, dim=-1)
    logits = (a_n * p_n).sum(dim=-1) / temperature
    targets = torch.tensor(
        [1.0 if same else 0.0 for same in same_state],
        dtype=logits.dtype,
        device=logits.device,
    )
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    return bce


__all__ = [
    "REPORT_SLOT_NAMES",
    "SelfReport",
    "SelfModelOutput",
    "SelfModel",
    "report_consistency_loss",
    "contrastive_report_loss",
    "contrastive_report_loss_pairwise",
]
