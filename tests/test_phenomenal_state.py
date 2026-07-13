"""Unit tests for PhenomenalState."""

from __future__ import annotations

import torch

from qualia.model.phenomenal_state import PAYLOAD_KEYS, PhenomenalState

PERCEPT_DIM = 16
WORKSPACE_DIM = 8
SELF_MODEL_DIM = 8
PAYLOAD_SLOTS = 4
BATCH = 3


def _make_state(seed: int = 0) -> PhenomenalState:
    torch.manual_seed(seed)
    return PhenomenalState(
        percept_dim=PERCEPT_DIM,
        workspace_dim=WORKSPACE_DIM,
        self_model_dim=SELF_MODEL_DIM,
        payload_slots=PAYLOAD_SLOTS,
    )


def test_state_is_queryable_and_well_typed() -> None:
    state = _make_state(seed=0)
    assert state.workspace.shape == (WORKSPACE_DIM,)
    assert state.self_model.shape == (SELF_MODEL_DIM,)
    payload = state.payload()
    assert set(payload.keys()) == set(PAYLOAD_KEYS)
    for value in payload.values():
        assert value.shape == (PAYLOAD_SLOTS,)
    snap = state.introspect()
    assert set(snap.keys()) == {"workspace", "self_model", "payload", "step"}
    assert len(snap["workspace"]) == WORKSPACE_DIM
    assert len(snap["self_model"]) == SELF_MODEL_DIM
    assert snap["step"] == 0


def test_encode_updates_state_deterministically() -> None:
    s1 = _make_state(seed=123)
    s2 = _make_state(seed=123)
    torch.manual_seed(7)
    percepts = torch.randn(BATCH, PERCEPT_DIM)
    s1.encode(percepts)
    s2.encode(percepts)
    snap1 = s1.introspect()
    snap2 = s2.introspect()
    assert snap1["step"] == snap2["step"] == 1
    assert snap1["workspace"] == snap2["workspace"]
    assert snap1["self_model"] == snap2["self_model"]
    for key in PAYLOAD_KEYS:
        assert snap1["payload"][key] == snap2["payload"][key]


def test_state_is_differentiable() -> None:
    state = _make_state(seed=1)
    percepts = torch.randn(BATCH, PERCEPT_DIM, requires_grad=True)
    out = state.encode(percepts)
    loss = out["workspace"].pow(2).mean() + out["self_model"].pow(2).mean()
    for value in out["payload"].values():
        loss = loss + value.pow(2).mean()
    loss.backward()
    assert percepts.grad is not None
    assert torch.isfinite(percepts.grad).all()

    params_with_grad = [
        name for name, p in state.named_parameters() if p.requires_grad and p.grad is not None
    ]
    assert params_with_grad, "expected gradients to flow into PhenomenalState parameters"


def test_modulate_action_uses_current_state() -> None:
    state = _make_state(seed=42)
    action_dim = WORKSPACE_DIM + SELF_MODEL_DIM
    action = torch.ones(5, action_dim)
    gated_default = state.modulate_action(action)

    torch.manual_seed(0)
    state.encode(torch.randn(BATCH, PERCEPT_DIM))
    gated_after = state.modulate_action(action)

    snap = state.introspect()
    assert snap["step"] == 1
    assert gated_default.shape == action.shape
    assert gated_after.shape == action.shape
    assert not torch.allclose(gated_default, gated_after)


def test_modulate_action_requires_correct_dim() -> None:
    state = _make_state(seed=0)
    bad_action = torch.zeros(2, 4)
    try:
        state.modulate_action(bad_action)
    except ValueError:
        return
    raise AssertionError("expected ValueError for wrong-dim action")


def test_modulate_action_is_differentiable_through_state() -> None:
    state = _make_state(seed=2)
    percepts = torch.randn(BATCH, PERCEPT_DIM)
    state.encode(percepts)
    action_dim = WORKSPACE_DIM + SELF_MODEL_DIM
    action = torch.randn(4, action_dim, requires_grad=True)
    gated = state.modulate_action(action)
    loss = gated.pow(2).mean()
    loss.backward()
    assert action.grad is not None
    assert torch.isfinite(action.grad).all()
    grads = {name: p.grad for name, p in state.named_parameters() if p.grad is not None}
    assert grads, "expected gradients to flow into PhenomenalState parameters via modulate_action"


def test_reset_state_clears_buffers() -> None:
    state = _make_state(seed=5)
    state.encode(torch.randn(BATCH, PERCEPT_DIM))
    state.reset_state()
    snap = state.introspect()
    assert snap["step"] == 0
    assert all(v == 0.0 for v in snap["workspace"])
    assert all(v == 0.0 for v in snap["self_model"])
    for key in PAYLOAD_KEYS:
        assert all(v == 0.0 for v in snap["payload"][key])
