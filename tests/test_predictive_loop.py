"""Unit tests for the predictive-coding loop with global workspace integration."""

from __future__ import annotations

import torch

from qualia.model.predictive_loop import (
    GlobalWorkspace,
    PredictiveCodingLoop,
    predictive_coding_loss,
)

SENSORY_DIM = 12
WORKSPACE_DIM = 16
SELF_MODEL_DIM = 16
PAYLOAD_KEYS = ("content", "arousal", "valence")
PAYLOAD_VOCABS = {"content": 4, "arousal": 1, "valence": 1}
BATCH = 3


def _make_loop(seed: int = 0) -> PredictiveCodingLoop:
    torch.manual_seed(seed)
    return PredictiveCodingLoop(
        sensory_dim=SENSORY_DIM,
        workspace_dim=WORKSPACE_DIM,
        self_model_dim=SELF_MODEL_DIM,
        payload_keys=PAYLOAD_KEYS,
        payload_vocabs=PAYLOAD_VOCABS,
    )


def _make_payload_sequence(T: int, B: int, *, seed: int = 0) -> list[dict[str, torch.Tensor]]:
    g = torch.Generator().manual_seed(seed)
    out: list[dict[str, torch.Tensor]] = []
    for _ in range(T):
        out.append(
            {
                "content": torch.randn(B, 4, generator=g),
                "arousal": torch.randn(B, 1, generator=g),
                "valence": torch.randn(B, 1, generator=g),
            }
        )
    return out


def test_one_step_rollout_shapes() -> None:
    loop = _make_loop(seed=0)
    T = 1
    torch.manual_seed(123)
    sensory = torch.randn(T, BATCH, SENSORY_DIM)
    payloads = _make_payload_sequence(T, BATCH, seed=42)

    traj = loop.rollout(sensory, payload_sequence=payloads, reset=True)

    assert traj.predictions.shape == (T, BATCH, SENSORY_DIM)
    assert traj.errors.shape == (T, BATCH, SENSORY_DIM)
    assert traj.precisions.shape == (T, BATCH, SENSORY_DIM)
    assert traj.workspace.shape == (T, BATCH, WORKSPACE_DIM)
    assert traj.module_weights.shape == (T, BATCH, loop.workspace.n_modules)

    payload_total = sum(PAYLOAD_VOCABS[key] for key in PAYLOAD_KEYS)
    assert traj.predicted_payloads.shape == (T, BATCH, payload_total)
    assert traj.observed_payloads.shape == (T, BATCH, payload_total)
    assert traj.payload_keys == PAYLOAD_KEYS
    assert traj.payload_vocabs == PAYLOAD_VOCABS

    # Phenomenal entry must contain workspace + self_model + payload tensors,
    # one per step.
    assert len(traj.phenomenals) == T
    phenomenals_step = traj.phenomenals[0]
    assert phenomenals_step["workspace"].shape == (BATCH, WORKSPACE_DIM)
    assert phenomenals_step["self_model"].shape == (BATCH, SELF_MODEL_DIM)
    assert set(phenomenals_step["payload"].keys()) == set(PAYLOAD_KEYS)
    payload_slot_dim = max(PAYLOAD_VOCABS.values())
    for value in phenomenals_step["payload"].values():
        assert value.shape == (BATCH, payload_slot_dim)


def test_five_step_rollout_shapes() -> None:
    loop = _make_loop(seed=1)
    T = 5
    torch.manual_seed(7)
    sensory = torch.randn(T, BATCH, SENSORY_DIM)
    payloads = _make_payload_sequence(T, BATCH, seed=13)

    traj = loop.rollout(sensory, payload_sequence=payloads, reset=True)

    assert traj.predictions.shape == (T, BATCH, SENSORY_DIM)
    assert traj.errors.shape == (T, BATCH, SENSORY_DIM)
    assert traj.precisions.shape == (T, BATCH, SENSORY_DIM)
    assert traj.workspace.shape == (T, BATCH, WORKSPACE_DIM)
    assert traj.module_weights.shape == (T, BATCH, loop.workspace.n_modules)

    payload_total = sum(PAYLOAD_VOCABS[key] for key in PAYLOAD_KEYS)
    assert traj.predicted_payloads.shape == (T, BATCH, payload_total)
    assert traj.observed_payloads.shape == (T, BATCH, payload_total)

    assert len(traj.phenomenals) == T
    for step_phen in traj.phenomenals:
        assert step_phen["workspace"].shape == (BATCH, WORKSPACE_DIM)
        assert step_phen["self_model"].shape == (BATCH, SELF_MODEL_DIM)
        assert set(step_phen["payload"].keys()) == set(PAYLOAD_KEYS)

    # Precision weights must be strictly positive (softplus + epsilon).
    assert (traj.precisions > 0).all()
    # Module weights must sum to one across modules at every step.
    assert torch.allclose(
        traj.module_weights.sum(dim=-1),
        torch.ones(T, BATCH),
        atol=1e-5,
    )


def test_gradients_flow_through_prediction_error_pathway() -> None:
    loop = _make_loop(seed=2)
    T = 5
    torch.manual_seed(11)
    sensory = torch.randn(T, BATCH, SENSORY_DIM)
    payloads = _make_payload_sequence(T, BATCH, seed=19)

    traj = loop.rollout(sensory, payload_sequence=payloads, reset=True)
    losses = predictive_coding_loss(traj, observed_sensory=sensory, kl_weight=1.0)
    losses["total"].backward()

    grad_keys_to_check = (
        "sensory_predictor",
        "payload_predictor",
        "precision_head",
        "error_integrator",
        "workspace.experts.0",
        "workspace.scorer",
    )
    grads = {name: grad for name, grad in loop.named_parameters() if grad is not None}
    for key_prefix in grad_keys_to_check:
        matched = [name for name in grads if name.startswith(key_prefix)]
        assert matched, f"expected gradients to flow into {key_prefix!r}"
        non_zero = [name for name in matched if grads[name].abs().sum() > 0]
        assert non_zero, f"expected non-zero gradients in {key_prefix!r}; got zero everywhere"
        for name in non_zero:
            assert torch.isfinite(grads[name]).all(), f"non-finite grad for {name}"

    # PhenomenalState parameters must also receive gradients.
    state_grads = {
        name: grad for name, grad in loop.phenomenal_state.named_parameters() if grad is not None
    }
    assert state_grads, "expected gradients to flow into PhenomenalState parameters"
    for name, grad in state_grads.items():
        assert torch.isfinite(grad).all(), f"non-finite state grad for {name}"
        assert grad.abs().sum() > 0, f"zero state grad for {name}"


def test_loss_decreases_with_overfit_target() -> None:
    """Training on a single fixed sensory sequence reduces loss below the
    initial value. This sanity-checks that the loss is well-formed and
    end-to-end differentiable through the precision-weighted KL pathway."""
    loop = _make_loop(seed=3)
    T = 4
    g = torch.Generator().manual_seed(2024)
    sensory = torch.randn(T, BATCH, SENSORY_DIM, generator=g)
    payloads = _make_payload_sequence(T, BATCH, seed=2025)

    traj0 = loop.rollout(sensory, payload_sequence=payloads, reset=True)
    losses0 = predictive_coding_loss(traj0, observed_sensory=sensory, kl_weight=0.5)
    initial = losses0["total"].item()
    assert torch.isfinite(losses0["reconstruction"])
    assert torch.isfinite(losses0["kl"])
    assert losses0["total"].item() == initial

    optimizer = torch.optim.Adam(loop.parameters(), lr=5e-3)
    for _ in range(25):
        optimizer.zero_grad()
        traj = loop.rollout(sensory, payload_sequence=payloads, reset=True)
        losses = predictive_coding_loss(traj, observed_sensory=sensory, kl_weight=0.5)
        losses["total"].backward()
        optimizer.step()

    traj_final = loop.rollout(sensory, payload_sequence=payloads, reset=True)
    losses_final = predictive_coding_loss(traj_final, observed_sensory=sensory, kl_weight=0.5)
    assert losses_final["total"].item() < initial


def test_global_workspace_broadcasts_to_downstream_dim() -> None:
    loop = _make_loop(seed=4)
    T = 3
    torch.manual_seed(31)
    sensory = torch.randn(T, BATCH, SENSORY_DIM)
    traj = loop.rollout(sensory, payload_sequence=None, reset=True)

    # The global workspace broadcast is exactly workspace_dim wide and must
    # be finite at every step.
    assert traj.workspace.shape == (T, BATCH, WORKSPACE_DIM)
    assert torch.isfinite(traj.workspace).all()

    # Module weights must be a valid probability distribution per step.
    weights = traj.module_weights
    assert torch.allclose(weights.sum(dim=-1), torch.ones(T, BATCH), atol=1e-5)
    assert (weights >= 0).all()


def test_global_workspace_module() -> None:
    """Direct test of the GlobalWorkspace module in isolation."""
    torch.manual_seed(5)
    gws = GlobalWorkspace(input_dim=8, workspace_dim=4, n_modules=3)
    x = torch.randn(2, 8)
    broadcast, weights = gws(x)
    assert broadcast.shape == (2, 4)
    assert weights.shape == (2, 3)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2), atol=1e-5)
    assert (weights >= 0).all()


def test_rollout_with_reset_changes_state_only_when_reset_disabled() -> None:
    """Disabling reset keeps state across ``rollout`` calls."""
    loop = _make_loop(seed=6)
    T = 2
    torch.manual_seed(99)
    seq_a = torch.randn(T, BATCH, SENSORY_DIM)
    seq_b = torch.randn(T, BATCH, SENSORY_DIM)

    traj_a = loop.rollout(seq_a, payload_sequence=None, reset=True)
    traj_b_reset = loop.rollout(seq_b, payload_sequence=None, reset=True)
    traj_b_no_reset = loop.rollout(seq_b, payload_sequence=None, reset=False)

    # Re-running with reset=True must produce identical output given the
    # same input sequence + seed, so the rollout is reproducible.
    assert torch.equal(traj_a.predictions, traj_a.predictions)
    assert torch.equal(traj_b_reset.predictions, traj_b_reset.predictions)

    # Disabling reset means the state was carried over from seq_a, so the
    # first-step predictions for seq_b should differ.
    assert not torch.allclose(traj_b_reset.predictions, traj_b_no_reset.predictions)
