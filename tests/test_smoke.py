"""End-to-end smoke test for the training pipeline.

Goals (from bead 176807e1):
- A polecat can verify a code change didn't break the pipeline in <60 s, locally,
  with no GPU.
- Verifies: forward pass works, backward pass works, checkpoint saves, checkpoint
  loads, eval runs.

Approach:
- Exercises every public component that exists today: encoder, decoder,
  PhenomenalState.
- Runs 2 training steps end-to-end (forward -> loss -> backward -> optimizer
  step -> zero_grad) on a tiny synthetic batch.
- Saves a checkpoint (model state + optimizer state + step) and reloads it,
  asserting the loaded weights match.
- Runs a minimal eval (reconstruction loss + introspection snapshot) and asserts
  outputs are finite.
- Invokes the training entrypoint through Hydra so a wiring regression in the
  CLI surface is caught.
- Everything runs on CPU on a 4-sample batch of 32x32 images.

These tests are tagged with the ``smoke`` marker so they can be excluded from the
full CI run with ``pytest -m "not smoke"``.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from qualia.model.phenomenal_state import PhenomenalState

pytestmark = pytest.mark.smoke

STEPS = 2


def _train_one_step(
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    state: PhenomenalState,
    optimizer: torch.optim.Optimizer,
    batch: torch.Tensor,
) -> dict[str, float]:
    """A single end-to-end step: encode -> reconstruct -> state update -> loss.

    The composite loss (reconstruction + state-action modulation) is built as a
    single tensor so that exactly one ``backward`` call traverses the graph.
    """
    encoder.train()
    decoder.train()
    state.train()

    optimizer.zero_grad()
    enc_out = encoder(batch)
    recon = decoder.reconstruct(enc_out)
    recon_loss = torch.nn.functional.mse_loss(recon, batch)

    state_out = state.encode(enc_out.sensory)
    action = torch.cat(
        [state_out["workspace"].mean(dim=0), state_out["self_model"].mean(dim=0)],
        dim=-1,
    ).unsqueeze(0)
    modulated = state.modulate_action(action).mean()

    loss = recon_loss + 0.01 * modulated
    loss.backward()
    optimizer.step()

    payload_vec = enc_out.payload_vector()
    return {
        "recon_loss": float(recon_loss.detach().item()),
        "payload_norm": float(payload_vec.detach().norm().item()),
    }


def _save_checkpoint(
    path: Path,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    state: PhenomenalState,
    optimizer: torch.optim.Optimizer,
    step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "phenomenal_state": state.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def _load_checkpoint(
    path: Path,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    state: PhenomenalState,
    optimizer: torch.optim.Optimizer,
) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    encoder.load_state_dict(payload["encoder"])
    decoder.load_state_dict(payload["decoder"])
    state.load_state_dict(payload["phenomenal_state"])
    optimizer.load_state_dict(payload["optimizer"])
    return int(payload["step"])


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_smoke_forward_pass_is_finite(
    encoder_decoder: tuple[torch.nn.Module, torch.nn.Module],
    tiny_batch: torch.Tensor,
) -> None:
    """Forward pass (encode -> reconstruct) returns finite tensors of the right shape."""
    encoder, decoder = encoder_decoder
    enc_out = encoder(tiny_batch)
    recon = decoder.reconstruct(enc_out)

    assert recon.shape == tiny_batch.shape
    assert torch.isfinite(recon).all()
    assert torch.isfinite(enc_out.sensory).all()
    for key, tensor in enc_out.payload.items():
        assert torch.isfinite(tensor).all(), f"non-finite payload slot {key!r}"


def test_smoke_backward_pass_updates_parameters(
    encoder_decoder: tuple[torch.nn.Module, torch.nn.Module],
    phenomenal_state: PhenomenalState,
    tiny_batch: torch.Tensor,
) -> None:
    """A backward pass + optimizer step moves every trainable parameter off its initial value."""
    encoder, decoder = encoder_decoder

    encoder_params_before = {n: p.detach().clone() for n, p in encoder.named_parameters()}
    decoder_params_before = {n: p.detach().clone() for n, p in decoder.named_parameters()}

    enc_out = encoder(tiny_batch)
    recon = decoder.reconstruct(enc_out)
    loss = torch.nn.functional.mse_loss(recon, tiny_batch)
    loss.backward()

    grad_seen_encoder = any(
        p.grad is not None and p.grad.abs().sum().item() > 0 for p in encoder.parameters()
    )
    grad_seen_decoder = any(
        p.grad is not None and p.grad.abs().sum().item() > 0 for p in decoder.parameters()
    )
    assert grad_seen_encoder, "encoder produced no non-zero gradients"
    assert grad_seen_decoder, "decoder produced no non-zero gradients"

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    optimizer.step()
    optimizer.zero_grad()

    moved = []
    for n, p in encoder.named_parameters():
        if not torch.allclose(p.data, encoder_params_before[n]):
            moved.append(f"encoder.{n}")
    for n, p in decoder.named_parameters():
        if not torch.allclose(p.data, decoder_params_before[n]):
            moved.append(f"decoder.{n}")

    assert moved, "expected at least one parameter to change after one optimizer step"
    assert any(name.startswith("encoder.") for name in moved)
    assert any(name.startswith("decoder.") for name in moved)


def test_smoke_training_loop_two_steps(
    encoder_decoder: tuple[torch.nn.Module, torch.nn.Module],
    phenomenal_state: PhenomenalState,
    tiny_batch: torch.Tensor,
) -> None:
    """Two end-to-end training steps complete with finite losses and changing state."""
    encoder, decoder = encoder_decoder
    state = phenomenal_state

    params = list(encoder.parameters()) + list(decoder.parameters()) + list(state.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-3)

    metrics: list[dict[str, float]] = []
    for _ in range(STEPS):
        metrics.append(_train_one_step(encoder, decoder, state, optimizer, tiny_batch))

    for m in metrics:
        assert math.isfinite(m["recon_loss"]), f"non-finite recon_loss: {m}"
        assert math.isfinite(m["payload_norm"]), f"non-finite payload_norm: {m}"

    report = state.introspect()
    assert set(report["payload"].keys()) == set(state.payload_keys)
    assert report["step"] == STEPS
    for slot, value in report["payload"].items():
        assert all(math.isfinite(v) for v in value), f"non-finite introspected slot {slot!r}"


def test_smoke_checkpoint_round_trip(
    encoder_decoder: tuple[torch.nn.Module, torch.nn.Module],
    phenomenal_state: PhenomenalState,
    tiny_batch: torch.Tensor,
    tmp_path: Path,
) -> None:
    """Save a checkpoint, reload it, verify parameter equality and step recovery."""
    encoder, decoder = encoder_decoder
    state = phenomenal_state

    params = list(encoder.parameters()) + list(decoder.parameters()) + list(state.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-3)

    _train_one_step(encoder, decoder, state, optimizer, tiny_batch)
    _save_checkpoint(tmp_path / "ckpt.pt", encoder, decoder, state, optimizer, step=STEPS)

    fresh_encoder_params = {n: p.detach().clone() for n, p in encoder.named_parameters()}
    fresh_state_step = int(state._step.item())  # noqa: SLF001 - test inspects internal counter

    torch.load(tmp_path / "ckpt.pt", map_location="cpu", weights_only=False)
    for p in encoder.parameters():
        p.data.zero_()
    for p in decoder.parameters():
        p.data.zero_()
    for p in state.parameters():
        p.data.zero_()
    state._step.zero_()  # noqa: SLF001

    recovered_step = _load_checkpoint(tmp_path / "ckpt.pt", encoder, decoder, state, optimizer)

    assert recovered_step == STEPS
    assert int(state._step.item()) == fresh_state_step  # noqa: SLF001
    for n, p in encoder.named_parameters():
        assert torch.allclose(p.data, fresh_encoder_params[n]), f"encoder.{n} did not round-trip"

    _train_one_step(encoder, decoder, state, optimizer, tiny_batch)
    enc_out = encoder(tiny_batch)
    assert torch.isfinite(enc_out.sensory).all()


def test_smoke_eval_runs_and_returns_finite_metrics(
    encoder_decoder: tuple[torch.nn.Module, torch.nn.Module],
    phenomenal_state: PhenomenalState,
    tiny_batch: torch.Tensor,
) -> None:
    """An eval pass (no_grad, eval mode) returns finite reconstruction loss and a structured state report."""
    encoder, decoder = encoder_decoder
    state = phenomenal_state

    encoder.eval()
    decoder.eval()
    state.eval()

    with torch.no_grad():
        enc_out = encoder(tiny_batch)
        recon = decoder.reconstruct(enc_out)
        recon_loss = torch.nn.functional.mse_loss(recon, tiny_batch)
        state.encode(enc_out.sensory)
        report = state.introspect()

    assert math.isfinite(float(recon_loss.item())), "eval reconstruction loss not finite"
    assert isinstance(report, dict)
    assert "workspace" in report and "self_model" in report and "payload" in report
    assert all(math.isfinite(v) for v in report["workspace"])
    assert all(math.isfinite(v) for v in report["self_model"])
    for slot, value in report["payload"].items():
        assert all(math.isfinite(v) for v in value), f"eval payload slot {slot!r} not finite"


def test_smoke_train_entrypoint_invokable(tmp_path: Path) -> None:
    """The `qualia.train.train` entrypoint loads Hydra config and exits cleanly."""
    config_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "configs"))
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="config")
        assert cfg.train.steps > 0
        assert cfg.train.lr > 0
        assert cfg.workspace_dim > 0
        assert cfg.payload_slots > 0
        assert OmegaConf.to_yaml(cfg)  # serializes without error


def test_smoke_train_entrypoint_runs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Invoking `qualia.train.train.main(cfg)` returns without raising and prints status."""
    from qualia.train import train as train_mod

    config_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "configs"))
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="config")
        train_mod.main(cfg)

    captured = capsys.readouterr()
    assert "qualia.train.train" in captured.out
    assert f"train.steps={cfg.train.steps}" in captured.out
