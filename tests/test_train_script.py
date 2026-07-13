"""Smoke tests for the end-to-end training script."""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from qualia.data.synthetic import ColoredShapesDataset
from qualia.model.predictive_loop import PredictiveCodingLoop
from qualia.model.self_model import SelfModel
from qualia.model.synthetic_pipeline import (
    PAYLOAD_KEYS,
    SyntheticDecoder,
    SyntheticEncoder,
)
from qualia.train import train as train_mod

_CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "configs")


def _build_small_pipeline():
    encoder = SyntheticEncoder(percept_dim=16, sensory_dim=8, payload_keys=PAYLOAD_KEYS)
    decoder = SyntheticDecoder(encoder)
    loop = PredictiveCodingLoop(
        sensory_dim=8,
        workspace_dim=8,
        self_model_dim=8,
        payload_keys=PAYLOAD_KEYS,
    )
    self_model = SelfModel(workspace_dim=8, self_model_dim=8, slot_dim=4)
    return encoder, decoder, loop, self_model


def test_train_step_runs_end_to_end() -> None:
    torch = pytest.importorskip("torch")
    encoder, decoder, loop, self_model = _build_small_pipeline()

    dataset = ColoredShapesDataset(num_samples=8, percept_dim=16, seed=0)
    batches = train_mod._batch_iter(dataset, batch_size=4, steps=2, seed=0)

    tc = train_mod.TrainConfig(
        percept_dim=16,
        sensory_dim=8,
        workspace_dim=8,
        self_model_dim=8,
        slot_dim=4,
        attention_schema_dim=8,
        payload_vocabs={"shape": 3, "color": 3},
        steps=2,
        log_every=1,
        batch_size=4,
        lr=1e-3,
        rollout_steps=2,
        kl_weight=0.1,
        report_weight=0.1,
        introspection_weight=0.1,
        seed=0,
    )
    optim = torch.optim.Adam(
        list(encoder.parameters())
        + list(decoder.parameters())
        + list(loop.parameters())
        + list(self_model.parameters()),
        lr=tc.lr,
    )

    prev_workspace = None
    prev_self_model = None
    prev_payload = None
    for batch in batches:
        out = train_mod._train_step(
            encoder,
            decoder,
            loop,
            self_model,
            tc,
            batch,
            prev_workspace=prev_workspace,
            prev_self_model=prev_self_model,
            prev_payload=prev_payload,
            optim=optim,
        )
        assert "loss" in out and out["loss"] == out["loss"]  # not NaN
        prev_workspace = out.pop("_prev_workspace")
        prev_self_model = out.pop("_prev_self_model")
        prev_payload = out.pop("_prev_payload")


def test_baseline_config_loads() -> None:
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        cfg = compose(config_name="baseline")
        assert cfg.train.steps == 100
        assert cfg.train.lr == 3e-4
        assert cfg.model.workspace_dim == 32
        assert cfg.eval.dataset.percept_dim == 16
