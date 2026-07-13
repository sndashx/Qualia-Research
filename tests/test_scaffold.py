"""Smoke tests for the scaffold."""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from qualia import __version__


def test_version() -> None:
    assert __version__ == "0.1.0"


def test_hydra_config_loads() -> None:
    config_dir = str(Path(__file__).resolve().parent.parent / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="config")
        assert cfg.workspace_dim == 32
        assert cfg.model.workspace_dim == 32
        assert cfg.payload_slots == 16
        assert cfg.train.steps == 100
        assert cfg.eval.report_consistency_threshold == 0.5
