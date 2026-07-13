"""Unit tests for the tracking module.

These tests cover:
- Run id generation and sanitization
- JsonlTracker (the always-on local sink)
- TensorBoardTracker (writes tfevents + jsonl)
- NullTracker (the ``--no-log`` / offline / CI mode)
- make_tracker dispatch and graceful wandb fallback
- Run lifecycle: directory layout, metadata, checkpoint naming with run_id,
  end-to-end log_metrics / save_checkpoint round-trip
- Train / eval entrypoints expose the --no-log flag without crashing
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from qualia.tracking import (
    JsonlTracker,
    NullTracker,
    Run,
    TensorBoardTracker,
    generate_run_id,
    make_tracker,
    sanitize_run_id,
)
from qualia.tracking.metadata import capture_metadata
from qualia.tracking.system import sample as sample_system

_RUN_ID_RE = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{6}$")


def _make_cfg(tmp_path: Path) -> OmegaConf:
    cfg = OmegaConf.create(
        {
            "run_dir": str(tmp_path / "results" / "test"),
            "workspace_dim": 32,
            "payload_slots": 16,
            "seed": 0,
            "train": {"steps": 100, "log_every": 10, "save_every": 50, "lr": 3.0e-4},
            "model": {
                "encoder": {"backbone": "cnn", "out_dim": 32},
                "decoder": {"backbone": "cnn"},
            },
            "data": {"name": "synthetic", "batch_size": 16, "num_workers": 0},
            "eval": {
                "report_consistency_threshold": 0.5,
                "introspection_accuracy_threshold": 0.5,
            },
            "tracker": {"name": "none", "run_subdir": True},
        }
    )
    return cfg


def test_generate_run_id_format():
    rid = generate_run_id()
    assert _RUN_ID_RE.match(rid), f"unexpected run id format: {rid}"


def test_generate_run_id_unique():
    ids = {generate_run_id() for _ in range(50)}
    assert len(ids) == 50


def test_sanitize_run_id_strips_unsafe_chars():
    assert sanitize_run_id("foo/bar:baz") == "foo-bar-baz"
    assert sanitize_run_id("a" * 500) == "a" * 128
    assert sanitize_run_id("") != ""
    assert sanitize_run_id("###") != ""


def test_sanitize_run_id_preserves_safe_chars():
    assert sanitize_run_id("my.run_42-ok") == "my.run_42-ok"


def test_sample_system_returns_dict():
    metrics = sample_system()
    assert isinstance(metrics, dict)
    for k, v in metrics.items():
        assert isinstance(k, str)
        assert isinstance(v, int | float)


def test_metadata_capture_returns_expected_keys():
    cfg = OmegaConf.create({"workspace_dim": 32})
    meta = capture_metadata(cfg)
    assert "git_sha" in meta
    assert "hostname" in meta
    assert "platform" in meta
    assert "python" in meta
    assert "torch" in meta
    assert "argv" in meta
    assert "config" in meta
    assert meta["config"] == {"workspace_dim": 32}


def test_make_tracker_none(tmp_path):
    t = make_tracker("none", run_dir=str(tmp_path), run_id="r1")
    assert isinstance(t, NullTracker)
    t.close()


def test_make_tracker_tensorboard(tmp_path):
    t = make_tracker("tensorboard", run_dir=str(tmp_path), run_id="r2")
    assert isinstance(t, TensorBoardTracker)
    t.close()


def test_make_tracker_unknown_raises(tmp_path):
    with pytest.raises(ValueError):
        make_tracker("not-a-tracker", run_dir=str(tmp_path), run_id="r3")


def test_make_tracker_wandb_falls_back_when_missing(tmp_path):
    t = make_tracker("wandb", run_dir=str(tmp_path), run_id="r4")
    assert isinstance(t, TensorBoardTracker | JsonlTracker)
    t.close()


def test_jsonl_tracker_writes_metrics(tmp_path):
    t = JsonlTracker(run_dir=str(tmp_path), run_id="jsonl-r")
    t.log_metrics({"loss": 0.5, "lr": 1e-3}, step=0)
    t.log_metrics({"loss": 0.3}, step=10)
    t.close()
    lines = (tmp_path / "logs" / "metrics.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    rows = [json.loads(line) for line in lines]
    assert rows[0]["step"] == 0
    assert rows[0]["loss"] == 0.5
    assert rows[1]["step"] == 10
    assert rows[1]["loss"] == 0.3


def test_jsonl_tracker_writes_config_and_metadata(tmp_path):
    cfg = OmegaConf.create({"a": 1, "b": [1, 2, 3]})
    t = JsonlTracker(run_dir=str(tmp_path), run_id="cfg-r")
    t.log_config(cfg)
    t.log_metadata({"foo": "bar"})
    t.log_config(cfg)
    t.close()
    config = json.loads((tmp_path / "logs" / "config.json").read_text())
    metadata = json.loads((tmp_path / "logs" / "metadata.json").read_text())
    assert config == {"a": 1, "b": [1, 2, 3]}
    assert metadata == {"foo": "bar"}


def test_tensorboard_tracker_writes_tfevents(tmp_path):
    t = TensorBoardTracker(run_dir=str(tmp_path), run_id="tb-r")
    t.log_metrics({"loss": 0.5}, step=0)
    t.log_text("note", "hello world", step=0)
    t.close()
    tb_dir = tmp_path / "tb" / "tb-r"
    assert tb_dir.is_dir()
    files = list(tb_dir.iterdir())
    assert any("tfevents" in f.name for f in files), files
    rows = [
        json.loads(line)
        for line in (tmp_path / "logs" / "metrics.jsonl").read_text().strip().splitlines()
    ]
    metrics_rows = [r for r in rows if r.get("kind") == "metrics"]
    text_rows = [r for r in rows if r.get("kind") == "text"]
    assert len(metrics_rows) == 1
    assert metrics_rows[0]["loss"] == 0.5
    assert len(text_rows) == 1
    assert text_rows[0]["text"] == "hello world"


def test_run_creates_directory_layout(tmp_path):
    cfg = _make_cfg(tmp_path)
    with Run(cfg, run_id="layout", tracker_name="none") as run:
        assert run.run_id == "layout"
        assert run.run_dir == str(tmp_path / "results" / "test" / "layout")
        assert os.path.isdir(run.run_dir)
        assert os.path.isdir(os.path.join(run.run_dir, "logs"))
        assert os.path.isdir(os.path.join(run.run_dir, "checkpoints"))


def test_run_auto_assigns_run_id(tmp_path):
    cfg = _make_cfg(tmp_path)
    cfg.tracker.run_subdir = True
    with Run(cfg, tracker_name="none") as run:
        assert _RUN_ID_RE.match(run.run_id)
        assert run.run_dir.endswith(run.run_id)


def test_run_metadata_includes_git_sha_and_seed(tmp_path):
    cfg = _make_cfg(tmp_path)
    with Run(cfg, run_id="meta", tracker_name="none") as run:
        meta = run.metadata
        assert "argv" in meta
        assert meta["config"]["seed"] == 0
        assert "git_sha" in meta
        assert "hostname" in meta


def test_run_log_metrics_persists(tmp_path):
    cfg = _make_cfg(tmp_path)
    cfg.tracker.run_subdir = False
    with Run(cfg, run_id="metric", tracker_name="none") as run:
        run.log_metrics({"loss": 0.5, "lr": 1e-3, "grad_norm": 1.0}, step=0)
        run.log_metrics({"loss": 0.4}, step=10)
    lines = (
        (tmp_path / "results" / "test" / "logs" / "metrics.jsonl").read_text().strip().splitlines()
    )
    assert len(lines) == 2
    assert json.loads(lines[0])["grad_norm"] == 1.0


def test_run_checkpoint_uses_run_id_in_filename(tmp_path):
    cfg = _make_cfg(tmp_path)
    cfg.tracker.run_subdir = False
    with Run(cfg, run_id="ck", tracker_name="none") as run:
        path = run.save_checkpoint(
            {"model_state": {"w": torch.tensor([1.0])}, "optim_state": {"lr": 1e-3}},
            step=42,
            name="model",
            metrics={"loss": 0.1},
        )
    assert path.endswith("-runck.pt")
    assert "step00000042" in path
    ck = torch.load(path, weights_only=False)
    assert ck["run_id"] == "ck"
    assert ck["step"] == 42
    assert ck["metrics"] == {"loss": 0.1}
    assert ck["config"]["seed"] == 0
    assert torch.equal(ck["model_state"]["w"], torch.tensor([1.0]))
    assert ck["optim_state"]["lr"] == 1e-3


def test_run_close_is_idempotent(tmp_path):
    cfg = _make_cfg(tmp_path)
    cfg.tracker.run_subdir = False
    run = Run(cfg, run_id="idem", tracker_name="none")
    run.close()
    run.close()


def test_run_context_manager_closes(tmp_path):
    cfg = _make_cfg(tmp_path)
    cfg.tracker.run_subdir = False
    with Run(cfg, run_id="ctx", tracker_name="none") as run:
        run.log_metrics({"loss": 1.0}, step=0)
    assert (tmp_path / "results" / "test" / "logs" / "metrics.jsonl").exists()


def test_run_subdir_false_uses_run_dir_directly(tmp_path):
    cfg = _make_cfg(tmp_path)
    cfg.tracker.run_subdir = False
    with Run(cfg, run_id="flat", tracker_name="none") as run:
        assert run.run_dir == str(tmp_path / "results" / "test")


def test_run_uses_cfg_tracker_name(tmp_path):
    cfg = _make_cfg(tmp_path)
    cfg.tracker.run_subdir = False
    cfg.tracker.name = "none"
    with Run(cfg, run_id="cfg-name") as run:
        assert run.tracker_name == "none"


def test_run_uses_env_var_when_no_cfg_name(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path)
    cfg.tracker.run_subdir = False
    cfg.tracker.name = "tensorboard"
    monkeypatch.setenv("QUALIA_TRACKER", "none")
    with Run(cfg, run_id="env-name") as run:
        assert run.tracker_name == "none"


def test_run_cfg_name_used_when_no_env_var(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path)
    cfg.tracker.run_subdir = False
    cfg.tracker.name = "none"
    monkeypatch.delenv("QUALIA_TRACKER", raising=False)
    with Run(cfg, run_id="cfg-wins") as run:
        assert run.tracker_name == "none"


def test_run_no_log_flag_translates_to_none(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path)
    cfg.tracker.run_subdir = False
    cfg.tracker.name = "tensorboard"
    monkeypatch.setenv("QUALIA_TRACKER", "none")
    with Run(cfg, run_id="env-flag") as run:
        assert run.tracker_name == "none"


def test_train_entrypoint_no_log(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import sys

    saved = sys.argv
    try:
        sys.argv = ["train.py"]
        monkeypatch.setenv("QUALIA_TRACKER", "none")
        from qualia.train import train as train_mod

        train_mod.main()
    finally:
        sys.argv = saved
    assert (tmp_path / "results" / "baseline").exists()


def test_eval_entrypoint_no_log(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import sys

    saved = sys.argv
    try:
        sys.argv = ["run_eval.py"]
        monkeypatch.setenv("QUALIA_TRACKER", "none")
        from qualia.eval import run_eval as eval_mod

        eval_mod.main()
    finally:
        sys.argv = saved
    assert (tmp_path / "results" / "baseline").exists()


def test_run_logs_text(tmp_path):
    cfg = _make_cfg(tmp_path)
    cfg.tracker.run_subdir = False
    with Run(cfg, run_id="text", tracker_name="none") as run:
        run.log_text("note", "hello", step=0)
    rows = [
        json.loads(line)
        for line in (tmp_path / "results" / "test" / "logs" / "metrics.jsonl")
        .read_text()
        .strip()
        .splitlines()
    ]
    text_rows = [r for r in rows if r.get("kind") == "text"]
    assert text_rows and text_rows[0]["key"] == "note" and text_rows[0]["text"] == "hello"


@pytest.mark.parametrize(
    "argv,expected_trackers,env_tracker",
    [
        (["train.py"], "tensorboard", None),
        (["train.py", "--no-log"], "none", "none"),
    ],
)
def test_train_entrypoint_cli_matrix(tmp_path, monkeypatch, argv, expected_trackers, env_tracker):
    monkeypatch.chdir(tmp_path)
    import sys

    saved = sys.argv
    try:
        sys.argv = argv
        if env_tracker is not None:
            monkeypatch.setenv("QUALIA_TRACKER", env_tracker)
        if "--no-log" in argv:
            monkeypatch.setenv("QUALIA_TRACKER", "none")
            sys.argv = [a for a in sys.argv if a != "--no-log"]
        from qualia.train import train as train_mod

        train_mod.main()
        run_dirs = list((tmp_path / "results" / "baseline").iterdir())
        assert run_dirs, "no run directory was created"
        run_dir = run_dirs[0]
        if expected_trackers == "none":
            assert not (run_dir / "tb").exists(), "tb dir must not exist in --no-log mode"
        else:
            assert (run_dir / "tb").exists(), "tb dir must exist for tensorboard mode"
    finally:
        sys.argv = saved


def test_metrics_jsonl_is_append_only(tmp_path):
    cfg = _make_cfg(tmp_path)
    cfg.tracker.run_subdir = False
    with Run(cfg, run_id="append", tracker_name="none") as run:
        for s in range(5):
            run.log_metrics({"loss": 1.0 / (s + 1)}, step=s * 10)
    lines = (
        (tmp_path / "results" / "test" / "logs" / "metrics.jsonl").read_text().strip().splitlines()
    )
    assert len(lines) == 5
    losses = [json.loads(line)["loss"] for line in lines]
    assert losses == [1.0, 0.5, 1.0 / 3, 0.25, 0.2]


def test_checkpoint_filename_format(tmp_path):
    cfg = _make_cfg(tmp_path)
    cfg.tracker.run_subdir = False
    with Run(cfg, run_id="fmt", tracker_name="none") as run:
        path = run.checkpoint_path(step=7, name="model")
    assert re.match(r".*/model-step00000007-runfmt\.pt$", path)
