"""Run context manager.

A :class:`Run` is the single entry-point the training and eval entrypoints
use. It:

1. Picks (or accepts) a unique ``run_id`` — a short, sortable, filesystem-safe
   identifier. By default ``YYYYMMDD-HHMMSS-<6 hex>``.
2. Creates the run directory layout (``results/<cfg.run_dir>/<run_id>/`` with
   ``logs/`` and ``checkpoints/`` subdirectories).
3. Captures git/host/env metadata and persists it as ``metadata.json``.
4. Instantiates the configured :class:`Tracker` (TensorBoard by default,
   opt-in W&B, ``none`` for ``--no-log`` offline runs).
5. Exposes :meth:`log_metrics`, :meth:`log_text`, :meth:`save_checkpoint`
   helpers that the training loop calls per step / per epoch / per save.

The same :class:`Run` is safe to use as a context manager::

    with Run(cfg) as run:
        for step in range(cfg.train.steps):
            ...
            run.log_metrics({"loss": ..., "lr": ..., "grad_norm": ...}, step=step)
            if step % cfg.train.save_every == 0:
                run.save_checkpoint(model, optimizer, step=step)

Usage::

    run = Run(cfg)
    run_id = run.run_id
    run_dir = run.run_dir
    tracker = run.tracker
    ...
    run.close()
"""

from __future__ import annotations

import os
import re
import secrets
import time
from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .metadata import capture_metadata
from .tracker import Tracker, make_tracker


def generate_run_id(now: float | None = None) -> str:
    """Build a run id of the form ``YYYYMMDD-HHMMSS-<6 hex>``."""
    t = time.gmtime(now if now is not None else time.time())
    stamp = time.strftime("%Y%m%d-%H%M%S", t)
    suffix = secrets.token_hex(3)
    return f"{stamp}-{suffix}"


_SAFE_RUN_ID = re.compile(r"[^A-Za-z0-9_.-]+")


def sanitize_run_id(s: str) -> str:
    cleaned = _SAFE_RUN_ID.sub("-", s.strip())
    return cleaned[:128] or generate_run_id()


@dataclass
class Run(AbstractContextManager["Run"]):
    cfg: Any
    run_id: str = ""
    run_dir: str = ""
    tracker: Tracker | None = None
    log_every: int = 1
    save_every: int | None = None
    checkpoint_dir: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    _closed: bool = False

    def __init__(
        self,
        cfg: Any,
        run_id: str | None = None,
        log_every: int = 1,
        save_every: int | None = None,
        tracker_name: str | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.log_every = max(1, int(log_every))
        self.save_every = int(save_every) if save_every is not None and save_every > 0 else None

        run_name = _cfg_get(cfg, "run.name")
        base = _cfg_get(cfg, "run_dir") or "results/baseline"
        base_path = Path(str(base))
        if run_name:
            base_path = base_path / str(run_name)

        resolved_run_id = sanitize_run_id(
            run_id or _cfg_get(cfg, "tracker.run_id") or run_name or generate_run_id()
        )
        self.run_id = resolved_run_id

        if _cfg_get(cfg, "tracker.run_subdir", default=True):
            base_path = base_path / self.run_id
        self.run_dir = str(base_path)
        self.checkpoint_dir = os.path.join(self.run_dir, "checkpoints")

        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, "logs"), exist_ok=True)

        try:
            self.metadata = capture_metadata(cfg, cwd=_find_repo_root())
        except Exception:
            self.metadata = {}

        requested = (
            tracker_name
            or os.environ.get("QUALIA_TRACKER")
            or _cfg_get(cfg, "tracker.name")
            or "tensorboard"
        )
        if isinstance(requested, str) and requested.lower() in (
            "none",
            "no-log",
            "nolog",
            "offline",
        ):
            requested = "none"
        try:
            self.tracker = make_tracker(requested, self.run_dir, self.run_id, cfg)
        except Exception as e:
            import warnings

            warnings.warn(
                f"tracker {requested!r} unavailable ({e}); using offline JSONL.",
                stacklevel=2,
            )
            self.tracker = make_tracker("none", self.run_dir, self.run_id, cfg)

        self.tracker.log_metadata({"run_id": self.run_id, "run_dir": self.run_dir, **self.metadata})
        self.tracker.log_config(_safe_config(cfg))

    @property
    def tracker_name(self) -> str:
        return getattr(self.tracker, "name", "unknown")

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        if self.tracker is None:
            return
        self.tracker.log_metrics(dict(metrics), int(step))

    def log_text(self, key: str, text: str, step: int | None = None) -> None:
        if self.tracker is None:
            return
        self.tracker.log_text(key, text, step=step)

    def checkpoint_path(self, step: int, name: str = "model") -> str:
        return os.path.join(self.checkpoint_dir, f"{name}-step{step:08d}-run{self.run_id}.pt")

    def save_checkpoint(
        self,
        state: Mapping[str, Any] | Any,
        step: int,
        name: str = "model",
        metrics: Mapping[str, float] | None = None,
    ) -> str:
        import torch

        path = self.checkpoint_path(step, name=name)
        payload: dict[str, Any] = {"step": int(step), "run_id": self.run_id}
        if metrics is not None:
            payload["metrics"] = {k: float(v) for k, v in metrics.items()}
        try:
            from omegaconf import OmegaConf

            payload["config"] = OmegaConf.to_container(self.cfg, resolve=True)
        except Exception:
            payload["config"] = None
        if isinstance(state, Mapping):
            payload.update(dict(state))
        else:
            payload["model_state"] = state
        torch.save(payload, path)
        if self.tracker is not None:
            try:
                self.tracker.save_artifact(path, name=os.path.basename(path))
            except Exception:
                pass
            try:
                self.tracker.log_text(
                    "checkpoint",
                    f"saved {os.path.basename(path)} step={step}",
                    step=step,
                )
            except Exception:
                pass
        return path

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.tracker is not None:
            try:
                self.tracker.close()
            except Exception:
                pass

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _cfg_get(cfg: Any, dotted: str, default: Any = None) -> Any:
    try:
        node = cfg
        for part in dotted.split("."):
            if node is None:
                return default
            if hasattr(node, "get"):
                node = node.get(part)
            else:
                node = node[part]
        if node is None:
            return default
        return node
    except Exception:
        return default


def _safe_config(cfg: Any) -> dict[str, Any]:
    try:
        from omegaconf import OmegaConf

        return OmegaConf.to_container(cfg, resolve=True) or {}
    except Exception:
        return {}


def _find_repo_root(start: str | None = None) -> str | None:
    cur = os.path.abspath(start or os.getcwd())
    for _ in range(8):
        if os.path.isdir(os.path.join(cur, ".git")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return None
        cur = parent
    return None
