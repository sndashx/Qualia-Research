"""Experiment tracking infrastructure for ``qualia``.

Public API:

- :class:`Run` (re-exported from :mod:`qualia.tracking.run`) — the context
  manager / lifecycle object every entry point should instantiate.
- :func:`make_tracker` — build a specific backend (TensorBoard / W&B / offline
  JSONL) directly.
- :func:`capture_metadata`, :func:`generate_run_id` — small helpers used by
  :class:`Run` but exported for testability and downstream reuse.
- :func:`sample_system_metrics` — sample CPU/RAM/GPU scalar metrics.

Backend choice: TensorBoard is the default because it has zero auth/network
dependencies, works fully offline, and the ``tfevents`` files are easy to ship
inside ``results/<run>/``. W&B is opt-in via ``tracker=wandb`` (CLI override)
or the ``QUALIA_TRACKER=wandb`` environment variable. ``--no-log`` (or
``tracker=none``) disables every external sink; the JSONL log file is always
written so a run can be recovered from disk alone.
"""

from __future__ import annotations

from .metadata import capture_metadata
from .run import Run, generate_run_id, sanitize_run_id
from .system import sample as sample_system_metrics
from .tracker import (
    JsonlTracker,
    NullTracker,
    TensorBoardTracker,
    Tracker,
    WandbTracker,
    make_tracker,
)

__all__ = [
    "Run",
    "Tracker",
    "JsonlTracker",
    "NullTracker",
    "TensorBoardTracker",
    "WandbTracker",
    "make_tracker",
    "capture_metadata",
    "generate_run_id",
    "sanitize_run_id",
    "sample_system_metrics",
]
