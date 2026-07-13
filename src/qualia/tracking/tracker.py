"""Tracker backends.

A ``Tracker`` is the narrow interface the training loop uses to record metrics.
Three concrete implementations are provided:

- ``TensorBoardTracker`` (default): writes ``tfevents`` files consumable by
  ``tensorboard --logdir=...``. Zero auth, no network, offline-friendly.
- ``WandbTracker`` (opt-in, ``tracker=wandb``): streams to Weights & Biases
  when ``wandb`` is installed and ``WANDB_API_KEY`` (or ``wandb login``) is set.
  Falls back gracefully to TensorBoard (which also writes the JSONL sink) when
  wandb is unavailable.
- ``JsonlTracker`` (always-on local sink): appends every metric dict to a
  newline-delimited JSON file under ``results/<run>/logs/metrics.jsonl``. This
  is the offline/CI fallback and is **always** written even when an external
  tracker is enabled, so a run is recoverable from disk alone.

The selection happens via ``make_tracker(name, run_dir, run_id, cfg)``.
``--no-log`` maps to ``name="none"`` which only writes the JSONL sink.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from .system import sample as sample_system


class Tracker(Protocol):
    name: str

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None: ...

    def log_text(self, key: str, text: str, step: int | None = None) -> None: ...

    def log_config(self, cfg: Mapping[str, Any]) -> None: ...

    def log_metadata(self, metadata: Mapping[str, Any]) -> None: ...

    def save_artifact(self, path: str, name: str | None = None) -> None: ...

    def close(self) -> None: ...


class JsonlTracker:
    """Append-only JSONL sink. Always available, no third-party deps."""

    name = "jsonl"

    def __init__(self, run_dir: str, run_id: str) -> None:
        self.run_dir = run_dir
        self.run_id = run_id
        self.log_dir = Path(run_dir) / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.log_dir / "metrics.jsonl"
        self.config_path = self.log_dir / "config.json"
        self.metadata_path = self.log_dir / "metadata.json"
        self._config_written = False
        self._metadata_written = False

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        payload: dict[str, Any] = {"step": int(step), "kind": "metrics"}
        payload.update({k: float(v) for k, v in metrics.items()})
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")

    def log_text(self, key: str, text: str, step: int | None = None) -> None:
        payload = {
            "step": int(step) if step is not None else None,
            "kind": "text",
            "key": key,
            "text": text,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")

    def log_config(self, cfg: Mapping[str, Any]) -> None:
        if self._config_written:
            return
        with self.config_path.open("w", encoding="utf-8") as f:
            json.dump(_to_jsonable(cfg), f, indent=2, sort_keys=True)
        self._config_written = True

    def log_metadata(self, metadata: Mapping[str, Any]) -> None:
        if self._metadata_written:
            return
        with self.metadata_path.open("w", encoding="utf-8") as f:
            json.dump(_to_jsonable(metadata), f, indent=2, sort_keys=True)
        self._metadata_written = True

    def save_artifact(self, path: str, name: str | None = None) -> None:
        import shutil

        src = Path(path)
        if not src.exists():
            return
        dest_dir = self.log_dir / "artifacts"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / (name or src.name)
        shutil.copy2(src, dest)

    def close(self) -> None:
        pass


class TensorBoardTracker(JsonlTracker):
    """TensorBoard ``SummaryWriter`` wrapper. Inherits the JSONL sink so the
    run is always recoverable from disk.
    """

    name = "tensorboard"

    def __init__(self, run_dir: str, run_id: str) -> None:
        super().__init__(run_dir=run_dir, run_id=run_id)
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as e:
            raise RuntimeError(
                "TensorBoardTracker requires `tensorboard`. Install with "
                "`pip install tensorboard`."
            ) from e
        tb_dir = Path(run_dir) / "tb" / run_id
        tb_dir.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(log_dir=str(tb_dir))

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        super().log_metrics(metrics, step)
        for k, v in metrics.items():
            try:
                self._writer.add_scalar(k, float(v), global_step=step)
            except Exception:
                pass
        sys_metrics = sample_system()
        if sys_metrics:
            for k, v in sys_metrics.items():
                try:
                    self._writer.add_scalar(k, float(v), global_step=step)
                except Exception:
                    pass

    def log_text(self, key: str, text: str, step: int | None = None) -> None:
        super().log_text(key, text, step=step)
        try:
            self._writer.add_text(key, text, global_step=step or 0)
        except Exception:
            pass

    def close(self) -> None:
        try:
            self._writer.flush()
            self._writer.close()
        except Exception:
            pass


class WandbTracker(JsonlTracker):
    """Weights & Biases wrapper. Inherits the JSONL sink."""

    name = "wandb"

    def __init__(self, run_dir: str, run_id: str, cfg: Any | None = None) -> None:
        super().__init__(run_dir=run_dir, run_id=run_id)
        try:
            import wandb
        except ImportError as e:
            raise RuntimeError(
                "WandbTracker requires `wandb`. Install with `pip install wandb`."
            ) from e
        self._wandb = wandb
        project = os.environ.get("WANDB_PROJECT", "qualia")
        entity = os.environ.get("WANDB_ENTITY") or None
        self._run = wandb.init(
            project=project,
            entity=entity,
            name=run_id,
            id=run_id,
            resume="allow",
            dir=str(Path(run_dir) / "wandb"),
            reinit=True,
        )

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        super().log_metrics(metrics, step)
        payload = {k: float(v) for k, v in metrics.items()}
        payload.update(sample_system())
        try:
            self._wandb.log(payload, step=int(step))
        except Exception:
            pass

    def log_text(self, key: str, text: str, step: int | None = None) -> None:
        super().log_text(key, text, step=step)
        try:
            self._wandb.log({key: text}, step=int(step) if step is not None else None)
        except Exception:
            pass

    def save_artifact(self, path: str, name: str | None = None) -> None:
        super().save_artifact(path, name=name)
        try:
            self._wandb.save(path, base_path=os.path.dirname(path) or ".")
        except Exception:
            pass

    def close(self) -> None:
        try:
            self._wandb.finish()
        except Exception:
            pass


class NullTracker(JsonlTracker):
    """Offline/CI tracker. Skips any external backend; only writes JSONL."""

    name = "none"

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        super().log_metrics(metrics, step)

    def log_text(self, key: str, text: str, step: int | None = None) -> None:
        super().log_text(key, text, step=step)

    def save_artifact(self, path: str, name: str | None = None) -> None:
        super().save_artifact(path, name=name)

    def close(self) -> None:
        pass


def make_tracker(name: str, run_dir: str, run_id: str, cfg: Any | None = None) -> Tracker:
    name = (name or "tensorboard").lower()
    if name in ("none", "offline", "no-log", "nolog"):
        return NullTracker(run_dir=run_dir, run_id=run_id)
    if name in ("tensorboard", "tb"):
        return TensorBoardTracker(run_dir=run_dir, run_id=run_id)
    if name in ("wandb", "weights_and_biases", "w&b"):
        try:
            return WandbTracker(run_dir=run_dir, run_id=run_id, cfg=cfg)
        except Exception as e:
            import warnings

            warnings.warn(
                f"wandb tracker unavailable ({e}); falling back to TensorBoard.",
                stacklevel=2,
            )
            return TensorBoardTracker(run_dir=run_dir, run_id=run_id)
    raise ValueError(f"Unknown tracker backend: {name!r}")


def _to_jsonable(obj: Any) -> Any:
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(obj):
            return OmegaConf.to_container(obj, resolve=True)
    except Exception:
        pass
    if isinstance(obj, Mapping):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, str | int | float | bool) or obj is None:
        return obj
    return str(obj)
