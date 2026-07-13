"""Training entrypoint (scaffold placeholder + tracking wiring + data cache).

The full training loop is implemented in a downstream bead. This stub:

- Loads Hydra config from ``configs/``.
- Instantiates a :class:`qualia.tracking.Run`, which assigns a ``run_id``,
  creates ``results/<cfg.run_dir>/<run_id>/{logs,checkpoints}/``, captures
  metadata (git SHA, hostname, env, config snapshot) and opens the configured
  tracker (TensorBoard by default, ``wandb`` opt-in, ``none`` for offline runs).
- Materializes the dataset via :func:`qualia.data.make_dataloader`. The
  dataset is cached on disk under ``$QUALIA_CACHE`` (default
  ``./.cache/qualia``); subsequent runs with an identical config hit the
  cache and reload in milliseconds. A ``--rebuild-cache`` flag (equivalent to
  ``data.rebuild_cache=true`` in the config) forces regeneration.

Tracker selection follows this priority:

1. CLI override ``tracker=<name>`` (Hydra-native).
2. Environment variable ``QUALIA_TRACKER``.
3. Config value ``cfg.tracker.name``.
4. Default ``tensorboard``.

Pass ``tracker=none`` (or ``QUALIA_TRACKER=none``) for offline / CI runs
(equivalent to the documented ``--no-log`` flag).

Once the loop bead lands, the surrounding ``tracker.log_metrics(...)`` and
``tracker.save_checkpoint(...)`` calls in this stub are the contract every
training step should follow.
"""

from __future__ import annotations

import os
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

from qualia.data import make_dataloader
from qualia.tracking import Run

_CONFIG_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "configs"))


def _resolve_tracker_name(cfg: DictConfig) -> str:
    env = os.environ.get("QUALIA_TRACKER")
    if env:
        if env.lower() in ("none", "no-log", "nolog", "offline"):
            return "none"
        return env
    cfg_name = None
    try:
        cfg_name = cfg.tracker.get("name")
    except (AttributeError, KeyError):
        cfg_name = None
    if cfg_name:
        return str(cfg_name)
    return "tensorboard"


def _should_rebuild_cache(cfg: DictConfig) -> bool:
    try:
        return bool(cfg.data.get("rebuild_cache", False))
    except (AttributeError, KeyError):
        return False


@hydra.main(version_base=None, config_path=_CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    tracker_name = _resolve_tracker_name(cfg)
    run_id_override = None
    try:
        if cfg.tracker.get("run_id"):
            run_id_override = str(cfg.tracker.run_id)
    except (AttributeError, KeyError):
        pass

    log_every = 10
    save_every = 50
    try:
        log_every = int(cfg.train.get("log_every", 10))
        save_every = int(cfg.train.get("save_every", log_every * 5))
    except (AttributeError, KeyError):
        pass

    rebuild_cache = _should_rebuild_cache(cfg)

    with Run(
        cfg,
        run_id=run_id_override,
        log_every=log_every,
        save_every=save_every,
        tracker_name=tracker_name,
    ) as run:
        run.log_text(
            "config_resolved",
            OmegaConf.to_yaml(cfg),
            step=0,
        )

        # Materialize the (cached) dataset. The CachedDataset constructor logs
        # cache HIT / MISS + timing on its own; we additionally persist it as
        # a structured text artifact in the run logs so it shows up alongside
        # other run-scoped metadata.
        dataset, loader = make_dataloader(cfg, rebuild_cache=rebuild_cache)
        stats = dataset.stats
        cache_log = (
            f"data_cache_hit={int(stats.hit)} "
            f"key={stats.cache_key[:12]} "
            f"n={stats.n_samples} "
            f"build_seconds={stats.build_seconds:.4f} "
            f"load_seconds={stats.load_seconds:.4f} "
            f"path={stats.cache_path}"
        )
        run.log_text("data_cache", cache_log, step=0)
        try:
            run.log_metrics(stats.log_fields(), step=0)
        except Exception:
            pass

        print(f"[qualia.train.train] run_id={run.run_id}")
        print(f"[qualia.train.train] run_dir={run.run_dir}")
        print(f"[qualia.train.train] tracker={run.tracker_name}")
        print(f"[qualia.train.train] workspace_dim={cfg.workspace_dim}")
        print(f"[qualia.train.train] payload_slots={cfg.payload_slots}")
        print(f"[qualia.train.train] train.steps={cfg.train.steps}")
        print(f"[qualia.train.train] data.dataset={dataset.cache_key[:12]} n={len(dataset)}")
        print(f"[qualia.train.train] data.cache {cache_log}")
        print("  (training loop is implemented in a downstream bead)")


if __name__ == "__main__":
    # Translate well-known CLI flags that predate the config schema. They are
    # mapped to their config equivalents and stripped from argv so Hydra
    # doesn't choke on unknown overrides.
    if "--no-log" in sys.argv:
        os.environ["QUALIA_TRACKER"] = "none"
        sys.argv = [a for a in sys.argv if a != "--no-log"]
    if "--rebuild-cache" in sys.argv:
        sys.argv = [a for a in sys.argv if a != "--rebuild-cache"]
        # Inject the matching config override so the resolved cfg carries
        # `data.rebuild_cache=true` end-to-end.
        sys.argv.append("data.rebuild_cache=true")
    main()
