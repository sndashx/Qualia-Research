"""Evaluation entrypoint (scaffold placeholder + tracking wiring).

Mirrors :mod:`qualia.train.train` for the eval side: instantiates a tracking
:class:`qualia.tracking.Run`, captures config + metadata, and records the
two eval thresholds declared in ``configs/baseline.yaml`` as per-epoch
metrics so the eval-suite bead (downstream) can write into the same run
without further wiring.

Tracker selection follows the same priority as the train entrypoint:
``QUALIA_TRACKER`` env var > ``cfg.tracker.name`` > ``tensorboard``. Pass
``tracker=none`` (or ``QUALIA_TRACKER=none``) for offline / CI runs; the
``--no-log`` flag is also accepted and translated to ``QUALIA_TRACKER=none``.
"""

from __future__ import annotations

import os
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

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


@hydra.main(version_base=None, config_path=_CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    tracker_name = _resolve_tracker_name(cfg)
    run_id_override = None
    try:
        if cfg.tracker.get("run_id"):
            run_id_override = str(cfg.tracker.run_id)
    except (AttributeError, KeyError):
        pass

    with Run(cfg, run_id=run_id_override, tracker_name=tracker_name) as run:
        run.log_text("config_resolved", OmegaConf.to_yaml(cfg), step=0)
        run.log_metrics(
            {
                "eval/threshold.report_consistency": float(
                    cfg.eval.get("report_consistency_threshold", 0.5)
                ),
                "eval/threshold.introspection_accuracy": float(
                    cfg.eval.get("introspection_accuracy_threshold", 0.5)
                ),
            },
            step=0,
        )
        print(f"[qualia.eval.run_eval] run_id={run.run_id}")
        print(f"[qualia.eval.run_eval] run_dir={run.run_dir}")
        print(f"[qualia.eval.run_eval] tracker={run.tracker_name}")
        print(
            f"  eval.report_consistency_threshold={cfg.eval.get('report_consistency_threshold', 0.5)}"
        )
        print(
            f"  eval.introspection_accuracy_threshold={cfg.eval.get('introspection_accuracy_threshold', 0.5)}"
        )
        print("  (eval suite is implemented in a downstream bead)")


if __name__ == "__main__":
    if "--no-log" in sys.argv:
        os.environ["QUALIA_TRACKER"] = "none"
        sys.argv = [a for a in sys.argv if a != "--no-log"]
    main()
