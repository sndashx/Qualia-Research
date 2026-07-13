"""Evaluation entrypoint: run the introspection & reportability suite.

Usage (from the project root, with the package installed in editable mode):

    python -m qualia.eval.run_eval                    # uses defaults
    python -m qualia.eval.run_eval --config-name=baseline

The script:
  1. Builds a synthetic ``ColoredShapesDataset`` from the Hydra config.
  2. Runs the three eval metrics (introspection accuracy, report
     consistency, downstream grounding).
  3. Asserts the report-consistency threshold from ``cfg.eval``.
  4. Prints a summary table.
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from qualia.data.synthetic import ColoredShapesDataset
from qualia.eval.metrics import run_eval_suite

_CONFIG_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "configs"))


def _build_dataset(cfg: DictConfig) -> ColoredShapesDataset:
    data_cfg = cfg.eval.dataset
    return ColoredShapesDataset(
        num_samples=int(data_cfg.num_samples),
        percept_dim=int(data_cfg.percept_dim),
        seed=int(cfg.seed),
        include_agency=bool(data_cfg.include_agency),
    )


def _suite_kwargs(cfg: DictConfig) -> dict[str, Any]:
    eval_cfg = cfg.eval
    return {
        "seed": int(cfg.seed),
        "workspace_dim": int(eval_cfg.workspace_dim),
        "self_model_dim": int(eval_cfg.self_model_dim),
        "payload_slots": int(eval_cfg.payload_slots),
        "slot_dim": int(eval_cfg.slot_dim),
        "attention_schema_dim": int(eval_cfg.attention_schema_dim),
        "epochs": int(eval_cfg.epochs),
        "lr": float(eval_cfg.lr),
    }


def _print_summary(metrics: dict[str, float], elapsed_s: float) -> None:
    print("[qualia.eval.run_eval] metrics:")
    for name, value in metrics.items():
        print(f"  {name:>24s} = {value:.4f}")
    print(f"  {'elapsed_s':>24s} = {elapsed_s:.2f}")


def _check_thresholds(metrics: dict[str, float], cfg: DictConfig) -> list[str]:
    failures: list[str] = []
    if not math.isfinite(metrics["introspection_accuracy"]):
        failures.append("introspection_accuracy is not finite")
    if not math.isfinite(metrics["report_consistency"]):
        failures.append("report_consistency is not finite")
    if not math.isfinite(metrics["downstream_grounding"]):
        failures.append("downstream_grounding is not finite")

    rc_threshold = float(cfg.eval.report_consistency_threshold)
    ia_threshold = float(cfg.eval.introspection_accuracy_threshold)
    if metrics["report_consistency"] < rc_threshold:
        failures.append(
            f"report_consistency {metrics['report_consistency']:.4f} "
            f"< threshold {rc_threshold:.4f}"
        )
    if metrics["introspection_accuracy"] < ia_threshold:
        failures.append(
            f"introspection_accuracy {metrics['introspection_accuracy']:.4f} "
            f"< threshold {ia_threshold:.4f}"
        )
    return failures


@hydra.main(version_base=None, config_path=_CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> dict[str, Any]:
    print("[qualia.eval.run_eval] starting introspection & reportability suite")
    print(OmegaConf.to_yaml(cfg.eval))

    dataset = _build_dataset(cfg)
    print(
        f"[qualia.eval.run_eval] dataset: ColoredShapesDataset "
        f"(n={len(dataset)}, percept_dim={dataset.percept_dim}, seed={dataset.seed})"
    )

    started = time.perf_counter()
    bundle = run_eval_suite(dataset, **_suite_kwargs(cfg))
    elapsed_s = time.perf_counter() - started

    metrics = bundle.as_dict()
    _print_summary(metrics, elapsed_s)

    failures = _check_thresholds(metrics, cfg)
    if failures:
        for failure in failures:
            print(f"[qualia.eval.run_eval] FAIL: {failure}")
        raise SystemExit(1)

    summary_path = os.path.join(os.getcwd(), "eval_metrics.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump({"metrics": metrics, "elapsed_s": elapsed_s}, fh, indent=2)
    print(f"[qualia.eval.run_eval] wrote {summary_path}")

    return {"metrics": metrics, "elapsed_s": elapsed_s}


if __name__ == "__main__":
    main()
