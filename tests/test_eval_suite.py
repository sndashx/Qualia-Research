"""End-to-end tests for the introspection & reportability eval suite.

These tests run the full suite on a tiny synthetic dataset and assert:

* All three metrics are finite.
* Report-consistency exceeds the spec-mandated ``0.5`` floor.

A 60-second wall-clock budget is enforced via ``pytest -x`` + a soft budget
assertion; on a modern CPU the suite finishes in a few seconds.
"""

from __future__ import annotations

import math
import time

import pytest

from qualia.data.synthetic import ColoredShapesDataset
from qualia.eval.metrics import (
    downstream_grounding,
    introspection_accuracy,
    report_consistency,
    run_eval_suite,
)

DATASET_KWARGS = {
    "num_samples": 24,
    "percept_dim": 16,
    "seed": 0,
}


def _dataset() -> ColoredShapesDataset:
    return ColoredShapesDataset(**DATASET_KWARGS)


def test_synthetic_dataset_is_deterministic() -> None:
    a = ColoredShapesDataset(**DATASET_KWARGS)
    b = ColoredShapesDataset(**DATASET_KWARGS)
    assert len(a) == len(b) == DATASET_KWARGS["num_samples"]
    for i in range(len(a)):
        sa, sb = a[i], b[i]
        assert sa.shape_idx == sb.shape_idx
        assert sa.color_idx == sb.color_idx
        assert sa.hue == pytest.approx(sb.hue)
        assert sa.brightness == pytest.approx(sb.brightness)
        assert sa.agency == pytest.approx(sb.agency)
        assert torch_allclose(sa.percept, sb.percept)


def torch_allclose(a, b) -> bool:
    import torch

    return torch.allclose(a, b)


def test_individual_metrics_are_finite() -> None:
    ds = _dataset()
    int_acc = introspection_accuracy(ds, seed=0)
    cons = report_consistency(ds, seed=0)
    ground = downstream_grounding(ds, seed=0)
    assert math.isfinite(int_acc)
    assert math.isfinite(cons)
    assert math.isfinite(ground)
    assert 0.0 <= int_acc <= 1.0
    assert -1.0 <= cons <= 1.0
    assert -1.0 <= ground <= 1.0


def test_report_consistency_exceeds_threshold() -> None:
    ds = _dataset()
    cons = report_consistency(ds, seed=0)
    assert cons > 0.5, f"report_consistency should exceed 0.5; got {cons:.4f}"


def test_full_eval_suite_finishes_under_60s() -> None:
    ds = _dataset()
    started = time.perf_counter()
    bundle = run_eval_suite(ds, seed=0)
    elapsed = time.perf_counter() - started

    metrics = bundle.as_dict()
    for name, value in metrics.items():
        assert math.isfinite(value), f"{name} must be finite; got {value!r}"
    assert (
        metrics["report_consistency"] > 0.5
    ), f"report_consistency should exceed 0.5; got {metrics['report_consistency']:.4f}"
    # Spec asks the suite to finish in under 60s; we expect a few seconds on CPU
    # but keep a generous upper bound for slow CI machines.
    assert elapsed < 60.0, f"eval suite took {elapsed:.1f}s (must be < 60s)"


def test_report_consistency_is_one_for_identical_seed() -> None:
    ds = _dataset()
    # Different seeds for the two pipelines would change the report; here we
    # verify the canonical seed yields a value very close to 1.0 (within
    # float32 tolerance) because both runs are fully deterministic.
    cons = report_consistency(ds, seed=42)
    assert cons == pytest.approx(1.0, abs=1e-5)


def test_introspection_accuracy_above_chance_for_large_enough_dataset() -> None:
    ds = ColoredShapesDataset(num_samples=64, percept_dim=16, seed=0)
    int_acc = introspection_accuracy(ds, seed=0, epochs=80, lr=1e-1)
    # 3-way shape + 3-way color + 2-way brightness = chance ~ (1/3 + 1/3 + 1/2) / 3 = 0.389
    # With a decently expressive self-model and 64 samples we should clear it.
    assert int_acc > 0.39, f"introspection accuracy {int_acc:.4f} should beat chance"


def test_metric_bundle_keys_and_types() -> None:
    ds = _dataset()
    bundle = run_eval_suite(ds, seed=0)
    snap = bundle.as_dict()
    assert set(snap.keys()) == {
        "introspection_accuracy",
        "report_consistency",
        "downstream_grounding",
    }
    for value in snap.values():
        assert isinstance(value, float)
        assert math.isfinite(value)
