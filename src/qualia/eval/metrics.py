"""Three introspection & reportability metrics.

Each metric takes the model + a synthetic dataset and returns a finite scalar
that the suite asserts on:

1. ``introspection_accuracy`` — given a held-out stimulus, can the model
   answer questions about its own phenomenal state? We use the ``content``
   slot of the self-report, projected through a learned linear head, to
   predict ground-truth attributes (shape / color / brightness) for held-out
   stimuli. Accuracy is the fraction of attributes predicted correctly.

2. ``report_consistency`` — across repeated runs with the same seed, do
   self-reports match within tolerance? We run the pipeline twice with the
   same seed, freeze the model weights, and compute the cosine similarity of
   the report vectors across runs; averaged over the dataset.

3. ``downstream_grounding`` — does the introspected state improve a downstream
   decision vs a control that ignores it? We fit a linear head that predicts
   the same ground-truth attributes from the self-report and compare it to a
   control head that predicts from a constant (seed-independent) vector. The
   metric is the accuracy gain over the control.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from qualia.data.synthetic import (
    COLOR_NAMES,
    SHAPE_NAMES,
    ColoredShapesDataset,
)
from qualia.model.phenomenal_state import PhenomenalState
from qualia.model.self_model import SelfModel, SelfReport

# Discrete attributes predicted by the introspection / downstream heads.
# We pick a subset of attributes that are categorical (so accuracy is a clean
# 0..1 number) plus the continuous ``brightness`` attribute (signed-error
# based "accuracy" via binning).
INTROSPECTABLE_ATTRS: tuple[str, ...] = ("shape", "color", "brightness")


@dataclass
class MetricBundle:
    """Container for the three eval metrics.

    All values are finite floats in ``[0, 1]`` (or close to it); see the
    docstrings of each metric function for details.
    """

    introspection_accuracy: float
    report_consistency: float
    downstream_grounding: float

    def as_dict(self) -> dict[str, float]:
        return {
            "introspection_accuracy": self.introspection_accuracy,
            "report_consistency": self.report_consistency,
            "downstream_grounding": self.downstream_grounding,
        }


def _build_pipeline(
    *,
    seed: int,
    percept_dim: int,
    workspace_dim: int,
    self_model_dim: int,
    payload_slots: int,
    slot_dim: int,
    attention_schema_dim: int,
) -> tuple[PhenomenalState, SelfModel]:
    """Build a fresh (PhenomenalState, SelfModel) pair seeded deterministically."""
    torch.manual_seed(seed)
    state = PhenomenalState(
        percept_dim=percept_dim,
        workspace_dim=workspace_dim,
        self_model_dim=self_model_dim,
        payload_slots=payload_slots,
    )
    model = SelfModel(
        workspace_dim=workspace_dim,
        self_model_dim=self_model_dim,
        slot_dim=slot_dim,
        attention_schema_dim=attention_schema_dim,
    )
    return state, model


def _run_pipeline(
    dataset: ColoredShapesDataset,
    state: PhenomenalState,
    model: SelfModel,
) -> list[SelfReport]:
    """Encode every sample in ``dataset`` and return the resulting self-reports.

    The state is detached between samples so that each sample's autograd
    graph is independent — without detaching, successive ``encode`` calls
    accumulate the computation graph and the third eval (which trains a head
    via backprop) fails with "Trying to backward through the graph a second
    time". Each sample is treated as an independent "stimulus presentation":
    the live state from the previous sample is carried over as a constant
    tensor (no gradient flow), which is the right semantics for a series of
    held-out stimuli evaluated post-hoc by a downstream head.
    """
    reports: list[SelfReport] = []
    state.reset_state()
    for sample in dataset:
        # Percept is (percept_dim,); encode expects a batch. Add a batch dim
        # and keep the corresponding ``SelfModelOutput`` per sample.
        state.encode(sample.percept.unsqueeze(0))
        out = model.forward_from_state(state)
        reports.append(out.report)
        # Detach the live state so the next sample starts a fresh graph while
        # preserving the carried-over "previous state" values.
        state._workspace_live = state.workspace.detach().clone()  # type: ignore[attr-defined]
        state._self_model_live = state.self_model.detach().clone()  # type: ignore[attr-defined]
        state._payload_live = {
            key: tensor.detach().clone() for key, tensor in state.payload().items()
        }
    return reports


class _IntrospectionHead(nn.Module):
    """Tiny linear head mapping a self-report vector to attribute predictions.

    The head predicts:
      * shape: 3-way logits (matches ``len(SHAPE_NAMES)``)
      * color: 3-way logits (matches ``len(COLOR_NAMES)``)
      * brightness: 2-bin logits (low / high)
    """

    def __init__(self, report_dim: int) -> None:
        super().__init__()
        self.report_dim = report_dim
        self.shape_logits = nn.Linear(report_dim, len(SHAPE_NAMES))
        self.color_logits = nn.Linear(report_dim, len(COLOR_NAMES))
        self.brightness_logits = nn.Linear(report_dim, 2)

    def forward(self, report_vec: Tensor) -> dict[str, Tensor]:
        return {
            "shape": self.shape_logits(report_vec),
            "color": self.color_logits(report_vec),
            "brightness": self.brightness_logits(report_vec),
        }


def _report_matrix(reports: list[SelfReport], slot_dim: int) -> Tensor:
    """Stack a list of ``SelfReport`` objects into a ``(N, report_dim)`` matrix.

    Reports are detached so callers can freely backprop through a downstream
    head without hitting "Trying to backward through the graph a second time".
    """
    if not reports:
        raise ValueError("reports must be non-empty")
    return torch.stack([r.vector(slot_dim).detach() for r in reports], dim=0)


def _attribute_targets(
    dataset: ColoredShapesDataset,
) -> dict[str, Tensor]:
    """Build ground-truth attribute targets for the dataset."""
    shape_targets = torch.tensor([sample.shape_idx for sample in dataset], dtype=torch.long)
    color_targets = torch.tensor([sample.color_idx for sample in dataset], dtype=torch.long)
    brightness_targets = torch.tensor(
        [int(sample.brightness >= 0.5) for sample in dataset], dtype=torch.long
    )
    return {
        "shape": shape_targets,
        "color": color_targets,
        "brightness": brightness_targets,
    }


def introspection_accuracy(
    dataset: ColoredShapesDataset,
    *,
    seed: int = 0,
    workspace_dim: int = 8,
    self_model_dim: int = 8,
    payload_slots: int = 4,
    slot_dim: int = 6,
    attention_schema_dim: int = 8,
    train_fraction: float = 0.75,
    epochs: int = 25,
    lr: float = 5e-2,
) -> float:
    """Fraction of (sample, attribute) pairs whose introspection matches the ground truth.

    Trains a tiny linear head on ``content``/``arousal``/``valence``/... slot
    vectors to predict ground-truth attributes, then evaluates it on the held
    fraction of the dataset. Returns a value in ``[0, 1]``.

    Args:
        dataset: Synthetic dataset of colored shapes with ground-truth labels.
        seed: Seed used to construct the pipeline.
        workspace_dim, self_model_dim, payload_slots, slot_dim,
        attention_schema_dim: Module sizes for the pipeline.
        train_fraction: Fraction of the dataset used to train the head (the
            remainder is the held-out evaluation set).
        epochs: Number of optimization steps for the head.
        lr: Learning rate for the head.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be in (0, 1); got {train_fraction}")
    if epochs <= 0:
        raise ValueError(f"epochs must be positive; got {epochs}")

    state, model = _build_pipeline(
        seed=seed,
        percept_dim=dataset.percept_dim,
        workspace_dim=workspace_dim,
        self_model_dim=self_model_dim,
        payload_slots=payload_slots,
        slot_dim=slot_dim,
        attention_schema_dim=attention_schema_dim,
    )
    reports = _run_pipeline(dataset, state, model)
    report_mat = _report_matrix(reports, slot_dim)
    targets = _attribute_targets(dataset)

    n = len(dataset)
    n_train = max(1, int(n * train_fraction))
    train_idx = list(range(0, n_train))
    test_idx = list(range(n_train, n))

    torch.manual_seed(seed + 17)
    head = _IntrospectionHead(report_dim=report_mat.shape[-1])
    optim = torch.optim.Adam(head.parameters(), lr=lr)

    train_reports = report_mat[train_idx]
    train_targets = {k: v[train_idx] for k, v in targets.items()}

    for _ in range(epochs):
        optim.zero_grad()
        preds = head(train_reports)
        loss = 0.0
        for attr in INTROSPECTABLE_ATTRS:
            loss = loss + torch.nn.functional.cross_entropy(preds[attr], train_targets[attr])
        loss.backward()
        optim.step()

    test_reports = report_mat[test_idx]
    test_targets = {k: v[test_idx] for k, v in targets.items()}
    with torch.no_grad():
        preds = head(test_reports)
        correct = 0
        total = 0
        for attr in INTROSPECTABLE_ATTRS:
            predicted = preds[attr].argmax(dim=-1)
            correct += int((predicted == test_targets[attr]).sum().item())
            total += int(test_targets[attr].numel())

    if total == 0:
        return 0.0
    return float(correct) / float(total)


def report_consistency(
    dataset: ColoredShapesDataset,
    *,
    seed: int = 0,
    workspace_dim: int = 8,
    self_model_dim: int = 8,
    payload_slots: int = 4,
    slot_dim: int = 6,
    attention_schema_dim: int = 8,
) -> float:
    """Cosine similarity of self-reports across two same-seed runs.

    The pipeline is constructed twice with the same seed; both runs encode the
    full dataset and we measure the mean cosine similarity of the per-sample
    self-report vectors. A value close to 1.0 means the system produces
    reproducible reports for reproducible stimuli.
    """
    state_a, model_a = _build_pipeline(
        seed=seed,
        percept_dim=dataset.percept_dim,
        workspace_dim=workspace_dim,
        self_model_dim=self_model_dim,
        payload_slots=payload_slots,
        slot_dim=slot_dim,
        attention_schema_dim=attention_schema_dim,
    )
    state_b, model_b = _build_pipeline(
        seed=seed,
        percept_dim=dataset.percept_dim,
        workspace_dim=workspace_dim,
        self_model_dim=self_model_dim,
        payload_slots=payload_slots,
        slot_dim=slot_dim,
        attention_schema_dim=attention_schema_dim,
    )

    reports_a = _report_matrix(_run_pipeline(dataset, state_a, model_a), slot_dim)
    reports_b = _report_matrix(_run_pipeline(dataset, state_b, model_b), slot_dim)

    sims = torch.nn.functional.cosine_similarity(reports_a, reports_b, dim=-1)
    return float(sims.mean().item())


def downstream_grounding(
    dataset: ColoredShapesDataset,
    *,
    seed: int = 0,
    workspace_dim: int = 8,
    self_model_dim: int = 8,
    payload_slots: int = 4,
    slot_dim: int = 6,
    attention_schema_dim: int = 8,
    train_fraction: float = 0.75,
    epochs: int = 25,
    lr: float = 5e-2,
) -> float:
    """Accuracy gain from introspected state over a constant-vector control.

    Both heads predict the same attributes from the same train/test split.
    The "grounded" head sees the self-report vectors; the "control" head sees
    a constant vector (all zeros) of the same dimensionality. The returned
    metric is ``grounded_accuracy - control_accuracy``.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be in (0, 1); got {train_fraction}")

    state, model = _build_pipeline(
        seed=seed,
        percept_dim=dataset.percept_dim,
        workspace_dim=workspace_dim,
        self_model_dim=self_model_dim,
        payload_slots=payload_slots,
        slot_dim=slot_dim,
        attention_schema_dim=attention_schema_dim,
    )
    reports = _run_pipeline(dataset, state, model)
    report_mat = _report_matrix(reports, slot_dim)
    targets = _attribute_targets(dataset)

    n = len(dataset)
    n_train = max(1, int(n * train_fraction))
    test_idx = list(range(n_train, n))

    train_reports = report_mat[:n_train]
    train_targets = {k: v[:n_train] for k, v in targets.items()}
    test_reports = report_mat[test_idx]
    test_targets = {k: v[test_idx] for k, v in targets.items()}

    def _train_and_eval(train_features: Tensor, test_features: Tensor) -> float:
        torch.manual_seed(seed + 31)
        head = _IntrospectionHead(report_dim=train_features.shape[-1])
        optim = torch.optim.Adam(head.parameters(), lr=lr)
        for _ in range(epochs):
            optim.zero_grad()
            preds = head(train_features)
            loss = 0.0
            for attr in INTROSPECTABLE_ATTRS:
                loss = loss + torch.nn.functional.cross_entropy(preds[attr], train_targets[attr])
            loss.backward()
            optim.step()
        with torch.no_grad():
            preds = head(test_features)
            correct = 0
            total = 0
            for attr in INTROSPECTABLE_ATTRS:
                predicted = preds[attr].argmax(dim=-1)
                correct += int((predicted == test_targets[attr]).sum().item())
                total += int(test_targets[attr].numel())
        if total == 0:
            return 0.0
        return float(correct) / float(total)

    grounded_acc = _train_and_eval(train_reports, test_reports)
    control_features = torch.zeros_like(train_reports)
    control_test = torch.zeros_like(test_reports)
    control_acc = _train_and_eval(control_features, control_test)
    return grounded_acc - control_acc


def run_eval_suite(
    dataset: ColoredShapesDataset,
    *,
    seed: int = 0,
    workspace_dim: int = 8,
    self_model_dim: int = 8,
    payload_slots: int = 4,
    slot_dim: int = 6,
    attention_schema_dim: int = 8,
    epochs: int = 25,
    lr: float = 5e-2,
) -> MetricBundle:
    """Run all three metrics on the given dataset and bundle the results."""
    int_acc = introspection_accuracy(
        dataset,
        seed=seed,
        workspace_dim=workspace_dim,
        self_model_dim=self_model_dim,
        payload_slots=payload_slots,
        slot_dim=slot_dim,
        attention_schema_dim=attention_schema_dim,
        epochs=epochs,
        lr=lr,
    )
    cons = report_consistency(
        dataset,
        seed=seed,
        workspace_dim=workspace_dim,
        self_model_dim=self_model_dim,
        payload_slots=payload_slots,
        slot_dim=slot_dim,
        attention_schema_dim=attention_schema_dim,
    )
    ground = downstream_grounding(
        dataset,
        seed=seed,
        workspace_dim=workspace_dim,
        self_model_dim=self_model_dim,
        payload_slots=payload_slots,
        slot_dim=slot_dim,
        attention_schema_dim=attention_schema_dim,
        epochs=epochs,
        lr=lr,
    )
    return MetricBundle(
        introspection_accuracy=float(int_acc),
        report_consistency=float(cons),
        downstream_grounding=float(ground),
    )


__all__ = [
    "INTROSPECTABLE_ATTRS",
    "MetricBundle",
    "downstream_grounding",
    "introspection_accuracy",
    "report_consistency",
    "run_eval_suite",
]
