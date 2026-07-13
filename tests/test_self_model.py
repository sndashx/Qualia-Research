"""Unit tests for the higher-order SelfModel and its losses."""

from __future__ import annotations

import torch
from qualia.model.phenomenal_state import PhenomenalState
from qualia.model.self_model import (
    REPORT_SLOT_NAMES,
    SelfModel,
    contrastive_report_loss,
    contrastive_report_loss_pairwise,
    report_consistency_loss,
)

WORKSPACE_DIM = 8
SELF_MODEL_DIM = 8
SLOT_DIM = 6
ATTN_SCHEMA_DIM = 12
PAYLOAD_SLOTS = 4
PERCEPT_DIM = 16
BATCH = 4
HIDDEN_DIM = 32


def _make_self_model(seed: int = 0) -> SelfModel:
    torch.manual_seed(seed)
    return SelfModel(
        workspace_dim=WORKSPACE_DIM,
        self_model_dim=SELF_MODEL_DIM,
        slot_dim=SLOT_DIM,
        attention_schema_dim=ATTN_SCHEMA_DIM,
        hidden_dim=HIDDEN_DIM,
    )


def _make_state(seed: int = 0) -> PhenomenalState:
    torch.manual_seed(seed)
    return PhenomenalState(
        percept_dim=PERCEPT_DIM,
        workspace_dim=WORKSPACE_DIM,
        self_model_dim=SELF_MODEL_DIM,
        payload_slots=PAYLOAD_SLOTS,
    )


def _encode_state(state: PhenomenalState, seed: int) -> dict:
    torch.manual_seed(seed)
    percepts = torch.randn(BATCH, PERCEPT_DIM)
    return state.encode(percepts)


def test_report_slots_are_well_formed_tensors() -> None:
    model = _make_self_model(seed=0)
    state = _make_state(seed=1)
    _encode_state(state, seed=2)
    out = model.forward_from_state(state)
    assert set(out.report.slots.keys()) == set(REPORT_SLOT_NAMES)
    for name in REPORT_SLOT_NAMES:
        slot = out.report[name]
        assert isinstance(slot, torch.Tensor)
        assert slot.dim() == 1
        assert slot.shape == (SLOT_DIM,)
        assert torch.isfinite(slot).all()
    assert out.attention_schema.shape == (ATTN_SCHEMA_DIM,)
    assert torch.isfinite(out.attention_schema).all()
    assert out.state_repr.shape == (WORKSPACE_DIM + SELF_MODEL_DIM + len(REPORT_SLOT_NAMES),)


def test_distinct_states_produce_distinct_reports() -> None:
    model = _make_self_model(seed=0)
    s1 = _make_state(seed=10)
    s2 = _make_state(seed=11)
    _encode_state(s1, seed=100)
    _encode_state(s2, seed=200)

    torch.manual_seed(999)
    extra = torch.randn(BATCH, PERCEPT_DIM)
    s1.encode(extra)
    s2.encode(extra)

    out1 = model.forward_from_state(s1)
    out2 = model.forward_from_state(s2)

    v1 = out1.report.vector(SLOT_DIM)
    v2 = out2.report.vector(SLOT_DIM)
    assert not torch.allclose(v1, v2), "distinct phenomenal states should yield distinct reports"

    expected_dim = len(REPORT_SLOT_NAMES) * SLOT_DIM
    assert v1.shape == (expected_dim,)
    assert v2.shape == (expected_dim,)

    snap1 = out1.report.to_snapshot()
    snap2 = out2.report.to_snapshot()
    assert set(snap1.keys()) == set(REPORT_SLOT_NAMES)
    assert snap1 != snap2


def test_gradients_flow_through_report_and_losses() -> None:
    model = _make_self_model(seed=0)
    state = _make_state(seed=1)
    percepts = torch.randn(BATCH, PERCEPT_DIM, requires_grad=True)
    state.encode(percepts)

    out = model.forward_from_state(state)
    primary_loss = out.report["arousal"].pow(2).mean()
    for name in REPORT_SLOT_NAMES:
        primary_loss = primary_loss + out.report[name].pow(2).mean()
    primary_loss = primary_loss + out.attention_schema.pow(2).mean()
    primary_loss.backward()
    assert percepts.grad is not None and torch.isfinite(percepts.grad).all()
    params_with_grad = {name: p.grad for name, p in model.named_parameters() if p.grad is not None}
    assert "trunk.0.weight" in params_with_grad
    assert "slot_heads.arousal.weight" in params_with_grad
    assert "attention_schema_head.0.weight" in params_with_grad


def test_gradients_flow_through_report_consistency_loss() -> None:
    model = _make_self_model(seed=3)
    s_prev = _make_state(seed=4)
    s_curr = _make_state(seed=5)
    _encode_state(s_prev, seed=10)
    _encode_state(s_curr, seed=11)

    out_curr = model.forward_from_state(s_curr)
    prev_payload = s_prev.payload()
    rc_loss = report_consistency_loss(
        model,
        s_prev.workspace.detach(),
        s_prev.self_model.detach(),
        {k: v.detach() for k, v in prev_payload.items()},
        out_curr.report,
    )
    assert rc_loss.dim() == 0
    assert torch.isfinite(rc_loss)
    rc_loss.backward()

    pred_grads = {
        name: p.grad for name, p in model.report_predictor.named_parameters() if p.grad is not None
    }
    assert pred_grads, "expected gradients to flow into report_predictor"

    report_grads = {
        name: p.grad for name, p in model.slot_heads.named_parameters() if p.grad is not None
    }
    assert report_grads, "expected gradients to flow into slot_heads via curr_report"


def test_gradients_flow_through_contrastive_loss() -> None:
    model = _make_self_model(seed=7)
    states = [_make_state(seed=100 + i) for i in range(3)]
    outputs = []
    for idx, st in enumerate(states):
        _encode_state(st, seed=200 + idx)
        outputs.append(model.forward_from_state(st))

    cl = contrastive_report_loss(
        [o.report for o in outputs],
        slot_dim=SLOT_DIM,
        positive_index=0,
        temperature=0.1,
    )
    assert cl.dim() == 0
    assert torch.isfinite(cl)
    cl.backward()
    grads = {name: p.grad for name, p in model.named_parameters() if p.grad is not None}
    assert grads, "expected gradients to flow into SelfModel parameters"


def test_contrastive_loss_pulls_positives_and_pushes_negatives() -> None:
    model = _make_self_model(seed=2)
    s = _make_state(seed=3)
    _encode_state(s, seed=4)
    same_report = model.forward_from_state(s).report

    other1 = _make_state(seed=5)
    _encode_state(other1, seed=6)
    diff_report_1 = model.forward_from_state(other1).report

    other2 = _make_state(seed=7)
    _encode_state(other2, seed=8)
    diff_report_2 = model.forward_from_state(other2).report

    positive_report = model.forward_from_state(s).report

    loss_same = contrastive_report_loss(
        [same_report, positive_report, diff_report_2],
        slot_dim=SLOT_DIM,
        positive_index=0,
    )
    loss_diff = contrastive_report_loss(
        [same_report, diff_report_1, diff_report_2],
        slot_dim=SLOT_DIM,
        positive_index=0,
    )
    assert torch.isfinite(loss_same) and torch.isfinite(loss_diff)
    assert loss_diff.item() > loss_same.item(), (
        "contrastive loss should be higher when the partner is a distinct "
        f"state rather than the same state (got loss_diff={loss_diff.item():.4f}, "
        f"loss_same={loss_same.item():.4f})"
    )

    v_same = same_report.vector(SLOT_DIM)
    v_same_partner = positive_report.vector(SLOT_DIM)
    v_diff_partner = diff_report_1.vector(SLOT_DIM)
    sim_same = torch.nn.functional.cosine_similarity(
        v_same.unsqueeze(0), v_same_partner.unsqueeze(0)
    ).item()
    sim_diff = torch.nn.functional.cosine_similarity(
        v_same.unsqueeze(0), v_diff_partner.unsqueeze(0)
    ).item()
    assert sim_same > sim_diff, (
        "cosine similarity between same-state reports should exceed that "
        f"between distinct-state reports (got sim_same={sim_same:.4f}, "
        f"sim_diff={sim_diff:.4f})"
    )


def test_pairwise_contrastive_loss_is_finite_and_differentiable() -> None:
    model = _make_self_model(seed=0)
    s = _make_state(seed=1)
    o = _make_state(seed=2)
    _encode_state(s, seed=10)
    _encode_state(o, seed=11)
    ra = model.forward_from_state(s).report
    rb = model.forward_from_state(o).report
    rc = model.forward_from_state(s).report

    loss = contrastive_report_loss_pairwise(
        anchors=[ra, ra],
        partners=[rc, rb],
        same_state=[True, False],
        slot_dim=SLOT_DIM,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_self_model_rejects_bad_dimensions() -> None:
    model = _make_self_model(seed=0)
    bad_workspace = torch.zeros(WORKSPACE_DIM - 1)
    try:
        model(bad_workspace, torch.zeros(SELF_MODEL_DIM))
    except ValueError:
        return
    raise AssertionError("expected ValueError for wrong-dim workspace")


def test_self_report_dataclass_validates_slots() -> None:
    from qualia.model.self_model import SelfReport

    slots = {name: torch.zeros(SLOT_DIM) for name in REPORT_SLOT_NAMES}
    rep = SelfReport(slots=slots)
    assert rep.vector(SLOT_DIM).shape == (len(REPORT_SLOT_NAMES) * SLOT_DIM,)

    try:
        SelfReport(slots={"arousal": torch.zeros(2)})
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for missing slots")

    extra_slots = dict(slots)
    extra_slots["rogue"] = torch.zeros(SLOT_DIM)
    try:
        SelfReport(slots=extra_slots)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for extra slots")


def test_self_report_vector_validates_slot_dim() -> None:
    from qualia.model.self_model import SelfReport

    slots = {name: torch.zeros(SLOT_DIM) for name in REPORT_SLOT_NAMES}
    rep = SelfReport(slots=slots)

    try:
        rep.vector(SLOT_DIM + 1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for mismatched slot_dim")

    try:
        rep.vector(0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for non-positive slot_dim")


def test_self_model_output_as_dict_is_serializable() -> None:
    model = _make_self_model(seed=0)
    state = _make_state(seed=1)
    _encode_state(state, seed=2)
    out = model.forward_from_state(state)

    snap = out.as_dict()
    assert set(snap["report"].keys()) == set(REPORT_SLOT_NAMES)
    for name in REPORT_SLOT_NAMES:
        assert isinstance(snap["report"][name], list)
        assert all(isinstance(x, float) for x in snap["report"][name])
    assert isinstance(snap["attention_schema"], list)
    assert all(isinstance(x, float) for x in snap["attention_schema"])
    assert isinstance(snap["state_repr"], list)
    assert all(isinstance(x, float) for x in snap["state_repr"])


def test_state_representation_requires_state_dim_with_empty_payload() -> None:
    model = _make_self_model(seed=0)
    workspace = torch.randn(WORKSPACE_DIM)
    self_model_vec = torch.randn(SELF_MODEL_DIM)
    state_repr = model.state_representation(workspace, self_model_vec, payload={})
    expected_dim = WORKSPACE_DIM + SELF_MODEL_DIM + len(REPORT_SLOT_NAMES)
    assert state_repr.shape == (expected_dim,)
    assert torch.isfinite(state_repr).all()
