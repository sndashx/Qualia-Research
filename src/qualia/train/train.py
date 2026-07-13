"""End-to-end training script.

Wires the four modules of the qualia architecture together and runs a small
training loop on the synthetic ``ColoredShapesDataset``:

    SyntheticEncoder -> PredictiveCodingLoop -> SelfModel -> SyntheticDecoder

Losses:

  * ``recon`` — MSE on the decoder's reconstruction of the current percept
    from ``(sensory, payload)``.
  * ``predictive`` — MSE on the predictive loop's prediction of the *next*
    sensory vector.
  * ``kl`` — precision-weighted KL on the predicted payload vs the observed
    payload.
  * ``report_consistency`` — MSE on the self-model's predictor head vs the
    current self-report.
  * ``introspection`` — cross-entropy on the categorical payload slots
    (``shape``, ``color``) against the dataset's ground-truth labels.

Eval (per the eval suite): introspection accuracy, report consistency,
downstream grounding.

Outputs land under ``${run_dir}`` (defaults to ``results/baseline``):

  * ``train_log.jsonl`` — per-step loss trace.
  * ``eval_metrics.json`` — eval-suite output.
  * ``RESULTS.md`` — human-readable summary table.
  * ``config.yaml`` — frozen copy of the resolved config.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.nn import functional as F

from qualia.data.synthetic import ColoredShapesDataset
from qualia.eval.metrics import run_eval_suite
from qualia.model.predictive_loop import PredictiveCodingLoop, predictive_coding_loss
from qualia.model.self_model import SelfModel, report_consistency_loss
from qualia.model.synthetic_pipeline import (
    PAYLOAD_KEYS,
    SyntheticDecoder,
    SyntheticEncoder,
)
from qualia.tracking import Run

_CONFIG_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "configs"))


# ---------------------------------------------------------------------------
# Pipeline assembly
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """Resolved train-time hyperparameters (parsed from the Hydra config)."""

    percept_dim: int
    sensory_dim: int
    workspace_dim: int
    self_model_dim: int
    slot_dim: int
    attention_schema_dim: int
    payload_vocabs: dict[str, int] = field(default_factory=dict)
    steps: int = 100
    log_every: int = 10
    batch_size: int = 16
    lr: float = 3e-4
    eval_every: int | None = None
    rollout_steps: int = 4
    kl_weight: float = 0.1
    report_weight: float = 0.1
    introspection_weight: float = 0.1
    seed: int = 0
    num_train_samples: int = 64
    num_eval_samples: int = 32


def _build_payload_vocabs() -> dict[str, int]:
    from qualia.data.synthetic import COLOR_NAMES, SHAPE_NAMES

    return {"shape": len(SHAPE_NAMES), "color": len(COLOR_NAMES)}


def _resolve_train_config(cfg: DictConfig) -> TrainConfig:
    model_cfg = cfg.model
    train_cfg = cfg.train
    data_cfg = cfg.data
    eval_cfg = cfg.eval
    return TrainConfig(
        percept_dim=int(eval_cfg.dataset.percept_dim),
        sensory_dim=int(model_cfg.encoder.out_dim),
        workspace_dim=int(model_cfg.workspace_dim),
        self_model_dim=int(model_cfg.workspace_dim),
        slot_dim=int(eval_cfg.slot_dim),
        attention_schema_dim=int(eval_cfg.attention_schema_dim),
        payload_vocabs=_build_payload_vocabs(),
        steps=int(train_cfg.steps),
        log_every=int(train_cfg.log_every),
        batch_size=int(data_cfg.batch_size),
        lr=float(train_cfg.lr),
        rollout_steps=int(getattr(train_cfg, "rollout_steps", 4)),
        kl_weight=float(getattr(train_cfg, "kl_weight", 0.1)),
        report_weight=float(getattr(train_cfg, "report_weight", 0.1)),
        introspection_weight=float(getattr(train_cfg, "introspection_weight", 0.1)),
        seed=int(cfg.seed),
        num_train_samples=int(getattr(data_cfg, "num_train_samples", 64)),
        num_eval_samples=int(eval_cfg.dataset.num_samples),
    )


def _build_pipeline(
    tc: TrainConfig,
) -> tuple[SyntheticEncoder, SyntheticDecoder, PredictiveCodingLoop, SelfModel]:
    encoder = SyntheticEncoder(
        percept_dim=tc.percept_dim,
        sensory_dim=tc.sensory_dim,
        payload_keys=PAYLOAD_KEYS,
        payload_vocabs=tc.payload_vocabs,
    )
    decoder = SyntheticDecoder(encoder)
    loop = PredictiveCodingLoop(
        sensory_dim=tc.sensory_dim,
        workspace_dim=tc.workspace_dim,
        self_model_dim=tc.self_model_dim,
        payload_keys=PAYLOAD_KEYS,
        payload_vocabs=tc.payload_vocabs,
    )
    self_model = SelfModel(
        workspace_dim=tc.workspace_dim,
        self_model_dim=tc.self_model_dim,
        slot_dim=tc.slot_dim,
        attention_schema_dim=tc.attention_schema_dim,
    )
    return encoder, decoder, loop, self_model


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------


def _introspection_loss(
    payload: dict[str, Tensor],
    targets: dict[str, Tensor],
    payload_vocabs: dict[str, int],
) -> Tensor:
    """Cross-entropy on the categorical payload slots.

    Only slots with ``vocab > 1`` contribute; continuous slots are ignored.
    The targets are long tensors of shape ``(B,)`` matching the dataset's
    ground-truth labels.
    """
    loss = payload["shape"].new_zeros(())
    n_terms = 0
    for key, vocab in payload_vocabs.items():
        if vocab <= 1:
            continue
        if key not in targets:
            continue
        loss = loss + F.cross_entropy(payload[key], targets[key])
        n_terms += 1
    if n_terms == 0:
        return loss
    return loss / n_terms


def _decoder_recon_loss(
    decoder: SyntheticDecoder,
    sensory: Tensor,
    payload: dict[str, Tensor],
    percept: Tensor,
) -> Tensor:
    """MSE between the decoder's reconstruction and the original percept.

    Detaches ``sensory`` so the gradient through this loss flows only into
    the decoder and the payload heads (not the encoder trunk that produced
    the sensory vector). The encoder's reconstruction pressure is supplied
    by the predictive loop's reconstruction term.
    """
    recon = decoder(sensory.detach(), payload)
    return F.mse_loss(recon, percept)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _batch_iter(
    dataset: ColoredShapesDataset,
    batch_size: int,
    steps: int,
    seed: int,
) -> list[dict[str, Tensor]]:
    """Pre-compute ``steps`` random batches from ``dataset``.

    Each batch is a dict with the percept tensor, categorical targets and
    continuous attribute targets, all on CPU. Doing the slicing up-front
    keeps the inner training loop tight and deterministic given the seed.
    """
    n = len(dataset)
    gen = torch.Generator().manual_seed(seed)
    batches: list[dict[str, Tensor]] = []
    for _ in range(steps):
        idx = torch.randint(0, n, (batch_size,), generator=gen).tolist()
        percepts = torch.stack([dataset[i].percept for i in idx], dim=0)
        shape_targets = torch.tensor([dataset[i].shape_idx for i in idx], dtype=torch.long)
        color_targets = torch.tensor([dataset[i].color_idx for i in idx], dtype=torch.long)
        hue_targets = torch.tensor([dataset[i].hue for i in idx], dtype=torch.float32).unsqueeze(-1)
        brightness_targets = torch.tensor(
            [dataset[i].brightness for i in idx], dtype=torch.float32
        ).unsqueeze(-1)
        agency_targets = torch.tensor(
            [dataset[i].agency for i in idx], dtype=torch.float32
        ).unsqueeze(-1)
        batches.append(
            {
                "percept": percepts,
                "targets": {
                    "shape": shape_targets,
                    "color": color_targets,
                    "hue": hue_targets,
                    "brightness": brightness_targets,
                    "agency": agency_targets,
                },
            }
        )
    return batches


def _train_step(
    encoder: SyntheticEncoder,
    decoder: SyntheticDecoder,
    loop: PredictiveCodingLoop,
    self_model: SelfModel,
    tc: TrainConfig,
    batch: dict[str, Tensor],
    *,
    prev_workspace: Tensor | None,
    prev_self_model: Tensor | None,
    prev_payload: dict[str, Tensor] | None,
    optim: torch.optim.Optimizer,
) -> dict[str, float]:
    """Run a single training step and return a flat dict of scalar losses."""
    percept = batch["percept"]
    targets = batch["targets"]

    enc_out = encoder(percept)
    sensory = enc_out.sensory
    payload = enc_out.payload

    rollout_input = sensory.unsqueeze(0).expand(tc.rollout_steps, sensory.shape[0], -1)
    payload_seq = [{key: payload[key] for key in PAYLOAD_KEYS} for _ in range(tc.rollout_steps)]
    trajectory = loop.rollout(rollout_input, payload_sequence=payload_seq)
    last_phenomenal = trajectory.phenomenals[-1]
    last_workspace = last_phenomenal["workspace"]
    last_self_model = last_phenomenal["self_model"]
    last_payload = last_phenomenal["payload"]

    pc_loss = predictive_coding_loss(
        trajectory,
        rollout_input,
        kl_weight=tc.kl_weight,
    )

    # Self-model expects 1-D workspace/self_model/payload (one summary per
    # batch). The predictive loop returns per-step, per-batch tensors; take
    # the batch mean of the last step's state to get a single summary.
    summary_workspace = last_workspace.mean(dim=0)
    summary_self_model = last_self_model.mean(dim=0)
    summary_payload = {key: last_payload[key].mean(dim=0) for key in last_payload}
    sm_out = self_model(summary_workspace, summary_self_model, summary_payload)
    report = sm_out.report
    if prev_workspace is not None and prev_self_model is not None and prev_payload is not None:
        rc_loss = report_consistency_loss(
            self_model,
            prev_workspace,
            prev_self_model,
            prev_payload,
            report,
        )
    else:
        rc_loss = summary_workspace.new_zeros(())

    intro_loss = _introspection_loss(payload, targets, tc.payload_vocabs)
    recon_loss = _decoder_recon_loss(decoder, sensory, payload, percept)

    total = (
        pc_loss["total"]
        + tc.report_weight * rc_loss
        + tc.introspection_weight * intro_loss
        + recon_loss
    )

    optim.zero_grad()
    total.backward()
    optim.step()

    return {
        "loss": float(total.detach().item()),
        "reconstruction": float(pc_loss["reconstruction"].detach().item()),
        "kl": float(pc_loss["kl"].detach().item()),
        "predictive_total": float(pc_loss["total"].detach().item()),
        "report_consistency": float(rc_loss.detach().item()),
        "introspection": float(intro_loss.detach().item()),
        "decoder_recon": float(recon_loss.detach().item()),
        "_prev_workspace": summary_workspace.detach(),
        "_prev_self_model": summary_self_model.detach(),
        "_prev_payload": {key: summary_payload[key].detach() for key in summary_payload},
    }


def _train(
    encoder: SyntheticEncoder,
    decoder: SyntheticDecoder,
    loop: PredictiveCodingLoop,
    self_model: SelfModel,
    tc: TrainConfig,
    train_dataset: ColoredShapesDataset,
    log_path: str,
) -> list[dict[str, float]]:
    """Run the training loop and write per-step losses to ``log_path``."""
    torch.manual_seed(tc.seed)
    params = list(encoder.parameters()) + list(decoder.parameters())
    params += list(loop.parameters()) + list(self_model.parameters())
    optim = torch.optim.Adam(params, lr=tc.lr)

    batches = _batch_iter(train_dataset, tc.batch_size, tc.steps, seed=tc.seed)
    history: list[dict[str, float]] = []

    prev_workspace: Tensor | None = None
    prev_self_model: Tensor | None = None
    prev_payload: dict[str, Tensor] | None = None

    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(json.dumps({"event": "start", "config": tc.__dict__}) + "\n")
        started = time.perf_counter()
        for step, batch in enumerate(batches):
            step_losses = _train_step(
                encoder,
                decoder,
                loop,
                self_model,
                tc,
                batch,
                prev_workspace=prev_workspace,
                prev_self_model=prev_self_model,
                prev_payload=prev_payload,
                optim=optim,
            )
            prev_workspace = step_losses.pop("_prev_workspace")
            prev_self_model = step_losses.pop("_prev_self_model")
            prev_payload = step_losses.pop("_prev_payload")

            step_losses["step"] = step
            step_losses["wall_s"] = float(time.perf_counter() - started)
            history.append(step_losses)
            log_file.write(json.dumps(step_losses) + "\n")

            if tc.log_every and (step % tc.log_every == 0 or step == tc.steps - 1):
                print(
                    f"[qualia.train.train] step={step:4d} "
                    f"loss={step_losses['loss']:.4f} "
                    f"recon={step_losses['reconstruction']:.4f} "
                    f"decoder_recon={step_losses['decoder_recon']:.4f} "
                    f"kl={step_losses['kl']:.4f} "
                    f"report={step_losses['report_consistency']:.4f} "
                    f"intro={step_losses['introspection']:.4f}"
                )

    return history


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------


def _run_eval(tc: TrainConfig, seed: int = 0) -> dict[str, float]:
    dataset = ColoredShapesDataset(
        num_samples=tc.num_eval_samples,
        percept_dim=tc.percept_dim,
        seed=seed,
        include_agency=True,
    )
    bundle = run_eval_suite(
        dataset,
        seed=seed,
        workspace_dim=tc.workspace_dim,
        self_model_dim=tc.self_model_dim,
        payload_slots=int(tc.payload_vocabs.get("shape", 4)),
        slot_dim=tc.slot_dim,
        attention_schema_dim=tc.attention_schema_dim,
        epochs=int(getattr(_config_eval(tc), "epochs", 80)),
        lr=float(getattr(_config_eval(tc), "lr", 0.1)),
    )
    return bundle.as_dict()


def _config_eval(tc: TrainConfig) -> Any:
    return tc


def _write_results_md(
    history: list[dict[str, float]],
    metrics: dict[str, float],
    summary: dict[str, float],
    path: str,
) -> None:
    first = history[0] if history else {}
    last = history[-1] if history else {}
    lines = [
        "# Baseline Experiment — Results",
        "",
        "End-to-end baseline run of the qualia architecture on the synthetic",
        "``ColoredShapesDataset`` (encoder + predictive loop + self-model + decoder).",
        "",
        "## Training losses",
        "",
        "| Metric                 |   step 0 |   final |",
        "|------------------------|---------:|--------:|",
        f"| train loss             | {first.get('loss', float('nan')):.4f} | {last.get('loss', float('nan')):.4f} |",
        f"| reconstruction (pred)  | {first.get('reconstruction', float('nan')):.4f} | {last.get('reconstruction', float('nan')):.4f} |",
        f"| decoder reconstruction | {first.get('decoder_recon', float('nan')):.4f} | {last.get('decoder_recon', float('nan')):.4f} |",
        f"| predictive-coding KL   | {first.get('kl', float('nan')):.4f} | {last.get('kl', float('nan')):.4f} |",
        f"| report consistency     | {first.get('report_consistency', float('nan')):.4f} | {last.get('report_consistency', float('nan')):.4f} |",
        f"| introspection loss     | {first.get('introspection', float('nan')):.4f} | {last.get('introspection', float('nan')):.4f} |",
        "",
        "## Eval suite",
        "",
        "| Metric                 |   value |",
        "|------------------------|--------:|",
        f"| report consistency     | {metrics['report_consistency']:.4f} |",
        f"| introspection accuracy | {metrics['introspection_accuracy']:.4f} |",
        f"| downstream grounding   | {metrics['downstream_grounding']:.4f} |",
        "",
        "## Run summary",
        "",
    ]
    for key, value in summary.items():
        lines.append(f"- **{key}**: `{value}`")
    lines.append("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


@hydra.main(version_base=None, config_path=_CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> dict[str, Any]:
    torch.manual_seed(int(cfg.seed))
    tc = _resolve_train_config(cfg)

    with Run(cfg, log_every=int(tc.log_every)) as run:
        run_dir = run.run_dir

        resolved = OmegaConf.to_container(cfg, resolve=True)
        with open(os.path.join(run_dir, "config.yaml"), "w", encoding="utf-8") as fh:
            json.dump(resolved, fh, indent=2, default=str)

        print(f"[qualia.train.train] run_dir = {run_dir}")
        print(f"[qualia.train.train] config = {OmegaConf.to_yaml(cfg)}")

        train_dataset = ColoredShapesDataset(
            num_samples=tc.num_train_samples,
            percept_dim=tc.percept_dim,
            seed=int(cfg.seed),
            include_agency=True,
        )

        encoder, decoder, loop, self_model = _build_pipeline(tc)

        log_path = os.path.join(run_dir, "train_log.jsonl")
        started = time.perf_counter()
        history = _train(encoder, decoder, loop, self_model, tc, train_dataset, log_path)
        train_elapsed = time.perf_counter() - started

        eval_started = time.perf_counter()
        metrics = _run_eval(tc, seed=int(cfg.seed))
        eval_elapsed = time.perf_counter() - eval_started

        metrics_path = os.path.join(run_dir, "eval_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as fh:
            json.dump(
                {"metrics": metrics, "eval_elapsed_s": eval_elapsed},
                fh,
                indent=2,
            )

        summary = {
            "steps": tc.steps,
            "batch_size": tc.batch_size,
            "lr": tc.lr,
            "train_elapsed_s": f"{train_elapsed:.2f}",
            "eval_elapsed_s": f"{eval_elapsed:.2f}",
            "final_train_loss": f"{history[-1]['loss']:.4f}" if history else "nan",
            "introspection_accuracy": f"{metrics['introspection_accuracy']:.4f}",
            "report_consistency": f"{metrics['report_consistency']:.4f}",
            "downstream_grounding": f"{metrics['downstream_grounding']:.4f}",
        }

        _write_results_md(history, metrics, summary, os.path.join(run_dir, "RESULTS.md"))

        print("[qualia.train.train] eval metrics:")
        for name, value in metrics.items():
            print(f"  {name:>24s} = {value:.4f}")
        print(f"[qualia.train.train] wrote {log_path}")
        print(f"[qualia.train.train] wrote {metrics_path}")
        print(f"[qualia.train.train] wrote {os.path.join(run_dir, 'RESULTS.md')}")

        return {
            "metrics": metrics,
            "history": history,
            "summary": summary,
        }


if __name__ == "__main__":
    main()
