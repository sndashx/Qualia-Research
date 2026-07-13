# Experiment Tracking

Every `qualia-train` and `qualia-eval` run is automatically wired to an
experiment tracker. This page covers:

- What gets logged and where the dashboards live.
- How to pick a backend (TensorBoard / W&B / offline).
- How to use the `Run` API from inside the training / eval loops.
- How to resume, reproduce, and match a checkpoint back to a dashboard.

## TL;DR

```bash
# Default (TensorBoard)
qualia-train
qualia-eval

# Force a custom run id (useful for sweeps / restarts)
qualia-train tracker.run_id=baseline-2026-07-13

# Offline / CI runs (no tfevents, no wandb, only the JSONL sink)
qualia-train --no-log
qualia-eval --no-log

# Weights & Biases (requires `pip install wandb` and `wandb login`)
QUALIA_TRACKER=wandb qualia-train

# Point TensorBoard at a specific run
tensorboard --logdir results/baseline/<run_id>/tb
```

Each run creates a directory of the form
`results/<cfg.run_dir>/<run_id>/{logs,checkpoints,tb,...}`. The run id is a
short, sortable, unique string (`YYYYMMDD-HHMMSS-<6 hex>` by default) and
appears in checkpoint filenames so a `.pt` file can be matched back to a
dashboard row.

## What gets logged

| Kind | Where | Notes |
|---|---|---|
| Hyperparameters / config | `logs/config.json` + tensorboard `hparams` (TB) + wandb `config` | Full resolved Hydra config (all overrides). |
| Run metadata | `logs/metadata.json` | Git SHA, branch, dirty flag, hostname, OS, Python, torch, CLI argv, env (subset). |
| Per-step metrics | `logs/metrics.jsonl` (always) + tensorboard scalars + wandb stream | `loss`, `lr`, `grad_norm`, etc., every `train.log_every` steps. |
| System metrics | tensorboard scalars only | CPU %, RAM used/total %, per-GPU memory used/total, GPU util % when CUDA is present. Sampled automatically on every `log_metrics` call. |
| Per-epoch eval | `logs/metrics.jsonl` + tensorboard + wandb | Whatever the eval suite emits — `eval/threshold.*`, `eval/accuracy`, `eval/perplexity`, etc. |
| Text annotations | `logs/metrics.jsonl` + tensorboard `add_text` + wandb | Free-form notes attached to a step (e.g. config dump at step 0). |
| Checkpoints | `checkpoints/<name>-step<NNNNNNNN>-run<run_id>.pt` | `state_dict` plus `step`, `run_id`, `config`, optional `metrics`. Also copied into `logs/artifacts/` and uploaded to wandb when that backend is selected. |

The JSONL sink is **always** written, even when `--no-log` is set, so a run
is recoverable from disk alone. If you delete `tb/` and `wandb/` directories,
nothing is lost.

## Backends

| Backend | How to select | Requires | Offline-friendly | Where to view |
|---|---|---|---|---|
| **tensorboard** (default) | omit override, or `tracker=tensorboard`, or `tracker=wandb` w/o `wandb` installed | `tensorboard` | yes | `tensorboard --logdir results/<cfg.run_dir>` or per-run `results/<cfg.run_dir>/<run_id>/tb` |
| **wandb** | `tracker=wandb` or `QUALIA_TRACKER=wandb` | `wandb`, `WANDB_API_KEY` (or `wandb login`) | no (syncs to wandb.ai) | wandb project page (project defaults to `qualia`, override with `WANDB_PROJECT`) |
| **none / offline** | `--no-log`, `tracker=none`, or `QUALIA_TRACKER=none` | nothing | yes | inspect `logs/metrics.jsonl` directly |

When `tracker=wandb` is requested but `wandb` is not installed (or
`WANDB_API_KEY` is missing), the tracker emits a warning and falls back to
TensorBoard automatically. The JSONL sink is unaffected.

## CLI flags

The train / eval entrypoints accept:

| Flag | Equivalent | Effect |
|---|---|---|
| `--no-log` | `tracker=none` | Skip TensorBoard / W&B. Only the JSONL sink is written. |
| `--tracker <name>` | `QUALIA_TRACKER=<name>` | Pick a backend explicitly (`tensorboard`, `wandb`, `none`). |
| `tracker.run_id=<id>` | `QUALIA_RUN_ID=<id>` | Override the auto-generated run id. The id is sanitized (unsafe chars → `-`) and truncated to 128 chars. |
| `tracker.run_subdir=false` | — | Write directly into `cfg.run_dir` instead of a per-run subdirectory. Useful for sweeps that already namespace their own directories. |

`--no-log` is translated to `QUALIA_TRACKER=none` and stripped from
`sys.argv` before Hydra parses the remaining overrides, so it composes
cleanly with Hydra-style overrides like `model.workspace_dim=64`.

## The `Run` API

The training loop and eval suite both construct a `qualia.tracking.Run`
context manager at startup. The same object exposes everything the
inner loop needs:

```python
from qualia.tracking import Run

with Run(cfg, run_id="my-experiment", tracker_name="tensorboard") as run:
    for step in range(cfg.train.steps):
        loss = train_one_step(...)
        if step % cfg.train.log_every == 0:
            run.log_metrics(
                {
                    "train/loss": float(loss),
                    "train/lr": float(optimizer.param_groups[0]["lr"]),
                    "train/grad_norm": float(grad_norm),
                },
                step=step,
            )
        if step % cfg.train.save_every == 0:
            run.save_checkpoint(
                {"model_state": model.state_dict(), "optim_state": optimizer.state_dict()},
                step=step,
                name="model",
                metrics={"train/loss": float(loss)},
            )
    run.log_metrics({"eval/report_consistency": 0.93, "eval/accuracy": 0.81}, step=cfg.train.steps)
```

Public methods on `Run`:

| Method | Purpose |
|---|---|
| `Run.log_metrics(metrics, step)` | Send a scalar dict to the configured tracker (and to JSONL). |
| `Run.log_text(key, text, step=...)` | Annotate a step with free-form text. |
| `Run.save_checkpoint(state, step, name, metrics=...)` | Persist a `state_dict`-like mapping to `checkpoints/<name>-step<NNNNNNNN>-run<run_id>.pt`; copies it into the artifact sink for the backend. |
| `Run.checkpoint_path(step, name)` | Return the deterministic path without writing. |
| `Run.close()` | Flush and close the tracker. Called automatically when used as a context manager. |

`state` may be either a mapping (the keys are merged into the checkpoint
file alongside `step`, `run_id`, `config`, `metrics`) or a raw object (stored
under `model_state`).

## Run ids and reproducibility

A run id is a short, human-readable, lexicographically-sortable token. By
default it is generated as `YYYYMMDD-HHMMSS-<6 hex>` from `time.gmtime()`,
which guarantees uniqueness across the millisecond it takes to dispatch
multiple polecats. Explicit ids supplied via `cfg.tracker.run_id` or
`QUALIA_RUN_ID` are sanitized:

```
"my run!"   -> "my-run-"
"a"*500    -> "a"*128  (truncated)
"###"      -> "-"       (sanitized, kept — the regex collapses runs of unsafe chars)
"   "      -> (auto-generated; sanitization yields an empty string)
```

The `metadata.json` written next to every run captures the git SHA, branch,
and dirty flag at run start, so any checkpoint can be matched back to the
exact commit that produced it.

## Resuming a run

Re-running the same command will create a **new** run directory; the
existing `metadata.json` and `metrics.jsonl` are not overwritten. To
resume a checkpoint deterministically, supply the original run id:

```bash
qualia-train tracker.run_id=20260713-022011-7a3f01 --cfg-job train.resume_from=results/baseline/20260713-022011-7a3f01/checkpoints/model-step00000100-run20260713-022011-7a3f01.pt
```

The resume-from path is the responsibility of the training-loop bead
(not yet implemented); the tracking infrastructure guarantees that any
`.pt` file under `checkpoints/` round-trips through `torch.load(...)` with
`run_id`, `step`, `config`, `model_state`, and `optim_state` intact.

## Inspecting the JSONL sink

Every run's metrics are recoverable from `results/<run>/logs/metrics.jsonl`:

```bash
tail -f results/baseline/20260713-022011-7a3f01/logs/metrics.jsonl | head -20
```

Each line is a JSON object with a `kind` discriminator:

```json
{"kind": "metrics", "step": 0, "loss": 0.5, "lr": 0.0003, "grad_norm": 1.0}
{"kind": "text", "step": 0, "key": "config_resolved", "text": "..."}
```

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `WandbTracker requires wandb` | `wandb` not installed | `pip install wandb` or use `tracker=tensorboard` / `--no-log` |
| `Unknown tracker backend: 'foo'` | typo | use one of `tensorboard`, `wandb`, `none` |
| `unrecognized arguments: --no-log` when called as a library function | the `__main__` strip is bypassed | set `QUALIA_TRACKER=none` directly |
| `tb/` directory is missing | `--no-log` was used | expected; metrics are still in `logs/metrics.jsonl` |
| `git_sha` is `null` | not inside a git repo (e.g. unpacked tarball) | run from a checkout, or accept `null` and use `metadata.json`'s other fields |

## Tests

The tracking module ships with a comprehensive unit suite
(`tests/test_tracking.py`, 32 tests) covering run-id generation,
sanitization, tracker dispatch, JSONL persistence, tfevents emission,
checkpoint round-trip, end-to-end CLI flows, and env-var precedence.