# Qualia-Research

A research repo investigating whether a neural network can represent its own internal "subjective" state as a **first-class, structured, introspectable object** — rather than as opaque activations — and use that representation to ground its outputs.

This is a **scaffold**: the directory layout, dependency set, config system, build targets, and CI are in place. The model is intentionally **not** implemented yet — downstream beads fill in the six modules (Encoder, PredictiveLoop, PhenomenalState, GlobalWorkspace, SelfModel, Decoder/Actor).

See [`docs/research-memo.md`](docs/research-memo.md) for the full research framing: prior-work survey, architectural hypothesis, training losses, and evaluation criteria.

## Project layout

```
qualia/
├── configs/                  Hydra configs (YAML)
├── docs/                     Research memo + notes
├── results/                  Output: logs, checkpoints, RESULTS.md
├── scripts/                  Entry-point shell scripts
├── src/qualia/
│   ├── model/                Encoder, Decoder, PredictiveLoop, PhenomenalState,
│   │                         GlobalWorkspace, SelfModel  (to be implemented)
│   ├── data/                 Datasets + synthetic generators  (to be implemented)
│   ├── train/                Training loop + Hydra entrypoint  (to be implemented)
│   ├── eval/                 Evaluation suite  (to be implemented)
│   └── metrics/              Metric implementations  (to be implemented)
└── tests/                    Pytest suite (one smoke test, to be expanded)
```

## Install

```bash
make setup        # creates .venv, installs package + dev deps, installs pre-commit
```

## Run

```bash
make test         # pytest
make train-toy    # trains on synthetic data (placeholder until training is wired)
make eval-toy     # runs eval suite on synthetic data (placeholder until wired)
make lint         # ruff + black --check
```

## Config system

Configs are Hydra-managed. The default config lives at `configs/config.yaml`. Override anything from the CLI, e.g.:

```bash
qualia-train model.workspace_dim=64 train.steps=1000
```

## Dependencies

Pinned in `pyproject.toml`:

| Package | Version |
|---|---|
| torch | 2.2.2 |
| numpy | 1.26.4 |
| scipy | 1.13.0 |
| hydra-core | 1.3.2 |
| omegaconf | 2.3.0 |
| pytest | 8.2.0 |

## CI

GitHub Actions workflow runs `pytest` and `ruff` on every push and PR.

## Status

This bead (scaffold) is **complete**. The architecture itself is implemented in downstream beads.

## License

MIT. See `LICENSE`.