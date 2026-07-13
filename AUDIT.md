# AUDIT — Qualia-Research Codebase

**Repo:** `sndashx/Qualia-Research` (branch `main` at audit time)
**Audit date:** 2026-07-13
**Auditor:** Toast (polecat)
**Scope:** Read-only audit. No code modified.

---

## TL;DR — One-paragraph summary

This repo is a **research scaffold** (NOT a runnable ML pipeline). It contains a `pyproject.toml`-installed Python package `qualia` (Hydra config system, ruff/black/pytest, GitHub Actions CI), three implemented modules (`QualiaEncoder`, `QualiaDecoder`, `PhenomenalState`), **zero implemented training loop**, **zero implemented data pipeline**, **zero implemented eval suite**, **zero implemented metric implementations**, **no experiment tracking**, **no data caching**, **no DataLoader**, **no checkpointing**, **no LR schedule**, **no AMP**, **no progress bar**, **no notebooks**, and tests covering only the encoder/decoder and the `PhenomenalState` module. The two entry points (`qualia-train`, `qualia-eval`) are explicit `print()` placeholders. `make train-toy` and `make eval-toy` run but do nothing. The README at line 5 calls this out explicitly: *"The model is intentionally **not** implemented yet — downstream beads fill in the six modules"*.

As a result, this audit cannot report on a "training pipeline" in the operational sense — only on **what is missing**, which is essentially everything needed for daily iteration. The top bottlenecks below are therefore best read as the **highest-leverage beads to write next**, not as performance issues in an existing pipeline.

---

## A) Directory structure

```
sndashx/Qualia-Research/
├── .github/workflows/ci.yml            — CI: ruff + black --check + pytest on push/PR, Python 3.10/3.11
├── .gitignore                          — ignores __pycache__, *.pt, results/*/checkpoints, .venv, etc.
├── .pre-commit-config.yaml             — ruff (with --fix) + ruff-format + black
├── LICENSE                             — MIT
├── Makefile                            — setup / test / train-toy / eval-toy / lint / clean targets
├── README.md                           — project description, layout, install, run, config notes
├── pyproject.toml                      — package metadata, deps, scripts, pytest/ruff/black config
├── configs/
│   ├── config.yaml                     — default Hydra config (defaults: baseline); seed=0
│   └── baseline.yaml                   — baseline Hydra config (run_dir, model, data, train, eval)
├── docs/
│   └── research-memo.md                — 283-line research memo: prior-work survey, architecture, losses, eval criteria
├── src/qualia/                         — Python package `qualia`
│   ├── __init__.py                     — `__version__ = "0.1.0"` (line 3)
│   ├── data/__init__.py                — 1-line placeholder: "Datasets and synthetic generators (to be implemented)"
│   ├── eval/
│   │   ├── __init__.py                 — 1-line placeholder: "Evaluation suite (to be implemented)"
│   │   └── run_eval.py                 — 26-line scaffold: prints eval thresholds, does nothing else
│   ├── metrics/__init__.py             — 1-line placeholder: "Metric implementations (to be implemented)"
│   ├── model/
│   │   ├── __init__.py                 — re-exports QualiaEncoder, QualiaDecoder, PhenomenalState + heads
│   │   ├── encoder.py                  — QualiaEncoder (CNN/ViT/audio backbones) + TinyConvNeXt + TinyViT + PayloadHead + SensoryHead + EncoderOutput
│   │   ├── decoder.py                  — QualiaDecoder + PayloadFusion + _CNNGenerator + _ViTGenerator + _AudioGenerator
│   │   └── phenomenal_state.py         — PhenomenalState (GRU-based), PhenomenalStateRecord
│   └── train/
│       ├── __init__.py                 — 1-line placeholder: "Training entrypoint (to be implemented)"
│       └── train.py                    — 27-line scaffold: prints config values, does nothing else
└── tests/
    ├── __init__.py                     — 1-line placeholder (package marker)
    ├── test_scaffold.py                — 23-line smoke test: version + Hydra config loads
    ├── test_phenomenal_state.py        — 124-line unit test of PhenomenalState module
    └── test_encoder_decoder.py         — 270-line unit test of QualiaEncoder + QualiaDecoder
```

**File-by-file purpose:**

| Path | LOC | Purpose |
|---|---|---|
| `.github/workflows/ci.yml` | 30 | Runs `ruff check`, `black --check`, and `pytest` on every push/PR against `main` and `convoy/**` branches, Python matrix 3.10/3.11. |
| `.pre-commit-config.yaml` | 12 | Pre-commit hooks: ruff (with --fix), ruff-format, black. |
| `.gitignore` | 20 | Standard Python ignores + `results/*/checkpoints/`, `results/*/logs/`, `*.pt`, `*.pth`, `.venv`. |
| `LICENSE` | — | MIT. |
| `Makefile` | 36 | `setup` (venv + install + pre-commit), `test`, `train-toy`, `eval-toy`, `lint`, `clean`. `train-toy`/`eval-toy` invoke `python -m qualia.train.train --config-name=baseline` and `python -m qualia.eval.run_eval --config-name=baseline`. |
| `README.md` | 72 | High-level description, layout, install, run, config (Hydra), dependencies, CI, status. Explicitly states the project is a scaffold. |
| `pyproject.toml` | 52 | Package `qualia==0.1.0`, deps (torch 2.2.2, numpy 1.26.4, scipy 1.13.0, hydra-core 1.3.2, omegaconf 2.3.0), dev extras (pytest 8.2.0, pytest-cov 5.0.0, ruff 0.4.4, black 24.4.2, pre-commit 3.7.1), console scripts `qualia-train` and `qualia-eval`, pytest config, ruff config. |
| `configs/config.yaml` | 31 | Root Hydra config; declares `defaults: [baseline]`; sets `workspace_dim=32`, `payload_slots=16`, `seed=0`; mirrors model/data/train/eval sections. |
| `configs/baseline.yaml` | 24 | Baseline run config: `run_dir: results/baseline`, encoder `convnext_tiny`, decoder `convnext_tiny`, `workspace_dim=32`, `payload_slots=16`, data `stochastic_colored_shapes`, `batch_size=16`, `num_workers=0`, `train.steps=100`, `train.log_every=10`, `train.lr=3.0e-4`, `eval.report_consistency_threshold=0.5`, `eval.introspection_accuracy_threshold=0.5`. |
| `docs/research-memo.md` | 283 | Long-form research framing: prior-work survey (GWT, IIT, Predictive Coding/FEP, AST, HOT, slot/object-centric), 6-module architecture hypothesis, 5 training losses, 7 evaluation criteria (IA, RC, Distinctness, DG, Manipulability, Robustness, Φ-proxy), minimal eval gate, downstream bead mapping. |
| `src/qualia/__init__.py` | 3 | Package marker; sets `__version__ = "0.1.0"`. |
| `src/qualia/data/__init__.py` | 1 | Placeholder; no Dataset, no DataLoader, no synthetic generator implemented. |
| `src/qualia/eval/__init__.py` | 1 | Placeholder; no eval metric implementations. |
| `src/qualia/eval/run_eval.py` | 26 | `@hydra.main` entrypoint that prints `cfg.eval.report_consistency_threshold` and `cfg.eval.introspection_accuracy_threshold`, then exits. |
| `src/qualia/metrics/__init__.py` | 1 | Placeholder; no metric implementations. |
| `src/qualia/model/__init__.py` | 33 | Public re-exports of the three implemented modules' classes. |
| `src/qualia/model/encoder.py` | 322 | `QualiaEncoder`: modality=`image`/`audio`, backbone=`cnn` (TinyConvNeXt) or `vit` (TinyViT), audio uses 1D conv stem; emits `EncoderOutput(sensory, payload, backbone_features)`. `PayloadHead` produces per-slot logits/continuous. `PAYLOAD_KEYS = ("shape","color","hue","brightness","agency")`. |
| `src/qualia/model/decoder.py` | 340 | `QualiaDecoder` reconstructing perception from `(sensory, payload)`. `PayloadFusion` (softmax + embed for categorical, linear for continuous, fuse MLP), `_CNNGenerator` (transposed-conv), `_ViTGenerator` (patch-decoding), `_AudioGenerator` (strided ConvTranspose1d). `use_payload=False` falls back to zeroed payload (ablation). |
| `src/qualia/model/phenomenal_state.py` | 216 | `PhenomenalState(nn.Module)`: workspace + self-model GRU cells, payload projections, gate network, `_workspace_live` / `_self_model_live` / `_payload_live` buffers, `encode(percepts)`, `introspect()`, `modulate_action(action)`, `reset_state()`. `PhenomenalStateRecord` is JSON-serializable snapshot. `PAYLOAD_KEYS = ("arousal","valence","content","agency","confidence")`. |
| `src/qualia/train/__init__.py` | 1 | Placeholder. |
| `src/qualia/train/train.py` | 27 | `@hydra.main` entrypoint that prints `cfg.workspace_dim`, `cfg.payload_slots`, `cfg.train.steps` and exits. |
| `tests/__init__.py` | 1 | Package marker. |
| `tests/test_scaffold.py` | 23 | Smoke test: `__version__` is `"0.1.0"` and Hydra config composes from `configs/config.yaml` with expected values. |
| `tests/test_phenomenal_state.py` | 124 | 7 tests: state shapes, deterministic `encode`, gradients flow into `PhenomenalState.parameters`, `modulate_action` changes with state, wrong-dim action raises ValueError, gradient flows through `modulate_action`, `reset_state` clears buffers. |
| `tests/test_encoder_decoder.py` | 270 | 7 tests (parametrized × 2 backbones): structured encoder output shapes, decoder reconstruction shape, sensory-only fallback, audio pipeline runs, payload size matches vocab sum, reconstruction loss is finite and gradients flow, payload carries mutual information with input (linear-probe MSE + InfoNCE MI lower bound). |

---

## B) The full training pipeline

### B.1 — Data loading

**Status: NOT IMPLEMENTED.**

- **Dataset class:** none. `src/qualia/data/__init__.py` is a single-line placeholder.
- **Raw data sources:** none on disk. The configs reference `data.name: stochastic_colored_shapes` (`configs/baseline.yaml:13`, `configs/config.yaml:20`) but no synthetic generator exists.
- **DataLoader config:** `num_workers: 0` is declared in config (`configs/baseline.yaml:15`, `configs/config.yaml:22`) but is currently dead config — no `DataLoader` is constructed anywhere in `src/`.
- **Caching:** none (no on-disk cache, no in-memory cache, no `arrow`/`parquet`/`feather`/`tfrecord` references anywhere in the repo).
- **Standardization / transforms:** none.
- **Train/val/test splits:** none.

### B.2 — Model definition

Three modules are implemented; three are not (per the research memo §3.2).

| Module | Status | File:line |
|---|---|---|
| **Encoder (E)** | implemented | `src/qualia/model/encoder.py:234` — `QualiaEncoder` class. `EncoderOutput` dataclass at `encoder.py:41`. `TinyConvNeXt` at `encoder.py:99`, `TinyViT` at `encoder.py:129`, `SensoryHead` at `encoder.py:213`, `PayloadHead` at `encoder.py:186`. `PAYLOAD_KEYS = ("shape","color","hue","brightness","agency")` at `encoder.py:32`. |
| **Decoder (D)** | implemented | `src/qualia/model/decoder.py:207` — `QualiaDecoder` class. `PayloadFusion` at `decoder.py:29`. `_CNNGenerator` at `decoder.py:93`. `_ViTGenerator` at `decoder.py:118`. `_AudioGenerator` at `decoder.py:165`. `reconstruct(encoder_out)` at `decoder.py:315`. |
| **PhenomenalState (Φ)** | implemented | `src/qualia/model/phenomenal_state.py:54` — `PhenomenalState` class. `encode(percepts)` at `phenomenal_state.py:141`, `introspect()` at `phenomenal_state.py:185`, `modulate_action(action)` at `phenomenal_state.py:195`, `reset_state()` at `phenomenal_state.py:127`. `PAYLOAD_KEYS = ("arousal","valence","content","agency","confidence")` at `phenomenal_state.py:30`. |
| **PredictiveLoop (P)** | NOT implemented | referenced in research memo §3.2 and `docs/research-memo.md:100`; no file. A downstream bead (open in convoy) will create `src/qualia/model/predictive_loop.py`. |
| **GlobalWorkspace (W)** | NOT implemented | referenced in research memo §3.2 and `docs/research-memo.md:102`; no file. |
| **SelfModel (S)** | NOT implemented | referenced in research memo §3.2 and `docs/research-memo.md:103`; no file. A downstream bead (open in convoy) will create `src/qualia/model/self_model.py`. |
| **Actor (decision head)** | NOT implemented | no file. |

The encoder and decoder are wired to share `payload_keys` / `payload_vocabs` via the `encoder=...` constructor arg to `QualiaDecoder` (`decoder.py:233-238`). `QualiaDecoder.reconstruct(encoder_out)` exists as a convenience entry point (`decoder.py:315`).

### B.3 — Training loop

**Status: NOT IMPLEMENTED.** `src/qualia/train/train.py` is 27 lines. The `@hydra.main` `main(cfg)` body at `train.py:18-23` is:

```python
print("[qualia.train.train] scaffold placeholder")
print(f"  workspace_dim={cfg.workspace_dim}")
print(f"  payload_slots={cfg.payload_slots}")
print(f"  train.steps={cfg.train.steps}")
print("  (training loop is implemented in a downstream bead)")
```

That is the entire "training loop".

There is therefore **no**:
- optimizer construction (no `torch.optim.*` reference anywhere in `src/`)
- loss function (no `nn.functional.*` loss reference anywhere in `src/`)
- LR schedule / warmup / decay (no `lr_scheduler` reference anywhere in `src/`; only `lr: 3.0e-4` is configured in YAML)
- gradient accumulation
- mixed-precision / AMP / autocast
- gradient clipping
- backward / step order
- gradient zeroing
- data fetching loop
- batch iteration
- progress bar
- eval-during-training
- early stopping
- epoch boundary handling
- seed control at run start (config declares `seed: 0` at `configs/config.yaml:8` but no code reads it)
- device placement (no `cuda`/`device`/`to(...)` reference anywhere in `src/` — only `model.parameters()` in tests)

### B.4 — Eval

**Status: NOT IMPLEMENTED.** `src/qualia/eval/run_eval.py` is 26 lines and mirrors the train scaffold:

```python
print("[qualia.eval.run_eval] scaffold placeholder")
print(f"  eval.report_consistency_threshold={cfg.eval.report_consistency_threshold}")
print(f"  eval.introspection_accuracy_threshold={cfg.eval.introspection_accuracy_threshold}")
print("  (eval suite is implemented in a downstream bead)")
```

`src/qualia/metrics/__init__.py` is a 1-line placeholder. `src/qualia/eval/__init__.py` is a 1-line placeholder. The `eval:` config section declares two thresholds (`report_consistency_threshold=0.5`, `introspection_accuracy_threshold=0.5`) but they are not consumed by any code. The research memo §4 enumerates seven metrics (IA, RC, Distinctness, DG, Manipulability, Robustness, Φ-proxy) — none are implemented.

### B.5 — Checkpointing

**Status: NOT IMPLEMENTED.** No `torch.save`, no `torch.load`, no `state_dict` reference anywhere in `src/`. `.gitignore` at line 17 already ignores `results/*/checkpoints/` and at lines 19-20 ignores `*.pt` and `*.pth`, indicating the intended output layout (`results/<run>/checkpoints/`), but no code writes to it. There is no:
- save format decision (state_dict vs full pickle)
- save frequency / `log_every` × step? (config has `log_every: 10` at `configs/baseline.yaml:19` but nothing reads it)
- keep-last-N policy
- optimizer state save
- RNG state save
- resume-from-checkpoint logic

---

## C) Entry points

| Entry point | Defined at | What it does | When to use |
|---|---|---|---|
| `qualia-train` (console script) | `pyproject.toml:31`, registered from `src/qualia/train/train.py:main` (`train.py:18`); also invokable as `python -m qualia.train.train` (`Makefile:25`). | Loads Hydra config from `configs/` (default `config.yaml`, overridable via `--config-name=`), prints `cfg.workspace_dim`, `cfg.payload_slots`, `cfg.train.steps`, and exits. **No model is constructed, no data is loaded, no step is run.** | Use today only to verify Hydra config wiring. Real training lives in a downstream bead. |
| `qualia-eval` (console script) | `pyproject.toml:32`, registered from `src/qualia/eval/run_eval.py:main` (`run_eval.py:18`); also invokable as `python -m qualia.eval.run_eval` (`Makefile:28`). | Loads Hydra config, prints the two eval thresholds, and exits. **No model is loaded, no metric is computed.** | Use today only to verify Hydra config wiring. |
| `make setup` | `Makefile:15-19` | Creates `.venv`, installs `-e ".[dev]"`, installs pre-commit hooks. | First-time setup, or after pulling changes that bump deps. |
| `make test` | `Makefile:21-22` | Runs `pytest` in `.venv`. Current tests: 3 files, ~16 test functions, all unit-level. | Before commit, in CI on every push/PR. |
| `make train-toy` | `Makefile:24-25` | Invokes `qualia.train.train --config-name=baseline`. Currently prints 4 lines and exits 0. | **Not yet functional** — placeholder until the training-loop bead lands. |
| `make eval-toy` | `Makefile:27-28` | Invokes `qualia.eval.run_eval --config-name=baseline`. Currently prints 3 lines and exits 0. | **Not yet functional** — placeholder until the eval-suite bead lands. |
| `make lint` | `Makefile:30-32` | `ruff check src tests` + `black --check src tests`. | Before commit (also enforced by CI and pre-commit). |
| `make clean` | `Makefile:34-36` | Removes `.pytest_cache`, `.ruff_cache`, `.mypy_cache`, `.coverage`, `htmlcov`, `build`, `dist`, and `__pycache__/` dirs. | Whenever caches go stale. |
| `make help` | `Makefile:6-13` | Echoes target descriptions. | Discoverability. |

**Notebooks:** **NONE.** No `.ipynb` files anywhere in the repo. No `notebooks/` or `scripts/` directories exist.

---

## D) Test coverage state

### D.1 — Test files

| File | LOC | What it covers |
|---|---|---|
| `tests/__init__.py` | 1 | Package marker only. |
| `tests/test_scaffold.py` | 23 | 2 tests: `test_version` (line 11), `test_hydra_config_loads` (line 15). Verifies `__version__ == "0.1.0"` and that `configs/config.yaml` composes with `workspace_dim=32`, `payload_slots=16`, `train.steps=100`, `eval.report_consistency_threshold=0.5`. |
| `tests/test_phenomenal_state.py` | 124 | 7 tests: state shapes / well-typed (line 25); deterministic encode with same seed (line 40); gradients flow into `PhenomenalState.parameters` (line 56); modulate_action uses current state (line 73); wrong-dim action raises ValueError (line 90); gradient flows through modulate_action (line 100); reset_state clears buffers (line 115). |
| `tests/test_encoder_decoder.py` | 270 | 7 tests: encoder produces structured outputs parametrized by `cnn`/`vit` (line 46); decoder reconstruction shape parametrized (line 68); sensory-only fallback (line 78); audio pipeline runs (line 87); payload size matches vocab sum (line 115); reconstruction loss is finite and gradients flow parametrized (line 129); payload carries MI with input (line 187). |

Total: **3 test files, 16 test functions, ~415 LOC of test code**, no coverage threshold configured (pytest-cov is a dev dependency at `pyproject.toml:24` but never invoked in CI or Makefile).

### D.2 — What is NOT covered by tests

- `src/qualia/data/__init__.py` — entire file is a placeholder.
- `src/qualia/eval/run_eval.py` — entire file is a placeholder.
- `src/qualia/metrics/__init__.py` — entire file is a placeholder.
- `src/qualia/train/train.py` — entire file is a placeholder.
- `src/qualia/model/encoder.py` — covered superficially (forward shape, sensory-only fallback, audio path, payload size, MI sanity), but **not** covered: error handling for invalid `modality` (line 263), invalid `backbone` (line 279), wrong-dim image input (line 301), wrong-dim audio input (line 297); `_ConvNeXtBlock` is exercised indirectly only.
- `src/qualia/model/decoder.py` — covered via `reconstruct`, sensory-only fallback, and audio length override, but **not** covered: invalid `modality` (line 274), invalid `backbone` (line 268), `PayloadFusion.payload_features` directly (line 74), `_AudioGenerator` length interpolation logic (line 198-204), `use_payload=False` zero-init path (line 299-309).
- `src/qualia/model/phenomenal_state.py` — covered well for forward / state / gradient behavior, but **not** covered: `_init_live_state` device handling (line 108-114), the `_step` counter increment (line 174, indirectly tested), the gate network with non-positive `workspace_dim` / `self_model_dim` / `payload_slots` (line 74-75), empty `payload_keys` (line 76-77).

**No tests exist for:**
- any dataset / DataLoader (none exists)
- any training step (none exists)
- any optimizer / scheduler (none exists)
- any checkpointing round-trip (none exists)
- any eval metric (none exists)
- the end-to-end `make train-toy && make eval-toy` flow (both are placeholders)
- seed reproducibility for the train entrypoint

### D.3 — Pytest / lint / typecheck configuration

| Tool | Config | Value |
|---|---|---|
| **pytest** | `pyproject.toml:37-40` | `testpaths = ["tests"]`, `addopts = "-ra --strict-markers"`, `pythonpath = ["src"]`. **No markers configured.** No smoke marker. No slow marker. The `--strict-markers` flag will fail any test using `@pytest.mark.something` unless that marker is registered in `pyproject.toml` — which means the **smoke marker required by a downstream bead cannot be added without also registering it in `tool.pytest.ini_options.markers`**. |
| **pytest-cov** | dev dep only | `pyproject.toml:24`. Never invoked in `Makefile` or `.github/workflows/ci.yml`. No coverage threshold set. |
| **ruff** | `pyproject.toml:46-52` | `line-length=100`, `target-version=py310`, selects `["E","F","I","B","UP","W"]`, ignores `E501`. Run via `ruff check src tests` in CI and Makefile. |
| **black** | `pyproject.toml:42-44` | `line-length=100`, `target-version=["py310"]`. Run via `black --check src tests` in CI and Makefile. |
| **pre-commit** | `.pre-commit-config.yaml` | ruff (with --fix), ruff-format, black (py3). |
| **mypy / pyright** | — | **NOT configured.** No `[tool.mypy]` or `[tool.pyright]` section in `pyproject.toml`. No `mypy` dependency. |
| **CI** | `.github/workflows/ci.yml` | Triggers: push to `main` / `convoy/**`, PR to same. Matrix: Python 3.10, 3.11 on `ubuntu-latest`. Steps: checkout, setup-python, `pip install -e ".[dev]"`, ruff check, black --check, pytest. **No coverage upload, no mypy, no smoke-vs-full split, no GPU runner.** |

### D.4 — Smoke tests / markers

- A `smoke` marker is **referenced** in the open convoy (downstream bead asks for `pytest -m "not smoke"`) but is **not registered** in `pyproject.toml`. Today no `@pytest.mark.smoke` decorator exists in any test.
- `tests/test_scaffold.py` is a de-facto smoke test (verifies the project is wired up at all) but is not marked as such.
- `tests/test_encoder_decoder.py::test_payload_carries_mutual_information_with_input` trains a 40-step linear probe twice (lines 224, 238) — this is the heaviest test in the suite but is fast (<1s on CPU) and is not marked.

---

## E) Pre-existing experiment tracking / caching / parallel eval tooling

| Capability | Present? | Evidence |
|---|---|---|
| **Experiment tracking (W&B / MLflow / TensorBoard / SwanLab / ClearML)** | **NO** | No references to `wandb`, `tensorboard`, `mlflow`, `tb_log`, `SummaryWriter`, `tracker`, `swanlab`, or `clearml` anywhere in the repo. Configs do not declare a tracking backend. |
| **Run metadata (git SHA, hostname, CLI argv, env snapshot)** | **NO** | No `subprocess.check_output(["git", ...])`, no `socket.gethostname()`, no `sys.argv` capture anywhere in `src/`. |
| **Structured per-step / per-epoch logging** | **NO** | `train.log_every: 10` is configured in `configs/baseline.yaml:19` but nothing reads it. |
| **Data caching (arrow / parquet / tfrecord / feather / pickle)** | **NO** | No `cache`, `arrow`, `parquet`, `tfrecord`, `feather`, `pickle`, or `joblib` references in `src/` or tests. `data/` is an empty placeholder. |
| **Dataset versioning / preprocessing-config hash** | **NO** | — |
| **Multi-GPU / distributed (DDP / FSDP)** | **NO** | No `torch.distributed`, no `DistributedDataParallel`, no `torchrun` references. |
| **Parallel dataloader** | **NO (and explicitly disabled)** | `data.num_workers: 0` at `configs/baseline.yaml:15`, `configs/config.yaml:22`. No `DataLoader` construction anywhere in `src/`. |
| **`pin_memory`, `persistent_workers`, `prefetch_factor`** | **NO** | — |
| **Mixed precision (`torch.cuda.amp` / `torch.autocast` / bf16)** | **NO** | No `amp`, `autocast`, `fp16`, `bf16`, `GradScaler` references in `src/` or tests. |
| **`torch.compile`** | **NO** | — |
| **Gradient accumulation** | **NO** | — |
| **Gradient clipping** | **NO** | — |
| **LR schedule / warmup / cosine** | **NO** | `train.lr: 3.0e-4` in config but no scheduler. |
| **EMA / SWA** | **NO** | — |
| **Checkpointing (save / load / resume)** | **NO** | `.gitignore` declares `results/*/checkpoints/` and `*.pt`/`*.pth` but no code writes them. |
| **Config snapshotting** | **NO** | Hydra would handle this automatically **if a real run wrote to `results/<run>/`** — currently nothing writes there. |
| **Eval-during-training** | **NO** | — |
| **Parallel eval** | **NO** | — |
| **Progress bar (tqdm / rich)** | **NO** | No `tqdm`, `rich`, `ProgressBar` references anywhere. |
| **Profiler (`torch.profiler` / `torch.cuda.profile`)** | **NO** | — |
| **Seed control at run start** | **NO** | `seed: 0` is configured at `configs/config.yaml:8` but no code reads it. Tests do call `torch.manual_seed(...)` directly (e.g. `tests/test_phenomenal_state.py:16`, `tests/test_encoder_decoder.py:22,37,88,199,233`). |
| **Determinism flags (`torch.use_deterministic_algorithms`)** | **NO** | — |
| **Device placement (CPU / CUDA / MPS)** | **NO** | No `torch.device`, `cuda`, `mps`, or `.to(...)` references in `src/`. |
| **CLI overrides (Hydra sweeper, `--multirun`)** | partial | `@hydra.main` is wired in `train.py:17` and `run_eval.py:17`; CLI overrides would work the moment a real `main` body exists. |
| **Doc generation (Sphinx / mkdocs)** | **NO** | Only the handwritten `docs/research-memo.md` exists. |
| **VSCode / devcontainer config** | **NO** | `.idea/` and `.vscode/` are ignored at `.gitignore:15-16` but never present. |

---

## F) Top 5 concrete bottlenecks ranked by expected impact on iteration speed

The current state is unusual: **there is no running pipeline to optimize**. The "bottlenecks" below are therefore the highest-leverage gaps to close, ranked by how much daily iteration speed they unlock once filled. Each finding includes file:line evidence.

---

### #1 — No training loop exists at all

**Impact: catastrophic / blocking.** Nothing else matters until this exists. The research program cannot iterate at any speed without it.

**Evidence:**
- `src/qualia/train/train.py:18-23` — `main(cfg)` is five `print()` statements.
- `src/qualia/train/__init__.py:1` — explicitly states *"to be implemented"*.
- Configs declare `train.steps=100`, `train.log_every=10`, `train.lr=3.0e-4` (`configs/baseline.yaml:17-20`) but no code consumes them.
- `Makefile:10-11` — `train-toy` is documented as *"placeholder until training is wired"*.

**What a developer hits daily:** every developer in the convoy cannot run a single gradient step on real data today. The 6 downstream beads (predictive loop, self-model, eval, train, cache, tracking) all depend on this one being filled first.

**Concrete fix shape:** `src/qualia/train/train.py` needs to (a) construct the encoder/decoder/PhenomenalState from `cfg` and (probably) the not-yet-existing PredictiveLoop + SelfModel; (b) load a Dataset/DataLoader (which also doesn't exist); (c) construct an optimizer (AdamW with `cfg.train.lr=3.0e-4`); (d) define the composite loss from research-memo §3.4 (reconstruction + prediction + KL + report-consistency + downstream); (e) loop over steps; (f) save a checkpoint. Estimated ~200 LOC.

---

### #2 — No Dataset, no DataLoader, no synthetic generator

**Impact: catastrophic / blocking.** Closely coupled with #1, but listed separately because it is a *data* bottleneck that has its own design choices (cache format, augmentation, splits).

**Evidence:**
- `src/qualia/data/__init__.py:1` — 1-line placeholder.
- `configs/baseline.yaml:12-15` and `configs/config.yaml:19-22` reference `data.name: stochastic_colored_shapes` and `batch_size: 16, num_workers: 0` — but no `__init__.py` even declares an enum / registry of available datasets.
- No `torch.utils.data.Dataset`, `IterableDataset`, `DataLoader`, `Sampler`, or `collate_fn` anywhere in `src/`.
- The encoder hard-codes an expectation of a specific synthetic dataset: payload slots are `("shape","color","hue","brightness","agency")` (`encoder.py:32-38`) — i.e. shapes with categorical shape and color + continuous hue/brightness/agency. The dataset class must emit these labels for any supervised loss on the payload to work.

**What a developer hits daily:** even if the training loop existed, there is no data to train on. Every downstream bead that needs a "real run" is blocked.

**Concrete fix shape:** implement `src/qualia/data/synthetic.py::StochasticColoredShapes` (or similar — the convoy open bead names this exactly) emitting `(image, label_dict)` pairs; implement a standard `torch.utils.data.Dataset` wrapper and a `make_dataloader(cfg)` factory in `src/qualia/data/__init__.py` or a sibling module. Estimated ~150 LOC.

---

### #3 — No experiment tracking, no checkpointing, no resume

**Impact: high.** The first run a developer kicks off today will produce zero machine-readable artifacts: no metrics log, no hyperparameters recorded with the run, no checkpoint, no way to compare two runs, no way to resume after a crash. The research memo §3.4 names **five** distinct loss terms — without a tracker there is no way to tell which term is misbehaving. The research memo §4 names **seven** evaluation criteria — without a tracker there is no way to compare across seeds.

**Evidence:**
- No `wandb`, `tensorboard`, `mlflow`, `SummaryWriter`, or equivalent import anywhere in `src/`.
- `.gitignore:17-20` already declares `results/*/checkpoints/`, `results/*/logs/`, `*.pt`, `*.pth` — the *layout* is committed but no code writes to it.
- No `torch.save`, `torch.load`, or `state_dict` reference anywhere in `src/`.
- `configs/baseline.yaml:1` declares `run_dir: results/baseline` but the training entrypoint does not create or write to it.
- `seed: 0` is in `configs/config.yaml:8` but no code reads it — first reproducibility bug guaranteed on day 1.

**What a developer hits daily:** "I just ran a new config for 4 hours, did it actually beat the baseline? which git SHA was that?" — unanswerable today.

**Concrete fix shape:**
1. Pick a tracker (open convoy bead allows W&B / MLflow / TensorBoard; recommendation: **TensorBoard** as the default since it has zero auth/network deps for offline use, with a `--tracker=wandb` opt-in). Wire `src/qualia/train/train.py` to (a) create `cfg.run_dir`, (b) snapshot `cfg` as YAML into it, (c) emit `git rev-parse HEAD`, (d) log per-step losses and per-eval metrics.
2. Add `torch.save({"model": state_dict, "optim": optim.state_dict, "step": step, "cfg": cfg}, path)` every `cfg.train.log_every` steps and at end of training.
3. Add `torch.load(...)` resume logic gated on a `--resume=<path>` CLI flag.
4. Wire `seed: 0` into a top-level `torch.manual_seed(cfg.seed)` call.

Estimated ~100 LOC.

---

### #4 — No data caching layer (every reload re-generates or re-decodes from scratch)

**Impact: high (after #1 and #2 land).** The first training-loop run will likely involve a synthetic dataset; reload time will be fast (microseconds). But the research memo §4.6 explicitly lists a Robustness eval under distribution shift, and §4.7 lists the Φ-proxy eval which requires bipartition-based statistics — both of which will iterate over the full eval set multiple times. The downstream bead on caching (open in convoy) explicitly anticipates that this is needed; the audit confirms there is **zero** scaffolding for it today.

**Evidence:**
- No `cache`, `arrow`, `parquet`, `feather`, `tfrecord`, `pickle`, `joblib`, or `pyarrow` reference anywhere in `src/` or tests.
- No `__getitem__` is overridden anywhere (because no Dataset exists).
- No `mmap_mode` / `prefetch_to_disk` / `LRU` / `functools.lru_cache` on data paths.

**What a developer hits daily:** if they bump `image_size` from 32 to 64, every run regenerates; if they iterate on a perturbation eval, every pass re-decodes the source images.

**Concrete fix shape:**
- Wrap the eventual `Dataset` with a cache that writes processed `(image, label)` tuples to `results/<run>/cache/<split>/<hash>.pt` (or arrow/parquet — arrow is recommended for heterogeneous label dicts).
- Hash the cache key from `(raw_data_path, preprocessing_cfg_hash)`, so a config change auto-invalidates.
- Log cache hit/miss + elapsed time per `__getitem__`.
- Add `--rebuild-cache` CLI flag.

Estimated ~80 LOC wrapping the eventual Dataset.

---

### #5 — No smoke test for the training pipeline; CI runs only the two module-level test files

**Impact: high.** Today `make test` runs in well under 5 seconds (no real data, no real training). The first time the training-loop bead lands, every CI run will become either (a) instant if no smoke test exists or (b) multi-minute if the full training loop is exercised. Without a smoke marker, every PR will be slow, every push will be slow, and polecats will start disabling CI checks.

**Evidence:**
- No `@pytest.mark.smoke` decorator anywhere.
- `pyproject.toml:39` has `addopts = "-ra --strict-markers"` — registering a `smoke` marker requires editing `pyproject.toml` to add `markers = ["smoke: <description>"]` under `[tool.pytest.ini_options]`.
- `.github/workflows/ci.yml:30` runs `pytest` unconditionally with no `-m` filter and no timeout.
- `Makefile:21-22` runs `pytest` unconditionally with no `-m` filter.
- No `pytest-timeout` in dev deps; no `pytest-xdist` (parallelism); no `pytest-mock` (mocks); no `pytest-benchmark`.
- No `conftest.py` (so no shared fixtures for `device`, `tiny_cfg`, `tiny_model`).

**What a developer hits daily:** the moment any change accidentally breaks the training loop (e.g. wrong device, shape mismatch in a new module), CI will hang or run the full pipeline rather than failing fast in 60s.

**Concrete fix shape:**
1. Register `smoke` (and probably `slow`) markers in `pyproject.toml`.
2. Write `tests/test_smoke.py` that (a) constructs the smallest encoder/decoder/PhenomenalState, (b) instantiates the dataset loader from a 4-sample synthetic batch, (c) runs 1-2 training steps, (d) runs eval, (e) saves and reloads a checkpoint, (f) asserts all metrics are finite. Target: <60s on CPU.
3. Update CI to run `pytest -m smoke` on every push and `pytest` (full) only on merges to `main` / nightly.
4. Add a `pytest --timeout=120` guard (needs `pytest-timeout` dev dep).

Estimated ~120 LOC including a small fixture conftest.

---

### Honorable mentions (not in top 5 but worth a future bead)

- **`data.num_workers=0` in config (`configs/baseline.yaml:15`, `configs/config.yaml:22`)** — defaults to zero. The moment real audio/image data lands this will become a CPU bottleneck. Should default to `os.cpu_count()` or `2` and expose `persistent_workers`/`pin_memory`.
- **No `device` abstraction** — `train.py` will need to choose CPU/CUDA/MPS; the encoder/decoder have no `.to(device)` path. Worth picking `cfg.device: auto` once.
- **No LR schedule** — `cfg.train.lr` is a scalar; research memo §3.4 and §4 imply the trainer needs at minimum a warmup. Cosine or OneCycle is the conventional choice for transformer-style training.
- **No `grad_clip_norm` config** — standard practice for transformer + RNN (PhenomenalState has GRUCells at `phenomenal_state.py:89-90`).
- **No progress bar** — once the loop exists, `tqdm` is a 2-line drop-in for fast feedback.
- **Encoder `backbone: convnext_tiny` in `configs/baseline.yaml:5,8`** — `convnext_tiny` is a torchvision name but **no `torchvision` dependency is declared in `pyproject.toml`** (lines 13-19). First time a developer tries to instantiate the model from the literal baseline config it will `NameError` on the encoder factory. Either (a) add `torchvision` to deps and wire an actual `convnext_tiny` factory in `encoder.py`, or (b) rewrite the config to use the already-implemented `cnn` / `vit` backbones.
- **Research memo §4.5 calls for an `intervene(slot, value)` method on `PhenomenalState`** — not present (compare `phenomenal_state.py:54-213` to the method list in `docs/research-memo.md:144-147`).
- **`PhenomenalState.encode` reduces per-batch to a single state vector via `mean(dim=0)` (`phenomenal_state.py:167-172`)** — once training runs in real DataLoader mode with batch>1, the per-sample batch dim is collapsed silently. This may be intended, but there is no docstring explaining why the per-batch `h_w, h_s` are discarded after a single state update.

---

## Appendix A — Open convoy beads and how this audit maps to them

These beads are open in the same convoy (per gastown context); this audit is sufficient to plan each:

| Bead | What the audit confirms about its starting state |
|---|---|
| **Add experiment tracking infrastructure** (`2d4476be…`) | See bottleneck #3. No tracker exists; `results/baseline/` is the declared `run_dir`; `seed: 0` is configured but unread. |
| **Add data caching for fast dataset reloads** (`fae9a030…`) | See bottleneck #4. No Dataset exists yet, so the cache layer wraps a class to-be-built. |
| **Add automated smoke test for training pipeline** (`176807e1…`) | See bottleneck #5. `smoke` marker is not registered in `pyproject.toml`; no `conftest.py`; CI runs full `pytest` with no `-m` filter. |
| **Implement predictive-coding loop** (`bd78deec…`) | `src/qualia/model/predictive_loop.py` does not exist; encoder `sensory` (`encoder.py:305`) and `PhenomenalState.encode` (`phenomenal_state.py:141`) are the inputs it must consume. |
| **Implement self-model / attention-schema head** (`7c09135a…`) | `src/qualia/model/self_model.py` does not exist; payload slot names `(arousal, valence, content, agency, confidence)` are pre-declared at `phenomenal_state.py:30`. |
| **Build introspection & reportability evaluation suite** (`cc284249…`) | `src/qualia/eval/` and `src/qualia/metrics/` are placeholders; the seven metrics from research memo §4 are unimplemented. |
| **Wire up end-to-end training script and run a baseline experiment** (`504e1aa1…`) | See bottleneck #1 + #2. |

---

## Appendix B — One-shot facts the next beads will need

- **Python:** `>=3.10` (`pyproject.toml:10`); CI tests 3.10 and 3.11 (`ci.yml:14`).
- **Torch:** pinned `2.2.2` (`pyproject.toml:14`).
- **Hydra / OmegaConf:** `hydra-core==1.3.2`, `omegaconf==2.3.0` (`pyproject.toml:17-18`). `@hydra.main(version_base=None, config_path=<abs path to configs>, config_name="config")` (`train.py:17`, `run_eval.py:17`).
- **CLI override pattern:** `qualia-train model.workspace_dim=64 train.steps=1000` (README:46).
- **Config layout:** `configs/config.yaml` is the root, `defaults: [baseline]`, both files declare overlapping keys (`workspace_dim`, `payload_slots`, `model.encoder.backbone`, etc.) — intentional so CLI overrides at the root level take effect.
- **Public API surface today:** `QualiaEncoder`, `QualiaDecoder`, `PhenomenalState`, `PhenomenalStateRecord`, `PayloadFusion`, `PayloadHead`, `SensoryHead`, `TinyConvNeXt`, `TinyViT`, `EncoderOutput`, `PAYLOAD_KEYS` (encoder), `PAYLOAD_KEYS` (PhenomenalState — note: this is a **second** `PAYLOAD_KEYS` tuple inside `phenomenal_state.py:30`, distinct from the encoder's at `encoder.py:32`; both are 5-tuples but the slot *names* differ).
- **No `torch.cuda` / `torch.backends` / `torch.compile` / `torch.jit` / `torch.distributed` / `torch.profiler`** anywhere in `src/` or tests.
- **No `conftest.py`** anywhere.
- **No `requirements.txt` / `requirements-dev.txt` / `uv.lock` / `poetry.lock` / `pdm.lock`** — `pyproject.toml` is the sole source of dependencies.
- **`Makefile` targets:** `help`, `setup`, `test`, `train-toy`, `eval-toy`, `lint`, `clean` (all `.PHONY`).
- **No `CONTRIBUTING.md`, no `CODE_OF_CONDUCT.md`, no `CHANGELOG.md`, no `SECURITY.md`.**
- **`results/` does not exist** as a directory yet, despite being mentioned in README:15 and `.gitignore:17-18`.