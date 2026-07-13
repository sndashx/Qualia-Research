# Automation — Productivity Boost Follow-up Convey

**Author:** Birch (polecat)
**Bead:** `329c288d-3548-4768-acb5-55fa444ba2a9` ("Automation: pre-stage follow-up improvement beads based on AUDIT.md")
**Source audit:** `AUDIT.md` (Toast, 2026-07-13) — top-5 iteration-speed bottlenecks at `AUDIT.md:277-380`.
**Initial convoy:** `fc2c7518-c81a-4d36-a9b9-3d89ecbf54f6` ("Productivity Boost — Audit + Tooling") covered bottlenecks #3, #4, #5.
**Follow-up convoy:** "Productivity Boost — Follow-up 1" — staged, `merge_mode = "review-then-land"`.

---

## 1. Bottlenecks remaining after the initial convoy

The initial convoy (3 issue beads + 1 audit) addressed:
- **#3** — Experiment tracking (`2d4476be…`)
- **#4** — Data caching (`fae9a030…`)
- **#5** — Smoke test / CI marker wiring (`176807e1…`)

Top-5 bottlenecks not addressed:
- **#1** — *No training loop exists at all.* `src/qualia/train/train.py:18-23` is five `print()` statements (`AUDIT.md:283-295`). Catastrophic / blocking.
- **#2** — *No Dataset, no DataLoader, no synthetic generator.* `src/qualia/data/__init__.py:1` is a placeholder; `configs/baseline.yaml:12-15` references `data.name: stochastic_colored_shapes` but nothing emits those samples (`AUDIT.md:297-311`). Catastrophic / blocking, paired with #1.

The follow-up convoy creates two beads — one per remaining bottleneck — so that closing them unblocks every downstream bead (predictive loop, self-model, eval suite, end-to-end baseline) in convoy `bafece7c`.

---

## 2. Why each new bead addresses one bottleneck

### Bead A — `synthetic-dataset-and-dataloader`
Targets **bottleneck #2** (`AUDIT.md:297-311`). Evidence:
- `src/qualia/data/__init__.py:1` placeholder.
- `configs/baseline.yaml:12-15` and `configs/config.yaml:19-22` reference `data.name: stochastic_colored_shapes`, `batch_size=16`, `num_workers=0` with no implementation.
- Encoder hard-codes payload slots `("shape","color","hue","brightness","agency")` at `src/qualia/model/encoder.py:32-38` — the dataset must emit exactly these labels for any supervised payload loss to function.

The bead fills the data bottleneck in isolation, before the trainer needs it, so the trainer bead (Bead B) can consume a real `DataLoader` instead of re-implementing it.

### Bead B — `wire-end-to-end-training-loop`
Targets **bottleneck #1** (`AUDIT.md:283-295`). Evidence:
- `src/qualia/train/train.py:18-23` is a 5-line `print()` scaffold.
- `src/qualia/train/__init__.py:1` says "to be implemented".
- `configs/baseline.yaml:17-20` declares `train.steps=100`, `train.log_every=10`, `train.lr=3.0e-4` with no consumer.
- `Makefile:10-11` documents `train-toy` as a placeholder.

The bead depends on Bead A (needs a real DataLoader) and on the three open convoy-beads that supply the not-yet-implemented modules (`bd78deec` → predictive_loop, `7c09135a` → self_model). It is the integration bead that turns the scaffold into a runnable baseline.

---

## 3. Dependency DAG

```
        (initial convoy beads: tracking, caching, smoke test)
                           │
                           ▼
                  Bead A — synthetic-dataset-and-dataloader   (no internal deps)
                           │
                           ▼
        Bead B — wire-end-to-end-training-loop
              ├── depends on Bead A
              ├── depends on bd78deec (predictive_loop, in convoy bafece7c)
              └── depends on 7c09135a (self_model, in convoy bafece7c)
```

Strict ordering: **A → B**. Within the follow-up convoy, no two beads can run in parallel safely — the trainer cannot be wired before data exists.

---

## 4. Estimated impact

| Bead | Bottleneck | Estimated LOC | Iteration-speed unlock |
|---|---|---|---|
| `synthetic-dataset-and-dataloader` | #2 | ~150 LOC (`AUDIT.md:311`) | Every downstream bead that needs a "real run" can finally start. Caching bead (`fae9a030…`) now has a Dataset to wrap. |
| `wire-end-to-end-training-loop` | #1 | ~200 LOC (`AUDIT.md:295`) | Researchers can run gradient steps for the first time. Closes the iteration loop that the tracking bead needs to log into. |

Together: converts the repo from "scaffold with three modules" to "runnable baseline on synthetic data with checkpointing + tracking + cache + smoke test", which is the prerequisite for every convoy-bafece7c deliverable.

---

## 5. Convey spec (for `gt_sling_batch --staged`)

Invocation:

```
gt_sling_batch \
  --rig <rig-id> \
  --staged=true \
  --merge_mode=review-then-land \
  --title "Productivity Boost — Follow-up 1" \
  --feature_branch "convoy/productivity-boost-followup-1/<convoy-id>/head" \
  --beads <bead-a-id> <bead-b-id>
```

### Bead A body — `synthetic-dataset-and-dataloader`

```
Title: Implement synthetic dataset, Dataset wrapper, and DataLoader factory

Source: AUDIT.md bottleneck #2 (AUDIT.md:297-311).

Targets the data bottleneck identified in the productivity-boost audit. The
repo currently has no Dataset, no DataLoader, and no synthetic generator
(src/qualia/data/__init__.py:1). configs/baseline.yaml:12-15 and
configs/config.yaml:19-22 reference data.name=stochastic_colored_shapes,
batch_size=16, num_workers=0 but nothing implements them. The encoder
hard-codes payload slots ("shape","color","hue","brightness","agency") at
src/qualia/model/encoder.py:32-38, so the dataset must emit exactly these
labels.

Acceptance:
1. src/qualia/data/synthetic.py implements StochasticColoredShapes emitting
   (image, label_dict) pairs where label_dict keys match encoder.py:32-38
   (shape ∈ {triangle, square, circle}, color ∈ {red, green, blue},
   hue ∈ [0, 1], brightness ∈ [0, 1], agency ∈ {0, 1}). Images are 3x32x32
   by default, configurable via cfg.data.image_size.
2. src/qualia/data/__init__.py exposes a make_dataloader(cfg) factory that
   returns a torch.utils.data.DataLoader honoring cfg.data.batch_size and
   cfg.data.num_workers.
3. Tests in tests/test_data_synthetic.py cover: dataset length matches
   cfg.data.num_samples; __getitem__ returns correctly-typed image tensor
   and label dict; collate_fn batches a list of samples into a single
   (image_batch, labels_batch) tuple with consistent shapes; DataLoader
   yields at least one batch of the configured size.
4. make train-toy no longer crashes at the "no data" step (it will still
   not train — that is Bead B's job — but it should not raise
   NotImplementedError on data construction).

Out of scope: caching (handled by the open caching bead fae9a030…).
Out of scope: any real-image or audio dataset. Only the synthetic generator
needed by the baseline config.
```

### Bead B body — `wire-end-to-end-training-loop`

```
Title: Wire end-to-end training loop using Hydra + existing encoder/decoder/PhenomenalState

Source: AUDIT.md bottleneck #1 (AUDIT.md:283-295).

Targets the training-loop bottleneck. src/qualia/train/train.py:18-23 is a
5-line print() scaffold. configs/baseline.yaml:17-20 declares
train.steps=100, train.log_every=10, train.lr=3.0e-4 with no consumer.
Makefile:10-11 documents train-toy as a placeholder. seed:0 is configured
at configs/config.yaml:8 but no code reads it.

Depends on:
- Bead A (synthetic-dataset-and-dataloader) — needs a real DataLoader.
- Open convoy bead bd78deec (predictive_loop) — needed for prediction
  error loss term from research memo §3.4.
- Open convoy bead 7c09135a (self_model) — needed for report-consistency
  loss term from research memo §3.4.
- Initial convoy bead 2d4476be (experiment tracking) — trainer logs into
  the tracker; if tracking is not yet merged, fall back to console logging
  via a --no-log flag and gate tracker import.

Acceptance:
1. src/qualia/train/train.py:main(cfg) constructs (encoder, decoder,
   PhenomenalState, predictive_loop, self_model) from cfg, instantiates
   the DataLoader from Bead A, builds an AdamW optimizer with
   cfg.train.lr, and loops for cfg.train.steps.
2. Composite loss = reconstruction (MSE on decoder output) + prediction
   error term from predictive_loop + report-consistency term from
   self_model + downstream-grounding term stub. Weights live in
   cfg.train.loss_weights.
3. Per-step metrics: total_loss, recon_loss, pred_loss, report_loss,
   lr, grad_norm — emitted to the tracker (or stdout if --no-log).
4. Every cfg.train.log_every steps and at end of training, save a
   checkpoint to cfg.run_dir/checkpoints/step_{N}.pt containing
   {model, optim, step, cfg}.
5. torch.manual_seed(cfg.seed) is called at the top of main(cfg).
6. Device placement honors cfg.device: auto|cpu|cuda|mps.
7. make train-toy runs end-to-end on the synthetic dataset for
   cfg.train.steps=20 (override-able) and exits 0; losses are finite.

Out of scope: eval-during-training (covered by open bead cc284249…).
Out of scope: gradient accumulation / AMP / torch.compile — listed as
honorable mentions in AUDIT.md:382-394, can be follow-up beads later.
```