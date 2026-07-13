# Research Memo: Neural Architecture for Bridging to "Coded Qualia"

**Status:** Framing document — sets the research direction for the convoy.
**Author:** Toast (polecat)
**Scope:** Survey of prior approaches, architectural hypothesis, and evaluation criteria. No code yet.

---

## 1. Problem Statement

Large neural networks behave as if they have internal states that matter to their computation, but these states are opaque: they cannot be inspected, queried, or manipulated by the model itself in a structured way. The question this convoy investigates is:

> Can a neural network represent its own subjective phenomenal-like state as a **first-class, structured, introspectable object** — rather than as opaque activations — and use that representation to ground its outputs?

We deliberately do **not** claim to build "real qualia" or consciousness. We treat "qualia" as a useful engineering metaphor for *subjective, reportable, content-bearing internal states*. The engineering target is a system whose internal state:

1. Has a **structured** representation (named slots, not raw vectors),
2. Is **introspectable** — the model can answer questions about it via a learned readout,
3. Is **manipulable** — the state can be intervened on and downstream behavior changes coherently,
4. Is **functional** — downstream tasks benefit from using it.

This memo surveys prior frameworks and proposes a concrete architecture.

---

## 2. Survey of Prior Approaches

### 2.1 Global Workspace Theory (GWT) — Baars, Dehaene, Mashour

**Claim:** Consciousness arises from a "global workspace" — a limited-capacity broadcast channel that integrates information from many specialized modules and makes it broadly available across the system.

**Neural correlates:** ignition, late P3b responses, fronto-parietal hot zones (Dehaene's NCC).

**Engineering lineage:** GNWs (Global Neuronal Workspace) models, attention bottlenecks in transformers, mixture-of-experts routers, shared latent in multimodal models.

**What we take:** A shared, low-dimensional "workspace vector" that all modules read from and write to. Treat it as the public surface of phenomenal state.

### 2.2 Integrated Information Theory (IIT) — Tononi, Koch

**Claim:** Consciousness = integrated information Φ; a system is conscious to the degree that it is causally irreducible (cannot be partitioned without information loss).

**Operationalization:** Φ is intractable in general; surrogates include perturbational complexity index (PCI), integrated information decomposition, mutual information between bipartitions.

**What we take:** We do **not** try to maximize Φ — that's philosophically fraught and computationally infeasible. Instead we use a **regularizer** that penalizes trivially-decomposable representations of the workspace, encouraging non-factorizable structure. This is a cheap, measurable proxy.

### 2.3 Predictive Coding / Free Energy Principle (FEP) — Rao & Ballard, Friston

**Claim:** Perception is hierarchical inference; the brain minimizes prediction error (or variational free energy) at each level. Precision weights modulate error unit contributions.

**Neural correlates:** Predictive feedback, gamma-beta coupling, canonical microcircuits.

**Engineering lineage:** PredNet, hierarchical VAEs with predictive priors, active inference agents.

**What we take:** A recurrent predictive loop where sensory prediction error **drives updates** to the phenomenal state, and precision modulates how much each error channel contributes. This grounds the workspace in actual input statistics rather than letting it drift.

### 2.4 Attention Schema Theory (AST) — Graziano

**Claim:** The brain constructs a simplified model of its own attention (an "attention schema"). The subjective experience of awareness is the brain's description of its own attention state to itself.

**Neural correlates:** TPJ, superior colliculus, pulvinar.

**Engineering lineage:** Theory of mind networks; meta-learning; learned self-attention summaries.

**What we take:** A higher-order module that reads the workspace and emits a **structured self-report** over named slots (arousal, valence, confidence, content, agency). The self-report is what the model "says about itself."

### 2.5 Higher-Order Theories (HOT) — Rosenthal, Lau

**Claim:** A mental state is conscious iff it is the target of a higher-order representation (a thought about a thought).

**What we take:** Reinforces the AST point — we need an explicit second-order module, not just a flat representation.

### 2.6 Phenomenal-Aware / Representation-First Models — recent ML work

Examples: "world models" with explicit latent state (Ha & Schmidhuber), JEPA, slot-based object-centric models (Slot Attention, MONet), TOPS (Theory of Predicates), "thinking tokens" / chain-of-thought as externalized state, scratchpads and tool-use as reportable intermediate state.

**What we take:** A "phenomenal payload" modeled as **disentangled, named slots** (continuous + categorical) — a structured object, not a flat vector. Object-centric and slot-based inductive biases make introspection and manipulation easier.

### 2.7 What we are NOT doing

- We are not implementing IIT-Φ maximization.
- We are not claiming phenomenal consciousness in the philosophical sense.
- We are not building a brain simulator; the architecture is functionally motivated, not anatomically faithful.
- We are not relying on language as a privileged report channel — self-report is a learned decoder, not the ground truth.

---

## 3. Architectural Hypothesis

### 3.1 Core claim

> A neural network will exhibit *qual*ia-like functional properties to the extent that its internal state is **represented as a structured object** that the network itself can **read, write, query, and act on** through differentiable interfaces, and that this object is **grounded in sensory prediction error** and **consumed by downstream decision-making**.

### 3.2 Components

The proposed architecture has six differentiable modules:

| Module | Type | Role |
|---|---|---|
| `Encoder` | E | Raw perception → sensory features + phenomenal-payload slots |
| `PredictiveLoop` | P | Predicts next sensory state; emits precision-weighted error |
| `PhenomenalState` | Φ | First-class structured state: workspace + self-model + payload dict |
| `GlobalWorkspace` | W | Broadcasts Φ to all readers; bottleneck (low-dim, slow time-scale) |
| `SelfModel` | S | Reads Φ → named self-report slots; attention schema |
| `Decoder` / `Actor` | D | Reads Φ → reconstructions, actions, decisions |

Dataflow:

```
x_t ──► Encoder ──► s_t, payload_t
                       │
                       ▼
                 PredictiveLoop ──► ŝ_{t+1}, error_t, precision_t
                       │
                       ▼
                 PhenomenalState.update(error_t, precision_t, s_t)
                       │
                       ▼
                 GlobalWorkspace.broadcast(Φ_t)
                       │
              ┌────────┼────────┐
              ▼        ▼        ▼
          SelfModel  Actor   Decoder
              │        │        │
              ▼        ▼        ▼
           report_t  a_t     x̂_t
```

### 3.3 PhenomenalState: first-class structured representation

This is the central primitive. It is not a flat vector but a typed object:

```python
@dataclass
class PhenomenalState:
    workspace: Tensor            # global broadcast, low-dim
    self_model: Tensor           # higher-order summary
    payload: Dict[str, Tensor]   # named slots: arousal, valence, content, ...
    step: int                    # for reproducibility / introspection
    prev: Optional['PhenomenalState']  # history for reportability
```

Operations:

- `update(error, precision, sensory)` — differentiable state transition.
- `introspect(query)` — returns a structured view keyed by slot name.
- `modulate_action(action)` — gates action by current state.
- `intervene(slot, value)` — sets a named slot; used for counterfactual evals.

This is the "first-class" claim: the payload has named, addressable slots that can be set, read, and reasoned about, mirroring how phenomenal states have *content*.

### 3.4 Training losses

A composite objective grounds the representation and forces it to be useful:

1. **Reconstruction loss** — `Decoder(sensory, payload)` reconstructs `x`. Forces payload to carry real information.
2. **Prediction loss** — `PredictiveLoop` minimizes precision-weighted sensory prediction error.
3. **KL term on payload** — encourages the workspace distribution to be structured (low effective dimensionality, near-factorized where possible — cheap IIT-style regularizer).
4. **Self-report consistency** — `SelfModel` should produce stable reports across repeated runs with same seed; distinct phenomenal states produce distinct reports (contrastive).
5. **Downstream task loss** — action/decision task that demonstrably benefits from the payload (ablation: without payload, performance drops).

### 3.5 What "introspection" means formally

A readout head `R` is trained on the workspace `Φ` with supervision from held-out queries about the input (e.g., "what color is the dominant object?", "are you confident?"). Introspection accuracy is `R`'s accuracy on queries it has not been trained on but that are predictable from the input and from the model's processing. This is a strict, testable definition — no homunculus required.

---

## 4. Evaluation Criteria

All metrics are measurable on synthetic and naturalistic data. None of these directly measure consciousness; they measure the engineering properties a "coded qualia" system should exhibit.

### 4.1 Introspection Accuracy (IA)

- **Definition:** Accuracy of a learned readout `R` on held-out queries about the input that are answerable from the model's internal state.
- **Why:** If the model cannot answer questions about its own state from its state, the state is not introspectable.
- **Pass bar:** > 0.8 accuracy on a held-out synthetic question set; > 0.6 on naturalistic data.
- **Baselines:** linear probe, MLP probe, random-vector control (scrambled workspace).

### 4.2 Report Consistency (RC)

- **Definition:** Given identical seed and input, repeated `introspect()` calls produce reports within tolerance (e.g., cosine sim > 0.95 for continuous slots, exact match for categorical).
- **Why:** A phenomenal state that is purely stochastic is not reportable.
- **Pass bar:** RC > 0.95 within-run, > 0.7 across re-seeds with same input.

### 4.3 Distinctness (D)

- **Definition:** Distinct phenomenal states (different inputs, contexts) produce distinct reports (mutual information between report slots and input labels > threshold).
- **Why:** A state that maps everything to the same report is not informative.
- **Pass bar:** MI(report_slots; input_labels) > 0.5 bits.

### 4.4 Downstream Grounding (DG)

- **Definition:** Ablation: task performance with `payload` ∈ Φ vs. without (zeroed / replaced by noise). Δ = performance_with − performance_without.
- **Why:** If the payload is decorative, removing it shouldn't hurt. We require it to *help*.
- **Pass bar:** Δ > 0 on a decision task where state matters (e.g., confusable inputs).

### 4.5 Manipulability (M)

- **Definition:** `intervene(slot=v)` produces predictable downstream behavior change. Measured by counterfactual accuracy: given a target slot value, set it and verify downstream output matches expected.
- **Why:** Structured, addressable, manipulable.
- **Pass bar:** > 0.7 counterfactual accuracy on synthetic targets.

### 4.6 Robustness (R)

- **Definition:** Introspection accuracy and report consistency under distribution shift (noise, occlusion, modality drop).
- **Why:** A brittle introspection mechanism is not introspective — it's a memorized feature detector.
- **Pass bar:** < 20% relative drop on standard shift suite.

### 4.7 Non-decomposability proxy (Φ-proxy, optional)

- **Definition:** For bipartitions of the workspace, measure how much information is lost when one half is randomized given the other. Penalize trivially-decomposable representations in training.
- **Why:** Cheap IIT-flavored regularizer; not a Φ claim.
- **Pass bar:** Proxy > baseline (a vanilla VAE), tracked as a curiosity metric, not a primary goal.

### 4.8 Minimal eval gate

A toy end-to-end run (`make train-toy && make eval-toy`) must produce, in under 60 seconds on CPU:

| Metric | Target |
|---|---|
| Reconstruction loss | finite, decreasing |
| Prediction loss | finite, decreasing |
| Report consistency | > 0.5 |
| Introspection accuracy | > 0.5 |
| Distinctness | > 0.3 |
| Δ (downstream grounding) | > 0 |
| Manipulability | > 0.5 |

These thresholds are deliberately permissive for the toy setting; the point is that **every metric is finite, monotonic where expected, and beats a sensible control**.

---

## 5. What This Memo Commits To

- We adopt a **structured, slot-based phenomenal payload** as our central primitive.
- We adopt **predictive-coding-driven updates** to ground the state in input statistics.
- We adopt **GWT-style broadcast** for the workspace.
- We adopt **AST-style self-reporting** as the introspection surface.
- We adopt **measurable, behavioral surrogates** (introspection accuracy, report consistency, manipulability, downstream grounding) as success criteria. We do not measure consciousness directly.
- We treat IIT's Φ only as a **cheap regularizer / curiosity metric**, not a goal.

## 6. What This Memo Does Not Commit To

- A claim that the system is conscious.
- A claim about the philosophical adequacy of these theories.
- A specific modality — the architecture is modality-agnostic; vision and audio are the first testbeds.
- Specific hyperparameters or backbones — left to the implementation beads.

---

## 7. Downstream Beads — How They Connect

This memo is the framing for every bead in the convoy:

| Bead | How this memo informs it |
|---|---|
| Scaffold | Layout matches the six modules in §3.2. |
| PhenomenalState | Implements §3.3 exactly. |
| Encoder/Decoder | Implements §3.2 E and D. |
| PredictiveLoop | Implements §3.2 P; defines precision weighting. |
| SelfModel | Implements §3.2 S; defines named report slots. |
| Eval suite | Implements §4 metrics end-to-end. |
| Training script | Wires losses from §3.4 and runs the minimal gate from §4.8. |

If a downstream decision contradicts this memo, this memo should be updated, not silently overridden.

---

## 8. References (no URLs required, but pointers)

- Baars, B. J. (1988). *A Cognitive Theory of Consciousness.*
- Dehaene, S. (2014). *Consciousness and the Brain.*
- Tononi, G. (2004). An information integration theory of consciousness. *BMC Neuroscience.*
- Friston, K. (2010). The free-energy principle. *Nature Reviews Neuroscience.*
- Rao, R. P. & Ballard, D. H. (1999). Predictive coding in the visual cortex. *Nature Neuroscience.*
- Graziano, M. S. (2013). *Consciousness and the Social Brain.*
- Rosenthal, D. (2005). *Consciousness and Mind.*
- Ha, D. & Schmidhuber, J. (2018). World Models. *NeurIPS.*
- Locatello et al. (2020). Object-Centric Learning with Slot Attention. *NeurIPS.*
- LeCun, Y. (2022). A Path Towards Autonomous Machine Intelligence.

---

*End of memo.*
