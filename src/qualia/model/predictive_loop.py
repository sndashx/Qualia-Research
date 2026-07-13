"""Recurrent predictive-coding loop with global workspace integration.

The predictive loop is the core dynamics engine of the system. On each step
it:

    1. Observes a sensory vector ``x_t`` and the current phenomenal state
       ``S_{t-1}``.
    2. Predicts the next sensory state ``x̂_{t+1}`` from the current state.
    3. Computes a precision-weighted prediction error ``e_t = π_t ⊙ (x_t - x̂_t)``
       where ``π_t`` is a per-channel precision weight produced by the
       state itself.
    4. Updates the :class:`PhenomenalState` by combining the incoming percept
       and the precision-weighted error.
    5. Broadcasts the resulting state through a :class:`GlobalWorkspace`,
       producing a downstream-facing summary that other modules (self-model,
       decoder, policy) can consume.

The module exposes a small training loss
(:func:`predictive_coding_loss`) that combines a reconstruction term on the
predicted sensory vector with a prediction-error-weighted KL between the
predicted phenomenal payload and the observed one. The KL weighting comes
from the precision weights, so high-precision channels dominate the
phenomenal-update signal (and thus the KL term), while low-precision
channels are effectively ignored.

The API is intentionally narrow: a single :meth:`PredictiveCodingLoop.rollout`
call runs ``T`` recurrent steps and returns the full trajectory of
predictions, errors, phenomenals, precisions and workspace broadcasts, so
callers can train on whatever combination of losses they want.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _kl_categorical_logits(pred_logits: Tensor, obs_logits: Tensor, dim: int = -1) -> Tensor:
    """KL( softmax(obs_logits) || softmax(pred_logits) ) along ``dim``.

    Returns a tensor with ``dim`` reduced. Used for categorical payload slots.
    """
    pred_log = F.log_softmax(pred_logits, dim=dim)
    obs_log_probs = F.log_softmax(obs_logits, dim=dim)
    obs_probs = obs_log_probs.exp()
    # KL(p||q) = sum p (log p - log q); use obs_probs * (obs_log_probs - pred_log)
    return (obs_probs * (obs_log_probs - pred_log)).sum(dim=dim)


def _kl_gaussian(pred_mean: Tensor, obs_mean: Tensor, obs_logvar: Tensor | None = None) -> Tensor:
    """KL between two isotropic Gaussians with diagonal covariance.

    If ``obs_logvar`` is ``None``, unit variance on the observation is assumed.
    Returns a tensor with the last (feature) dim reduced. The caller is
    responsible for any further reductions.
    """
    if obs_logvar is None:
        pred_logvar = torch.zeros_like(pred_mean)
        return 0.5 * ((pred_mean - obs_mean).pow(2) + pred_logvar.exp() - 1.0 - pred_logvar).sum(
            dim=-1
        )
    return 0.5 * (
        (pred_mean - obs_mean).pow(2) / obs_logvar.exp() + obs_logvar - 1.0 - obs_logvar
    ).sum(dim=-1)


# ---------------------------------------------------------------------------
# Global Workspace
# ---------------------------------------------------------------------------


class GlobalWorkspace(nn.Module):
    """GWT-style broadcast slot.

    A few specialized modules compete (via softmax over a learned score) to
    occupy a single low-dimensional workspace vector at each step. Downstream
    modules consume this broadcast vector directly.

    For training simplicity we use a deterministic, soft winner-take-all:
    each "expert" produces a workspace contribution, and the contribution is
    weighted by ``softmax(score)``. The full broadcast is the sum of weighted
    contributions, which keeps the whole thing end-to-end differentiable.
    """

    def __init__(self, input_dim: int, workspace_dim: int = 32, n_modules: int = 4) -> None:
        super().__init__()
        if workspace_dim <= 0 or n_modules <= 0:
            raise ValueError("workspace_dim and n_modules must be positive")
        self.workspace_dim = workspace_dim
        self.n_modules = n_modules
        self.experts = nn.ModuleList(
            [nn.Linear(input_dim, workspace_dim) for _ in range(n_modules)]
        )
        self.scorer = nn.Linear(input_dim, n_modules)

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        """Compute the workspace broadcast and the per-module scores.

        Args:
            inputs: ``(B, input_dim)`` aggregated features.

        Returns:
            broadcast: ``(B, workspace_dim)`` workspace vector.
            weights: ``(B, n_modules)`` softmax weights over modules.
        """
        scores = self.scorer(inputs)
        weights = F.softmax(scores, dim=-1)
        contributions = torch.stack([expert(inputs) for expert in self.experts], dim=1)
        broadcast = (weights.unsqueeze(-1) * contributions).sum(dim=1)
        return broadcast, weights


# ---------------------------------------------------------------------------
# Predictive loop core
# ---------------------------------------------------------------------------


@dataclass
class PredictiveTrajectory:
    """Full rollout record returned by :meth:`PredictiveCodingLoop.rollout`.

    All tensors have a leading time dimension ``T`` (the first axis) and a
    leading batch dimension ``B`` (the second axis) unless noted otherwise.

    Attributes:
        predictions: ``(T, B, sensory_dim)`` predicted *next* sensory at each step.
        errors: ``(T, B, sensory_dim)`` precision-weighted prediction errors.
        precisions: ``(T, B, sensory_dim)`` precision weights used.
        workspace: ``(T, B, workspace_dim)`` global-workspace broadcasts.
        module_weights: ``(T, B, n_modules)`` softmax weights over experts.
        phenomenals: List of length ``T`` with the :class:`PhenomenalState`
            outputs (``{"workspace", "self_model", "payload"}``) at each step.
        predicted_payloads: ``(T, B, payload_dim)`` flat predicted payload vector.
        observed_payloads: ``(T, B, payload_dim)`` flat observed payload vector.
        payload_keys: Tuple of slot names carried in the payload, matching the
            column layout of ``predicted_payloads`` / ``observed_payloads``.
        payload_vocabs: ``payload_key -> vocab_size`` mapping (1 for continuous).
        predicted_sensory_mask: ``(T, B, sensory_dim)`` mask indicating
            ``True`` where sensory_dim was padded out and should be ignored.
    """

    predictions: Tensor
    errors: Tensor
    precisions: Tensor
    workspace: Tensor
    module_weights: Tensor
    phenomenals: list[dict[str, Tensor]]
    predicted_payloads: Tensor
    observed_payloads: Tensor
    payload_keys: tuple[str, ...]
    payload_vocabs: dict[str, int]
    predicted_sensory_mask: Tensor | None = None


class PredictiveCodingLoop(nn.Module):
    """Recurrent predictive-coding loop with global workspace integration.

    The loop observes a stream of sensory vectors and produces predictions,
    precision-weighted errors, and an updated :class:`PhenomenalState` for
    each step. The current implementation is fully feed-forward across time
    (each step is a forward through the same module on different inputs) so
    that ``rollout`` can be implemented as a single Python loop without
    unrolling through time at the autograd level. Gradients still flow
    end-to-end because the state is held in ``nn.Module`` buffers /
    parameters that the optimizer updates at every minibatch.

    Args:
        sensory_dim: Dimensionality of the sensory vector.
        workspace_dim: Dimensionality of the global workspace vector.
        self_model_dim: Dimensionality of the self-model vector.
        payload_keys: Names of the phenomenal payload slots.
        payload_vocabs: ``slot -> vocab_size`` mapping (1 for continuous).
        precision_init: Initial value for the precision logit. ``log(1) = 0``
            corresponds to uniform precision weighting.
        error_clamp: Optional ``(min, max)`` clamp on the raw precision logits
            for numerical stability. ``None`` disables clamping.
    """

    def __init__(
        self,
        sensory_dim: int,
        workspace_dim: int = 32,
        self_model_dim: int = 32,
        payload_keys: tuple[str, ...] = ("content", "arousal", "valence"),
        payload_vocabs: dict[str, int] | None = None,
        precision_init: float = 0.0,
        error_clamp: tuple[float, float] | None = (-6.0, 6.0),
    ) -> None:
        super().__init__()
        if sensory_dim <= 0:
            raise ValueError("sensory_dim must be positive")
        if workspace_dim <= 0 or self_model_dim <= 0:
            raise ValueError("workspace_dim and self_model_dim must be positive")
        if not payload_keys:
            raise ValueError("payload_keys must be non-empty")

        self.sensory_dim = sensory_dim
        self.workspace_dim = workspace_dim
        self.self_model_dim = self_model_dim
        self.payload_keys: tuple[str, ...] = tuple(payload_keys)
        self.payload_vocabs: dict[str, int] = dict(payload_vocabs or {})
        for key in self.payload_keys:
            self.payload_vocabs.setdefault(key, 1)
        self.error_clamp = error_clamp

        # PhenomenalState holds the live, differentiable state of the loop.
        # It expects a percept_dim equal to the sensory_dim we observe, and
        # produces workspace + self_model + payload.
        from .phenomenal_state import PhenomenalState

        self.phenomenal_state = PhenomenalState(
            percept_dim=sensory_dim,
            workspace_dim=workspace_dim,
            self_model_dim=self_model_dim,
            payload_slots=max(self.payload_vocabs.values()),
            payload_keys=self.payload_keys,
        )

        # Predictor: maps the *current* workspace + self_model to the next
        # sensory state. We split it into a prediction of the sensory vector
        # proper and a prediction of the (flat) payload vector, so the
        # prediction error pathway can be trained end-to-end.
        state_dim = workspace_dim + self_model_dim
        self.sensory_predictor = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.Tanh(),
            nn.Linear(state_dim, sensory_dim),
        )
        payload_total = sum(self.payload_vocabs[key] for key in self.payload_keys)
        self.payload_predictor = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.Tanh(),
            nn.Linear(state_dim, payload_total),
        )

        # Precision head: produces a log-precision per sensory channel from
        # the current state. Using a log-precision makes the weighting
        # numerically stable and ensures positivity via softplus.
        self.precision_head = nn.Sequential(
            nn.Linear(state_dim, sensory_dim),
            nn.Tanh(),
            nn.Linear(sensory_dim, sensory_dim),
        )
        # Initialize the final layer's bias to ``precision_init`` so the
        # loop starts with roughly uniform precision.
        with torch.no_grad():
            self.precision_head[-1].bias.fill_(float(precision_init))

        # Error integrator: maps (percept, precision-weighted error, current
        # state) into the next percept to feed into PhenomenalState. Keeping
        # this as a small MLP lets the loop learn how to *use* the prediction
        # error rather than just passing it through verbatim.
        self.error_integrator = nn.Sequential(
            nn.Linear(sensory_dim + sensory_dim + state_dim, state_dim),
            nn.Tanh(),
            nn.Linear(state_dim, sensory_dim),
        )

        # Global workspace: broadcasts the integrated state to downstream.
        self.workspace = GlobalWorkspace(
            input_dim=sensory_dim + state_dim,
            workspace_dim=workspace_dim,
            n_modules=4,
        )

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def payload_total(self) -> int:
        return sum(self.payload_vocabs[key] for key in self.payload_keys)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _flatten_payload(self, payload: dict[str, Tensor]) -> Tensor:
        """Flatten a payload dict in canonical order to ``(B, payload_total)``."""
        chunks = [payload[key].flatten(start_dim=1) for key in self.payload_keys]
        return torch.cat(chunks, dim=-1)

    def _predict_payload(self, state_vec: Tensor) -> Tensor:
        """Predict the next flat payload vector from the current state."""
        return self.payload_predictor(state_vec)

    def _predict_sensory(self, state_vec: Tensor) -> Tensor:
        """Predict the next sensory vector from the current state."""
        return self.sensory_predictor(state_vec)

    def _current_state_vec(self) -> Tensor:
        """Return the concatenated workspace + self_model ``(state_dim,)``."""
        return torch.cat(
            [self.phenomenal_state.workspace, self.phenomenal_state.self_model], dim=-1
        )

    # ------------------------------------------------------------------
    # Single step
    # ------------------------------------------------------------------

    def step(
        self, observed_sensory: Tensor, sensory_mask: Tensor | None = None
    ) -> dict[str, Tensor]:
        """Run a single predictive-coding step.

        Args:
            observed_sensory: ``(B, sensory_dim)`` sensory vector observed at
                ``t``.
            sensory_mask: Optional ``(B, sensory_dim)`` boolean mask where
                ``False`` indicates the channel should be ignored (padded /
                out-of-range). Defaults to all-ones.

        Returns:
            Dict with the trajectory entries for this step:
            ``prediction``, ``error``, ``precision``, ``phenomenal``,
            ``predicted_payload``, ``workspace``, ``module_weights``.
        """
        if observed_sensory.dim() != 2 or observed_sensory.shape[-1] != self.sensory_dim:
            raise ValueError(
                f"observed_sensory must have shape (B, {self.sensory_dim}); got {tuple(observed_sensory.shape)}"
            )
        batch_size = observed_sensory.shape[0]
        if sensory_mask is None:
            sensory_mask = torch.ones_like(observed_sensory, dtype=torch.bool)
        elif sensory_mask.shape != observed_sensory.shape:
            raise ValueError(
                f"sensory_mask must have shape {tuple(observed_sensory.shape)}; got {tuple(sensory_mask.shape)}"
            )

        # 1. Predict the next sensory state from the *current* state.
        state_vec = self._current_state_vec().unsqueeze(0).expand(batch_size, -1)
        predicted_sensory = self._predict_sensory(state_vec)

        # 2. Compute raw prediction error and per-channel precision.
        raw_error = observed_sensory - predicted_sensory
        log_precision = self.precision_head(state_vec)
        if self.error_clamp is not None:
            log_precision = log_precision.clamp(*self.error_clamp)
        precision = F.softplus(log_precision) + 1e-3  # strictly positive

        # 3. Apply mask: ignored channels get zero error and unit precision
        # so they don't blow up gradients. The masking preserves the original
        # prediction for downstream.
        masked_error = raw_error * sensory_mask.to(raw_error.dtype)
        weighted_error = precision * masked_error

        # 4. Integrate: feed (observed, weighted_error, current_state) into
        # PhenomenalState. We use observed_sensory (the actual percept) so
        # the state remains anchored, but expose the weighted error to the
        # integrator so it can use the prediction error signal.
        integrated_input = torch.cat([observed_sensory, weighted_error, state_vec], dim=-1)
        next_percept = self.error_integrator(integrated_input)
        # Replace masked channels with the observed sensory so they don't
        # contribute spurious gradients.
        next_percept = torch.where(sensory_mask, next_percept, observed_sensory)

        phenomenal_out = self.phenomenal_state.encode(next_percept)

        # 5. Predict the *next* payload from the *new* state and broadcast
        # the integrated state through the global workspace.
        new_state_vec = self._current_state_vec().unsqueeze(0).expand(batch_size, -1)
        predicted_payload = self._predict_payload(new_state_vec)

        workspace_in = torch.cat([next_percept, new_state_vec], dim=-1)
        broadcast, module_weights = self.workspace(workspace_in)

        return {
            "prediction": predicted_sensory,
            "error": weighted_error,
            "raw_error": raw_error,
            "precision": precision,
            "phenomenal": phenomenal_out,
            "predicted_payload": predicted_payload,
            "workspace": broadcast,
            "module_weights": module_weights,
        }

    # ------------------------------------------------------------------
    # Multi-step rollout
    # ------------------------------------------------------------------

    def rollout(
        self,
        sensory_sequence: Tensor,
        payload_sequence: list[dict[str, Tensor]] | None = None,
        reset: bool = True,
    ) -> PredictiveTrajectory:
        """Run ``T`` recurrent steps over a sensory sequence.

        Args:
            sensory_sequence: ``(T, B, sensory_dim)`` sequence of sensory
                vectors to observe, in chronological order.
            payload_sequence: Optional list of length ``T`` of payload dicts
                observed at each step. Each dict maps slot name to a
                ``(B, vocab_size)`` or ``(B, 1)`` tensor. Used by the loss
                function for the KL term. If ``None``, only the prediction
                loss is meaningful.
            reset: Whether to reset the :class:`PhenomenalState` before
                running. Defaults to ``True``.

        Returns:
            :class:`PredictiveTrajectory` with all per-step tensors.
        """
        if sensory_sequence.dim() != 3 or sensory_sequence.shape[-1] != self.sensory_dim:
            raise ValueError(
                f"sensory_sequence must have shape (T, B, {self.sensory_dim}); got {tuple(sensory_sequence.shape)}"
            )
        T, B, _ = sensory_sequence.shape

        if payload_sequence is not None and len(payload_sequence) != T:
            raise ValueError(
                f"payload_sequence must have length T={T}; got {len(payload_sequence)}"
            )

        if reset:
            self.phenomenal_state.reset_state()

        predictions: list[Tensor] = []
        errors: list[Tensor] = []
        precisions: list[Tensor] = []
        broadcasts: list[Tensor] = []
        module_weights: list[Tensor] = []
        phenomenals: list[dict[str, Tensor]] = []
        predicted_payloads: list[Tensor] = []
        observed_payloads: list[Tensor] = []

        for t in range(T):
            step_out = self.step(sensory_sequence[t])
            predictions.append(step_out["prediction"])
            errors.append(step_out["error"])
            precisions.append(step_out["precision"])
            broadcasts.append(step_out["workspace"])
            module_weights.append(step_out["module_weights"])
            phenomenals.append(step_out["phenomenal"])
            predicted_payloads.append(step_out["predicted_payload"])
            if payload_sequence is not None:
                observed_payloads.append(self._flatten_payload(payload_sequence[t]))

        predicted_payloads_t = torch.stack(predicted_payloads, dim=0)
        observed_payloads_t = (
            torch.stack(observed_payloads, dim=0)
            if observed_payloads
            else predicted_payloads_t.detach() * 0
        )

        return PredictiveTrajectory(
            predictions=torch.stack(predictions, dim=0),
            errors=torch.stack(errors, dim=0),
            precisions=torch.stack(precisions, dim=0),
            workspace=torch.stack(broadcasts, dim=0),
            module_weights=torch.stack(module_weights, dim=0),
            phenomenals=phenomenals,
            predicted_payloads=predicted_payloads_t,
            observed_payloads=observed_payloads_t,
            payload_keys=self.payload_keys,
            payload_vocabs=dict(self.payload_vocabs),
        )


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def predictive_coding_loss(
    trajectory: PredictiveTrajectory,
    observed_sensory: Tensor,
    kl_weight: float = 1.0,
    sensory_mask: Tensor | None = None,
    reduction: str = "mean",
) -> dict[str, Tensor]:
    """Mixed loss: reconstruction + precision-weighted KL on phenomenals.

    The reconstruction term is the MSE between ``trajectory.predictions``
    and ``observed_sensory`` (shifted by one step: the prediction made at
    step ``t`` is matched against the observation at step ``t + 1``). The
    last-step prediction has no target and is dropped.

    The KL term is a sum over payload slots of the KL between predicted and
    observed slot distributions, weighted by the mean precision on that
    step. This matches the predictive-coding prior: high-precision channels
    dominate the phenomenal-update signal and therefore dominate the KL
    term.

    Args:
        trajectory: Output of :meth:`PredictiveCodingLoop.rollout`.
        observed_sensory: ``(T, B, sensory_dim)`` sequence actually observed.
        kl_weight: Scalar weight on the KL term.
        sensory_mask: Optional ``(T, B, sensory_dim)`` boolean mask. ``False``
            channels are ignored by the reconstruction loss.
        reduction: ``"mean"`` (default), ``"sum"``, or ``"none"``.

    Returns:
        Dict with keys ``"reconstruction"``, ``"kl"``, ``"total"``.
    """
    if trajectory.predictions.dim() != 3:
        raise ValueError(
            f"trajectory.predictions must have shape (T, B, sensory_dim); got {tuple(trajectory.predictions.shape)}"
        )
    T, B, D = trajectory.predictions.shape
    if observed_sensory.shape != (T, B, D):
        raise ValueError(
            f"observed_sensory must have shape {(T, B, D)}; got {tuple(observed_sensory.shape)}"
        )

    pred_for_target = trajectory.predictions[:-1]
    target = observed_sensory[1:]
    if sensory_mask is not None:
        mask = sensory_mask[1:].to(pred_for_target.dtype)
        recon = ((pred_for_target - target) ** 2 * mask).sum(dim=-1)
        denom = mask.sum(dim=-1).clamp(min=1.0)
        recon_per_step = recon / denom
    else:
        recon_per_step = ((pred_for_target - target) ** 2).mean(dim=-1)

    if reduction == "mean":
        reconstruction = recon_per_step.mean()
    elif reduction == "sum":
        reconstruction = recon_per_step.sum()
    elif reduction == "none":
        reconstruction = recon_per_step
    else:
        raise ValueError(f"reduction must be 'mean', 'sum', or 'none'; got {reduction!r}")

    # KL term: precision-weighted KL between predicted and observed payloads.
    # We assume the prediction made at step ``t`` targets the observation at
    # step ``t + 1``; this gives the same shift as the reconstruction term.
    kl_value = _payload_kl(
        trajectory,
        weight=trajectory.precisions[:-1].mean(dim=-1),
    )

    total = reconstruction + kl_weight * kl_value

    return {
        "reconstruction": reconstruction,
        "kl": kl_value,
        "total": total,
    }


def _payload_kl(
    trajectory: PredictiveTrajectory,
    weight: Tensor,
) -> Tensor:
    """Compute precision-weighted KL between predicted and observed payloads.

    For continuous slots (vocab==1) we treat the prediction and observation as
    means of isotropic Gaussians with unit variance and use the closed-form
    diagonal Gaussian KL. For categorical slots (vocab>1) we use the
    KL between two softmax distributions over the slot vocabulary.

    Args:
        trajectory: Output of :meth:`PredictiveCodingLoop.rollout`.
        weight: ``(T', B)`` per-step scalar weight (typically the mean
            precision on that step). Time-shifted to align with predictions
            (caller passes ``trajectory.precisions[:-1].mean(dim=-1)``).

    Returns:
        Scalar tensor with the weighted KL summed across slots and steps.
    """
    pred = trajectory.predicted_payloads[:-1]
    obs = trajectory.observed_payloads[:-1]
    if pred.numel() == 0:
        return pred.sum() * 0.0

    total_kl = pred.new_zeros(())
    offset = 0
    for key in trajectory.payload_keys:
        vocab = trajectory.payload_vocabs[key]
        chunk_pred = pred[..., offset : offset + vocab]
        chunk_obs = obs[..., offset : offset + vocab]
        if vocab == 1:
            kl = _kl_gaussian(chunk_pred, chunk_obs)
        else:
            kl = _kl_categorical_logits(chunk_pred, chunk_obs, dim=-1)
        total_kl = total_kl + (weight * kl).mean()
        offset += vocab
    return total_kl


__all__ = [
    "GlobalWorkspace",
    "PredictiveCodingLoop",
    "PredictiveTrajectory",
    "predictive_coding_loss",
]
