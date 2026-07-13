"""Synthetic-data pipeline modules.

Small CPU-friendly encoder / decoder pair that operate on the 1-D percept
vectors produced by :class:`qualia.data.synthetic.ColoredShapesDataset`.

The full qualia stack (:class:`qualia.model.encoder.QualiaEncoder`,
:class:`qualia.model.decoder.QualiaDecoder`) is built around RGB images and
audio waveforms. The synthetic dataset emits ``(percept_dim,)`` feature
vectors, so this module provides a thin, equivalent API on top of plain MLPs:

  - :class:`SyntheticEncoder` maps a percept to ``(sensory, payload)`` where
    the payload matches the canonical slot names
    ``("shape", "color", "hue", "brightness", "agency")``.
  - :class:`SyntheticDecoder` reconstructs percepts from
    ``(sensory, payload)``.

Both modules expose the same call signatures the rest of the qualia stack
expects (``forward(percept) -> EncoderOutput``-like object, ``forward(sensory,
payload) -> Tensor``), so they can be dropped into the training loop in place
of the image/audio variants.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

PAYLOAD_KEYS: tuple[str, ...] = (
    "shape",
    "color",
    "hue",
    "brightness",
    "agency",
)


@dataclass
class SyntheticEncoderOutput:
    """Structured output of :class:`SyntheticEncoder`.

    Attributes:
        sensory: ``(B, sensory_dim)`` global sensory feature.
        payload: Dict mapping each named slot to its tensor. Categorical
            slots are ``(B, vocab_size)`` logits; continuous slots are
            ``(B, 1)``.
    """

    sensory: Tensor
    payload: dict[str, Tensor]


class SyntheticEncoder(nn.Module):
    """Tiny MLP encoder for 1-D percept vectors.

    Args:
        percept_dim: Dimensionality of incoming percept vectors.
        sensory_dim: Dimensionality of the produced sensory representation.
        payload_keys: Names of the structured phenomenal-payload slots.
        payload_vocabs: Mapping ``slot -> vocab_size`` for categorical slots.
            Slots missing from this map default to ``vocab=1`` (continuous).
        hidden_dim: Width of the shared MLP trunk.
    """

    def __init__(
        self,
        percept_dim: int,
        sensory_dim: int = 32,
        payload_keys: tuple[str, ...] = PAYLOAD_KEYS,
        payload_vocabs: dict[str, int] | None = None,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        if percept_dim <= 0:
            raise ValueError(f"percept_dim must be positive; got {percept_dim}")
        if sensory_dim <= 0:
            raise ValueError(f"sensory_dim must be positive; got {sensory_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive; got {hidden_dim}")

        self.percept_dim = int(percept_dim)
        self.sensory_dim = int(sensory_dim)
        self.hidden_dim = int(hidden_dim)
        self.payload_keys: tuple[str, ...] = tuple(payload_keys)
        # Default to categorical vocabs for the two categorical slot names so
        # that ``payload_vocabs=None`` produces an encoder with the right
        # output shape out of the box (continuous slots still default to 1).
        self.payload_vocabs: dict[str, int] = dict(payload_vocabs or {"shape": 4, "color": 4})
        for key in self.payload_keys:
            self.payload_vocabs.setdefault(key, 1)

        self.trunk = nn.Sequential(
            nn.Linear(self.percept_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.sensory_head = nn.Linear(self.hidden_dim, self.sensory_dim)
        self.payload_heads = nn.ModuleDict(
            {key: nn.Linear(self.hidden_dim, self.payload_vocabs[key]) for key in self.payload_keys}
        )

    def forward(self, percept: Tensor) -> SyntheticEncoderOutput:
        if percept.dim() != 2 or percept.shape[-1] != self.percept_dim:
            raise ValueError(
                f"percept must have shape (B, {self.percept_dim}); got {tuple(percept.shape)}"
            )
        features = self.trunk(percept)
        sensory = self.sensory_head(features)
        payload = {key: self.payload_heads[key](features) for key in self.payload_keys}
        return SyntheticEncoderOutput(sensory=sensory, payload=payload)


class SyntheticDecoder(nn.Module):
    """Tiny MLP decoder that reconstructs percepts from ``(sensory, payload)``.

    Mirrors :class:`SyntheticEncoder`: continuous payload slots are passed
    through a learned linear projection, categorical slots are softmaxed and
    embedded, and the resulting features are concatenated with the sensory
    vector before decoding back to ``percept_dim``.
    """

    def __init__(
        self,
        encoder: SyntheticEncoder,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.percept_dim = encoder.percept_dim
        self.sensory_dim = encoder.sensory_dim
        self.payload_keys = encoder.payload_keys
        self.payload_vocabs = dict(encoder.payload_vocabs)
        self.hidden_dim = int(hidden_dim or encoder.hidden_dim)

        self.embeddings = nn.ModuleDict(
            {
                key: nn.Linear(vocab, self.hidden_dim)
                for key, vocab in self.payload_vocabs.items()
                if vocab > 1
            }
        )
        self.cont_embeds = nn.ModuleDict(
            {
                key: nn.Linear(1, self.hidden_dim)
                for key, vocab in self.payload_vocabs.items()
                if vocab == 1
            }
        )
        self.body = nn.Sequential(
            nn.Linear(
                self.sensory_dim + len(self.payload_keys) * self.hidden_dim,
                self.hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.percept_dim),
        )

    def payload_features(self, payload: dict[str, Tensor]) -> Tensor:
        feats: list[Tensor] = []
        for key in self.payload_keys:
            slot = payload[key]
            if slot.dim() == 1:
                slot = slot.unsqueeze(0)
            vocab = self.payload_vocabs[key]
            if vocab == 1:
                feats.append(self.cont_embeds[key](slot))
            else:
                feats.append(self.embeddings[key](slot.softmax(dim=-1)))
        return torch.cat(feats, dim=-1)

    def forward(self, sensory: Tensor, payload: dict[str, Tensor]) -> Tensor:
        if sensory.dim() != 2 or sensory.shape[-1] != self.sensory_dim:
            raise ValueError(
                f"sensory must have shape (B, {self.sensory_dim}); got {tuple(sensory.shape)}"
            )
        payload_feat = self.payload_features(payload)
        return self.body(torch.cat([sensory, payload_feat], dim=-1))


__all__ = [
    "PAYLOAD_KEYS",
    "SyntheticDecoder",
    "SyntheticEncoder",
    "SyntheticEncoderOutput",
]
