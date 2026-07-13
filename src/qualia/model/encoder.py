"""Qualia-aware encoder.

Maps raw perception (RGB images or audio) into:
  (a) a sensory representation: a global feature vector describing the
      "what is being sensed".
  (b) a phenomenal-payload: a structured, named dict of slots where each slot
      carries either a categorical (logits over a small vocabulary) or a
      continuous (real-valued) attribute. The payload is the introspectable,
      structured "what-it-is-like" content that downstream modules
      (self-model, predictive loop) consume.

The decoder (see :mod:`qualia.model.decoder`) reconstructs the perception
*from* ``(sensory, phenomenal-payload)``. That reconstruction pressure is what
forces the payload to be informative rather than decorative: if the decoder
can recover the input better when given the payload than when it is not, the
payload carries mutual information with the input.

Backbones (small, CPU-friendly):
  - ``cnn``: a tiny ConvNeXt-style stem with two depthwise-separable blocks.
  - ``vit``: a ViT-style patch encoder with a small number of patches.

The same encoder head produces sensory + payload for either backbone.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

PAYLOAD_KEYS: tuple[str, ...] = (
    "shape",  # categorical: which shape class is present
    "color",  # categorical: which color class is present
    "hue",  # continuous: hue value in [0, 1]
    "brightness",  # continuous: scalar brightness
    "agency",  # continuous: scalar agency / self-causation
)


@dataclass
class EncoderOutput:
    """Structured output of the encoder.

    Attributes:
        sensory: ``(B, sensory_dim)`` global sensory feature.
        payload: Dict mapping each named slot to its tensor. Categorical slots
            are ``(B, vocab_size)`` logits; continuous slots are ``(B, 1)``.
        backbone_features: ``(B, feature_dim)`` raw backbone feature, kept for
            the decoder's skip-style conditioning.
    """

    sensory: Tensor
    payload: dict[str, Tensor]
    backbone_features: Tensor

    def payload_vector(self) -> Tensor:
        """Concatenate all payload slots into a single flat vector.

        Continuous slots are kept as-is (1 dim each); categorical slots are
        represented by their full logits (vocab dims each). Useful for mutual
        information estimation and the decoder's payload-conditioning path.
        """
        return torch.cat(
            [self.payload[key].flatten(start_dim=1) for key in PAYLOAD_KEYS],
            dim=-1,
        )


# ---------------------------------------------------------------------------
# Backbones
# ---------------------------------------------------------------------------


class _ConvNeXtBlock(nn.Module):
    """A small ConvNeXt-style block: depthwise 7x7 conv + pointwise MLP."""

    def __init__(self, dim: int, expansion: int = 4) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pw1 = nn.Linear(dim, dim * expansion)
        self.act = nn.GELU()
        self.pw2 = nn.Linear(dim * expansion, dim)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, H, W) -> (B, H, W, C)
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        x = x.permute(0, 3, 1, 2)
        return residual + x


class TinyConvNeXt(nn.Module):
    """A small ConvNeXt-style backbone for 32x32 RGB inputs."""

    def __init__(self, in_channels: int = 3, dims: tuple[int, ...] = (32, 64)) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=3, stride=1, padding=1),
            nn.GELU(),
        )
        blocks: list[nn.Module] = []
        for i in range(len(dims)):
            blocks.append(_ConvNeXtBlock(dims[i]))
            if i + 1 < len(dims):
                blocks.append(
                    nn.Sequential(
                        nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                        nn.GELU(),
                    )
                )
        self.body = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = dims[-1]

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.body(x)
        x = self.pool(x).flatten(1)
        return x


class TinyViT(nn.Module):
    """A small ViT-style patch encoder.

    Splits the input into a grid of ``patch_size x patch_size`` patches,
    linearly projects each patch, adds learned positional embeddings and a CLS
    token, then runs a few Transformer encoder layers.
    """

    def __init__(
        self,
        in_channels: int = 3,
        image_size: int = 32,
        patch_size: int = 8,
        dim: int = 64,
        depth: int = 2,
        heads: int = 4,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.dim = dim
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 2,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.out_dim = dim

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            raise ValueError(
                f"expected input of shape (B, C, {self.image_size}, {self.image_size}); "
                f"got {tuple(x.shape)}"
            )
        patches = self.proj(x)
        patches = patches.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        tokens = torch.cat([cls, patches], dim=1)
        tokens = tokens + self.pos_embed
        tokens = self.encoder(tokens)
        return tokens[:, 0]


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------


class PayloadHead(nn.Module):
    """Produces a structured phenomenal-payload from a feature vector.

    For each named slot, either an ``(B, vocab)`` categorical head (when
    ``vocab > 1``) or an ``(B, 1)`` continuous head (when ``vocab == 1``).
    """

    def __init__(
        self,
        in_dim: int,
        keys: tuple[str, ...] = PAYLOAD_KEYS,
        vocabs: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        if vocabs is None:
            vocabs = {"shape": 4, "color": 4}
        self.keys = tuple(keys)
        self.vocabs = dict(vocabs)
        # Validate: all keys must have an entry (defaulting vocab=1 = continuous).
        for key in self.keys:
            self.vocabs.setdefault(key, 1)
        self.projs = nn.ModuleDict(
            {key: nn.Linear(in_dim, self.vocabs[key]) for key in self.keys}
        )

    def forward(self, feature: Tensor) -> dict[str, Tensor]:
        return {key: self.projs[key](feature) for key in self.keys}


class SensoryHead(nn.Module):
    """Maps a backbone feature vector to the sensory representation."""

    def __init__(self, in_dim: int, sensory_dim: int = 64) -> None:
        super().__init__()
        self.sensory_dim = sensory_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, sensory_dim),
            nn.GELU(),
            nn.Linear(sensory_dim, sensory_dim),
        )

    def forward(self, feature: Tensor) -> Tensor:
        return self.net(feature)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class QualiaEncoder(nn.Module):
    """Qualia-aware encoder.

    Args:
        modality: One of ``"image"`` or ``"audio"``. Image uses a CNN or ViT
            backbone over ``image_size x image_size`` inputs. Audio uses a 1D
            CNN stem treating the waveform as a single-channel signal.
        backbone: ``"cnn"`` (TinyConvNeXt) or ``"vit"`` (TinyViT). Ignored for
            audio.
        in_channels: Number of input channels (3 for RGB, 1 for audio).
        image_size: Spatial size for image inputs (used by ViT).
        sensory_dim: Dimensionality of the sensory representation.
        payload_keys: Names of the phenomenal-payload slots to emit.
        payload_vocabs: Mapping ``slot -> vocab_size`` for categorical slots.
            Slots missing from this map default to ``vocab=1`` (continuous).
    """

    def __init__(
        self,
        modality: str = "image",
        backbone: str = "cnn",
        in_channels: int = 3,
        image_size: int = 32,
        sensory_dim: int = 64,
        payload_keys: tuple[str, ...] = PAYLOAD_KEYS,
        payload_vocabs: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        if modality not in {"image", "audio"}:
            raise ValueError(f"modality must be 'image' or 'audio'; got {modality!r}")
        self.modality = modality
        self.backbone_name = backbone
        self.sensory_dim = sensory_dim
        self.payload_keys = tuple(payload_keys)
        self.payload_vocabs: dict[str, int] = dict(payload_vocabs or {"shape": 4, "color": 4})
        # Every key needs an entry (default 1 = continuous slot).
        for key in self.payload_keys:
            self.payload_vocabs.setdefault(key, 1)

        if modality == "image":
            if backbone == "cnn":
                self.backbone: nn.Module = TinyConvNeXt(in_channels=in_channels)
            elif backbone == "vit":
                self.backbone = TinyViT(in_channels=in_channels, image_size=image_size)
            else:
                raise ValueError(f"backbone must be 'cnn' or 'vit'; got {backbone!r}")
        else:
            self.backbone = nn.Sequential(
                nn.Conv1d(in_channels, 32, kernel_size=8, stride=4, padding=4),
                nn.GELU(),
                nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=2),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(1),
            )

        feature_dim = int(getattr(self.backbone, "out_dim", 64))
        self.sensory_head = SensoryHead(feature_dim, sensory_dim=sensory_dim)
        self.payload_head = PayloadHead(feature_dim, keys=self.payload_keys, vocabs=self.payload_vocabs)

    def forward(self, x: Tensor) -> EncoderOutput:
        if self.modality == "audio":
            if x.dim() != 3:
                raise ValueError(
                    f"audio input must have shape (B, C, T); got {tuple(x.shape)}"
                )
            feature = self.backbone(x).flatten(1)
        else:
            if x.dim() != 4:
                raise ValueError(
                    f"image input must have shape (B, C, H, W); got {tuple(x.shape)}"
                )
            feature = self.backbone(x)

        sensory = self.sensory_head(feature)
        payload = self.payload_head(feature)
        return EncoderOutput(sensory=sensory, payload=payload, backbone_features=feature)

    def payload_size(self) -> int:
        """Total dim of the concatenated payload vector."""
        return sum(self.payload_vocabs[key] for key in self.payload_keys)


__all__ = [
    "EncoderOutput",
    "PAYLOAD_KEYS",
    "QualiaEncoder",
    "SensoryHead",
    "PayloadHead",
    "TinyConvNeXt",
    "TinyViT",
]
