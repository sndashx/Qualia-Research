"""Qualia-aware decoder.

Reconstructs the original perception (RGB image or audio waveform) from
``(sensory, phenomenal-payload)``.

The reconstruction objective is what gives the phenomenal-payload its meaning:
if the decoder can recover the input using the payload, then the payload must
carry information about the input. If the decoder can recover the input
*better* given the payload than without it, the payload has positive mutual
information with the input.

The decoder is composed of:
  - a *fusion* module that combines ``sensory`` with the payload into a
    conditioning vector,
  - a *prior* branch that reconstructs from ``sensory`` alone (for ablation /
    MI sanity checks),
  - a *backbone-specific generator* (transposed-conv stack for the CNN path,
    patch-decoding ViT for the ViT path).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .encoder import PAYLOAD_KEYS, EncoderOutput


class PayloadFusion(nn.Module):
    """Fuse sensory vector with payload slots into a single conditioning vec.

    Continuous payload slots are passed through; categorical payload slots are
    soft-maxed (so gradients flow cleanly) and embedded via a per-slot
    embedding layer.
    """

    def __init__(
        self,
        sensory_dim: int,
        payload_keys: tuple[str, ...] = PAYLOAD_KEYS,
        payload_vocabs: dict[str, int] | None = None,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        if payload_vocabs is None:
            payload_vocabs = {"shape": 4, "color": 4}
        self.payload_keys = tuple(payload_keys)
        self.payload_vocabs = dict(payload_vocabs)
        for key in self.payload_keys:
            self.payload_vocabs.setdefault(key, 1)

        self.embeddings = nn.ModuleDict(
            {key: nn.Linear(vocab, hidden_dim) for key, vocab in self.payload_vocabs.items()}
        )
        self.cont_embeds = nn.ModuleDict(
            {key: nn.Linear(1, hidden_dim) for key in self.payload_keys}
        )
        self.fuse = nn.Sequential(
            nn.Linear(sensory_dim + len(self.payload_keys) * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.hidden_dim = hidden_dim
        self.out_dim = hidden_dim

    def payload_features(self, payload: dict[str, Tensor]) -> Tensor:
        """Return ``(B, len(keys) * hidden_dim)`` features from the payload."""
        feats: list[Tensor] = []
        for key in self.payload_keys:
            slot = payload[key]
            if slot.dim() == 1:
                slot = slot.unsqueeze(0)
            if slot.shape[-1] == 1:
                feats.append(self.cont_embeds[key](slot))
            else:
                probs = slot.softmax(dim=-1)
                feats.append(self.embeddings[key](probs))
        return torch.cat(feats, dim=-1)

    def forward(self, sensory: Tensor, payload: dict[str, Tensor]) -> Tensor:
        payload_feat = self.payload_features(payload)
        return self.fuse(torch.cat([sensory, payload_feat], dim=-1))


class _CNNGenerator(nn.Module):
    """Transposed-conv decoder for the TinyConvNeXt encoder."""

    def __init__(self, cond_dim: int, out_channels: int = 3, base_size: int = 4) -> None:
        super().__init__()
        self.base_size = base_size
        self.proj = nn.Linear(cond_dim, 128 * base_size * base_size)
        # Three transposed-conv blocks: 4 -> 8 -> 16 -> 32.
        self.body = nn.Sequential(
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, cond: Tensor) -> Tensor:
        x = self.proj(cond)
        x = x.view(x.shape[0], 128, self.base_size, self.base_size)
        return self.body(x)


class _ViTGenerator(nn.Module):
    """Patch-decoding ViT generator matching the TinyViT encoder.

    Produces a feature grid shaped like the encoder's pre-pool patch grid, then
    decodes each position back to ``patch_size x patch_size`` pixels with a
    shared linear decoder.
    """

    def __init__(
        self,
        cond_dim: int,
        out_channels: int = 3,
        image_size: int = 32,
        patch_size: int = 8,
        dim: int = 64,
        depth: int = 2,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.dim = dim
        self.proj = nn.Linear(cond_dim, dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 2,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.unproj = nn.Linear(dim, patch_size * patch_size * out_channels)

    def forward(self, cond: Tensor) -> Tensor:
        tokens = self.proj(cond).unsqueeze(1).expand(-1, self.num_patches, -1)
        tokens = tokens + self.pos_embed
        tokens = self.encoder(tokens)
        patches = self.unproj(tokens)
        b = cond.shape[0]
        h = self.image_size // self.patch_size
        patches = patches.view(b, h, h, self.patch_size, self.patch_size, -1)
        patches = patches.permute(0, 5, 1, 3, 2, 4).contiguous()
        return patches.view(b, -1, self.image_size, self.image_size)


class _AudioGenerator(nn.Module):
    """1D transposed-conv decoder for the audio modality."""

    def __init__(self, cond_dim: int, out_channels: int = 1, upsample: int = 16) -> None:
        super().__init__()
        self.proj = nn.Linear(cond_dim, 64)
        self.body = nn.Sequential(
            nn.GELU(),
            nn.ConvTranspose1d(64, 32, kernel_size=8, stride=4, padding=2),
            nn.GELU(),
            nn.ConvTranspose1d(32, out_channels, kernel_size=8, stride=4, padding=2),
        )
        self.upsample = upsample

    def forward(self, cond: Tensor) -> Tensor:
        x = self.proj(cond).unsqueeze(-1)
        return self.body(x)


class QualiaDecoder(nn.Module):
    """Decoder that reconstructs perception from ``(sensory, payload)``.

    Args:
        encoder: The encoder whose outputs this decoder consumes. The decoder
            copies ``sensory_dim`` and ``payload_vocabs`` from the encoder to
            stay in sync.
        modality / backbone / in_channels / image_size: Mirrors of the encoder
            settings (inferred from ``encoder`` by default).
    """

    def __init__(
        self,
        encoder: nn.Module | None = None,
        *,
        sensory_dim: int = 64,
        modality: str = "image",
        backbone: str = "cnn",
        in_channels: int = 3,
        image_size: int = 32,
        payload_keys: tuple[str, ...] = PAYLOAD_KEYS,
        payload_vocabs: dict[str, int] | None = None,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        if encoder is not None:
            sensory_dim = getattr(encoder, "sensory_dim", sensory_dim)
            modality = getattr(encoder, "modality", modality)
            backbone = getattr(encoder, "backbone_name", backbone)
            payload_vocabs = dict(getattr(encoder, "payload_vocabs", payload_vocabs or {}))
            payload_keys = tuple(getattr(encoder, "payload_keys", payload_keys))

        if payload_vocabs is None:
            payload_vocabs = {"shape": 4, "color": 4}

        self.sensory_dim = sensory_dim
        self.modality = modality
        self.backbone_name = backbone
        self.in_channels = in_channels
        self.image_size = image_size
        self.payload_keys = tuple(payload_keys)
        self.payload_vocabs = dict(payload_vocabs)

        self.fusion = PayloadFusion(
            sensory_dim=sensory_dim,
            payload_keys=self.payload_keys,
            payload_vocabs=self.payload_vocabs,
            hidden_dim=hidden_dim,
        )

        cond_dim = self.fusion.out_dim
        if modality == "image":
            if backbone == "cnn":
                self.generator: nn.Module = _CNNGenerator(cond_dim, out_channels=in_channels)
            elif backbone == "vit":
                self.generator = _ViTGenerator(
                    cond_dim, out_channels=in_channels, image_size=image_size
                )
            else:
                raise ValueError(f"backbone must be 'cnn' or 'vit'; got {backbone!r}")
        elif modality == "audio":
            self.generator = _AudioGenerator(cond_dim, out_channels=in_channels)
        else:
            raise ValueError(f"modality must be 'image' or 'audio'; got {modality!r}")

    def forward(
        self,
        sensory: Tensor,
        payload: dict[str, Tensor] | None = None,
        *,
        use_payload: bool = True,
    ) -> Tensor:
        """Reconstruct perception.

        Args:
            sensory: ``(B, sensory_dim)`` sensory representation.
            payload: Phenomenal-payload dict. If ``None`` (or
                ``use_payload=False``), the decoder falls back to a sensory-only
                reconstruction, useful as an ablation baseline when measuring
                whether the payload adds information.
            use_payload: If ``False``, ignore ``payload`` even if provided.
        """
        if payload is None or not use_payload:
            empty_payload = {
                key: torch.zeros(
                    sensory.shape[0],
                    self.payload_vocabs[key],
                    device=sensory.device,
                    dtype=sensory.dtype,
                )
                for key in self.payload_keys
            }
            payload = empty_payload
        cond = self.fusion(sensory, payload)
        return self.generator(cond)

    def reconstruct(
        self,
        encoder_out: EncoderOutput,
        *,
        use_payload: bool = True,
    ) -> Tensor:
        """Convenience: decode directly from an :class:`EncoderOutput`."""
        return self.forward(
            encoder_out.sensory,
            encoder_out.payload,
            use_payload=use_payload,
        )


__all__ = ["PayloadFusion", "QualiaDecoder"]
