"""Shared fixtures for the test suite.

These fixtures keep tests concise and ensure that smoke and full tests share the
same canonical "tiny" configuration (small image, small batch, few steps, few
parameters) so that smoke stays under the 60-second budget even on cold CPU.
"""

from __future__ import annotations

import pytest
import torch
from qualia.model.decoder import QualiaDecoder
from qualia.model.encoder import PAYLOAD_KEYS, QualiaEncoder
from qualia.model.phenomenal_state import PAYLOAD_KEYS as PS_PAYLOAD_KEYS
from qualia.model.phenomenal_state import PhenomenalState

IMG_SIZE = 32
SENSORY_DIM = 32
WORKSPACE_DIM = 16
SELF_MODEL_DIM = 16
PAYLOAD_SLOTS = 8
BATCH = 4
VOCABS: dict[str, int] = {"shape": 4, "color": 4}


@pytest.fixture(scope="session")
def device() -> torch.device:
    """Always CPU for smoke tests; no GPU required."""
    return torch.device("cpu")


@pytest.fixture()
def encoder_decoder() -> tuple[QualiaEncoder, QualiaDecoder]:
    """The smallest encoder + decoder pair: CNN backbone, 32x32 RGB, 4 classes."""
    torch.manual_seed(0)
    encoder = QualiaEncoder(
        modality="image",
        backbone="cnn",
        in_channels=3,
        image_size=IMG_SIZE,
        sensory_dim=SENSORY_DIM,
        payload_keys=PAYLOAD_KEYS,
        payload_vocabs=VOCABS,
    )
    decoder = QualiaDecoder(encoder=encoder)
    return encoder, decoder


@pytest.fixture()
def phenomenal_state() -> PhenomenalState:
    """A tiny PhenomenalState matching the encoder's sensory dim."""
    torch.manual_seed(0)
    return PhenomenalState(
        percept_dim=SENSORY_DIM,
        workspace_dim=WORKSPACE_DIM,
        self_model_dim=SELF_MODEL_DIM,
        payload_slots=PAYLOAD_SLOTS,
        payload_keys=PS_PAYLOAD_KEYS,
    )


@pytest.fixture()
def tiny_batch() -> torch.Tensor:
    """A 4-sample batch of 32x32 RGB images in [-1, 1]."""
    torch.manual_seed(1)
    return torch.rand(BATCH, 3, IMG_SIZE, IMG_SIZE) * 2 - 1
