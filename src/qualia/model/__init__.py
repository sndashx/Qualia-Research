"""Model modules: Encoder, Decoder, PredictiveLoop, PhenomenalState, GlobalWorkspace, SelfModel.

Public modules:
  - :class:`qualia.model.phenomenal_state.PhenomenalState`
  - :class:`qualia.model.encoder.QualiaEncoder` (and backbones / heads)
  - :class:`qualia.model.decoder.QualiaDecoder`
"""

from .decoder import PayloadFusion, QualiaDecoder
from .encoder import (
    PAYLOAD_KEYS,
    EncoderOutput,
    PayloadHead,
    QualiaEncoder,
    SensoryHead,
    TinyConvNeXt,
    TinyViT,
)
from .phenomenal_state import PhenomenalState, PhenomenalStateRecord

__all__ = [
    "EncoderOutput",
    "PAYLOAD_KEYS",
    "PayloadFusion",
    "PayloadHead",
    "PhenomenalState",
    "PhenomenalStateRecord",
    "QualiaDecoder",
    "QualiaEncoder",
    "SensoryHead",
    "TinyConvNeXt",
    "TinyViT",
]
