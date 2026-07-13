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
from .predictive_loop import (
    GlobalWorkspace,
    PredictiveCodingLoop,
    PredictiveTrajectory,
    predictive_coding_loss,
)
from .self_model import (
    REPORT_SLOT_NAMES,
    SelfModel,
    SelfModelOutput,
    SelfReport,
    contrastive_report_loss,
    contrastive_report_loss_pairwise,
    report_consistency_loss,
)

__all__ = [
    "EncoderOutput",
    "GlobalWorkspace",
    "PAYLOAD_KEYS",
    "PayloadFusion",
    "PayloadHead",
    "PhenomenalState",
    "PhenomenalStateRecord",
    "PredictiveCodingLoop",
    "PredictiveTrajectory",
    "QualiaDecoder",
    "QualiaEncoder",
    "REPORT_SLOT_NAMES",
    "SensoryHead",
    "SelfModel",
    "SelfModelOutput",
    "SelfReport",
    "TinyConvNeXt",
    "TinyViT",
    "contrastive_report_loss",
    "contrastive_report_loss_pairwise",
    "predictive_coding_loss",
    "report_consistency_loss",
]
