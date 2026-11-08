"""External classification baselines for the transfer study.

Comparators CLARiTy is measured against: the published TorchXRayVision
DenseNet121-NIH checkpoint loaded unchanged, and RAD-DINO as a frozen encoder
behind a linear probe, scored on the study's own split and label space with the
study's own metric. :mod:`clarity.external.localization` reads both through HiResCAM
so they can be localized by the same OneBox evaluation.
"""

from clarity.external.checkpoints import (
    XRV_CHECKPOINTS,
    XrvCheckpoint,
    load_xrv_model,
    xrv_checkpoint,
)
from clarity.external.evaluate import (
    build_arm_document,
    evaluate_xrv,
    label_matrix,
    load_split_filenames,
    split_sha256,
    study_metrics,
)
from clarity.external.labels import (
    XRV_LABEL_MAP,
    finding_indices,
    map_to_study_labels,
)
from clarity.external.preprocess import XRVPreprocessor
from clarity.external.probe import (
    RADDINO_C_GRID,
    RaddinoProbe,
    build_raddino_arm_document,
    evaluate_raddino_probe,
)
from clarity.external.raddino import (
    RADDINO_MODEL_ID,
    RADDINO_RESOLUTION,
    RaddinoEncoder,
    load_raddino,
)

__all__ = [
    "RADDINO_C_GRID",
    "RADDINO_MODEL_ID",
    "RADDINO_RESOLUTION",
    "RaddinoEncoder",
    "RaddinoProbe",
    "XRV_CHECKPOINTS",
    "XRV_LABEL_MAP",
    "XRVPreprocessor",
    "XrvCheckpoint",
    "build_arm_document",
    "build_raddino_arm_document",
    "evaluate_raddino_probe",
    "evaluate_xrv",
    "finding_indices",
    "label_matrix",
    "load_split_filenames",
    "load_raddino",
    "load_xrv_model",
    "map_to_study_labels",
    "split_sha256",
    "study_metrics",
    "xrv_checkpoint",
]
