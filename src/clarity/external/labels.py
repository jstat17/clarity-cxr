"""Map a published model's label space onto the study's 11 VinDr findings.

Derived by name from the checkpoint's own label list, never from hard-coded
indices: the three checkpoints differ (`densenet121-res224-nih` leaves four slots
blank, `resnet50-res512-all` fills all eighteen) and a silent mis-index would
produce a plausible but wrong row.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch as th

from clarity.data.vindr import VINDR_DISEASES

#: our finding -> the source model's label name(s); two names are fused with max
XRV_LABEL_MAP: dict[str, tuple[str, ...]] = {
    "Atelectasis": ("Atelectasis",),
    "Cardiomegaly": ("Cardiomegaly",),
    "Consolidation": ("Consolidation",),
    "Edema": ("Edema",),
    "Pleural effusion": ("Effusion",),
    "Emphysema": ("Emphysema",),
    "Pulmonary fibrosis": ("Fibrosis",),
    "Infiltration": ("Infiltration",),
    "Nodule/Mass": ("Nodule", "Mass"),
    "Pleural thickening": ("Pleural_Thickening",),
    "Pneumothorax": ("Pneumothorax",),
}


def finding_indices(labels: Sequence[str]) -> dict[str, tuple[int, ...]]:
    """`finding -> positions in `labels``, in the study's finding order."""
    positions: dict[str, tuple[int, ...]] = {}
    missing: list[str] = []
    for finding in VINDR_DISEASES:
        wanted = XRV_LABEL_MAP[finding]
        found = tuple(index for index, label in enumerate(labels) if label in wanted)
        # every declared member must be present: a fused max over one of two rows
        # silently changes what the finding means
        if len(found) != len(set(wanted)):
            missing.append(finding)
            continue
        positions[finding] = found
    if missing:
        raise ValueError(
            f"the checkpoint does not emit every member of {missing} "
            f"(needs {[XRV_LABEL_MAP[m] for m in missing]}); it emits {list(labels)}"
        )
    return positions


def map_to_study_labels(outputs: th.Tensor, labels: Sequence[str]) -> th.Tensor:
    """Reduce a `[..., classes, ...]` output to the 11 findings (class axis 1).

    The one many-to-one entry (`Nodule/Mass`) is fused with `max`, the same rule
    A1 (NIH-calibrated) and A2 (Target-calibrated) apply at the model-output boundary; single-member findings pass through.
    """
    if outputs.ndim < 2:
        raise ValueError(f"expected at least 2 dimensions, got {tuple(outputs.shape)}")
    if outputs.shape[1] != len(labels):
        raise ValueError(
            f"expected {len(labels)} classes per the checkpoint's label list, "
            f"got {outputs.shape[1]}"
        )
    positions = finding_indices(labels)

    pieces = []
    for finding in VINDR_DISEASES:
        indices = positions[finding]
        if len(indices) == 1:
            pieces.append(outputs[:, indices[0]].unsqueeze(1))
        else:
            stacked = th.stack([outputs[:, index] for index in indices], dim=1)
            pieces.append(stacked.max(dim=1, keepdim=True).values)
    return th.cat(pieces, dim=1)
