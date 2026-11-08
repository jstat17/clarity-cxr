"""Names of the transfer-study evaluation arms.

The checkpoint is always the one trained on ChestX-ray14 (NIH). The two arms
differ only in where the OneBox localization calibration (heatmap thresholds
and box parameters) comes from:

* ``A1`` -- **NIH-calibrated**: the calibration fitted on the NIH validation
  split is frozen and applied unchanged to the target dataset (strict
  zero-shot; ``--frozen-onebox-calibration`` in ``clarity-evaluate``).
* ``A2`` -- **Target-calibrated**: the OneBox calibration is refitted on the
  validation split of the target dataset (VinDr-CXR or RSNA Pneumonia).

The short ids ``A1``/``A2`` are used in file names and tables; the long names
are the ones used in the paper.
"""

from __future__ import annotations

ARM_A1 = "A1"
ARM_A2 = "A2"

#: short id -> paper name
ARM_NAMES = {
    ARM_A1: "NIH-calibrated",
    ARM_A2: "Target-calibrated",
}

#: paper name -> short id
ARM_IDS = {name: arm for arm, name in ARM_NAMES.items()}


def arm_label(arm: str) -> str:
    """Return ``"A1 (NIH-calibrated)"`` style text for an arm id."""
    return f"{arm} ({ARM_NAMES[arm]})"
