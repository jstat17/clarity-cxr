"""Soft Dice segmentation loss (paper Eq. L_Dice)."""

from __future__ import annotations

import textwrap

import torch as th
from torch import nn

from .registry import register_loss
from .utils import reduce


@register_loss("dice")
class SoftDiceLoss(nn.Module):
    """Soft Dice loss on heatmap planes against binary segmentation masks.

    The paper computes the loss directly at the image resolution, per
    ``(image, class)`` plane, over the set ``Omega`` of planes whose
    ground-truth mask contains at least one foreground pixel:

        L_Dice = 1 - 1/|Omega| * sum_{c in Omega}
                 (2 * sum H_c S_c + eps) / (sum H_c + sum S_c + eps)

    with ``eps = 1e-6``. Planes with an empty target do not contribute.
    """

    def __init__(self, reduction: str = "mean", **kwargs) -> None:
        super().__init__()

        self.reduction = reduction

    def __str__(self) -> str:
        s = textwrap.dedent(f"""\
                SoftDiceLoss(
                    reduction={self.reduction}
                )""")

        return s

    def dict_repr(self) -> dict[str, dict]:
        d = {"SoftDiceLoss": {"reduction": self.reduction}}

        return d

    def forward(self, heatmaps: th.Tensor, segmentations: th.Tensor) -> th.Tensor:
        # heatmap planes in [0, 1] (raw sigmoid output of the heatmap head);
        # segmentations are boolean masks of the same shape (B, C, H, W)
        targets = segmentations.float()

        # per-plane intersection and cardinality terms (B, C)
        intersection = th.sum(heatmaps * targets, dim=(2, 3))
        cardinality = th.sum(heatmaps, dim=(2, 3)) + th.sum(targets, dim=(2, 3))

        # soft Dice coefficient per plane; epsilon keeps the division finite
        # in backward for empty/zero planes (their forward value is masked out)
        dice = (2.0 * intersection + 1e-6) / (cardinality + 1e-6)

        # only planes whose target contains at least one foreground pixel
        # contribute to the average over Omega
        plane_has_target = th.sum(targets, dim=(2, 3)) > 0  # (B, C)
        selected = dice[plane_has_target]
        if selected.numel() == 0:
            return th.sum(heatmaps) * 0.0

        return 1.0 - reduce(selected, self.reduction)
