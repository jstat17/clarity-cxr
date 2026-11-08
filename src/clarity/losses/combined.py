"""The combined CLARiTy loss from the paper.

The paper writes the objective as

    L = alpha_C * L_CLS^C
        + alpha_P * L_CLS^P
        + alpha_f * L_CLS^f
        + beta * L_BAS
        + gamma * L_Dice

where ``L_CLS`` is the weighted binary cross-entropy (WBCE), ``L_BAS`` is the
background activation suppression loss, and ``L_Dice`` is the soft Dice loss.
The coefficient names mirror the paper symbols (alpha_C, alpha_P, alpha_f,
beta, gamma); the classification loss can be swapped to MLSM for the paper's
classification-loss ablation.
"""

from __future__ import annotations

import textwrap

import torch as th
from torch import nn

from .registry import get_loss, register_loss


@register_loss("combined")
class CombinedLoss(nn.Module):
    """The paper's combined CLARiTy loss.

    Args:
        cls_loss: classification-loss registry name; the paper uses ``wbce``
            and reports an ``mlsm`` ablation.
        alpha_C: weight of the class-token classification loss
            :math:`\\alpha_C`.
        alpha_P: weight of the patch-token classification loss
            :math:`\\alpha_P`.
        alpha_f: weight of the foreground classification loss
            :math:`\\alpha_f`.
        beta: weight of the background activation suppression loss
            :math:`\\beta`.
        gamma: weight of the soft Dice loss :math:`\\gamma`.
        reduction: reduction used by the component losses.
        **kwargs: ``cls_*`` entries are forwarded to the classification loss
            (e.g. ``cls_pred_background``); other entries are ignored so the
            training setup can pass generic loss arguments.
    """

    def __init__(
        self,
        cls_loss: str = "wbce",
        alpha_C: float = 2.2674146440622804,
        alpha_P: float = 2.4677773704169415,
        alpha_f: float = 3.967003879741848,
        beta: float = 0.4376612021802623,
        gamma: float = 0.1426684236243471,
        reduction: str = "mean",
        **kwargs,
    ) -> None:
        super().__init__()

        # collect classification-loss options; generic setup arguments such
        # as ``labels``/``device`` are intentionally not forwarded
        cls_kwargs = {}
        for key, value in kwargs.items():
            if key.startswith("cls_"):
                cls_kwargs[key[4:]] = value
        if "pred_background" in kwargs:
            cls_kwargs.setdefault("pred_background", kwargs["pred_background"])

        self.cls_loss_name = cls_loss
        self.cls_loss = get_loss(cls_loss, **cls_kwargs)
        self.bas_loss = get_loss("bas", reduction=reduction)
        self.dice_loss = get_loss("dice", reduction=reduction)
        self.alpha_C = alpha_C
        self.alpha_P = alpha_P
        self.alpha_f = alpha_f
        self.beta = beta
        self.gamma = gamma
        self.reduction = reduction

    def __str__(self) -> str:
        s = textwrap.dedent(f"""\
                CombinedLoss(
                    cls_loss={self.cls_loss},
                    alpha_C={self.alpha_C},
                    alpha_P={self.alpha_P},
                    alpha_f={self.alpha_f},
                    beta={self.beta},
                    gamma={self.gamma},
                    reduction={self.reduction!r}
                )""")

        return s

    def dict_repr(self) -> dict[str, dict]:
        d = {
            "CombinedLoss": {
                "cls_loss": self.cls_loss.dict_repr(),
                "cls_loss_name": self.cls_loss_name,
                "alpha_C": self.alpha_C,
                "alpha_P": self.alpha_P,
                "alpha_f": self.alpha_f,
                "beta": self.beta,
                "gamma": self.gamma,
                "reduction": self.reduction,
            }
        }

        return d

    def forward(
        self,
        inputs: tuple[tuple[th.Tensor, ...], tuple[th.Tensor, ...]],
        targets: tuple[th.Tensor, th.Tensor],
    ) -> th.Tensor:
        # logits: (class token, patch token, foreground) or (patch, foreground)
        logits = inputs[0]
        match len(logits):
            case 3:
                logits_class, logits_patch, logits_foreground = logits
            case 2:
                logits_class = None
                logits_patch, logits_foreground = logits
            case _:
                raise ValueError(
                    "combined loss expects two or three logit components, "
                    f"got {len(logits)}"
                )

        # SegmentCAM outputs: heatmaps, patch-token logits, background logits
        segment_outputs = inputs[1]
        if segment_outputs is None:
            heatmaps, logits_patch_bas, logits_background = None, None, None
        else:
            if len(segment_outputs) < 3:
                raise ValueError(
                    "combined loss expects heatmaps, patch-token activations "
                    "and background activations from SegmentCAM"
                )
            heatmaps, logits_patch_bas, logits_background = segment_outputs[:3]

        labels, segmentations = targets

        terms: list[th.Tensor] = []

        # classification losses: class token, patch token and foreground
        if logits_class is not None:
            terms.append(self.alpha_C * self.cls_loss(logits_class, labels))
        if logits_patch is not None:
            terms.append(self.alpha_P * self.cls_loss(logits_patch, labels))
        if logits_foreground is not None:
            terms.append(self.alpha_f * self.cls_loss(logits_foreground, labels))

        # background activation suppression (only for SegmentCAM runs)
        if self.beta != 0.0 and logits_patch_bas is not None and logits_background is not None:
            terms.append(self.beta * self.bas_loss(logits_patch_bas, logits_background, labels))

        # soft Dice segmentation loss at the heatmap/segmentation resolution
        if self.gamma != 0.0 and heatmaps is not None and segmentations is not None:
            if heatmaps.shape[-2:] != segmentations.shape[-2:]:
                raise ValueError(
                    "Dice is computed directly at the heatmap resolution; got "
                    f"heatmaps {tuple(heatmaps.shape[-2:])} and segmentations "
                    f"{tuple(segmentations.shape[-2:])}"
                )
            terms.append(self.gamma * self.dice_loss(heatmaps, segmentations))

        if not terms:
            raise ValueError("combined loss received no active loss components")

        return sum(terms)
