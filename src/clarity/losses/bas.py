"""Background activation suppression loss (paper Eq. L_BAS)."""

from __future__ import annotations

import textwrap

import torch as th
import torch.nn.functional as F
from torch import nn

from .registry import register_loss


@register_loss("bas")
class BackgroundActivationSuppressionLoss(nn.Module):
    """Background activation suppression, adapted to multilabel targets.

    The paper applies the loss only to positive classes and uses softplus
    instead of ReLU so the ratio stays finite and differentiable:

        L_BAS = 1 / max(sum_c y_c, 1)
                * sum_c y_c * softplus(l_b^c) / (softplus(l_P^c) + eps)

    where ``l_b`` are the background logits and ``l_P`` the patch-token logits.
    """

    def __init__(self, epsilon: float = 1e-8, reduction: str = "mean", **kwargs) -> None:
        super().__init__()

        self.epsilon = epsilon
        self.reduction = reduction

    def __str__(self) -> str:
        s = textwrap.dedent(f"""\
                BackgroundActivationSuppressionLoss(
                    epsilon={self.epsilon},
                    reduction={self.reduction}
                )""")

        return s

    def dict_repr(self) -> dict[str, dict]:
        d = {
            "BackgroundActivationSuppressionLoss": {
                "epsilon": self.epsilon,
                "reduction": self.reduction,
            }
        }

        return d

    def forward(
        self, activations: th.Tensor, activations_background: th.Tensor, labels: th.Tensor
    ) -> th.Tensor:
        # paper ratio: softplus(background) / softplus(patch) + epsilon
        ratio = F.softplus(activations_background) / (
            F.softplus(activations) + self.epsilon
        )

        # apply the loss only to positive classes
        mask = (labels > 0.5).to(ratio.dtype)
        ratio = ratio * mask

        match self.reduction:
            case "mean":
                # paper normalisation: 1 / max(sum_c y_c, 1)
                n = th.clamp(mask.sum(), min=1.0)
                return ratio.sum() / n
            case "sum":
                return ratio.sum()
            case "none":
                return ratio
            case _:
                raise ValueError(
                    f"unsupported reduction {self.reduction!r}; "
                    "expected 'mean', 'sum' or 'none'"
                )
