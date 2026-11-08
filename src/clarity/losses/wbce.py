import torch as th
import torch.nn as nn
import torch.nn.functional as F
import textwrap

from clarity.losses.utils import get_binary_batch_weights
from clarity.losses.registry import register_loss


@register_loss("wbce")
class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, pred_background: bool = False, **kwargs) -> None:
        super().__init__()

        self.pred_background = pred_background

    def __str__(self) -> str:
        s = textwrap.dedent(f"""\
                WeightedBCEWithLogitsLoss(
                    pred_background={self.pred_background}
                )""")

        return s

    def dict_repr(self) -> dict[str, dict]:
        d = {"WeightedBCEWithLogitsLoss": {"pred_background": self.pred_background}}

        return d

    def forward(self, inputs: th.Tensor, targets: th.Tensor) -> th.Tensor:
        # create a tensor of weights
        weights = get_binary_batch_weights(targets, pred_background=self.pred_background)

        # calculate the loss
        loss = F.binary_cross_entropy_with_logits(inputs, targets, weight=weights, reduction="mean")

        return loss
