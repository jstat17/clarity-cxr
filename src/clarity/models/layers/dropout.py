import torch as th
from torch import nn
from torch.nn import functional as F
from typing import Literal


class ChannelDropout(nn.Dropout2d):
    """``nn.Dropout2d`` that keeps working on inputs without two spatial dims.

    Recent PyTorch versions emit a deprecation warning when ``nn.Dropout2d``
    receives inputs that are not 3-D/4-D. The historical behaviour for those
    ranks is preserved here, warning-free:

    - 2-D input ``[N, C]``: the op degenerates to element-wise dropout, so
      ``F.dropout`` is applied directly (identical to the deprecated path);
    - 3-D input ``[N, C, L]``: historical ``nn.Dropout2d`` performs 1-D
      channel-wise dropout, so ``F.dropout1d`` is applied directly;
    - any other rank is delegated to ``nn.Dropout2d`` unchanged.
    """

    def forward(self, input: th.Tensor) -> th.Tensor:
        if input.dim() == 2:
            return F.dropout(input, self.p, self.training, self.inplace)
        if input.dim() == 3:
            return F.dropout1d(input, self.p, self.training, self.inplace)
        return super().forward(input)


def get_dropout(
    dropout: Literal["", "pixel", "channel", "pixelchannel"], drop_rate: float
) -> nn.Module:
    # set dropout
    match dropout:
        case "":
            dropout = nn.Identity()
        case "pixel":
            dropout = nn.Dropout(drop_rate)
        case "channel":
            dropout = ChannelDropout(drop_rate)
        case "pixelchannel":
            eff_drop_rate = 1 - (1 - drop_rate) ** 0.5
            dropout = nn.Sequential(*[nn.Dropout(eff_drop_rate), ChannelDropout(eff_drop_rate)])

    return dropout
