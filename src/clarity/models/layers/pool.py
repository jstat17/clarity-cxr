import torch as th
from torch import nn
import torch.nn.functional as F
from timm.models._efficientnet_builder import efficientnet_init_weights
from timm.models._efficientnet_blocks import EdgeResidual
from typing import Literal

from .norm import SoftmaxSpatial2d
from .dropout import get_dropout


class LogSumExpPool2DStable(nn.Module):
    """Numerically stable version of Log-Sum-Exp pooling from arXiv:1705.02315.
    r is set to 5 in the original unstable version in arXiv:1411.6228.
    """

    def __init__(self, r: float = 5.0, flatten: bool = True, shape: str = "BCHW") -> None:
        super().__init__()

        self.r = r
        self.shape = shape

        if flatten:
            self.flatten = nn.Flatten()
        else:
            self.flatten = nn.Identity()

    def forward(self, x: th.Tensor) -> th.Tensor:
        if self.shape == "BHWC":
            x = x.permute(0, 3, 1, 2)

        # x shape (B, C, H, W)
        S = x.shape[2] * x.shape[3]
        feature_map_absmax = th.amax(th.abs(x), dim=(2, 3), keepdim=True)

        x = self.r * (x - feature_map_absmax)
        x = th.exp(x)
        x = th.sum(x, dim=(2, 3), keepdim=True)
        x = x / S
        x = x + 1e-38  # add small value to prevent log(0.) = -inf
        x = th.log(x) / self.r
        x = x + feature_map_absmax

        x = self.flatten(x)

        return x


class AdaptiveAvgMaxPool1d(nn.Module):
    def __init__(
        self,
        output_size: int | tuple[int, int],
        alpha: float = 0.5,
        flatten: bool = True,
        shape: str = "BCL",
    ):
        super().__init__()

        # set params
        self.alpha = alpha
        self.shape = shape

        if flatten:
            self.flatten = nn.Flatten()
        else:
            self.flatten = nn.Identity()

        # set pooling components
        self.avg = nn.AdaptiveAvgPool1d(output_size)
        self.max = nn.AdaptiveMaxPool1d(output_size)

    def forward(self, x: th.Tensor) -> th.Tensor:
        if self.shape == "BLC":
            x = x.permute(0, 2, 1)

        # (B, C, L) -> (B, C, 1)
        x = self.alpha * self.avg(x) + (1 - self.alpha) * self.max(x)

        x = self.flatten(x)

        return x


class AdaptiveAvgMaxPool2d(nn.Module):
    def __init__(
        self,
        output_size: int | tuple[int, int],
        alpha: float = 0.5,
        flatten: bool = True,
        shape: str = "BCHW",
    ):
        super().__init__()

        # set params
        self.alpha = alpha
        self.shape = shape

        if flatten:
            self.flatten = nn.Flatten()
        else:
            self.flatten = nn.Identity()

        # set pooling components
        self.avg = nn.AdaptiveAvgPool2d(output_size)
        self.max = nn.AdaptiveMaxPool2d(output_size)

    def forward(self, x: th.Tensor) -> th.Tensor:
        if self.shape == "BHWC":
            x = x.permute(0, 3, 1, 2)

        # (B, C, H, W) -> (B, C, 1, 1)
        x = self.alpha * self.avg(x) + (1 - self.alpha) * self.max(x)

        x = self.flatten(x)

        return x


class AttentionPool1d(nn.Module):
    def __init__(
        self,
        num_classes: int,
        embed_dim: int,
        dropout: Literal["", "pixel", "channel", "pixelchannel"] = "pixelchannel",
        drop_rate: float = 0.0,
    ):
        super().__init__()

        # learnable attention weights for each class, shape (C, L)
        self.attention_weights = nn.Parameter(th.randn(num_classes, embed_dim))

        # set dropout
        self.dropout = get_dropout(dropout, drop_rate)

    def forward(self, x: th.Tensor) -> th.Tensor:
        """Perform learned attention pooling.

        Args:
            x (th.Tensor): Input tensor, shape (B, C, L).

        Returns:
            th.Tensor: Pooled output, shape (B, C).
        """
        # apply dropout to attention weights
        attention_weights = self.dropout(self.attention_weights)

        # compute attention probabilities: softmax over embedding dimension (L)
        attention_probs = F.softmax(attention_weights, dim=1)  # (C, L)

        # compute weighted sum for each class token
        # x: (B, C, L), attention_probs: (C, L) -> (B, C)
        x = th.einsum("bcd,cd->bc", x, attention_probs)

        return x


class AttentionPool2d(AttentionPool1d):
    def __init__(self, num_classes: int, size: tuple[int, int], drop_rate: float = 0.0):
        # get flattened embedding dim
        embed_dim = size[0] * size[1]

        super().__init__(num_classes, embed_dim, drop_rate)

    def forward(self, x: th.Tensor) -> th.Tensor:
        # flatten tensor
        B, C, H, W = x.shape
        x = x.view(B, C, H * W)

        # apply attention weights
        x = super().forward(x)

        # reshape 2d
        x = x.view(B, C, H, W)

        return x


class ConvAttentionPool2d(nn.Module):
    def __init__(
        self,
        chs: int,
        exp_ratio: float = 1.0,
    ):
        super().__init__()

        # create conv
        self.conv = EdgeResidual(
            in_chs=chs,
            out_chs=chs,
            exp_ratio=exp_ratio,
            act_layer=nn.SiLU,
        )

        # create spatial softmax
        self.softmax = SoftmaxSpatial2d()

        efficientnet_init_weights(self)

    def forward(self, x: th.Tensor) -> th.Tensor:
        # apply conv and spatial softmax to get attention weights
        attention_weights = self.softmax(self.conv(x.clone()))
        # element-wise multiply and sum spatially:
        # (B, C, H, W) -> (B, C)
        x = th.sum(x * attention_weights, dim=(2, 3))

        return x


class MaskedAvgPool2d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: th.Tensor, mask: th.Tensor) -> th.Tensor:
        # cast mask to float
        mask = mask.to(th.float32)
        # get number of non-zeroed pixels
        n_nonzero = th.sum(mask, dim=(2, 3))  # (B, C)
        # get feature map sum
        feature_sum = th.sum(x, dim=(2, 3))  # (B, C)
        # get average
        x = feature_sum / (n_nonzero + 1)
        # replace NaN
        x = th.nan_to_num(x, nan=0.0, posinf=0.0)

        return x
