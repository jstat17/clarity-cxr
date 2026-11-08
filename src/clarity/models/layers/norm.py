import torch as th
from torch import nn
import torch.nn.functional as F


class Norm2d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tensor: th.Tensor) -> th.Tensor:
        # get max tensor across 2D feature maps
        # (B, C, H, W) -> (B, C, 1, 1)
        tensor_max = th.amax(tensor, dim=(2, 3), keepdim=True)

        # get min tensor across 2D feature maps
        # (B, C, H, W) -> (B, C, 1, 1)
        tensor_min = th.amin(tensor, dim=(2, 3), keepdim=True)

        # get denominator
        scale = tensor_max - tensor_min
        # Min-max normalization with epsilon softening:
        #   normalized = (t - min + eps) / (scale + eps)
        # Healthy planes (scale >> eps) are virtually unchanged. Constant
        # (sigmoid-saturated) planes map to 1.0 instead of 0: a constant-zero
        # output is the degenerate optimum of the area/confinement losses and
        # creates a run-away basin where every heatmap plane collapses and
        # the foreground-logit path is severed. A value of
        # 1.0 makes saturation maximally unattractive while remaining fully
        # differentiable (no hard th.where switch, no zero-gradient dead
        # zones, no 1/0 in the backward pass).
        eps = 1e-2
        tensor = (tensor - tensor_min + eps) / (scale + eps)
        tensor = tensor.clamp(min=0.0, max=1.0)

        return tensor


class SoftmaxSpatial2d(nn.Module):
    def __init__(self, shape: str = "BCHW"):
        super().__init__()

        # set params
        self.shape = shape

    def forward(self, x: th.Tensor) -> th.Tensor:
        if self.shape == "BHWC":
            x = x.permute(0, 3, 1, 2)

        # get tensor shape
        B, C, H, W = x.shape

        # reshape tensor
        x = x.view(B, C, H * W)
        # apply softmax spatially
        x = F.softmax(x, dim=2)
        # reshape back
        x = x.view(B, C, H, W)

        return x
