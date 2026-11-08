import torch as th
from torch import nn
import torch.nn.functional as F
from timm.models._efficientnet_blocks import EdgeResidual


class ProgressiveFusionHead(nn.Module):
    """Fused multi-stage heatmap head and upsampler.

    Maps (in_channels, in_size) -> (num_classes, in_size * upsample) in a
    single module, combining the roles of the heatmap head and the feature
    upsampler. Structure:

      - main path: n_stages = log2(upsample) - 1 stages of
        [bilinear x2 -> conv3x3 -> SiLU -> BatchNorm], with channel taper
        w_1 = max(4C, 64), w_k = max(3C, 48), ..., final stage outputs
        num_classes at in_size * upsample / 2 (all learned computation runs
        at or below half the target resolution);
      - low-res skip: 1x1 conv in_channels -> num_classes on the input
        tokens, bilinearly resampled by upsample/2 and added at the
        second-to-last resolution, restoring global class context that the
        narrowed main path loses (the 'F' variant of the fused-head study);
      - finish: plain bilinear x2 (parameter-free) + sigmoid.

    The design follows the progressive-refinement pattern of feature-pyramid
    decoders (FPN-style lateral skip + staged refinement); the closest named
    relatives are the progressive upsampling modules of fast super-resolution
    decoders and FPN-hybrid segmentation heads.

    Args:
        in_channels: input feature channels (e.g. ViT embed_dim = 384).
        num_classes: output map channels (C); all widths derive from C.
        in_size: input spatial size (G, G), G = img_size // patch_size.
        upsample: total spatial upsample factor; must be a power of two >= 2
            (e.g. 16 for both 224/16 and 512/16 models).
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        in_size: tuple[int, int],
        upsample: int,
        act_layer: nn.Module = nn.SiLU,
    ) -> None:
        super().__init__()

        # validate upsample factor
        if upsample < 2 or (upsample & (upsample - 1)) != 0:
            raise ValueError(f"upsample must be a power of two >= 2, got {upsample}")

        # store params
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.in_size = in_size
        self.upsample = upsample

        # output resolution
        self.out_size = (int(self.in_size[0] * self.upsample), int(self.in_size[1] * self.upsample))
        self.out_channels = num_classes

        # number of conv stages: all but the final plain x2
        n_stages = (upsample.bit_length() - 1) - 1  # log2(S) - 1

        # channel taper: first stage keeps the widest layer, tapering to C
        # widths benchmarked at C=14: [64, 48, 14]
        widths = []
        w = max(4 * num_classes, 64)
        for i in range(n_stages):
            if i == n_stages - 1:
                widths.append(num_classes)
            elif i == 0:
                widths.append(w)
            else:
                w = max(3 * num_classes, 48)
                widths.append(w)

        # create main-path stages: bilinear x2 -> conv3x3 -> SiLU -> BN
        stages = []
        cin = in_channels
        for cout in widths:
            stages.append(
                nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), act_layer(), nn.BatchNorm2d(cout))
            )
            cin = cout
        self.stages = nn.ModuleList(stages)
        self.stage_upsample = 2

        # low-res skip: 1x1 projection to num_classes on the raw tokens
        self.skip = nn.Conv2d(in_channels, num_classes, 1)

        # finishing factor (the one plain, parameter-free x2)
        self.finish_upsample = 2

    def get_outdim(self) -> tuple[tuple[int, int], int]:
        return self.out_size, self.out_channels

    def forward(self, x: th.Tensor) -> th.Tensor:
        # low-res skip: project raw tokens to class maps and enlarge
        skip = resample(
            self.skip(x), size=(self.out_size[0] // 2, self.out_size[1] // 2), mode="bilinear"
        )

        # main path: staged refinement at <= half the target resolution
        half = (self.out_size[0] // 2, self.out_size[1] // 2)
        for stage in self.stages:
            if x.shape[-2] < half[0] or x.shape[-1] < half[1]:
                x = resample(
                    x,
                    size=(min(x.shape[-2] * 2, half[0]), min(x.shape[-1] * 2, half[1])),
                    mode="bilinear",
                )
            x = stage(x)

        # fuse skip and finish at image resolution
        x = x + skip
        x = resample(x, size=self.out_size, mode="bilinear")

        return x.sigmoid()


class ConvBilinearUpsample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        in_size: tuple[int, int],
        sampling_rate: float = 2.0,
        exp_ratio: float = 1.0,
        act_layer: nn.Module = nn.SiLU,
    ) -> None:
        super().__init__()

        # store params
        self.in_channels = in_channels
        self.in_size = in_size
        self.sampling_rate = sampling_rate

        # set upsampled spatial resolution, and downsampled depth
        self.out_size = (
            int(self.in_size[0] * self.sampling_rate),
            int(self.in_size[1] * self.sampling_rate),
        )
        self.out_channels = int(self.in_channels / self.sampling_rate)

        # create linear depth-downsampler for skip features
        self.depth_downsampler_1x1 = nn.Conv2d(
            in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1
        )

        # create conv depth-downsampler
        self.conv = nn.Sequential(
            EdgeResidual(
                in_chs=self.in_channels,
                out_chs=self.out_channels,
                exp_ratio=exp_ratio,
                act_layer=act_layer,
                noskip=True,
                group_size=self.in_channels,
            ),
            nn.BatchNorm2d(self.out_channels),
        )

        # create post-residual normalization
        self.norm = nn.BatchNorm2d(self.out_channels)

    def get_outdim(self) -> tuple[tuple[int, int], int]:
        return self.out_size, self.out_channels

    def forward_features(self, x: th.Tensor) -> th.Tensor:
        # bilinear upsample feature map
        # [sampling_rate = S]
        # (B, C, H, W) -> (B, C, S*H, S*W)
        x = resample(x, size=self.out_size, mode="bilinear")

        # clone feature map and get skip features
        # (B, C, S*H, S*W) -> (B, C/S, S*H, S*W)
        skip = self.depth_downsampler_1x1(x.clone())

        # apply conv and normalize
        # (B, C, S*H, S*W) -> (B, C/S, S*H, S*W)
        x = self.conv(x) + skip
        x = self.norm(x)

        return x

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.forward_features(x)
        return x


class PixelShuffleUpsample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        in_size: tuple[int, int],
        out_channels_frac: float | None = None,
        sampling_rate: int = 2,
        kernel_size: int = 3,
        **kwargs,
    ):
        super().__init__()

        # set out_channels default
        if out_channels_frac is None:
            out_channels_frac = 1 / (sampling_rate**2)

        # set params
        self.in_channels = in_channels
        self.in_size = in_size
        self.out_channels = int(self.in_channels * out_channels_frac)
        self.sampling_rate = sampling_rate
        self.out_size = (self.in_size[0] * self.sampling_rate, self.in_size[1] * self.sampling_rate)
        self.kernel_size = kernel_size
        self.intermediate_channels = int(self.out_channels * (self.sampling_rate**2))

        # create convolution to increase from in_channels to intermediate_channels
        self.conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.intermediate_channels,
            kernel_size=self.kernel_size,
            padding="same",
        )

        # create pixel shuffle
        self.pixel_shuffle = nn.PixelShuffle(self.sampling_rate)

        # create activation and batch norm
        self.act = nn.SiLU()
        self.bn = nn.BatchNorm2d(self.out_channels)

    def get_outdim(self) -> tuple[tuple[int, int], int]:
        return self.out_size, self.out_channels

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.act(x)
        x = self.bn(x)
        return x


def resample(x: th.Tensor, size: tuple[int, int], mode: str) -> th.Tensor:
    x = F.interpolate(x, size=size, mode=mode)

    return x
