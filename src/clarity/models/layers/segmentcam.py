import torch as th
from torch import nn
import torch.nn.functional as F
from timm.models._efficientnet_builder import efficientnet_init_weights
from timm.models._efficientnet_blocks import EdgeResidual
from timm.layers.typing import LayerType
from typing import Literal

from .upsample import ConvBilinearUpsample, PixelShuffleUpsample, ProgressiveFusionHead, resample
from .pool import LogSumExpPool2DStable, AdaptiveAvgMaxPool2d, ConvAttentionPool2d, MaskedAvgPool2d
from .norm import Norm2d, SoftmaxSpatial2d
from .dropout import get_dropout
from .init import _init_weights

SegmentCAMReturnType = tuple[tuple[th.Tensor, th.Tensor], tuple[th.Tensor, th.Tensor, th.Tensor]]


class SegmentCAM(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        in_size: tuple[int, int],
        upsample_method: str | None = None,
        n_upsample: int = 0,
        heatmap_head_upsample: int | None = None,
        concat_channels: int = 0,
        head_type: Literal["1x1", "conv"] = "conv",
        head_drop_rate: float = 0.0,
        head_act_layer: LayerType = nn.SiLU,
        head_exp_ratio: float = 1.0,
        heatmap_prior: Literal[
            "softmax", "sigmoid"
        ] = "sigmoid",  # relative activation vs likelihood of class
        background_suppressor_type: Literal["conv", ""] = "conv",
        background_suppressor_n_layers: int = 3,
        background_suppressor_exp_ratio: float = 1.0,
        background_suppressor_act_layer: LayerType = nn.SiLU,
        mask_method: Literal["multiply", "gate", "mask", ""] = "mask",
        norm_heatmap: bool = True,
        pool: Literal["avg", "avgmax", "maskedavg", "lse", "convattention"] = "avg",
        lse_r: float = 5.0,
        softplus_beta: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__()

        # store params
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.in_size = in_size
        self.upsample_method = (
            upsample_method.casefold() if isinstance(upsample_method, str) else None
        )
        self.n_upsample = n_upsample
        self.heatmap_head_upsample = heatmap_head_upsample
        self.concat_channels = concat_channels
        self.head_exp_ratio = head_exp_ratio
        self.background_suppressor_exp_ratio = background_suppressor_exp_ratio
        self.mask_method = mask_method
        self.norm_heatmap_enabled = norm_heatmap

        # select upsampling method
        if self.upsample_method is not None:
            match self.upsample_method:
                case "convbilinear":
                    self.upsample, in_channels = self._create_convbilinear_upsample(**kwargs)
                case "pixelshuffle":
                    self.upsample, in_channels = self._create_pixelshuffle_upsample(**kwargs)

        else:
            self.upsample = nn.Identity()

        # fused heatmap head + upsampler: produces image-resolution heatmaps
        # directly (fused heatmap head). When set, the heatmap branch
        # below is replaced; the feature head / BAS / pools remain at in_size.
        if self.heatmap_head_upsample is not None:
            self.head_heatmap = ProgressiveFusionHead(
                in_channels=in_channels + self.concat_channels,
                num_classes=self.num_classes,
                in_size=self.in_size,
                upsample=self.heatmap_head_upsample,
            )
            # keep heatmap planes in [0, 1] with a per-plane min-max scale,
            # exactly like the low-res path (sigmoid + Norm2d below). Without
            # it the raw sigmoid output can collapse to all-zero planes: the
            # one-sided segmentation losses are then simultaneously zero and
            # the foreground-logit gradient path is severed, which is a
            # degenerate attractor of the summed objective.
            # Disable (norm_heatmap=False) when the segmentation loss is
            # two-sided (e.g. soft Dice): the loss then has no zero basin, and
            # the min-max norm de-calibrates the model's fixed 0.5 inference
            # threshold (it maps a peaked low-level plane onto a tiny mask).
            self.norm_heatmap = Norm2d() if norm_heatmap else None

        # create feature and heatmap heads
        heads = []
        for head_type, drop_rate in zip(2 * [head_type], 2 * [head_drop_rate]):
            match head_type:
                case "1x1":
                    head = self._create_head_1x1(
                        in_channels=in_channels + self.concat_channels,
                        dropout="pixelchannel",
                        drop_rate=drop_rate,
                    )
                case "conv":
                    head = self._create_head_conv(
                        in_channels=in_channels + self.concat_channels,
                        dropout="pixelchannel",
                        drop_rate=drop_rate,
                        act_layer=head_act_layer,
                    )

            heads.append(head)

        match heatmap_prior:
            case "softmax":
                act = [SoftmaxSpatial2d(), Norm2d()]
            case "sigmoid":
                act = [nn.Sigmoid()]

        self.head_feature = heads[0]
        if self.heatmap_head_upsample is None:
            self.head_heatmap = nn.Sequential(*[heads[1], *act])
        else:
            # keep the unused low-res head construction out of the module tree
            # so old checkpoints match strictly except the new fused head
            self.act_heatmap = act

        # create pooling layers
        pool_layers = []
        for _ in range(3):
            match pool:
                case "avg":
                    pool = nn.Sequential(*[nn.AdaptiveAvgPool2d(1), nn.Flatten()])
                case "max":
                    pool = nn.Sequential(*[nn.AdaptiveMaxPool2d(1), nn.Flatten()])
                case "avgmax":
                    pool = AdaptiveAvgMaxPool2d(output_size=1, flatten=True)
                case "maskedavg":
                    pool = MaskedAvgPool2d()
                case "lse":
                    pool = LogSumExpPool2DStable(r=lse_r, flatten=True)
                case "convattention":
                    pool = ConvAttentionPool2d(chs=self.num_classes, exp_ratio=2.0)

            pool_layers.append(pool)

        self.pool, self.pool_fg, self.pool_bg = pool_layers

        # create background_suppressor
        match background_suppressor_type:
            case "conv":
                self.background_suppressor = self._create_background_suppressor(
                    n_layers=background_suppressor_n_layers,
                    act_layer=background_suppressor_act_layer,
                )
            case "":
                self.background_suppressor = nn.Identity()

        # create activation in [0, inf]
        self.act = nn.Softplus(beta=softplus_beta)

        efficientnet_init_weights(self)

    def _create_convbilinear_upsample(self, **kwargs) -> nn.Sequential:
        blocks = []
        in_size, in_channels = self.in_size, self.in_channels
        for _ in range(self.n_upsample):
            # create block
            block = ConvBilinearUpsample(in_channels=in_channels, in_size=in_size, **kwargs)
            # get next block's input dimensions
            in_size, in_channels = block.get_outdim()

            blocks.append(block)

        return nn.Sequential(*blocks), in_channels

    def _create_pixelshuffle_upsample(
        self,
        **kwargs,
    ) -> tuple[PixelShuffleUpsample, int]:
        block = PixelShuffleUpsample(in_channels=self.in_channels, in_size=self.in_size, **kwargs)
        _out_size, out_channels = block.get_outdim()

        return block, out_channels

    def _create_background_suppressor(self, n_layers: int, act_layer: LayerType) -> nn.Sequential:
        layers = []
        for _ in range(n_layers):
            layer = EdgeResidual(
                in_chs=self.num_classes,
                out_chs=self.num_classes,
                exp_ratio=self.background_suppressor_exp_ratio,
                act_layer=act_layer,
            )
            layers.append(layer)

        return nn.Sequential(*layers)

    def _create_head_conv(
        self,
        in_channels: int,
        dropout: Literal["", "pixel", "channel", "pixelchannel"] = "",
        drop_rate: float = 0.0,
        act_layer: LayerType = None,
    ) -> nn.Sequential:
        # create convolution
        # (B, in_channels, H, W) -> (B, num_classes, H, W)
        conv = EdgeResidual(
            in_chs=in_channels,
            out_chs=self.num_classes,
            exp_ratio=self.head_exp_ratio,
            act_layer=act_layer,
        )

        # set dropout
        dropout = get_dropout(dropout, drop_rate)

        # create head as sequential
        head = nn.Sequential(*[dropout, conv])

        return head

    def _create_head_1x1(
        self,
        in_channels: int,
        dropout: Literal["", "pixel", "channel", "pixelchannel"] = "",
        drop_rate: float = 0.0,
    ) -> nn.Sequential:
        # create pointwise (1x1) convolution
        # (B, in_channels, H, W) -> (B, num_classes, H, W)
        conv1x1 = nn.Conv2d(in_channels=in_channels, out_channels=self.num_classes, kernel_size=1)

        # set dropout
        dropout = get_dropout(dropout, drop_rate)

        # set batch norm
        bn = nn.BatchNorm2d(self.num_classes)

        # create head as sequential
        head = nn.Sequential(*[dropout, conv1x1, bn])

        return head

    def _suppress_background(
        self, features: th.Tensor, heatmaps: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        # create fore/background masks
        match self.mask_method:
            case "multiply":
                masks_foreground = heatmaps
                masks_background = 1 - heatmaps
            case "gate":
                masks_foreground = heatmaps * (heatmaps >= 0.5)
                masks_background = heatmaps * (heatmaps < 0.5)
            case "mask":
                masks_foreground = (heatmaps >= 0.5).to(heatmaps.dtype)
                masks_background = (heatmaps < 0.5).to(heatmaps.dtype)
            case "":
                return [features] * 3

        # mask background features
        features_background = features.clone() * masks_background

        # suppress background
        features = self.background_suppressor(features)
        features_background = self.background_suppressor(features_background)

        # mask foreground features
        features_foreground = features.clone() * masks_foreground

        return features, features_foreground, features_background

    def _pool_features(
        self,
        features: th.Tensor,
        features_foreground: th.Tensor,
        features_background: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        # get classification logits from features
        logits = self.pool(features)
        # get classification logits from foreground features
        logits_foreground = self.pool_fg(features_foreground)

        # get pooled activations
        activations = self.act(logits.clone())
        # get pooled background activations
        activations_background = self.act(self.pool_bg(features_background))

        return logits, logits_foreground, activations, activations_background

    def forward(self, features: th.Tensor, *args: th.Tensor) -> SegmentCAMReturnType:
        # upsample features
        features = self.upsample(features)

        # concat other features along channel dimension
        for arg in args:
            features = th.concat([features, arg], dim=1)

        # apply heatmap head
        if self.heatmap_head_upsample is not None:
            # fused head+upsampler: image-resolution heatmaps, bypassing the
            # low-res path entirely (the low-res heatmaps are not needed for
            # suppression since the fused head consumed the same features)
            heatmaps = self.head_heatmap(features)
            if self.norm_heatmap is not None:
                heatmaps = self.norm_heatmap(heatmaps)

            # apply feature head (stays at in_size for BAS/pooling)
            features = self.head_feature(features)

            # suppression masks from the image-res heatmaps downsampled back
            # to in_size, preserving alignment with the feature maps. Not
            # detached: gradient flow matches the original architecture.
            # mask_method='mask' thresholds to binary, so no grad reaches the
            # heatmap head through the masks (as before); soft methods
            # ('multiply', 'gate') pass grad into the head exactly as the
            # unfused architecture did.
            heatmaps_low = resample(heatmaps, size=self.in_size, mode="bilinear")
            features, features_foreground, features_background = self._suppress_background(
                features, heatmaps_low
            )
        else:
            heatmaps = self.head_heatmap(features.clone())

            # apply feature head
            features = self.head_feature(features)

            # fuse masks with features and suppress background
            features, features_foreground, features_background = self._suppress_background(
                features, heatmaps
            )

        # pool features
        logits, logits_foreground, activations, activations_background = self._pool_features(
            features, features_foreground, features_background
        )

        return (logits, logits_foreground), (heatmaps, activations, activations_background)


class PatchClassifier(nn.Module):
    """
    Create the CLARiTy patch-classification head.
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        pooling_strategy: str = "gwrp",
        decay_parameter: float = 0.996,
    ) -> None:
        # the pooling strategy is one of gap or grwp
        super().__init__()

        # validate the pooling strategy is accepted
        if pooling_strategy not in ["gap", "gwrp"]:
            raise ValueError(
                f"Pooling Strategy {pooling_strategy} is not accepted in the CAM Module"
            )

        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.pooling_strategy = pooling_strategy
        self.decay_parameter = decay_parameter

        # create the convolutional layer
        self.conv = nn.Conv2d(self.embed_dim, self.num_classes, kernel_size=3, stride=1, padding=1)

        # create the GAP/GWRP layer
        if self.pooling_strategy == "gap":
            # input is:  B x C x W x H
            # output is: B x C x 1 x 1
            # i.e. we are getting a single value per class per image
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif self.pooling_strategy != "gwrp":
            raise NotImplementedError(f"{self.pooling_strategy} is not yet implemented")

        # intialise the module
        self.conv.apply(_init_weights)

    def forward(
        self,
        x_patch_tokens: th.Tensor,
        w: int,
        h: int,
        patch_size: tuple[int, int],
        device: str = "cpu",
    ):
        # get device
        device = x_patch_tokens.device

        # n is number of images, p is number of patch tokens (per image), c is the dimensionality of the patch tokens (num classes)
        n, p, c = x_patch_tokens.shape

        # reshape the patch tokens into a 2D feature map: NxNxD
        if w != h:
            # account for the fact that the image may not be square (e.g. when we are performing inference on a single image)
            w0 = w // patch_size[0]
            h0 = h // patch_size[0]
            x_patch_tokens = th.reshape(x_patch_tokens, [n, w0, h0, c])
        else:
            x_patch_tokens = th.reshape(x_patch_tokens, [n, int(p**0.5), int(p**0.5), c])

        # change the patch token dimensionality: BxWxHxC -> BxCxWxH
        x_patch_tokens = x_patch_tokens.permute([0, 3, 1, 2])

        # move the patch tokens to contiguous memory - stops all the above reshapes from throwing the data around
        x_patch_tokens = x_patch_tokens.contiguous()

        # pass the reshaped patch tokens through a convolutional layer (the output of this will be used to get the CAMs)
        x_patch_tokens = self.conv(x_patch_tokens)

        # perform GAP/GWRP on the patch tokens
        if self.pooling_strategy == "gap":
            # squeeze the final 1 x 1 dimensions away
            x_patch_logits = self.pool(x_patch_tokens).squeeze(3).squeeze(2)
        elif self.pooling_strategy == "gwrp":
            # flatten patch tokens into 2 main dimensions (3rd is the embedding)
            x_patch_flattened = x_patch_tokens.view(
                x_patch_tokens.shape[0], x_patch_tokens.shape[1], -1
            ).permute(0, 2, 1)

            # sort the patches according to their activations
            sorted_patch_token, indices = th.sort(x_patch_flattened, -2, descending=True)

            # assign weights based on the activation ranking
            weights = th.logspace(
                start=0,
                end=x_patch_flattened.size(-2) - 1,
                steps=x_patch_flattened.size(-2),
                base=self.decay_parameter,
            ).to(device)

            # take the weighted sum to get the patch logits
            x_patch_logits = (
                th.sum(sorted_patch_token * weights.unsqueeze(0).unsqueeze(-1), dim=-2)
                / weights.sum()
            )
        else:
            raise NotImplementedError(f"{self.pooling_strategy} is not implemented")

        # get the feature maps to use for the CAM
        feature_map = (
            x_patch_tokens.detach().clone()
        )  # B * C * 14 * 14 (14*14 is just the patch size, which could be 16*16)
        feature_map = F.relu(feature_map)

        return feature_map, x_patch_logits


#: Historical name kept as an alias for external checkpoints/imports.
PatchCAM = PatchClassifier
