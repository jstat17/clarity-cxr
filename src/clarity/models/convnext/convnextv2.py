"""ConvNeXtV2-B teacher model (paper baseline; public companion).

ImageNet-pretrained ConvNeXtV2 teachers used to generate distillation labels.
Other ConvNeXt variants are not part of this release.
"""

from __future__ import annotations

from typing import Literal

import timm
import torch as th
from torch import nn

from clarity.utils.types import LogitMask
from clarity.models.layers.pool import LogSumExpPool2DStable
from clarity.models._weights import linearize_state_dict


class ConvNeXtV2(nn.Module):
    def __init__(
        self,
        model_size: Literal["B", "L"],
        num_classes: int,
        weights: str | None,
        r: float = 0.0,
        img_size: int = 224,
        drop_rate: float = 0.1,  # Head pre-classifier dropout rate.
        drop_path_rate: float = 0.1,  # Stochastic depth drop rate.
        **kwargs,
    ) -> None:
        super().__init__()

        # set model size
        match model_size:
            case "B":
                self.model_info = "convnextv2_base"
                self.fullname = "ConvNeXtV2-B"
            case "L":
                self.model_info = "convnextv2_large"
                self.fullname = "ConvNeXtV2-L"

        # specify weights
        if weights is not None:
            timm_model = self.model_info + weights
            pretrained = True
        else:
            timm_model = self.model_info
            pretrained = False

        # create timm model
        self.model = timm.create_model(
            model_name=timm_model,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )

        # set cfg
        self.model.default_cfg["crop_pct"] = 1.0
        self.model.default_cfg["test_crop_pct"] = 1.0
        self.model.default_cfg["input_size"] = (3, img_size, img_size)
        self.model.default_cfg["test_input_size"] = (3, img_size, img_size)
        self.model.default_cfg["mean"] = (0.0, 0.0, 0.0)
        self.model.default_cfg["std"] = (1.0, 1.0, 1.0)

        if r > 0:
            # use Log-Sum-Exp pooling
            self.model.head.global_pool = LogSumExpPool2DStable(r=r, flatten=False)

        # Optional active-class mask, matching CLARiTy's ``logit_mask`` so a
        # 14-class ChestX-ray14 checkpoint can serve as a teacher (or a
        # comparator) on a single-finding target dataset such as RSNA, where
        # the distillation labels are 1-dimensional and must match the logits.
        self.logit_mask = LogitMask(kwargs.get("logit_mask"))
        self.num_classes = num_classes

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.model(x)
        return x[:, self.logit_mask.indices]


class ConvNeXtV2_B(ConvNeXtV2):
    def __init__(self, **kwargs) -> None:
        super().__init__(model_size="B", **kwargs)


def _get_convnextv2_b(
    num_classes: int, weights: str | dict | None = None, img_size: int = 224, **kwargs
) -> ConvNeXtV2_B:
    """Build the ConvNeXtV2-B teacher with optional ImageNet weights."""
    if isinstance(weights, str):
        match weights.casefold():
            case "imagenet":
                loaded_weights = (
                    ".fcmae_ft_in22k_in1k" if img_size == 224 else ".fcmae_ft_in22k_in1k_384"
                )
            case _:
                loaded_weights = weights
    else:
        loaded_weights = None
    model = ConvNeXtV2_B(
        num_classes=num_classes,
        weights=loaded_weights,
        img_size=img_size,
        **kwargs,
    )
    if isinstance(weights, dict):
        model.load_state_dict(linearize_state_dict(weights))
    return model


def get_convnextv2_b(
    num_classes: int, weights: str | dict | None = None, img_size: int = 224, **kwargs
) -> ConvNeXtV2_B:
    return _get_convnextv2_b(
        num_classes=num_classes,
        weights=weights,
        img_size=img_size,
        **kwargs,
    )
