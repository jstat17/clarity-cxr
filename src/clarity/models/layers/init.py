"""Shared module weight initialisation helpers."""

from __future__ import annotations

import torch.nn as nn
from timm.layers import trunc_normal_


def _init_weights(module: nn.Module) -> None:
    """Initialise transformer Linear/LayerNorm weights (truncated normal)."""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.bias, 0)
        nn.init.constant_(module.weight, 1.0)
