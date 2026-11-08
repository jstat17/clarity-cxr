"""CLARiTy weight-initialization helpers and positional-embedding resize.

No weight download or model import
happens here.
"""

from __future__ import annotations

import math
from functools import partial
from typing import Callable

import torch
import torch.nn as nn
from timm.layers import lecun_normal_, resample_abs_pos_embed, trunc_normal_


def init_weights_vit_timm(module: nn.Module, name: str = "") -> None:
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, "init_weights"):
        module.init_weights()


def init_weights_vit_jax(module: nn.Module, name: str = "", head_bias: float = 0.0) -> None:
    """ViT weight initialization, matching JAX (Flax) impl"""
    if isinstance(module, nn.Linear):
        if name.startswith("head"):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        else:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                (
                    nn.init.normal_(module.bias, std=1e-6)
                    if "mlp" in name
                    else nn.init.zeros_(module.bias)
                )
    elif isinstance(module, nn.Conv2d):
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, "init_weights"):
        module.init_weights()


def init_weights_vit_moco(module: nn.Module, name: str = "") -> None:
    """ViT weight initialization, matching moco-v3 impl minus fixed PatchEmbed"""
    if isinstance(module, nn.Linear):
        if "qkv" in name:
            # treat the weights of Q, K, V separately
            val = math.sqrt(6.0 / float(module.weight.shape[0] // 3 + module.weight.shape[1]))
            nn.init.uniform_(module.weight, -val, val)
        else:
            nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, "init_weights"):
        module.init_weights()


def get_init_weights_vit(mode: str = "jax", head_bias: float = 0.0) -> Callable:
    if "jax" in mode:
        return partial(init_weights_vit_jax, head_bias=head_bias)
    elif "moco" in mode:
        return init_weights_vit_moco
    else:
        return init_weights_vit_timm


def resize_pos_embed(
    posemb: torch.Tensor,
    posemb_new: torch.Tensor,
    num_prefix_tokens: int = 1,
    gs_new: tuple[int, int] = (),
    interpolation: str = "bicubic",
    antialias: bool = False,
) -> torch.Tensor:
    """Rescale the grid of position embeddings when loading from state_dict.
    *DEPRECATED* This function is being deprecated in favour of using resample_abs_pos_embed
    """
    ntok_new = posemb_new.shape[1] - num_prefix_tokens
    ntok_old = posemb.shape[1] - num_prefix_tokens
    gs_old = [int(math.sqrt(ntok_old))] * 2
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    return resample_abs_pos_embed(
        posemb,
        gs_new,
        gs_old,
        num_prefix_tokens=num_prefix_tokens,
        interpolation=interpolation,
        antialias=antialias,
        verbose=True,
    )


__all__ = [
    "get_init_weights_vit",
    "init_weights_vit_jax",
    "init_weights_vit_moco",
    "init_weights_vit_timm",
    "resize_pos_embed",
]
