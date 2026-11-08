"""CLARiTy: the paper's Class Localizing and Attention Refining Transformer.

Public surface: the model class and the three factories.  Importing this
package does not download weights (the ``requests``/timm-hub loaders are lazy).
"""

from __future__ import annotations

from clarity.models.clarity.factory import get_clarity_s_8, get_clarity_s_16, get_clarity_t_16
from clarity.models.clarity.init import (
    get_init_weights_vit,
    init_weights_vit_jax,
    init_weights_vit_moco,
    init_weights_vit_timm,
    resize_pos_embed,
)
from clarity.models.clarity.model import (
    CLARiTy,
    CLARiTyReturnType,
    Attention,
    Block,
    LayerScale,
    global_pool_nlc,
)

__all__ = [
    "Attention",
    "Block",
    "CLARiTy",
    "CLARiTyReturnType",
    "LayerScale",
    "get_clarity_s_8",
    "get_clarity_s_16",
    "get_clarity_t_16",
    "get_init_weights_vit",
    "global_pool_nlc",
    "init_weights_vit_jax",
    "init_weights_vit_moco",
    "init_weights_vit_timm",
    "resize_pos_embed",
]
