"""CLARiTy Vision Transformer in PyTorch

A PyTorch implement of Vision Transformers as described in:

'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale'
    - https://arxiv.org/abs/2010.11929

`How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers`
    - https://arxiv.org/abs/2106.10270

`FlexiViT: One Model for All Patch Sizes`
    - https://arxiv.org/abs/2212.08013

The official jax code is released and available at
  * https://github.com/google-reference/vision_transformer
  * https://github.com/google-reference/big_vision

Based on the timm library's implementation of the Vision Transformer
    - [https://github.com/huggingface/pytorch-image-models]
"""

import math
from functools import partial
from typing import Callable, Iterable, Literal, Optional, Type, Union

import torch
import torch.nn as nn
from torch.jit import Final
import torch.nn.functional as F
from timm.layers import (
    AttentionPoolLatent,
    DropPath,
    LayerType,
    Mlp,
    PatchDropout,
    PatchEmbed,
    get_act_layer,
    get_norm_layer,
    resample_abs_pos_embed,
    trunc_normal_,
    use_fused_attn,
)
from timm.models._features import feature_take_indices
from timm.models._manipulate import checkpoint_seq, named_apply

from clarity.data.augment import fuse_multiply
from clarity.utils.types import LogitMask
from clarity.models.layers.segmentcam import PatchClassifier, SegmentCAM
from clarity.models.layers.norm import Norm2d
from clarity.models.layers.pool import AdaptiveAvgMaxPool1d, AttentionPool1d
from clarity.models.clarity.init import get_init_weights_vit, init_weights_vit_timm

CLARiTyReturnType = Union[
    torch.Tensor,
    tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ],
    tuple[torch.Tensor, torch.Tensor],
]


def _slice_class_dim(tensor: Optional[torch.Tensor], cls_mask) -> Optional[torch.Tensor]:
    """Select the active classes from a ``[B, C, ...]`` tensor."""
    return None if tensor is None else tensor[:, cls_mask]

class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        num_prefix_tokens: int = 0,
        num_class_token: int = 0,
        mask_cls_to_cls: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        # handle masking args
        self.num_prefix_tokens = num_prefix_tokens
        self.num_class_token = num_class_token
        self.mask_cls_to_cls = mask_cls_to_cls

    def forward(
        self, x: torch.Tensor, full_attn: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Attention forward with two exact modes.

        Fast mode (default, used by all training/inference/sweep paths):
        the value-weighted output is computed with fused scaled-dot-product
        attention (the ``[B, heads, N, N]`` score matrix is never
        materialized), and the returned attention is the compact
        class/register-to-patch rows ``[B, heads, T, P]``. Softmax is
        row-wise, so these rows are exactly the corresponding rows of the
        full-matrix attention; the patch columns are sliced AFTER the
        softmax over all N keys so the row normalization is unchanged.

        Full mode (``full_attn=True``, used by ``forward_intermediates``):
        the legacy eager computation returning the complete
        ``[B, heads, N, N]`` attention matrix; bitwise-identical to the
        pre-optimization implementation.
        """
        B, N, C = x.shape
        device = x.device
        dtype = x.dtype
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        # get mask (over all N rows; only needed when class-to-class
        # attention masking is enabled, otherwise it is all zeros)
        apply_mask = self.num_prefix_tokens > 0 and self.mask_cls_to_cls
        mask = torch.zeros(N, N, device=device, dtype=dtype)
        if apply_mask:
            class_indices = torch.arange(self.num_prefix_tokens, device=device)
            mask[class_indices[:, None], class_indices] = float("-inf")
            mask[class_indices, class_indices] = 0  # allow self-attention

        if full_attn:
            # legacy full-matrix path (bitwise-identical to the
            # pre-optimization implementation)
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn + mask
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            out = attn @ v
            out = out.transpose(1, 2).reshape(B, N, C)
            out = self.proj(out)
            out = self.proj_drop(out)
            return out, attn

        # fast path: value output via fused SDPA. SDPA scales q internally,
        # so the unscaled q is passed with an explicit matching scale. The
        # all-zeros mask is an exact no-op, so it is only passed when the
        # class-to-class mask is active.
        attn_drop_p = self.attn_drop.p if self.training else 0.0
        out = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask[None, None, :, :] if apply_mask else None,
            dropout_p=attn_drop_p,
            scale=self.scale,
        )

        # compact class/register-to-patch attention rows: only these rows
        # are ever consumed downstream (inference fusion, localization
        # sweep). Row-slice the scores, softmax over
        # ALL N keys (row-wise ⇒ exactly the full-matrix rows), then slice
        # the patch columns after the softmax.
        q_cls = q[:, :, self.num_class_token : self.num_prefix_tokens, :] * self.scale
        rows = q_cls @ k.transpose(-2, -1)  # [B, heads, T, N]
        if apply_mask:
            rows = rows + mask[self.num_class_token : self.num_prefix_tokens, :][None, None, :, :]
        rows = rows.softmax(dim=-1)
        rows = rows[..., self.num_prefix_tokens :]  # [B, heads, T, P]
        # NOTE: with attn_drop > 0 in training, the dropout draws here are
        # applied to a [B, heads, T, P] tensor instead of the legacy
        # [B, heads, N, N] one, so results are statistically equivalent to
        # the legacy path but not bitwise (identical when attn_drop == 0).
        rows = self.attn_drop(rows)

        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        return out, rows

class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: float = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_bias: bool = True,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        mlp_layer: Type[nn.Module] = Mlp,
        num_prefix_tokens: int = 0,
        num_class_token: int = 0,
        mask_cls_to_cls: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            num_prefix_tokens=num_prefix_tokens,
            num_class_token=num_class_token,
            mask_cls_to_cls=mask_cls_to_cls,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            bias=proj_bias,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self, x: torch.Tensor, full_attn: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # no clone needed: norm/ls/drop_path allocate new tensors, nothing
        # mutates x in place (B0 optimization, bitwise-identical)
        shortcut = x
        x = self.norm1(x)
        x, attn = self.attn(x, full_attn=full_attn)
        x = self.ls1(x)
        x = self.drop_path1(x)
        x = shortcut + x

        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.ls2(x)
        x = self.drop_path2(x)
        x = shortcut + x

        return x, attn

def global_pool_nlc(
    x: torch.Tensor,
    pool_type: str = "token",
    num_prefix_tokens: int = 1,
    reduce_include_prefix: bool = False,
):
    if not pool_type:
        return x

    if pool_type == "token":
        x = x[:, 0]  # class token
    else:
        x = x if reduce_include_prefix else x[:, num_prefix_tokens:]
        if pool_type == "avg":
            x = x.mean(dim=1)
        elif pool_type == "avgmax":
            x = 0.5 * (x.amax(dim=1) + x.mean(dim=1))
        elif pool_type == "max":
            x = x.amax(dim=1)
        else:
            assert not pool_type, f"Unknown pool type {pool_type}"

    return x

class CLARiTy(nn.Module):
    """Class Localizing and Attention Refining Image Transformer (CLARiTy)"""

    dynamic_img_size: Final[bool]

    # Documented capability attribute (see localization.sweep's
    # SWEEP_CAPABILITY_ATTRIBUTE documentation): this class supports the
    # opt-in multi-variant localization attention sweep via the keyword-only
    # ``localization_layer_counts`` argument of :meth:`forward`.
    supports_localization_sweep = True
    # Convenience alias with identical semantics.
    supports_localization_attention_layer_sweep = True

    #: Inference attention types accepted by the sweep-mode override (and the
    #: ordinary inference fusion path). The standard acceptance path is
    #: ``mask``.
    SUPPORTED_INFERENCE_ATTENTION_TYPES = frozenset(
        {"attention", "cam", "multiply", "gate", "mask"}
    )

    def __init__(
        self,
        img_size: Union[int, tuple[int, int]] = 224,
        patch_size: Union[int, tuple[int, int]] = 16,
        model_size: Literal["T", "S"] = "S",
        in_chans: int = 3,
        num_classes: int = 1000,
        global_pool: Literal["", "avg", "avgmax", "max", "token", "map"] = "token",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        proj_bias: bool = True,
        init_values: Optional[float] = None,
        class_token: bool = True,
        pos_embed: str = "learn",
        no_embed_class: bool = False,
        reg_tokens: int = 0,
        pre_norm: bool = False,
        final_norm: bool = True,
        fc_norm: Optional[bool] = None,
        dynamic_img_size: bool = False,
        dynamic_img_pad: bool = False,
        drop_rate: float = 0.0,
        pos_drop_rate: float = 0.0,
        patch_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        weight_init: Literal["skip", "jax", "jax_nlhb", "moco", ""] = "",
        fix_init: bool = False,
        embed_layer: Callable = PatchEmbed,
        embed_norm_layer: Optional[LayerType] = None,
        norm_layer: Optional[LayerType] = None,
        act_layer: Optional[LayerType] = None,
        block_fn: Type[nn.Module] = Block,
        mlp_layer: Type[nn.Module] = Mlp,
        use_segment_cam: bool = True,
        segment_cam_head_type: Literal["1x1", "conv"] = "conv",
        segment_cam_head_exp_ratio: float = 2.0,
        segment_cam_pool: Literal["avg", "avgmax", "maskedavg", "lse", "convattention"] = "avg",
        segment_cam_background_suppressor_type: Literal["conv", ""] = "conv",
        segment_cam_background_suppressor_exp_ratio: float = 2.0,
        segment_cam_mask_method: Literal["multiply", "gate", "mask", ""] = "mask",
        segment_cam_upsample: int | None = 16,
        segment_cam_norm_heatmap: bool = True,
        use_patch_cam: bool = False,
        class_token_pool: Literal["avg", "avgmax", "attention"] = "attention",
        only_classify: bool = False,
        n_attn_layers_avg: int = 5,
        inference_attention_type: Literal["attention", "cam", "multiply", "gate", "mask", ""] = "",
        mask_cls_to_cls: bool = False,
        use_foreground_logits_for_preds: bool = True,
        logit_weights: list[float] = [1.0, 1.0, 1.0],  # aggregate token, class tokens, segmentcam
        logit_mask: Iterable[int] | str | slice | None = None,
        **kwargs,
    ) -> None:
        """
        Args:
            img_size: Input image size.
            patch_size: Patch size.
            in_chans: Number of image input channels.
            num_classes: Number of classes for classification head.
            logit_mask: Optional active-class mask applied to logits, heatmaps
                and attention maps. Accepts the forms documented by the
                ``LogitMask`` helper; also stored in the model for
                introspection and checkpoint compatibility.
            global_pool: Type of global pooling for final sequence (default: 'token').
            embed_dim: Transformer embedding dimension.
            depth: Depth of transformer.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: Enable bias for qkv projections if True.
            init_values: Layer-scale init values (layer-scale enabled if not None).
            class_token: Use class token.
            no_embed_class: Don't include position embeddings for class (or reg) tokens.
            reg_tokens: Number of register tokens.
            pre_norm: Enable norm after embeddings, before transformer blocks (standard in CLIP ViT).
            final_norm: Enable norm after transformer blocks, before head (standard in most ViT).
            fc_norm: Move final norm after pool (instead of before), if None, enabled when global_pool == 'avg'.
            drop_rate: Head dropout rate.
            pos_drop_rate: Position embedding dropout rate.
            attn_drop_rate: Attention dropout rate.
            drop_path_rate: Stochastic depth rate.
            weight_init: Weight initialization scheme.
            fix_init: Apply weight initialization fix (scaling w/ layer index).
            embed_layer: Patch embedding layer.
            embed_norm_layer: Normalization layer to use / override in patch embed module.
            norm_layer: Normalization layer.
            act_layer: MLP activation layer.
            block_fn: Transformer block layer.
        """
        super().__init__()
        global_pool = global_pool if class_token else ""
        assert global_pool in ("", "avg", "avgmax", "max", "token", "map")
        assert class_token or global_pool != "token"
        assert pos_embed in ("", "none", "learn")
        use_fc_norm = global_pool in ("avg", "avgmax", "max") if fc_norm is None else fc_norm
        norm_layer = get_norm_layer(norm_layer) or partial(nn.LayerNorm, eps=1e-6)
        embed_norm_layer = get_norm_layer(embed_norm_layer)
        act_layer = get_act_layer(act_layer) or nn.GELU

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.head_hidden_size = self.embed_dim = (
            embed_dim  # for consistency with other models
        )
        self.num_prefix_tokens = 1 if class_token else 0
        self.num_prefix_tokens += reg_tokens
        self.num_reg_tokens = reg_tokens
        self.has_class_token = class_token
        self.num_class_token = int(class_token)
        self.no_embed_class = no_embed_class  # don't embed prefix positions (includes reg)
        self.dynamic_img_size = dynamic_img_size
        self.grad_checkpointing = False
        self.fullname = f"CLARiTy-{model_size}-{patch_size}"
        self.spatial_dim_len = int(img_size // patch_size)
        self.use_segment_cam = use_segment_cam
        self.use_patch_cam = use_patch_cam
        self.only_classify = only_classify
        self.n_attn_layers_avg = n_attn_layers_avg
        self.inference_attention_type = inference_attention_type.casefold()
        self.mask_cls_to_cls = mask_cls_to_cls
        self.use_foreground_logits_for_preds = use_foreground_logits_for_preds
        self.logit_weights = logit_weights
        self.logit_mask = LogitMask(logit_mask)
        if isinstance(self.logit_mask.indices, list) and self.logit_mask.indices:
            invalid = [index for index in self.logit_mask.indices if index >= num_classes]
            if invalid:
                raise ValueError(
                    f"logit_mask indices {invalid} are outside the configured "
                    f"num_classes={num_classes}"
                )

        embed_args = {}
        if dynamic_img_size:
            # flatten deferred until after pos embed
            embed_args.update(dict(strict_img_size=False, output_fmt="NHWC"))
        if embed_norm_layer is not None:
            embed_args["norm_layer"] = embed_norm_layer
        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
            dynamic_img_pad=dynamic_img_pad,
            **embed_args,
        )
        num_patches = self.patch_embed.num_patches
        reduction = (
            self.patch_embed.feat_ratio() if hasattr(self.patch_embed, "feat_ratio") else patch_size
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        self.reg_token = nn.Parameter(torch.zeros(1, reg_tokens, embed_dim)) if reg_tokens else None
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        if not pos_embed or pos_embed == "none":
            self.pos_embed = None
        else:
            self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * 0.02)
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
            )
        else:
            self.patch_drop = nn.Identity()
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.Sequential(
            *[
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    proj_bias=proj_bias,
                    init_values=init_values,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    mlp_layer=mlp_layer,
                    num_prefix_tokens=self.num_prefix_tokens,
                    num_class_token=self.num_class_token,
                    mask_cls_to_cls=mask_cls_to_cls,
                )
                for i in range(depth)
            ]
        )
        self.feature_info = [
            dict(module=f"blocks.{i}", num_chs=embed_dim, reduction=reduction) for i in range(depth)
        ]
        self.norm = norm_layer(embed_dim) if final_norm and not use_fc_norm else nn.Identity()

        # Classifier Head
        if global_pool == "map":
            self.attn_pool = AttentionPoolLatent(
                self.embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
                act_layer=act_layer,
            )
        else:
            self.attn_pool = None
        self.fc_norm = norm_layer(embed_dim) if final_norm and use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = (
            nn.Linear(self.embed_dim, num_classes)
            if (num_classes > 0 and self.has_class_token)
            else nn.Identity()
        )

        if weight_init != "skip":
            self.init_weights(weight_init)
        if fix_init:
            self.fix_init_weight()

        # create SegmentCAM module
        if self.use_segment_cam:
            self.cam = SegmentCAM(
                in_channels=embed_dim,
                num_classes=self.num_classes,
                in_size=(self.spatial_dim_len, self.spatial_dim_len),
                head_drop_rate=drop_rate,
                head_type=segment_cam_head_type,
                head_exp_ratio=segment_cam_head_exp_ratio,
                background_suppressor_type=segment_cam_background_suppressor_type,
                background_suppressor_exp_ratio=segment_cam_background_suppressor_exp_ratio,
                mask_method=segment_cam_mask_method,
                pool=segment_cam_pool,
                heatmap_head_upsample=segment_cam_upsample,
                norm_heatmap=segment_cam_norm_heatmap,
            )

        # create PatchCAM module
        elif self.use_patch_cam:
            self.cam = PatchClassifier(embed_dim=embed_dim, num_classes=self.num_classes)

        # create attention norm
        self.norm_attn = Norm2d()

        # create class token head
        match class_token_pool:
            case "avg":
                self.class_head = nn.Sequential(
                    *[nn.Dropout(drop_rate), nn.AdaptiveAvgPool1d(1), nn.Flatten()]
                )
            case "avgmax":
                self.class_head = nn.Sequential(
                    *[
                        nn.Dropout(drop_rate),
                        AdaptiveAvgMaxPool1d(1, flatten=True),
                    ]
                )
            case "attention":
                self.class_head = AttentionPool1d(
                    num_classes=self.num_classes, embed_dim=embed_dim, drop_rate=drop_rate
                )

    def fix_init_weight(self):
        def rescale(param, _layer_id):
            param.div_(math.sqrt(2.0 * _layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def init_weights(self, mode: str = "") -> None:
        assert mode in ("jax", "jax_nlhb", "moco", "")
        head_bias = -math.log(self.num_classes) if "nlhb" in mode else 0.0
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        if self.reg_token is not None:
            nn.init.normal_(self.reg_token, std=1e-6)
        named_apply(get_init_weights_vit(mode, head_bias), self)

    def _init_weights(self, m: nn.Module) -> None:
        # this fn left here for compat with downstream users
        init_weights_vit_timm(m)

    @torch.jit.ignore
    def no_weight_decay(self) -> set:
        return {"pos_embed", "cls_token", "dist_token"}

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> dict:
        return dict(
            stem=r"^cls_token|pos_embed|patch_embed",  # stem and embed
            blocks=[(r"^blocks\.(\d+)", None), (r"^norm", (99999,))],
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self.grad_checkpointing = enable
        if hasattr(self.patch_embed, "set_grad_checkpointing"):
            self.patch_embed.set_grad_checkpointing(enable)

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.head

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ("", "avg", "avgmax", "max", "token", "map")
            if global_pool == "map" and self.attn_pool is None:
                assert False, "Cannot currently add attention pooling in reset_classifier()."
            elif global_pool != "map" and self.attn_pool is not None:
                self.attn_pool = None  # remove attention pooling
            self.global_pool = global_pool
        self.head = (
            nn.Linear(self.embed_dim, num_classes)
            if (num_classes > 0 and self.has_class_token)
            else nn.Identity()
        )

    def set_input_size(
        self,
        img_size: Optional[tuple[int, int]] = None,
        patch_size: Optional[tuple[int, int]] = None,
    ):
        """Method updates the input image resolution, patch size

        Args:
            img_size: New input resolution, if None current resolution is used
            patch_size: New patch size, if None existing patch size is used
        """
        prev_grid_size = self.patch_embed.grid_size
        self.patch_embed.set_input_size(img_size=img_size, patch_size=patch_size)
        if self.pos_embed is not None:
            num_prefix_tokens = 0 if self.no_embed_class else self.num_prefix_tokens
            num_new_tokens = self.patch_embed.num_patches + num_prefix_tokens
            if num_new_tokens != self.pos_embed.shape[1]:
                self.pos_embed = nn.Parameter(
                    resample_abs_pos_embed(
                        self.pos_embed,
                        new_size=self.patch_embed.grid_size,
                        old_size=prev_grid_size,
                        num_prefix_tokens=num_prefix_tokens,
                        verbose=True,
                    )
                )

        # Keep the SegmentCAM head's spatial bookkeeping in sync with the new
        # resolution. Its convolutions (ProgressiveFusionHead stages, feature
        # head, background suppressor) are resolution-agnostic, but the
        # in_size/out_size metadata drives the resample targets at forward
        # time: a model built at one resolution and resized via
        # set_input_size (the weights_img_size path) would otherwise produce
        # heatmaps at the build-time resolution while the features flow at
        # the new one, crashing the background-suppression mul (observed at
        # 512: features 32x32 vs heatmaps_low 14x14).
        if getattr(self, "use_segment_cam", False):
            new_grid = int(self.patch_embed.grid_size[0])
            self.img_size = img_size if img_size is not None else self.img_size
            self.spatial_dim_len = new_grid
            self.cam.in_size = (new_grid, new_grid)
            head_heatmap = getattr(self.cam, "head_heatmap", None)
            if head_heatmap is not None:
                head_heatmap.in_size = self.cam.in_size
                head_heatmap.out_size = (
                    new_grid * head_heatmap.upsample,
                    new_grid * head_heatmap.upsample,
                )

    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        if self.pos_embed is None:
            return x.view(x.shape[0], -1, x.shape[-1])

        if self.dynamic_img_size:
            B, H, W, C = x.shape
            prev_grid_size = self.patch_embed.grid_size
            pos_embed = resample_abs_pos_embed(
                self.pos_embed,
                new_size=(H, W),
                old_size=prev_grid_size,
                num_prefix_tokens=0 if self.no_embed_class else self.num_prefix_tokens,
            )
            x = x.view(B, -1, C)
        else:
            pos_embed = self.pos_embed

        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        if self.reg_token is not None:
            to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))

        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + pos_embed
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
            x = x + pos_embed

        return self.pos_drop(x)

    def forward_intermediates(
        self,
        x: torch.Tensor,
        indices: Optional[Union[int, list[int]]] = None,
        return_prefix_tokens: bool = False,
        norm: bool = False,
        stop_early: bool = False,
        output_fmt: str = "NCHW",
        intermediates_only: bool = False,
    ) -> Union[list[torch.Tensor], tuple[torch.Tensor, list[torch.Tensor]]]:
        """Forward features that returns intermediates.

        Args:
            x: Input image tensor
            indices: Take last n blocks if int, all if None, select matching indices if sequence
            return_prefix_tokens: Return both prefix and spatial intermediate tokens
            norm: Apply norm layer to all intermediates
            stop_early: Stop iterating over blocks when last desired intermediate hit
            output_fmt: Shape of intermediate feature outputs
            intermediates_only: Only return intermediate features
        Returns:

        """
        assert output_fmt in ("NCHW", "NLC"), "Output format must be one of NCHW or NLC."
        reshape = output_fmt == "NCHW"
        intermediates = []
        attns = []
        take_indices, max_index = feature_take_indices(len(self.blocks), indices)

        # forward pass
        B, _, height, width = x.shape
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        if torch.jit.is_scripting() or not stop_early:  # can't slice blocks in torchscript
            blocks = self.blocks
        else:
            blocks = self.blocks[: max_index + 1]
        for i, blk in enumerate(blocks):
            # run every block in full-matrix mode: this legacy diagnostic API
            # returns complete [B, heads, N, N] attention matrices, and any
            # fast-mode block earlier in the stack would contaminate the
            # token stream of downstream full-mode blocks with fp-level
            # reassociation noise (mid-1e-7). Full mode for all blocks
            # restores the pre-optimization behaviour bit-for-bit. Cost is
            # irrelevant here (diagnostics-only path).
            x, attn = blk(x, full_attn=True)
            if i in take_indices:
                # normalize intermediates with final norm layer if enabled
                intermediates.append(self.norm(x) if norm else x)
                # take attention
                attns.append(attn)

        # process intermediates
        if self.num_prefix_tokens:
            # split prefix (e.g. class, distill) and spatial feature tokens
            prefix_tokens = [y[:, 0 : self.num_prefix_tokens] for y in intermediates]
            intermediates = [y[:, self.num_prefix_tokens :] for y in intermediates]
        else:
            prefix_tokens = None

        if reshape:
            # reshape to BCHW output format
            H, W = self.patch_embed.dynamic_feat_size((height, width))
            intermediates = [
                y.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for y in intermediates
            ]
        if not torch.jit.is_scripting() and return_prefix_tokens and prefix_tokens is not None:
            # return_prefix not support in torchscript due to poor type handling
            intermediates = list(zip(intermediates, prefix_tokens))

        if intermediates_only:
            return intermediates

        x = self.norm(x)

        return x, intermediates, attns

    def prune_intermediate_layers(
        self,
        indices: Union[int, list[int]] = 1,
        prune_norm: bool = False,
        prune_head: bool = True,
    ):
        """Prune layers not required for specified intermediates."""
        take_indices, max_index = feature_take_indices(len(self.blocks), indices)
        self.blocks = self.blocks[: max_index + 1]  # truncate blocks
        if prune_norm:
            self.norm = nn.Identity()
        if prune_head:
            self.fc_norm = nn.Identity()
            self.reset_classifier(0, "")
        return take_indices

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        x = self.norm(x)
        return x

    def pool(self, x: torch.Tensor, pool_type: Optional[str] = None) -> torch.Tensor:
        if self.attn_pool is not None:
            x = self.attn_pool(x)
            return x
        pool_type = self.global_pool if pool_type is None else pool_type
        x = global_pool_nlc(x, pool_type=pool_type, num_prefix_tokens=self.num_prefix_tokens)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = self.pool(x)
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def _forward_sweep_backbone(
        self,
        x: torch.Tensor,
        H: int,
        W: int,
    ):
        """Run the optimized single-pass backbone loop.

        Executes ``patch_embed`` and every transformer block and SegmentCAM
        exactly once (shared by the ordinary and sweep paths). During each
        block:

        - the class/register-token features needed by training losses that
          consume class-token intermediates are retained;
        - the class-to-patch attention is sliced immediately from the full
          ``[N, N]`` matrix and reduced to its compact
          ``[B, num_heads, 1, P]`` form, after which the full matrix is
          discarded before the next block;
        - only the FINAL patch tokens needed by SegmentCAM are retained
          (the unused per-layer ``_patches`` stack of the previous
          ``forward`` implementation is no longer constructed).

        Returns:
            ``(x, class_patch_attns, classes, patch_tokens)`` where

            - ``x``: final normalized tokens ``[B, N, embed_dim]``;
            - ``class_patch_attns``: list of per-layer compact attention
              reductions ``[B, num_heads, num_class_rows, P]`` (one entry
              per block, in order; ``num_class_rows`` is the number of
              register/class-token rows sliced from the full matrix);
            - ``classes``: per-layer class/register-token intermediates
              ``[B, depth, num_register_tokens, embed_dim]``;
            - ``patch_tokens``: final patch tokens
              ``[B, embed_dim, H, W]``.
        """
        # embed
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        P = x.shape[1] - self.num_prefix_tokens
        del P
        class_patch_attns = []
        class_token_intermediates = []
        for blk in self.blocks:
            # fast mode: attn is already the compact class/register-to-patch
            # rows [B, heads, num_class_rows, P] (identical slice of the full
            # matrix the legacy path would have produced); the full [B, heads,
            # N, N] matrix is never materialized
            x, attn = blk(x)
            class_patch_attns.append(attn)
            # retain the class/register-token features needed by training
            # losses that consume class-token intermediates (normalized
            # exactly like the legacy forward_intermediates(norm=True) path)
            class_token_intermediates.append(
                self.norm(x)[:, self.num_class_token : self.num_prefix_tokens]
            )

        # apply final norm
        x = self.norm(x)

        # [B, depth, num_register_tokens, embed_dim]
        classes = torch.stack(class_token_intermediates, dim=1)
        del class_token_intermediates

        # retain only the final patch tokens needed by SegmentCAM
        # [B, embed_dim, H, W]
        patch_tokens = x[:, self.num_prefix_tokens :]
        patch_tokens = patch_tokens.reshape(x.shape[0], H, W, -1).permute(0, 3, 1, 2).contiguous()

        return x, class_patch_attns, classes, patch_tokens

    def _compute_single_variant_maps(
        self,
        logits: torch.Tensor,
        heatmaps: Optional[torch.Tensor],
        attn: torch.Tensor,
        inference_attention_type: str,
    ) -> torch.Tensor:
        """Compute one normalized localization map from shared components.

        This is the per-variant body of the existing inference fusion path in
        :meth:`forward` (a ``logits, infer_attn`` return). ``logits``,
        ``heatmaps`` and ``attn`` (the normalized class-to-patch attention at
        patch resolution) are computed once by the caller and reused for
        every requested variant.

        Args:
            logits: Weighted-average classification logits (unused by the
                fusion math itself; accepted for contract symmetry).
            heatmaps: SegmentCAM heatmap planes (image or patch resolution).
            attn: Normalized class-to-patch attention ``[B, C, H, W]`` at
                patch resolution.
            inference_attention_type: One of the supported inference types
                (``attention``, ``cam``, ``multiply``, ``gate``, ``mask``).

        Returns:
            The normalized localization map ``[B, C, H', W']`` for the
            requested type (patch resolution for ``attention``, otherwise the
            heatmap resolution), identical to the corresponding legacy
            single-value ``forward`` result.
        """
        # When the heatmap branch runs at image resolution (fused upsampling
        # head), the class-to-patch attention is still at patch resolution.
        # Upsample it (bicubic) before fusion.
        fusion_attn = attn
        if (
            heatmaps is not None
            and fusion_attn is not None
            and heatmaps.shape[-2:] != fusion_attn.shape[-2:]
        ):
            fusion_attn = (
                F.interpolate(
                    fusion_attn.float(),
                    size=heatmaps.shape[-2:],
                    mode="bicubic",
                    align_corners=False,
                )
                .clamp(min=0.0, max=1.0)
                .to(fusion_attn.dtype)
            )

        match inference_attention_type:
            case "attention":
                infer_attn = attn if fusion_attn is attn else fusion_attn
            case "cam":
                infer_attn = heatmaps
            case "multiply":
                fused_heatmaps = fuse_multiply(heatmaps, fusion_attn)
                infer_attn = fused_heatmaps
            case "gate":
                gated_heatmaps = heatmaps * (heatmaps > 0.5)
                fused_heatmaps = fuse_multiply(gated_heatmaps, fusion_attn)
                infer_attn = fused_heatmaps
            case "mask":
                foreground_mask = heatmaps > 0.5
                infer_attn = fuse_multiply(foreground_mask, fusion_attn)
            case _:
                infer_attn = None

        if infer_attn is None:
            raise ValueError(f"inference_attention_type {inference_attention_type!r} not handled")

        # normalize each variant independently, exactly like the legacy
        # single-value path (self.norm_attn applies an identical min-max
        # scale per batch/class plane, so flattening [V, B] into one call is
        # equivalent to normalizing each variant separately)
        infer_attn = self.norm_attn(infer_attn)

        return infer_attn

    def forward(
        self,
        x: torch.Tensor,
        *,
        localization_layer_counts: tuple[int, ...] | None = None,
        inference_attention_type_override: str | None = None,
    ) -> CLARiTyReturnType:
        """Run CLARiTy.

        Ordinary call (``model(images)``) preserves the historical return
        contract exactly:

        - ``only_classify=True``: ``logits``;
        - training / validation, no inference attention type: the nested
          tuple consumed by the training criterion (logit components,
          heatmaps / activations / background activations, the canonical
          class-to-patch attention map averaged over
          ``self.n_attn_layers_avg`` layers, and all class-token
          intermediates);
        - otherwise: ``(logits, infer_attn)`` with the fusion behavior of the
          configured ``self.inference_attention_type``.

        Sweep mode (keyword-only, opt-in):

        ``localization_layer_counts`` requests attention maps for several
        suffix-layer counts from a single backbone/SegmentCAM pass. It is
        valid only in evaluation mode and when ``only_classify`` is false.
        Counts are validated against ``len(self.blocks)``. The override must
        not mutate ``self.n_attn_layers_avg`` or
        ``self.inference_attention_type``.

        Returns a fixed tensor (not a dict):
        ``(logits, variant_maps)`` where ``variant_maps`` is
        ``[V, B, C, H, W]`` in the same order as the requested tuple.

        ``inference_attention_type_override`` (keyword-only) temporarily
        selects the map fusion type for a single call (e.g. the standard
        ``mask`` path) without mutating ``self.inference_attention_type``.
        It may only be combined with ``localization_layer_counts``.

        All requested variant maps, logits, masks, and heatmaps are computed
        from ONE backbone/SegmentCAM pass; the per-block class-to-patch
        attention is sliced and reduced immediately so the full ``[N, N]``
        attention matrix is never retained across blocks and the unused
        per-layer patch-token stack is never constructed.

        Args:
            x: Input image tensor ``[B, C, img_H, img_W]``.
            localization_layer_counts: Optional tuple of suffix-layer counts
                (1-based, each in ``[1, len(self.blocks)]``) for sweep mode.
            inference_attention_type_override: Optional single-call map
                fusion type override (sweep mode only).

        Returns:
            See the docstring above; the return type depends on the mode.
        """
        ## Validate sweep arguments
        if (localization_layer_counts is not None) != (
            inference_attention_type_override is not None
        ):
            raise ValueError(
                "localization_layer_counts and inference_attention_type_override "
                "must be supplied together (sweep mode requires both)"
            )
        if localization_layer_counts is not None:
            if not isinstance(localization_layer_counts, tuple):
                raise TypeError(
                    "localization_layer_counts must be a tuple of ints, got "
                    f"{type(localization_layer_counts).__name__}"
                )
            if len(localization_layer_counts) == 0:
                raise ValueError("localization_layer_counts must not be empty")
            for count in localization_layer_counts:
                if isinstance(count, bool) or not isinstance(count, int):
                    raise TypeError(
                        "localization_layer_counts must contain only ints, got " f"{count!r}"
                    )
            depth = len(self.blocks)
            for count in localization_layer_counts:
                if count < 1 or count > depth:
                    raise ValueError(
                        f"localization layer count {count} is outside the valid "
                        f"range [1, {depth}]"
                    )
            if self.training:
                raise ValueError(
                    "localization_layer_counts is only supported in "
                    "evaluation mode (call model.eval() first)"
                )
            if self.only_classify:
                raise ValueError(
                    "localization_layer_counts is not supported when " "only_classify=True"
                )

            inference_attention_type = inference_attention_type_override.casefold()
            if inference_attention_type not in self.SUPPORTED_INFERENCE_ATTENTION_TYPES:
                raise ValueError(
                    f"inference_attention_type_override "
                    f"{inference_attention_type_override!r} is not one of the "
                    f"supported inference attention types "
                    f"{sorted(self.SUPPORTED_INFERENCE_ATTENTION_TYPES)}"
                )

        ## Get features
        # get image shape
        B, _, img_H, img_W = x.shape
        # get feature shape
        H, W = self.patch_embed.dynamic_feat_size((img_H, img_W))

        # forward through the optimized backbone loop: returns the final
        # normalized tokens, per-layer CLASS-TO-PATCH attention reductions
        # (each [B, heads, num_class_rows, P], already sliced from the full
        # [N, N] matrix, which is discarded immediately inside every block),
        # the class-token intermediates, and the FINAL patch tokens
        x, class_patch_attns, classes, patch_tokens = self._forward_sweep_backbone(x, H, W)

        ## Get localization maps from the shared components
        # get SegmentCAM outputs
        if self.use_segment_cam:
            cls_outputs, segm_outputs = self.cam(patch_tokens)

            # unpack outputs
            logits_cam, logits_foreground = cls_outputs
            heatmaps, activations, activations_background = segm_outputs

        # get PatchCAM outputs
        elif self.use_patch_cam:
            # reshape patch_tokens to [B, N, embed_dim]
            patch_tokens = patch_tokens.reshape(B, -1, self.embed_dim).contiguous()
            heatmaps, logits_cam = self.cam(patch_tokens, img_W, img_H, self.patch_size)

            # create outputs
            logits_cam, logits_foreground = logits_cam, None
            heatmaps, activations, activations_background = heatmaps, None, None

        else:
            logits_cam, logits_foreground = None, None
            heatmaps, activations, activations_background = None, None, None

        # apply the active-class mask to every class-indexed map/logit before
        # assembling the components (a no-op for the default full mask)
        cls_mask = self.logit_mask.indices
        logits_cam = _slice_class_dim(logits_cam, cls_mask)
        logits_foreground = _slice_class_dim(logits_foreground, cls_mask)
        heatmaps = _slice_class_dim(heatmaps, cls_mask)
        activations = _slice_class_dim(activations, cls_mask)
        activations_background = _slice_class_dim(activations_background, cls_mask)
        # `classes`/`attn` carry one row per register token, which the factory
        # sets to num_classes; guard unusual custom constructions.
        if classes.shape[2] == self.num_classes:
            classes = classes[:, :, cls_mask]

        # [B, num_register_tokens, embed_dim]
        class_tokens = x[:, self.num_class_token : self.num_prefix_tokens]

        ## Get outputs
        # store logits and logit weights
        component_logits, component_weights = [], []

        # get aggregate token logits
        if self.has_class_token:
            logits_agg = self.forward_head(x)[:, cls_mask]
            component_logits.append(logits_agg)
            component_weights.append(self.logit_weights[0])

        # get class token logits
        logits_class = self.class_head(class_tokens)[:, cls_mask]
        component_logits.append(logits_class)
        component_weights.append(self.logit_weights[1])

        if self.use_segment_cam:
            if self.use_foreground_logits_for_preds:
                component_logits.append(logits_foreground)
            else:
                component_logits.append(logits_cam)
            component_weights.append(self.logit_weights[2])

        elif self.use_patch_cam:
            component_logits.append(logits_cam)
            component_weights.append(self.logit_weights[2])

        # get model logits by weighted average across logit components
        logits_sum, weights_sum = 0.0, 0.0
        for logits, weight in zip(component_logits, component_weights):
            logits_sum += logits * weight
            weights_sum += weight
        logits = logits_sum / weights_sum

        if self.only_classify:
            return logits

        # [B, L, heads, T, P] (T = num_prefix - num_class_token rows) ->
        # last n layers only
        attn = torch.stack(class_patch_attns, dim=1)[:, -self.n_attn_layers_avg :, :, :, :]
        # average across layers and heads, then reshape to 2D
        # [B, T, H, W]
        attn = attn.mean(dim=(1, 2)).reshape(B, -1, H, W)
        # normalize attention in [0,1]
        attn = self.norm_attn(attn)
        # mask active classes
        if attn.shape[1] == self.num_classes:
            attn = attn[:, cls_mask]

        ## Sweep mode: one pass, several suffix-layer means
        if localization_layer_counts is not None:
            # retain only the requested compact suffixes
            # [B, L, heads, T, P]
            attn_stack = torch.stack(class_patch_attns, dim=1)
            depth = len(self.blocks)
            suffix_attns = [
                attn_stack[:, depth - count :, :, :, :] for count in localization_layer_counts
            ]
            del attn_stack, class_patch_attns

            # mean over the suffix layers and heads, reshape to 2D, then
            # normalize independently per variant (equivalent to the legacy
            # single-count path)
            # [V, B, T, H, W]
            variant_attns = torch.stack(
                [suffix.mean(dim=(1, 2)).reshape(B, -1, H, W) for suffix in suffix_attns], dim=0
            )
            del suffix_attns
            # [V, B, T, H, W] -> [V*B, T, H, W], one norm_attn call
            V = variant_attns.shape[0]
            T = variant_attns.shape[2]
            flat = variant_attns.reshape(V * B, T, H, W)
            flat = self.norm_attn(flat)
            variant_attns = flat.reshape(V, B, T, H, W)
            del flat
            # apply the same active-class mask as the ordinary path
            if variant_attns.shape[2] == self.num_classes:
                variant_attns = variant_attns[:, :, cls_mask]

            # per-variant localization maps via the ONE shared per-variant
            # routine (the same code path as the ordinary single-value
            # inference fusion); variant attention is already normalized and
            # fusion inputs are shared across variants
            variant_maps = []
            for variant_attn in variant_attns.unbind(dim=0):
                variant_maps.append(
                    self._compute_single_variant_maps(
                        logits=logits,
                        heatmaps=heatmaps,
                        attn=variant_attn,
                        inference_attention_type=inference_attention_type,
                    )
                )

            # [V, B, C, H', W']
            variant_maps = torch.stack(variant_maps, dim=0)

            return logits, variant_maps

        ## Return per the ordinary contract
        if self.inference_attention_type == "":
            if self.training:
                return (
                    (logits_class, logits_cam, logits_foreground),
                    (heatmaps, activations, activations_background, attn, classes),
                )
            else:
                # During validation the loss may still be evaluated, so dummy
                # activation tensors must retain the pooled [batch, classes]
                # shape expected by BAS. Heatmaps are spatial
                # [batch, classes, height, width] tensors. With SegmentCAM
                # off (use_segment_cam=False) heatmaps/activations are None;
                # the dummy BAS tensors then fall back to the logits' pooled
                # [batch, classes] shape — every BAS term is skipped by the
                # loss's activations-is-None guard anyway.
                dummy_activations = (
                    torch.ones_like(activations)
                    if activations is not None
                    else torch.ones_like(logits)
                )
                dummy_background = (
                    torch.zeros_like(activations_background)
                    if activations_background is not None
                    else torch.zeros_like(logits)
                )
                return (
                    (logits, logits, logits),
                    (heatmaps, dummy_activations, dummy_background, attn, classes),
                )
        else:
            infer_attn = self._compute_single_variant_maps(
                logits=logits,
                heatmaps=heatmaps,
                attn=attn,
                inference_attention_type=self.inference_attention_type,
            )

            return logits, infer_attn
