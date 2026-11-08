"""CLARiTy checkpoint loading and download helpers.

Download libraries are imported
lazily so importing :mod:`clarity.models.clarity` stays network/dependency-light.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import numpy as np
import torch
from torch import nn

from clarity.models._weights import linearize_state_dict, load_compatible_weights

if TYPE_CHECKING:
    from clarity.models.clarity.model import CLARiTy

logger = logging.getLogger(__name__)


def _load_warm_start_weights(model: "CLARiTy", weights: dict) -> "CLARiTy":
    """Load warm-start weights, tolerating checkpoints that predate newer
    modules (e.g. SegmentCAM's fused heatmap head). Missing keys stay
    freshly initialized; dropped keys are the checkpoint's outdated ones.
    Warns loudly when any weights were NOT loaded, so silent random-init
    components cannot go unnoticed in evaluation workflows.
    """
    weights = linearize_state_dict(weights)
    result = model.load_state_dict(weights, strict=False)
    missing, unexpected = result.missing_keys, result.unexpected_keys
    if missing or unexpected:
        logger.info(
            f"[CLARiTy] warm-start load: {len(missing)} key(s) not in "
            f"checkpoint (freshly initialized, e.g. "
            f"{sorted(missing)[0] if missing else 'n/a'}), "
            f"{len(unexpected)} checkpoint key(s) not in model "
            f"(dropped, e.g. "
            f"{sorted(unexpected)[0] if unexpected else 'n/a'})"
        )
    return model


def _load_weights_from_radiodino(model: CLARiTy) -> CLARiTy:
    """Initialize CLARiTy from the RadioDINO-S/16 timm checkpoint."""
    import timm

    radiodino = timm.create_model(
        "hf_hub:Snarcy/RadioDino-s16",
        pretrained=True,
    )
    return load_compatible_weights(model, radiodino.state_dict())


def _bfloat16_to_float32(array) -> np.ndarray:
    """Convert bfloat16-backed arrays to float32 without jax.

    NumPy has no native bfloat16 dtype.  If the checkpoint was loaded through
    an ml_dtypes-aware NumPy the array already carries a bfloat16 dtype;
    otherwise raise a clear error instead of importing jax.
    """
    array = np.asarray(array)
    try:
        return array.astype(np.float32)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "bfloat16 checkpoint arrays require ml_dtypes installed in the "
            "active NumPy environment"
        ) from exc


@torch.no_grad()
def _load_weights_from_npz_url(
    model: nn.Module,
    weights_url: str,
    prefix: str = "",
    load_bfloat16: bool = False,
    print_warnings: bool = False,
) -> nn.Module:
    """
    Download and load weights from a URL into a PyTorch model, ignoring incompatible keys.

    Args:
        model: PyTorch model (nn.Module)
        weights_url: URL to the .npz checkpoint file
        prefix: Prefix for weight keys in the checkpoint (default: '')
        load_bfloat16: Whether to load weights as bfloat16 (default: False)
    """
    # Define Hugging Face cache directory
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    os.makedirs(cache_dir, exist_ok=True)

    # Extract filename from URL
    parsed_url = urlparse(weights_url)
    filename = os.path.basename(parsed_url.path)
    local_path = os.path.join(cache_dir, filename)

    # Download the file if it doesn't exist.  ``requests`` is imported here,
    # not at module import, so importing the model/package never pulls it in.
    if not os.path.exists(local_path):
        import requests

        logger.info(f"Downloading weights from {weights_url} to {local_path}")
        response = requests.get(weights_url, stream=True)
        if response.status_code == 200:
            with open(local_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        else:
            raise RuntimeError(
                f"Failed to download weights from {weights_url}. Status code: {response.status_code}"
            )

    # Helper function to convert numpy weights to PyTorch tensors
    def _n2p(_w, t=True, idx=None):
        if idx is not None:
            _w = _w[idx]

        if load_bfloat16:
            _w = _bfloat16_to_float32(_w)

        if _w.ndim == 4 and _w.shape[0] == _w.shape[1] == _w.shape[2] == 1:
            _w = _w.flatten()
        if t:
            if _w.ndim == 4:
                _w = _w.transpose([3, 2, 0, 1])
            elif _w.ndim == 3:
                _w = _w.transpose([2, 0, 1])
            elif _w.ndim == 2:
                _w = _w.transpose([1, 0])

        _w = torch.from_numpy(_w)
        return _w

    # Load the .npz file.  bfloat16 arrays need an ml_dtypes-aware NumPy;
    # jax is deliberately not required (and not imported at all).
    try:
        w = np.load(local_path, allow_pickle=True)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "cannot load this npz checkpoint with NumPy; bfloat16 weights "
            "need ml_dtypes installed (no jax dependency)"
        ) from exc

    # Determine prefix based on checkpoint structure
    big_vision = False
    if not prefix:
        if "opt/target/embedding/kernel" in w:
            prefix = "opt/target/"
        elif "params/embedding/kernel" in w:
            prefix = "params/"
            big_vision = True
        elif "params/img/embedding/kernel" in w:
            prefix = "params/img/"
            big_vision = True

    # Create a dictionary to store converted weights
    weights_dict = {}

    # Helper function to add weights to dictionary with error handling
    def add_weight(model_key, checkpoint_key, transpose=True, idx=None, flatten=None):
        try:
            weight = _n2p(w[checkpoint_key], t=transpose, idx=idx)
            if flatten is not None:
                weight = weight.flatten(flatten)
            weights_dict[model_key] = weight
            return True
        except KeyError:
            if print_warnings:
                logger.warning("key %s not found in checkpoint, skipping", checkpoint_key)
            return False
        except Exception as e:
            if print_warnings:
                logger.warning("error processing %s: %s, skipping", checkpoint_key, e)
            return False

    # Load weights for patch embedding
    add_weight("patch_embed.proj.weight", f"{prefix}embedding/kernel")
    add_weight("patch_embed.proj.bias", f"{prefix}embedding/bias")

    # Load cls_token if it exists
    if hasattr(model, "cls_token") and model.cls_token is not None:
        add_weight("cls_token", f"{prefix}cls", transpose=False)

    # Load positional embedding
    pos_embed_key = (
        f"{prefix}pos_embedding"
        if big_vision
        else f"{prefix}Transformer/posembed_input/pos_embedding"
    )
    add_weight("pos_embed", pos_embed_key, transpose=False)

    # Load transformer encoder norm
    add_weight("norm.weight", f"{prefix}Transformer/encoder_norm/scale")
    add_weight("norm.bias", f"{prefix}Transformer/encoder_norm/bias")

    # Load classification head
    if hasattr(model, "head") and isinstance(model.head, torch.nn.Linear):
        add_weight("head.weight", f"{prefix}head/kernel")
        add_weight("head.bias", f"{prefix}head/bias")

    # Load transformer blocks
    mha_sub, b_sub, ln1_sub = (0, 0, 1) if big_vision else (1, 3, 2)
    for i in range(len(model.blocks)):
        block_prefix = (
            f"{prefix}Transformer/encoderblock_{i}/"
            if f"{prefix}Transformer/encoderblock_{i}/" in w
            else f"{prefix}Transformer/encoderblock/"
        )
        idx = i if f"{prefix}Transformer/encoderblock/LayerNorm_0/scale" in w else None
        mha_prefix = block_prefix + f"MultiHeadDotProductAttention_{mha_sub}/"

        # LayerNorm 1
        add_weight(f"blocks.{i}.norm1.weight", f"{block_prefix}LayerNorm_0/scale", idx=idx)
        add_weight(f"blocks.{i}.norm1.bias", f"{block_prefix}LayerNorm_0/bias", idx=idx)

        # Attention: Load qkv weights and biases individually
        qkv_weights = []
        qkv_biases = []
        for n in ("query", "key", "value"):
            # Load kernel
            if add_weight(
                f"temp_{n}_kernel", f"{mha_prefix}{n}/kernel", transpose=False, idx=idx, flatten=1
            ):
                qkv_weights.append(weights_dict.pop(f"temp_{n}_kernel").T)
            # Load bias
            if add_weight(
                f"temp_{n}_bias", f"{mha_prefix}{n}/bias", transpose=False, idx=idx, flatten=-1
            ):
                qkv_biases.append(weights_dict.pop(f"temp_{n}_bias"))

        # Combine qkv weights and biases if all components are present
        if len(qkv_weights) == 3:
            weights_dict[f"blocks.{i}.attn.qkv.weight"] = torch.cat(qkv_weights)
        if len(qkv_biases) == 3:
            weights_dict[f"blocks.{i}.attn.qkv.bias"] = torch.cat(qkv_biases)

        # Attention projection
        add_weight(f"blocks.{i}.attn.proj.weight", f"{mha_prefix}out/kernel", idx=idx, flatten=1)
        add_weight(f"blocks.{i}.attn.proj.bias", f"{mha_prefix}out/bias", idx=idx)

        # LayerNorm 2
        add_weight(f"blocks.{i}.norm2.weight", f"{block_prefix}LayerNorm_{ln1_sub}/scale", idx=idx)
        add_weight(f"blocks.{i}.norm2.bias", f"{block_prefix}LayerNorm_{ln1_sub}/bias", idx=idx)

        # MLP
        for r in range(2):
            add_weight(
                f"blocks.{i}.mlp.fc{r + 1}.weight",
                f"{block_prefix}MlpBlock_{b_sub}/Dense_{r}/kernel",
                idx=idx,
            )
            add_weight(
                f"blocks.{i}.mlp.fc{r + 1}.bias",
                f"{block_prefix}MlpBlock_{b_sub}/Dense_{r}/bias",
                idx=idx,
            )

    # Load weights into model, ignoring incompatible keys
    model = load_compatible_weights(model, weights_dict)
    return model


__all__ = [
    "_bfloat16_to_float32",
    "_load_warm_start_weights",
    "_load_weights_from_npz_url",
    "_load_weights_from_radiodino",
]
