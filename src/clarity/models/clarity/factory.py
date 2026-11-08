"""CLARiTy model factories (t-16, s-16, s-8)."""

from __future__ import annotations

import torch

from clarity.models._weights import load_compatible_weights
from clarity.models.clarity.model import CLARiTy
from clarity.models.clarity.weights import (
    _load_warm_start_weights,
    _load_weights_from_npz_url,
    _load_weights_from_radiodino,
)


def get_clarity_t_16(
    num_classes: int,
    weights: str | dict[str, torch.Tensor] | None = None,
    img_size: int = 224,
    weights_img_size: int | None = None,
    aggregate_token: bool = False,
    patch_size: int = 16,
    embed_dim: int = 192,
    depth: int = 12,
    num_heads: int = 3,
    drop_rate: float = 0.1,
    proj_drop_rate: float = 0.1,
    attn_drop_rate: float = 0.1,
    drop_path_rate: float = 0.1,
    **kwargs,
) -> CLARiTy:
    # specify image size of model to create
    if weights_img_size is not None:
        model_img_size = weights_img_size
    else:
        model_img_size = img_size

    # declare the model
    model = CLARiTy(
        img_size=model_img_size,
        patch_size=patch_size,
        model_size="T",
        num_classes=num_classes,
        class_token=aggregate_token,
        reg_tokens=num_classes,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        drop_rate=drop_rate,
        proj_drop_rate=proj_drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        **kwargs,
    )

    # handle weights
    match weights:
        case str():
            match weights.casefold():
                case "imagenet":
                    weights_url = "https://storage.googleapis.com/vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0.npz"

                    model = _load_weights_from_npz_url(model, weights_url=weights_url)

        case dict():
            model = _load_warm_start_weights(model, weights)

        case None | _:
            pass

    # update image and patch sizes after loading weights
    model.set_input_size(img_size, patch_size)

    return model


def get_clarity_s_16(
    num_classes: int,
    weights: str | dict[str, torch.Tensor] | None = None,
    img_size: int = 224,
    weights_img_size: int | None = None,
    aggregate_token: bool = False,
    patch_size: int = 16,
    embed_dim: int = 384,
    depth: int = 12,
    num_heads: int = 6,
    drop_rate: float = 0.1,
    proj_drop_rate: float = 0.1,
    attn_drop_rate: float = 0.1,
    drop_path_rate: float = 0.1,
    **kwargs,
) -> CLARiTy:
    # specify image size of model to create
    if weights_img_size is not None:
        model_img_size = weights_img_size
    else:
        model_img_size = img_size

    # declare the model
    model = CLARiTy(
        img_size=model_img_size,
        patch_size=patch_size,
        model_size="S",
        num_classes=num_classes,
        class_token=aggregate_token,
        reg_tokens=num_classes,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        drop_rate=drop_rate,
        proj_drop_rate=proj_drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        **kwargs,
    )

    # handle weights
    match weights:
        case str():
            match weights.casefold():
                case "imagenet":
                    match img_size:
                        case 224:
                            weights_url = "https://storage.googleapis.com/vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz"
                        case n if n >= 384:
                            weights_url = "https://storage.googleapis.com/vit_models/augreg/S_16-i1k-300ep-lr_0.001-aug_medium2-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_384.npz"

                    model = _load_weights_from_npz_url(model, weights_url=weights_url)

                case "dino":
                    weights_url = "https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"

                    # download weights
                    weights = torch.hub.load_state_dict_from_url(
                        url=weights_url,
                        map_location="cpu",
                        check_hash=True,
                    )

                    model = load_compatible_weights(model, weights)

                case "radiodino":
                    model = _load_weights_from_radiodino(model)

        case dict():
            model = _load_warm_start_weights(model, weights)

        case None | _:
            pass

    # update image and patch sizes after loading weights
    model.set_input_size(img_size, patch_size)

    return model


def get_clarity_s_8(
    num_classes: int,
    weights: str | dict[str, torch.Tensor] | None = None,
    weights_img_size: int | None = None,
    img_size: int = 224,
    aggregate_token: bool = False,
    patch_size: int = 8,
    embed_dim: int = 384,
    depth: int = 12,
    num_heads: int = 6,
    drop_rate: float = 0.1,
    proj_drop_rate: float = 0.1,
    attn_drop_rate: float = 0.1,
    drop_path_rate: float = 0.1,
    **kwargs,
) -> CLARiTy:
    # specify image size of model to create
    if weights_img_size is not None:
        model_img_size = weights_img_size
    else:
        model_img_size = img_size

    # declare the model
    model = CLARiTy(
        img_size=model_img_size,
        patch_size=patch_size,
        model_size="S",
        num_classes=num_classes,
        class_token=aggregate_token,
        reg_tokens=num_classes,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        drop_rate=drop_rate,
        proj_drop_rate=proj_drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        **kwargs,
    )

    # handle weights
    match weights:
        case str():
            match weights.casefold():
                case "imagenet":
                    weights_url = "https://storage.googleapis.com/vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz"

                    model = _load_weights_from_npz_url(model, weights_url=weights_url)

                case "dino":
                    weights_url = "https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth"

                    # download weights
                    weights = torch.hub.load_state_dict_from_url(
                        url=weights_url,
                        map_location="cpu",
                        check_hash=True,
                    )

                    model = load_compatible_weights(model, weights)

        case dict():
            model = _load_warm_start_weights(model, weights)

        case None | _:
            pass

    # update image and patch sizes after loading weights
    model.set_input_size(img_size, patch_size)

    return model
